
import os
import math
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# =========================
# 0) Hyper-params
# =========================
HIST_LEN = 96          # 历史长度（一天，15min 粒度）
PRED_LEN = 1           # 未来 15min，一步预测
STRIDE_STEPS = 1       # 每15min起报一次
BATCH_SIZE = 64
EPOCHS = 150
LR = 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATIENCE = 80

torch.manual_seed(0)
np.random.seed(0)


# =========================
# 1) Time features
# =========================
def build_time_features(timestamps: pd.Series):
    ts = pd.to_datetime(timestamps)
    month = ts.dt.month.values
    weekday = ts.dt.weekday.values
    hour = ts.dt.hour.values + ts.dt.minute.values / 60.0
    day_of_year = ts.dt.dayofyear.values

    feat = np.stack([
        np.sin(2 * np.pi * month / 12.0),
        np.cos(2 * np.pi * month / 12.0),
        np.sin(2 * np.pi * weekday / 7.0),
        np.cos(2 * np.pi * weekday / 7.0),
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * day_of_year / 365.0),
        np.cos(2 * np.pi * day_of_year / 365.0),
    ], axis=1).astype(np.float32)
    return feat


# =========================
# 2) RPE modules
# =========================
class RelativePositionBias(nn.Module):
    def __init__(self, n_heads=8, num_buckets=32, max_distance=128):
        super().__init__()
        self.n_heads = n_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, n_heads)

    def _relative_position_bucket(self, relative_position):
        num_buckets = self.num_buckets
        max_distance = self.max_distance

        n = -relative_position
        half = num_buckets // 2

        positive = (n > 0).long()
        n = torch.abs(n)

        max_exact = half // 2
        is_small = n < max_exact

        val_large = max_exact + (
            torch.log(n.float() / max_exact + 1e-6) /
            math.log(max_distance / max_exact) *
            (half - max_exact)
        ).long()
        val_large = torch.clamp(val_large, max=half - 1)

        buckets = torch.where(is_small, n, val_large)
        buckets = buckets + positive * half
        return buckets

    def forward(self, q_len, k_len):
        device = self.relative_attention_bias.weight.device
        context_position = torch.arange(q_len, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(k_len, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        rp_bucket = self._relative_position_bucket(relative_position)
        rp_bucket = rp_bucket.to(device)
        values = self.relative_attention_bias(rp_bucket)  # [q_len, k_len, heads]
        values = values.permute(2, 0, 1).contiguous()  # [heads, q_len, k_len]
        return values


class MultiHeadAttentionRPE(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, rpe_bias):
        B, Lq, _ = Q.size()
        _, Lk, _ = K.size()

        q = self.Wq(Q).view(B, Lq, self.n_heads, self.d_head).transpose(1, 2)
        k = self.Wk(K).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)
        v = self.Wv(V).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)
        logits = logits + rpe_bias.unsqueeze(0).to(logits.device)

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        out = self.Wo(out)
        return out


class EncoderLayerRPE(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttentionRPE(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rpe_bias):
        h = self.attn(x, x, x, rpe_bias)
        x = self.norm1(x + self.dropout(h))
        h2 = self.ffn(x)
        x = self.norm2(x + self.dropout(h2))
        return x


# =========================
# 3) Model
# =========================
class PatchTST_ST15(nn.Module):
    """
    与原方法保持一致：
    - 历史输入：天气 + 出力 + 时间
    - 未来输入：未来天气 + 未来时间
    - 历史 encoder + future <- history cross-attention
    仅将任务改为一步预测，因此 future 侧 patch_len=1。
    """
    def __init__(self, enc_in_hist, enc_in_future,
                 hist_len=96, pred_len=1,
                 d_model=128, n_heads=8, e_layers=3,
                 patch_len=1, dropout=0.1):
        super().__init__()

        assert hist_len % patch_len == 0, "hist_len must be divisible by patch_len"
        assert pred_len % patch_len == 0, "pred_len must be divisible by patch_len"

        self.hist_len = hist_len
        self.pred_len = pred_len
        self.patch_len = patch_len

        self.n_patch_hist = hist_len // patch_len
        self.n_patch_future = pred_len // patch_len

        self.hist_proj = nn.Linear(enc_in_hist, d_model)
        self.future_proj = nn.Linear(enc_in_future, d_model)

        self.patch_embed_hist = nn.Linear(patch_len * d_model, d_model)
        self.patch_embed_future = nn.Linear(patch_len * d_model, d_model)

        self.rpe_hist = RelativePositionBias(n_heads=n_heads)
        self.rpe_fut = RelativePositionBias(n_heads=n_heads)

        self.encoder_layers = nn.ModuleList([
            EncoderLayerRPE(d_model, n_heads, dropout)
            for _ in range(e_layers)
        ])

        self.cross_attn = MultiHeadAttentionRPE(d_model, n_heads, dropout)
        self.patch_proj = nn.Linear(d_model, patch_len * d_model)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, hist_x, fut_x):
        B = hist_x.size(0)

        hist = self.hist_proj(hist_x)
        fut = self.future_proj(fut_x)

        hist = hist.view(B, self.n_patch_hist, self.patch_len, -1)
        hist = self.patch_embed_hist(hist.reshape(B, self.n_patch_hist, -1))

        fut = fut.view(B, self.n_patch_future, self.patch_len, -1)
        fut = self.patch_embed_future(fut.reshape(B, self.n_patch_future, -1))

        rpe_hist = self.rpe_hist(self.n_patch_hist, self.n_patch_hist)
        for layer in self.encoder_layers:
            hist = layer(hist, rpe_hist)

        rpe_fut = self.rpe_fut(self.n_patch_future, self.n_patch_hist)
        fut = self.cross_attn(fut, hist, hist, rpe_fut)

        fut_seq = self.patch_proj(fut)
        fut_seq = fut_seq.view(B, self.n_patch_future, self.patch_len, -1)
        fut_seq = fut_seq.reshape(B, self.pred_len, -1)

        return self.output_proj(fut_seq)


# =========================
# 4) Dataset
# =========================
class PVShortTermDataset(Dataset):
    def __init__(self, Xh, Xf, Y):
        self.Xh = torch.tensor(Xh, dtype=torch.float32)
        self.Xf = torch.tensor(Xf, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.Xh)

    def __getitem__(self, idx):
        return self.Xh[idx], self.Xf[idx], self.Y[idx]


# =========================
# 5) Main
# =========================
def run_one_site(site_id, capacity):
    df = pd.read_excel(
        f"Solar station site {site_id} (Nominal capacity-{capacity}MW).xlsx"
    )
    df.columns = [str(c).strip() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if site_id in [1, 2, 3]:
        phys_features = ["TSI", "DNI", "GHI", "AT", "hPa"]
    else:
        phys_features = ["TSI", "DNI", "GHI", "AT", "hPa", "RH"]

    target = "Power"

    for c in phys_features + [target]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[phys_features] = df[phys_features].ffill().bfill()
    df[target] = df[target].ffill().bfill()
    df = df.dropna(subset=phys_features + [target]).reset_index(drop=True)

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_t = MinMaxScaler()

    x_scaled = scaler_x.fit_transform(df[phys_features]).astype(np.float32)
    y_scaled = scaler_y.fit_transform(df[[target]]).astype(np.float32)
    t_scaled = scaler_t.fit_transform(build_time_features(df["timestamp"])).astype(np.float32)

    def create_st15_samples(x, y, t, timestamps,
                            hist_len=HIST_LEN, pred_len=PRED_LEN, stride_steps=STRIDE_STEPS):
        X_hist, X_fut, Ys, pred_times = [], [], [], []
        N = len(x)

        for i in range(0, N - hist_len - pred_len + 1, stride_steps):
            x_hist = x[i:i + hist_len]
            y_hist = y[i:i + hist_len]
            t_hist = t[i:i + hist_len]

            x_fut = x[i + hist_len:i + hist_len + pred_len]
            y_fut = y[i + hist_len:i + hist_len + pred_len]
            t_fut = t[i + hist_len:i + hist_len + pred_len]

            hist_in = np.concatenate([x_hist, y_hist, t_hist], axis=-1)
            fut_in = np.concatenate([x_fut, t_fut], axis=-1)

            X_hist.append(hist_in)
            X_fut.append(fut_in)
            Ys.append(y_fut)
            pred_times.append(pd.to_datetime(timestamps.iloc[i + hist_len]))

        return (
            np.array(X_hist, dtype=np.float32),
            np.array(X_fut, dtype=np.float32),
            np.array(Ys, dtype=np.float32),
            pd.to_datetime(pred_times)
        )

    X_hist_all, X_fut_all, Y_all, pred_times_all = create_st15_samples(
        x_scaled, y_scaled, t_scaled, df["timestamp"]
    )

    N_samples = len(X_hist_all)
    if N_samples == 0:
        raise ValueError(f"[DATA] site {site_id}: no valid short-term samples.")

    split_idx = int(N_samples * 0.7)

    X_hist_train = X_hist_all[:split_idx]
    X_fut_train = X_fut_all[:split_idx]
    Y_train = Y_all[:split_idx]

    X_hist_val = X_hist_all[split_idx:]
    X_fut_val = X_fut_all[split_idx:]
    Y_val = Y_all[split_idx:]
    pred_times_val = pred_times_all[split_idx:]

    train_loader = DataLoader(
        PVShortTermDataset(X_hist_train, X_fut_train, Y_train),
        batch_size=BATCH_SIZE, shuffle=True
    )

    Cw = x_scaled.shape[1]
    Ct = t_scaled.shape[1]
    enc_in_hist = Cw + 1 + Ct
    enc_in_future = Cw + Ct

    model = PatchTST_ST15(
        enc_in_hist=enc_in_hist,
        enc_in_future=enc_in_future,
        hist_len=HIST_LEN,
        pred_len=PRED_LEN,
        d_model=128,
        n_heads=8,
        e_layers=3,
        patch_len=1,
        dropout=0.1
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    def evaluate_on_months(model_):
        model_.eval()
        preds_list, trues_list = [], []

        with torch.no_grad():
            for Xh, Xf, Y in zip(X_hist_val, X_fut_val, Y_val):
                Xh_t = torch.tensor(Xh, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                Xf_t = torch.tensor(Xf, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                out = model_(Xh_t, Xf_t)
                pred = out.squeeze(0).squeeze(-1).cpu().numpy()
                true = Y.squeeze(-1)

                preds_list.append(pred)
                trues_list.append(true)

        preds_arr = np.array(preds_list)
        trues_arr = np.array(trues_list)

        preds_inv = scaler_y.inverse_transform(preds_arr.reshape(-1, 1)).reshape(-1)
        trues_inv = scaler_y.inverse_transform(trues_arr.reshape(-1, 1)).reshape(-1)

        months = pd.Series(pred_times_val).dt.to_period("M")

        results = {}
        rmse_acc_list = []
        for m in np.unique(months):
            mask = (months == m).values
            if mask.sum() == 0:
                continue

            pred_m = preds_inv[mask]
            true_m = trues_inv[mask]

            mae = np.mean(np.abs(pred_m - true_m))
            rmse = np.sqrt(np.mean((pred_m - true_m) ** 2))

            acc_mae = 1 - mae / capacity
            acc_rmse = 1 - rmse / capacity

            results[str(m)] = {
                "1-MAE/Cap": acc_mae,
                "1-RMSE/Cap": acc_rmse
            }
            rmse_acc_list.append(acc_rmse)

        mean_rmse_acc = np.mean(rmse_acc_list) if len(rmse_acc_list) > 0 else -1e9
        return mean_rmse_acc, results

    best_acc = -1e9
    best_state = None
    patience_cnt = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for Xh_b, Xf_b, Y_b in train_loader:
            Xh_b = Xh_b.to(DEVICE)
            Xf_b = Xf_b.to(DEVICE)
            Y_b = Y_b.to(DEVICE)

            optimizer.zero_grad()
            out = model(Xh_b, Xf_b)
            loss = criterion(out, Y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / max(1, len(train_loader))
        val_mean_acc, _ = evaluate_on_months(model)
        print(f"Epoch {epoch+1}/{EPOCHS}   Loss = {train_loss:.6f}   Val_mean_1-RMSE/Cap = {val_mean_acc:.6f}")

        if val_mean_acc > best_acc:
            best_acc = val_mean_acc
            best_state = deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch+1}, best mean 1-RMSE/Cap = {best_acc:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds_list, trues_list = [], []

    with torch.no_grad():
        for Xh, Xf, Y in zip(X_hist_val, X_fut_val, Y_val):
            Xh_t = torch.tensor(Xh, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            Xf_t = torch.tensor(Xf, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            out = model(Xh_t, Xf_t)
            pred = out.squeeze(0).squeeze(-1).cpu().numpy()
            true = Y.squeeze(-1)

            preds_list.append(pred)
            trues_list.append(true)

    preds_arr = np.array(preds_list)
    trues_arr = np.array(trues_list)

    preds_inv = scaler_y.inverse_transform(preds_arr.reshape(-1, 1)).reshape(-1)
    trues_inv = scaler_y.inverse_transform(trues_arr.reshape(-1, 1)).reshape(-1)

    months = pd.Series(pred_times_val).dt.to_period("M")
    results = {}
    for m in np.unique(months):
        mask = (months == m).values
        if mask.sum() == 0:
            continue

        pred_m = preds_inv[mask]
        true_m = trues_inv[mask]

        mae = np.mean(np.abs(pred_m - true_m))
        rmse = np.sqrt(np.mean((pred_m - true_m) ** 2))

        acc_mae = 1 - mae / capacity
        acc_rmse = 1 - rmse / capacity

        results[str(m)] = {
            "1-MAE/Cap": acc_mae,
            "1-RMSE/Cap": acc_rmse
        }

    if len(preds_inv) > 0:
        print("示例 pred_inv[:10] =", preds_inv[:10])
        print("示例 true_inv[:10] =", trues_inv[:10])

    result_df = pd.DataFrame({
        "month": list(results.keys()),
        "1-MAE/Cap": [v["1-MAE/Cap"] for v in results.values()],
        "1-RMSE/Cap": [v["1-RMSE/Cap"] for v in results.values()],
    })

    print("\n========== PatchTST 风格（历史+未来条件 15min 预测 + 按月测试 + EarlyStopping）==========")
    print(result_df)

    save_dir = "15min_PatchTST_short_term"
    os.makedirs(save_dir, exist_ok=True)
    result_df.to_excel(f"{save_dir}/PREPatchTST_site_{site_id}_15min_earlystop.xlsx", index=False)

    with open(f"{save_dir}/PatchTST_site_{site_id}_15min_data.pkl", "wb") as f:
        pickle.dump({
            "pred": preds_inv,
            "true": trues_inv,
            "pred_time": pd.Series(pred_times_val),
            "month_metrics": results,
            "best_mean_1-RMSE/Cap": best_acc,
        }, f)


site_info = {
    1: 50,
    2: 130,
    # 3: 30,
    4: 130,
    5: 110,
    6: 35,
    7: 30,
    8: 30,
}

if __name__ == "__main__":
    for sid, cap in site_info.items():
        print(f"\n========== Running site {sid} ==========")
        run_one_site(sid, cap)
