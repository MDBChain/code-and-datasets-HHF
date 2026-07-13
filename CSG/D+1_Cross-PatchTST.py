import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import pickle
from copy import deepcopy
import math

# =============== 参数 ===============
HIST_LEN = 96       # 历史长度（一天）
PRED_LEN = 96       # 预测 D+1（一天）
STRIDE_DAYS = 1     # 两天一组，按天滑动
BATCH_SIZE = 64
EPOCHS = 150
LR = 5e-4
# site_id = 2
# capacity = 130
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATIENCE = 80       # 早停


def build_time_features(timestamps: pd.Series):
    """
    最优时间特征编码：
    - month（12周期）
    - weekday（7周期）
    - hour（24周期）
    - day_of_year（365周期）
    全部采用 sin/cos 编码，让模型感知周期性结构。
    """
    ts = timestamps.dt

    # 基本字段
    month = ts.month.values
    weekday = ts.weekday.values
    hour = ts.hour.values
    day_of_year = ts.dayofyear.values

    # sin/cos 编码（周期特征）
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    weekday_sin = np.sin(2 * np.pi * weekday / 7)
    weekday_cos = np.cos(2 * np.pi * weekday / 7)

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    doy_sin = np.sin(2 * np.pi * day_of_year / 365)
    doy_cos = np.cos(2 * np.pi * day_of_year / 365)

    # 组合特征
    features = np.stack(
        [
            month_sin, month_cos,
            weekday_sin, weekday_cos,
            hour_sin, hour_cos,
            doy_sin, doy_cos
        ],
        axis=1
    ).astype(np.float32)

    return features

class RelativePositionBias(nn.Module):
    """
    T5 风格 RPE，为时间序列优化过：
    - 永远不会产生负索引
    - 永远不会越界
    """
    def __init__(self, num_buckets=32, max_distance=128, n_heads=8):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads

        self.relative_attention_bias = nn.Embedding(num_buckets, n_heads)

    def forward(self, q_len, k_len):
        device = self.relative_attention_bias.weight.device

        context = torch.arange(q_len, device=device)[:, None]
        memory  = torch.arange(k_len, device=device)[None, :]
        relative_position = memory - context  # [q,k]

        rp_bucket = self._relative_position_bucket(relative_position)
        # 放到 GPU
        rp_bucket = rp_bucket.to(device)

        values = self.relative_attention_bias(rp_bucket)  # [q,k,h]
        return values.permute(2,0,1)                      # [h,q,k]

    def _relative_position_bucket(self, relative_position):
        """
        标准 T5 bucket，无符号！
        """
        n = torch.abs(relative_position)
        max_exact = self.num_buckets // 2

        # small
        is_small = n < max_exact

        # large
        val_large = max_exact + (
            torch.log(n.float() / max_exact + 1e-6) /
            math.log(self.max_distance / max_exact)
            * (self.num_buckets - max_exact)
        ).long()
        val_large = torch.clamp(val_large, max=self.num_buckets - 1)

        buckets = torch.where(is_small, n, val_large)
        return buckets

class MultiHeadAttentionRPE(nn.Module):
    """
    自定义 MultiheadAttention，显式添加 RPE bias 到 attention logits。
    支持任意 Q/K 序列长度。
    """
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
        """
        Q: [B, Lq, d]
        K: [B, Lk, d]
        V: [B, Lk, d]
        rpe_bias: [heads, Lq, Lk]
        """
        B, Lq, _ = Q.size()
        _, Lk, _ = K.size()

        # 1) 线性变换
        q = self.Wq(Q).view(B, Lq, self.n_heads, self.d_head).transpose(1, 2)
        k = self.Wk(K).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)
        v = self.Wv(V).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)

        # 2) Scaled dot-product attention logits
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)
        # logits: [B, heads, Lq, Lk]

        # 3) 加 RPE bias（自动 broadcast）
        logits = logits + rpe_bias.unsqueeze(0)

        # 4) softmax
        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        # 5) 加权求值
        out = torch.matmul(attn, v)  # [B, heads, Lq, d_head]

        # 6) 拼回输出
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
        """
        x: [B, L, d]
        rpe_bias: [heads, L, L]
        """
        # Attention + RPE
        h = self.attn(x, x, x, rpe_bias)
        x = self.norm1(x + self.dropout(h))

        # FFN
        h2 = self.ffn(x)
        x = self.norm2(x + self.dropout(h2))

        return x
class PatchTST_D1(nn.Module):
    def __init__(self, enc_in_hist, enc_in_future,
                 hist_len=96, pred_len=96,
                 d_model=128, n_heads=8, e_layers=3,
                 patch_len=12, dropout=0.1):
        super().__init__()

        self.hist_len = hist_len
        self.pred_len = pred_len
        self.patch_len = patch_len

        self.n_patch_hist = hist_len // patch_len
        self.n_patch_future = pred_len // patch_len

        # 输入投影
        self.hist_proj = nn.Linear(enc_in_hist, d_model)
        self.future_proj = nn.Linear(enc_in_future, d_model)

        # Patch embedding
        self.patch_embed_hist = nn.Linear(patch_len * d_model, d_model)
        self.patch_embed_future = nn.Linear(patch_len * d_model, d_model)

        # ======= RPE Bias =======
        self.rpe_hist = RelativePositionBias(n_heads=n_heads)
        self.rpe_fut  = RelativePositionBias(n_heads=n_heads)

        # ======= Encoder Layers with RPE =======
        self.encoder_layers = nn.ModuleList([
            EncoderLayerRPE(d_model, n_heads, dropout)
            for _ in range(e_layers)
        ])

        # ======= Cross Attention with RPE =======
        self.cross_attn = MultiHeadAttentionRPE(d_model, n_heads, dropout)

        # ======= 输出层 =======
        self.patch_proj = nn.Linear(d_model, patch_len * d_model)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, hist_x, fut_x):
        B = hist_x.size(0)

        # --- 1. 线性投影 ---
        hist = self.hist_proj(hist_x)
        fut  = self.future_proj(fut_x)

        # --- 2. 切片为 Patch ---
        hist = hist.view(B, self.n_patch_hist, self.patch_len, -1)
        hist = self.patch_embed_hist(hist.reshape(B, self.n_patch_hist, -1))

        fut = fut.view(B, self.n_patch_future, self.patch_len, -1)
        fut = self.patch_embed_future(fut.reshape(B, self.n_patch_future, -1))

        # --- 3. RPE for Encoder ---
        rpe_hist = self.rpe_hist(self.n_patch_hist, self.n_patch_hist)

        # --- 4. 历史 Encoder ---
        for layer in self.encoder_layers:
            hist = layer(hist, rpe_hist)

        # --- 5. Cross Attention (future ← history) ---
        rpe_fut = self.rpe_fut(self.n_patch_future, self.n_patch_hist)
        fut = self.cross_attn(fut, hist, hist, rpe_fut)

        # --- 6. 还原 patch → 序列 ---
        fut_seq = self.patch_proj(fut)
        fut_seq = fut_seq.view(B, self.n_patch_future, self.patch_len, -1)
        fut_seq = fut_seq.reshape(B, self.pred_len, -1)

        # --- 7. 输出到功率 ---
        return self.output_proj(fut_seq)


def run_one_site(site_id, capacity):


    # =============== 加载数据 ===============
    df = pd.read_excel(
        f"Solar station site {site_id} (Nominal capacity-{capacity}MW).xlsx"
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    if site_id in [1, 2, 3]:
        phys_features = ["TSI", "DNI", "GHI", "AT", "hPa"]
    else:
        phys_features = ["TSI", "DNI", "GHI", "AT", "hPa", "RH"]
        # phys_features = ["TSI", "DNI", "GHI", "AT", "hPa"]

    target = "Power"

    # =============== 标准化 ===============
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_t = MinMaxScaler()

    x_scaled = scaler_x.fit_transform(df[phys_features])                    # [N, Cw]
    y_scaled = scaler_y.fit_transform(df[[target]])                         # [N, 1]
    t_scaled = scaler_t.fit_transform(build_time_features(df["timestamp"])) # [N, Ct]


    # =============== 构建 D+1 样本：两天一组 ===============
    def create_D1_samples(x, y, t, timestamps,
                          hist_len=HIST_LEN, pred_len=PRED_LEN):
        """
        每个样本：前一天 (hist_len) + D+1 一天 (pred_len) 连续 192 条记录
        历史输入：天气 + 出力 + 时间
        未来输入：天气 ++ 时间
        label:   D+1 出力 [pred_len,1]
        """
        X_hist, X_fut, Ys, fut_dates = [], [], [], []
        N = len(x)
        step = hist_len * STRIDE_DAYS

        for i in range(0, N - hist_len - pred_len + 1, step):
            # 历史
            x_hist = x[i:i+hist_len]
            y_hist = y[i:i+hist_len]
            t_hist = t[i:i+hist_len]

            # D+1
            x_fut = x[i+hist_len:i+hist_len+pred_len]
            y_fut = y[i+hist_len:i+hist_len+pred_len]   # label
            t_fut = t[i+hist_len:i+hist_len+pred_len]

            # 历史输入：
            # hist_in = np.concatenate([x_hist, y_hist, t_hist], axis=-1)
            # hist_in = np.concatenate([y_hist, x_hist,y_hist,t_hist], axis=-1)#95
            hist_in = np.concatenate([x_hist, y_hist, t_hist], axis=-1)#95.1

            # 未来输入：未来出力未知 → 必须设为 0
            zero_future = np.zeros_like(y_fut)
            fut_in = np.concatenate([x_fut,  t_fut], axis=-1)
            # fut_in = np.concatenate([x_fut], axis=-1)

            X_hist.append(hist_in)
            X_fut.append(fut_in)
            Ys.append(y_fut)

            fut_dates.append(timestamps.iloc[i+hist_len])

        return (
            np.array(X_hist, dtype=np.float32),
            np.array(X_fut, dtype=np.float32),
            np.array(Ys, dtype=np.float32),
            pd.to_datetime(fut_dates)
        )



    X_hist_all, X_fut_all, Y_all, fut_dates_all = create_D1_samples(
        x_scaled, y_scaled, t_scaled, df["timestamp"]
    )

    N_samples = len(X_hist_all)
    split_idx = int(N_samples * 0.7)

    X_hist_train = X_hist_all[:split_idx]
    X_fut_train  = X_fut_all[:split_idx]
    Y_train      = Y_all[:split_idx]

    X_hist_val = X_hist_all[split_idx:]
    X_fut_val  = X_fut_all[split_idx:]
    Y_val      = Y_all[split_idx:]
    dates_val  = fut_dates_all[split_idx:]


    # =============== Dataset ===============
    class PVPatchD1Dataset(Dataset):
        def __init__(self, Xh, Xf, Y):
            self.Xh = torch.tensor(Xh, dtype=torch.float32)
            self.Xf = torch.tensor(Xf, dtype=torch.float32)
            self.Y  = torch.tensor(Y,  dtype=torch.float32)

        def __len__(self):
            return len(self.Xh)

        def __getitem__(self, idx):
            return self.Xh[idx], self.Xf[idx], self.Y[idx]


    train_loader = DataLoader(
        PVPatchD1Dataset(X_hist_train, X_fut_train, Y_train),
        batch_size=BATCH_SIZE, shuffle=True
    )


    # =============== 模型 / 优化器 / 损失 ===============
    Cw = x_scaled.shape[1]
    Ct = t_scaled.shape[1]
    enc_in_hist   = Cw + 1 + Ct    # 历史：天气 + 出力 + 时间
    enc_in_future = Cw + Ct     # 未来：天气 + 出力(这里用真值，可改为0/预测) + 时间
    # enc_in_hist   = Cw + 2    # 历史：天气 + 出力 + 时间
    # enc_in_future = Cw       # 未来：天气 + 出力(这里用真值，可改为0/预测) + 时间
    model = PatchTST_D1(
        enc_in_hist=enc_in_hist,
        enc_in_future=enc_in_future,
        hist_len=HIST_LEN,
        pred_len=PRED_LEN,
        d_model=128,
        n_heads=8,
        e_layers=3,
        patch_len=6,
        dropout=0.1
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()


    # =============== 验证函数：按 D+1 天评估 ===============
    def evaluate_on_days(model):
        model.eval()
        preds_list, trues_list = [], []

        with torch.no_grad():
            for Xh, Xf, Y in zip(X_hist_val, X_fut_val, Y_val):
                Xh_t = torch.tensor(Xh, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # [1,96,C_hist]
                Xf_t = torch.tensor(Xf, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # [1,96,C_fut]
                Y_t  = torch.tensor(Y,  dtype=torch.float32).unsqueeze(0).to(DEVICE)  # [1,96,1]

                out = model(Xh_t, Xf_t)                # [1,96,1]
                pred = out.squeeze(0).cpu().numpy()    # [96,1]
                true = Y_t.squeeze(0).cpu().numpy()    # [96,1]

                preds_list.append(pred.squeeze(-1))    # [96]
                trues_list.append(true.squeeze(-1))    # [96]

        preds_arr = np.array(preds_list)  # [N_val,96]
        trues_arr = np.array(trues_list)  # [N_val,96]

        # 反归一化
        preds_inv = scaler_y.inverse_transform(preds_arr.reshape(-1,1)).reshape(preds_arr.shape)
        trues_inv = scaler_y.inverse_transform(trues_arr.reshape(-1,1)).reshape(trues_arr.shape)

        # 按月计算 ACC = 1 - RMSE/Cap
        dates_series = pd.Series(dates_val)
        months = dates_series.dt.to_period("M")

        results = {}
        for m in np.unique(months):
            mask = (months == m).values
            rmse = np.sqrt(np.mean((preds_inv[mask] - trues_inv[mask])**2))
            acc = 1 - rmse / capacity
            results[str(m)] = acc

        mean_acc = np.mean(list(results.values()))
        return mean_acc


    # =============== 训练 + Early Stopping ===============
    best_acc = -1e9
    best_state = None
    patience_cnt = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for Xh_b, Xf_b, Y_b in train_loader:
            Xh_b = Xh_b.to(DEVICE)
            Xf_b = Xf_b.to(DEVICE)
            Y_b  = Y_b.to(DEVICE)

            optimizer.zero_grad()
            out = model(Xh_b, Xf_b)       # [B,96,1]
            loss = criterion(out, Y_b)    # label 就是 D+1 的 96×1
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_mean_acc = evaluate_on_days(model)

        print(f"Epoch {epoch+1}/{EPOCHS}   Loss = {train_loss:.6f}   Val_mean_Acc = {val_mean_acc:.6f}")

        # Early Stopping
        if val_mean_acc > best_acc:
            best_acc = val_mean_acc
            best_state = deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch+1}, best mean acc = {best_acc:.6f}")
                break

    # 恢复最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)

    # =============== 最终按天评估并输出每月指标 ===============
    model.eval()
    preds_day, trues_day = [], []

    with torch.no_grad():
        for Xh, Xf, Y in zip(X_hist_val, X_fut_val, Y_val):
            Xh_t = torch.tensor(Xh, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            Xf_t = torch.tensor(Xf, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            out = model(Xh_t, Xf_t)                    # [1,96,1]
            pred = out.squeeze(0).squeeze(-1).cpu().numpy()  # [96]
            true = Y.squeeze(-1)                             # [96]

            preds_day.append(pred)
            trues_day.append(true)

    preds_day = np.array(preds_day)   # [N_val,96]
    trues_day = np.array(trues_day)   # [N_val,96]

    # 反归一化
    preds_inv = scaler_y.inverse_transform(preds_day.reshape(-1,1)).reshape(preds_day.shape)
    trues_inv = scaler_y.inverse_transform(trues_day.reshape(-1,1)).reshape(trues_day.shape)

    dates_test = pd.Series(dates_val)
    months = dates_test.dt.to_period("M")

    results = {}
    for m in np.unique(months):
        mask = (months == m).values
        rmse = np.sqrt(np.mean((preds_inv[mask] - trues_inv[mask])**2))
        print("示例 pred_inv[0, :10] =", preds_inv[0][:10])
        print("示例 true_inv[0, :10] =", trues_inv[0][:10])

        acc = 1 - rmse / capacity
        results[str(m)] = acc

    d1_df = pd.DataFrame({
        "month": list(results.keys()),
        "D+1_RMSE/Cap": list(results.values())   # 保持你原来的列名格式
    })

    print("\n========== PatchTST 风格 (历史+未来条件 D+1 预测 + 按天测试 + EarlyStopping) ==========")
    print(d1_df)

    os.makedirs("D+D+1_PatchTST95PRE_noRH+futuretime循环", exist_ok=True)
    d1_df.to_excel(f"D+D+1_PatchTST95PRE_noRH+futuretime循环/PREPatchTST_site_{site_id}_D+1_autoreg_earlystop.xlsx", index=False)

    with open(f"D+D+1_PatchTST95PRE_noRH+futuretime循环/PatchTST_site_{site_id}_D+1_data.pkl","wb") as f:
        pickle.dump({
            "pred": preds_inv,
            "true": trues_inv,
            "date": dates_test,
            "month_metrics": results,
            "best_mean_acc": best_acc,
        }, f)
# =============== ⭐ 只加循环，其余不动 ===============
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

for sid, cap in site_info.items():
    run_one_site(sid, cap)