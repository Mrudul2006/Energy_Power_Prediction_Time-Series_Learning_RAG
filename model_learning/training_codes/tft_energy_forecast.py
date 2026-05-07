"""
tft_energy_forecast.py
══════════════════════════════════════════════════════════════════════════════
Temporal Fusion Transformer (TFT) for hourly energy consumption forecasting.

Architecture
  • Gated Linear Units (GLU)
  • Gated Residual Networks (GRN) — core non-linear building block
  • Variable Selection Networks (VSN) — soft per-feature importance
  • LSTM encoder → decoder with hidden-state hand-off
  • Interpretable Multi-Head Temporal Self-Attention (IMHA, shared values)
  • Quantile regression outputs  [q10, q50, q90]

Training
  • Combined pinball + Huber loss
  • AdamW + OneCycleLR, gradient clipping
  • Temporal train / val split  (val = last 3 months of training data)
  • Early stopping on validation loss

Hyperparameter optimisation
  • Optuna TPE sampler + Hyperband pruner (50 configurable trials)
  • Searches: encoder_length, hidden_size, num_heads, lstm_layers,
    dropout, lr, weight_decay, batch_size

Forecasting
  • Direct multi-step (1 pass, teacher-forced lag features) — used for
    per-horizon evaluation over the full test set
  • Truly recursive 168-step decoder — lag/rolling stats recalculated
    from model outputs at every step; used for final showcase forecast

Online learning
  • Page-Hinkley test for concept-drift detection
  • Sliding-window fine-tuning triggered on drift
  • Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting

Explainability (XAI)
  • Variable importance from VSN softmax weights (encoder & decoder)
  • Temporal attention heatmap  (decoder → encoder attention flow)
  • SHAP DeepExplainer integration (requires `pip install shap`)
  • Quantile prediction-interval coverage analysis

Evaluation
  • Per-horizon MAE, RMSE, MAPE, sMAPE  (identical grid to new_model.py)
  • Overall 7-day recursive MAE / RMSE
  • Diebold-Mariano test vs. seasonal-naïve baseline
  • Residual ACF and QQ-plot

Usage
  python tft_energy_forecast.py
"""

# ── 0. IMPORTS ────────────────────────────────────────────────────────────────
import warnings; warnings.filterwarnings("ignore")
import os, sys, math, time, copy, json, pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import periodogram
from statsmodels.tsa.stattools import acf as tsacf

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    import shap as _shap; HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[INFO] pip install shap  to enable SHAP explainability")

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {DEVICE}")

# ── 1. CONFIGURATION ──────────────────────────────────────────────────────────
@dataclass
class TFTConfig:
    # data
    csv_file:        str   = "cleaned_energy_data_model.csv"
    timestamp_col:   str   = "start_time"
    target_col:      str   = "consumption"
    test_cutoff:     str   = "2021-01-01"
    val_cutoff:      str   = "2020-10-01"   # last 3 months of train = validation

    # sequence
    encoder_length:  int   = 336            # 2-week lookback (tuned by Optuna)
    decoder_length:  int   = 168            # fixed 7-day horizon

    # model
    hidden_size:     int   = 128
    num_heads:       int   = 4
    num_lstm_layers: int   = 2
    dropout:         float = 0.1
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    # training
    batch_size:      int   = 32
    lr:              float = 1e-3
    weight_decay:    float = 1e-4
    max_epochs:      int   = 60
    patience:        int   = 10
    grad_clip:       float = 1.0

    # optuna
    n_trials:        int   = 50

    # online learning
    ol_window:       int   = 720            # 30-day sliding fine-tune window
    ph_delta:        float = 0.005          # Page-Hinkley sensitivity
    ph_lambda:       float = 50.0           # Page-Hinkley alarm threshold
    ewc_lambda:      float = 5_000.0        # EWC regularisation strength

    # output
    artifact_dir:    str   = "tft_artifacts"


# ── 2. KNOWN FEATURE LISTS ────────────────────────────────────────────────────
# Features that are computable for any future timestamp (no target leakage)
KNOWN_FUTURE_COLS = [
    'hour', 'day_of_week', 'month', 'year', 'season',
    'is_weekend', 'is_holiday', 'is_first_day', 'is_last_day',
]
# Features derived from past consumption (unknown in the future)
PAST_OBSERVED_COLS = [
    'lag_1h', 'lag_24h', 'lag_48h', 'lag_72h', 'lag_168h',
    'prev_daily_mean', 'prev_daily_max',
    'rolling_mean_24h', 'rolling_mean_168h', 'rolling_std_24h',
    'ewm_24h', 'how_historical_mean', 'resid_vs_how_mean',
]


# ── 3. DATA LOADING ───────────────────────────────────────────────────────────
def load_data(cfg: TFTConfig):
    df = pd.read_csv(cfg.csv_file)
    df[cfg.timestamp_col] = pd.to_datetime(df[cfg.timestamp_col])
    df.set_index(cfg.timestamp_col, inplace=True)
    df.sort_index(inplace=True)
    if 'end_time_utc' in df.columns:
        df.drop(columns=['end_time_utc'], inplace=True)

    # ── cyclic sin/cos encodings (added to known_future)
    cyc = [('hour', 24), ('day_of_week', 7), ('month', 12)]
    for base, period in cyc:
        if base in df.columns:
            df[f'{base}_sin'] = np.sin(2 * np.pi * df[base] / period)
            df[f'{base}_cos'] = np.cos(2 * np.pi * df[base] / period)

    df.dropna(inplace=True)

    # finalise feature lists (only keep columns that exist)
    known_future  = [c for c in KNOWN_FUTURE_COLS if c in df.columns]
    known_future += [c for c in df.columns if c.endswith('_sin') or c.endswith('_cos')]
    past_observed = [c for c in PAST_OBSERVED_COLS if c in df.columns]

    feat_cfg = {
        'known_future':  known_future,
        'past_observed': past_observed,
        'n_kf':  len(known_future),
        'n_po':  len(past_observed),
    }
    n_enc = len(known_future) + len(past_observed) + 1   # +1 for target
    n_dec = len(known_future) + len(past_observed)        # no target in decoder

    print(f"[Data]  rows={len(df):,}  |  known_future={len(known_future)}  "
          f"past_observed={len(past_observed)}")
    print(f"        enc_feats={n_enc}  dec_feats={n_dec}")

    # ── normalise (fit only on train portion)
    all_feat_cols = known_future + past_observed + [cfg.target_col]
    train_mask = df.index < cfg.test_cutoff

    scalers: Dict[str, RobustScaler] = {}
    df_sc = df.copy()
    for col in all_feat_cols:
        sc = RobustScaler()
        sc.fit(df.loc[train_mask, [col]])
        df_sc[col] = sc.transform(df[[col]]).ravel()
        scalers[col] = sc

    train_df = df_sc[df_sc.index <  cfg.val_cutoff]
    val_df   = df_sc[(df_sc.index >= cfg.val_cutoff) & (df_sc.index < cfg.test_cutoff)]
    test_df  = df_sc[df_sc.index >= cfg.test_cutoff]

    print(f"[Split] train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")
    return df, df_sc, train_df, val_df, test_df, scalers, feat_cfg


# ── 4. PYTORCH DATASET ────────────────────────────────────────────────────────
class TFTDataset(Dataset):
    """
    Yields (enc_x, dec_x, target) for sequence-to-sequence training.

    enc_x  : [encoder_length, n_enc]   known_future + past_observed + target(t-1..t-T)
    dec_x  : [decoder_length, n_dec]   known_future + past_observed (teacher-forced)
    target : [decoder_length]          scaled consumption
    """
    def __init__(self, df_sc: pd.DataFrame, cfg: TFTConfig, fc: dict, stride: int = 1):
        enc_cols = fc['known_future'] + fc['past_observed'] + [cfg.target_col]
        dec_cols = fc['known_future'] + fc['past_observed']

        self.enc_arr = df_sc[enc_cols].values.astype(np.float32)
        self.dec_arr = df_sc[dec_cols].values.astype(np.float32)
        self.tgt_arr = df_sc[cfg.target_col].values.astype(np.float32)
        self.n_enc   = len(enc_cols)
        self.n_dec   = len(dec_cols)

        enc_len = cfg.encoder_length
        dec_len = cfg.decoder_length
        max_start = len(df_sc) - enc_len - dec_len
        self.starts = list(range(0, max(0, max_start), stride))
        self.enc_len = enc_len
        self.dec_len = dec_len

    def __len__(self): return len(self.starts)

    def __getitem__(self, idx):
        i = self.starts[idx]
        enc_x = self.enc_arr[i: i + self.enc_len]
        dec_x = self.dec_arr[i + self.enc_len: i + self.enc_len + self.dec_len]
        tgt   = self.tgt_arr[i + self.enc_len: i + self.enc_len + self.dec_len]
        return torch.from_numpy(enc_x.copy()), torch.from_numpy(dec_x.copy()), torch.from_numpy(tgt.copy())


# ── 5. TFT BUILDING BLOCKS ────────────────────────────────────────────────────
class GLU(nn.Module):
    """Gated Linear Unit."""
    def __init__(self, d: int):
        super().__init__()
        self.fc = nn.Linear(d, 2 * d)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GRN(nn.Module):
    """
    Gated Residual Network.
    η₁ = ELU(W₁·x + W_c·ctx + b₁)
    η₂ = Dropout(W₂·η₁ + b₂)
    ỹ  = GLU(η₂)
    output = LayerNorm(ỹ + skip(x))
    """
    def __init__(self, in_d: int, hid_d: int, out_d: int,
                 dropout: float = 0.0, ctx_d: Optional[int] = None):
        super().__init__()
        self.fc1  = nn.Linear(in_d, hid_d)
        self.fc2  = nn.Linear(hid_d, hid_d)
        self.gate = nn.Linear(hid_d, 2 * out_d)
        self.ln   = nn.LayerNorm(out_d)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Linear(in_d, out_d) if in_d != out_d else nn.Identity()
        self.ctx_fc = nn.Linear(ctx_d, hid_d, bias=False) if ctx_d else None

    def forward(self, x, ctx=None):
        h = F.elu(self.fc1(x))
        if ctx is not None and self.ctx_fc is not None:
            h = h + self.ctx_fc(ctx)
        h = self.drop(self.fc2(h))
        a, b = self.gate(h).chunk(2, dim=-1)
        return self.ln(a * torch.sigmoid(b) + self.skip(x))


class VSN(nn.Module):
    """
    Variable Selection Network.
    Input  : [B, T, V, H]  (temporal) or [B, V, H]  (static)
    Output : ([B, T, H] or [B, H],  weights [B, T, V] or [B, V])
    """
    def __init__(self, n_vars: int, embed_d: int, hid_d: int,
                 dropout: float = 0.0, ctx_d: Optional[int] = None):
        super().__init__()
        self.n_vars  = n_vars
        self.var_grns = nn.ModuleList([
            GRN(embed_d, hid_d, hid_d, dropout) for _ in range(n_vars)
        ])
        self.sel_grn = GRN(n_vars * embed_d, hid_d, n_vars, dropout, ctx_d)

    def forward(self, x, ctx=None):
        temporal = (x.dim() == 4)
        if temporal:
            B, T, V, D = x.shape
            flat  = x.reshape(B * T, V * D)
            ctx_r = (ctx.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
                     if ctx is not None else None)
            w = F.softmax(self.sel_grn(flat, ctx_r), dim=-1)          # [B*T, V]
            xr = x.reshape(B * T, V, D)
            pv = torch.stack([self.var_grns[i](xr[:, i, :]) for i in range(V)], dim=1)
            out = (pv * w.unsqueeze(-1)).sum(1).reshape(B, T, -1)
            return out, w.reshape(B, T, V)
        else:
            B, V, D = x.shape
            flat = x.reshape(B, V * D)
            w  = F.softmax(self.sel_grn(flat, ctx), dim=-1)            # [B, V]
            pv = torch.stack([self.var_grns[i](x[:, i, :]) for i in range(V)], dim=1)
            out = (pv * w.unsqueeze(-1)).sum(1)
            return out, w


class IMHA(nn.Module):
    """
    Interpretable Multi-Head Attention (Bryan Lim et al., 2021).
    All heads share a single value projection; attention weights are
    averaged across heads to give one interpretable attention map.
    """
    def __init__(self, n_heads: int, d: int, dropout: float = 0.0):
        super().__init__()
        assert d % n_heads == 0, "hidden_size must be divisible by num_heads"
        self.h = n_heads; self.dk = d // n_heads; self.d = d
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(d, self.dk)      # shared across heads
        self.Wo = nn.Linear(self.dk, d)
        self.drop = nn.Dropout(dropout)
        self.ln   = nn.LayerNorm(d)

    def forward(self, Q, K, V, mask=None):
        B, Tq, _ = Q.shape; Tk = K.shape[1]
        q = self.Wq(Q).reshape(B, Tq, self.h, self.dk).transpose(1, 2)   # [B,h,Tq,dk]
        k = self.Wk(K).reshape(B, Tk, self.h, self.dk).transpose(1, 2)   # [B,h,Tk,dk]
        v = self.Wv(V).unsqueeze(1).expand(-1, self.h, -1, -1)            # [B,h,Tk,dk]

        sc   = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        if mask is not None:
            sc = sc.masked_fill(mask == 0, float('-inf'))
        attn = self.drop(F.softmax(sc, dim=-1))                           # [B,h,Tq,Tk]

        ctx      = (attn @ v).mean(1)          # average heads → [B,Tq,dk]
        attn_avg = attn.mean(1)                # [B,Tq,Tk]

        out = self.ln(self.Wo(ctx) + Q)
        return out, attn_avg


# ── 6. TEMPORAL FUSION TRANSFORMER ────────────────────────────────────────────
class TFT(nn.Module):
    """
    Full TFT model.

    Flow
    ────
    Encoder path
      per-variable linear embed → VSN  →  LSTM encoder
    Decoder path
      per-variable linear embed → VSN  →  LSTM decoder  (init from encoder state)
    Fusion
      gated skip (VSN output + LSTM output) → LayerNorm
      → causal IMHA  → gated skip → LayerNorm
      → GRN (position-wise FF) → gated skip → LayerNorm
      → linear  → [B, T_dec, n_quantiles]
    """
    def __init__(self, cfg: TFTConfig, n_enc_feats: int, n_dec_feats: int):
        super().__init__()
        self.cfg = cfg
        H  = cfg.hidden_size
        nq = len(cfg.quantiles)

        # per-variable embed projections
        self.enc_projs = nn.ModuleList([nn.Linear(1, H) for _ in range(n_enc_feats)])
        self.dec_projs = nn.ModuleList([nn.Linear(1, H) for _ in range(n_dec_feats)])

        # variable selection
        self.enc_vsn = VSN(n_enc_feats, H, H, cfg.dropout)
        self.dec_vsn = VSN(n_dec_feats, H, H, cfg.dropout)

        # LSTM encoder / decoder
        lstm_kw = dict(batch_first=True,
                       dropout=cfg.dropout if cfg.num_lstm_layers > 1 else 0.0)
        self.enc_lstm = nn.LSTM(H, H, cfg.num_lstm_layers, **lstm_kw)
        self.dec_lstm = nn.LSTM(H, H, cfg.num_lstm_layers, **lstm_kw)

        # post-LSTM gated add & norm
        self.lstm_glu = GLU(H)
        self.lstm_ln  = nn.LayerNorm(H)

        # temporal self-attention
        self.attn     = IMHA(cfg.num_heads, H, cfg.dropout)
        self.attn_glu = GLU(H)
        self.attn_ln  = nn.LayerNorm(H)

        # position-wise feed-forward (GRN)
        self.ff        = GRN(H, H * 4, H, cfg.dropout)
        self.ff_glu    = GLU(H)
        self.ff_ln     = nn.LayerNorm(H)

        # quantile output head
        self.out_fc = nn.Linear(H, nq)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, p in m.named_parameters():
                    if 'weight_ih' in name: nn.init.xavier_uniform_(p)
                    elif 'weight_hh' in name: nn.init.orthogonal_(p)
                    elif 'bias' in name: nn.init.zeros_(p)

    def _embed(self, x, projs):
        """[B, T, F] → [B, T, F, H]  via per-variable projections."""
        return torch.stack([projs[i](x[..., i:i+1]) for i in range(len(projs))], dim=-2)

    def forward(self, enc_x, dec_x):
        """
        enc_x : [B, T_enc, n_enc]
        dec_x : [B, T_dec, n_dec]
        returns preds [B, T_dec, nq], interpretability dict
        """
        B, T_enc, _ = enc_x.shape
        T_dec = dec_x.shape[1]
        H = self.cfg.hidden_size

        # embed
        enc_emb = self._embed(enc_x, self.enc_projs)   # [B, T_enc, n_enc, H]
        dec_emb = self._embed(dec_x, self.dec_projs)   # [B, T_dec, n_dec, H]

        # variable selection
        enc_sel, enc_vw = self.enc_vsn(enc_emb)        # [B,T_enc,H], [B,T_enc,n_enc]
        dec_sel, dec_vw = self.dec_vsn(dec_emb)        # [B,T_dec,H], [B,T_dec,n_dec]

        # LSTM
        enc_out, (h_n, c_n) = self.enc_lstm(enc_sel)                   # [B,T_enc,H]
        dec_out, _           = self.dec_lstm(dec_sel, (h_n, c_n))      # [B,T_dec,H]

        # gated add-norm after LSTM
        combined = torch.cat([enc_out, dec_out], dim=1)                 # [B,T_all,H]
        vsn_all  = torch.cat([enc_sel,  dec_sel], dim=1)
        gated    = self.lstm_ln(self.lstm_glu(combined) + vsn_all)      # [B,T_all,H]

        # causal temporal self-attention
        T_all = T_enc + T_dec
        mask  = torch.tril(
            torch.ones(T_all, T_all, device=enc_x.device)
        ).unsqueeze(0)                                                   # [1,T_all,T_all]
        attn_out, attn_w = self.attn(gated, gated, gated, mask)
        attn_gated = self.attn_ln(self.attn_glu(attn_out) + gated)

        # position-wise GRN
        ff_out    = self.ff(attn_gated)
        ff_gated  = self.ff_ln(self.ff_glu(ff_out) + attn_gated)

        # output — decoder positions only
        preds = self.out_fc(ff_gated[:, T_enc:, :])                     # [B,T_dec,nq]

        return preds, {
            'enc_var_weights': enc_vw,                   # [B, T_enc, n_enc]
            'dec_var_weights': dec_vw,                   # [B, T_dec, n_dec]
            'attn_weights':    attn_w[:, T_enc:, :T_enc],# [B, T_dec, T_enc]
        }


# ── 7. LOSS FUNCTIONS ─────────────────────────────────────────────────────────
def pinball_loss(pred: torch.Tensor, target: torch.Tensor,
                 quantiles: List[float]) -> torch.Tensor:
    """Quantile / pinball loss averaged across quantiles and timesteps."""
    tgt = target.unsqueeze(-1).expand_as(pred)
    q   = torch.tensor(quantiles, device=pred.device, dtype=pred.dtype).view(1, 1, -1)
    err = tgt - pred
    return torch.where(err >= 0, q * err, (q - 1) * err).mean()


def combined_loss(pred, target, quantiles, huber_delta=1.0):
    """Pinball across all quantiles + Huber on median for extra stability."""
    pb = pinball_loss(pred, target, quantiles)
    q50_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
    huber = F.huber_loss(pred[:, :, q50_idx], target, delta=huber_delta)
    return pb + 0.3 * huber


# ── 8. TRAINER ────────────────────────────────────────────────────────────────
class Trainer:
    def __init__(self, model: TFT, cfg: TFTConfig,
                 train_loader: DataLoader, val_loader: DataLoader):
        self.model = model.to(DEVICE)
        self.cfg   = cfg
        self.tl    = train_loader
        self.vl    = val_loader
        total_steps = len(train_loader) * cfg.max_epochs
        self.opt   = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.sched = OneCycleLR(self.opt, max_lr=cfg.lr, total_steps=total_steps,
                                pct_start=0.05, anneal_strategy='cos')
        self.best_val   = float('inf')
        self.best_state = None
        self.patience   = cfg.patience

    def _run(self, loader, train: bool) -> float:
        self.model.train(train)
        total = 0.
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for enc_x, dec_x, tgt in loader:
                enc_x, dec_x, tgt = enc_x.to(DEVICE), dec_x.to(DEVICE), tgt.to(DEVICE)
                if train: self.opt.zero_grad()
                pred, _ = self.model(enc_x, dec_x)
                loss = combined_loss(pred, tgt, self.cfg.quantiles)
                if train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.opt.step(); self.sched.step()
                total += loss.item() * enc_x.size(0)
        return total / len(loader.dataset)

    def fit(self, verbose=True) -> float:
        patience_left = self.patience
        for epoch in range(1, self.cfg.max_epochs + 1):
            tr  = self._run(self.tl, True)
            val = self._run(self.vl, False)
            if verbose and epoch % 5 == 0:
                print(f"  epoch {epoch:3d}/{self.cfg.max_epochs}  "
                      f"train={tr:.5f}  val={val:.5f}")
            if val < self.best_val:
                self.best_val   = val
                self.best_state = copy.deepcopy(self.model.state_dict())
                patience_left   = self.patience
            else:
                patience_left -= 1
                if patience_left == 0:
                    if verbose: print(f"  [early stop] epoch {epoch}")
                    break
        self.model.load_state_dict(self.best_state)
        return self.best_val


# ── 9. OPTUNA HYPERPARAMETER SEARCH ───────────────────────────────────────────
def _optuna_objective(trial, cfg: TFTConfig, df_sc: pd.DataFrame, fc: dict) -> float:
    enc_len = trial.suggest_int('encoder_length', 168, 504, step=24)
    hid     = trial.suggest_categorical('hidden_size', [64, 128, 192, 256])
    heads   = trial.suggest_categorical('num_heads', [2, 4, 8])
    layers  = trial.suggest_int('num_lstm_layers', 1, 3)
    drop    = trial.suggest_float('dropout', 0.0, 0.4)
    lr      = trial.suggest_float('lr', 5e-5, 5e-3, log=True)
    wd      = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    bsz     = trial.suggest_categorical('batch_size', [16, 32, 64])

    if hid % heads != 0:
        raise optuna.exceptions.TrialPruned()

    tcfg = TFTConfig(
        encoder_length=enc_len, hidden_size=hid, num_heads=heads,
        num_lstm_layers=layers, dropout=drop, lr=lr, weight_decay=wd,
        batch_size=bsz, max_epochs=15, patience=4
    )

    tr_part = df_sc[df_sc.index <  cfg.val_cutoff]
    va_part = df_sc[(df_sc.index >= cfg.val_cutoff) & (df_sc.index < cfg.test_cutoff)]

    try:
        tr_ds = TFTDataset(tr_part, tcfg, fc, stride=6)
        va_ds = TFTDataset(va_part, tcfg, fc, stride=6)
        if len(tr_ds) < 8 or len(va_ds) < 4:
            raise optuna.exceptions.TrialPruned()

        tr_dl = DataLoader(tr_ds, batch_size=bsz, shuffle=True,  num_workers=0, pin_memory=False)
        va_dl = DataLoader(va_ds, batch_size=bsz, shuffle=False, num_workers=0, pin_memory=False)

        model   = TFT(tcfg, tr_ds.n_enc, tr_ds.n_dec)
        trainer = Trainer(model, tcfg, tr_dl, va_dl)
        val     = trainer.fit(verbose=False)

        trial.report(val, 0)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        return val
    except (RuntimeError, ValueError):
        raise optuna.exceptions.TrialPruned()


def run_optuna(cfg: TFTConfig, df_sc: pd.DataFrame, fc: dict) -> dict:
    print(f"\n[Optuna] {cfg.n_trials} trials  |  TPE sampler  |  Hyperband pruner")
    sampler = TPESampler(seed=SEED, n_startup_trials=10, multivariate=True)
    pruner  = HyperbandPruner(min_resource=3, max_resource=15, reduction_factor=3)
    study   = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
    study.optimize(lambda t: _optuna_objective(t, cfg, df_sc, fc),
                   n_trials=cfg.n_trials, show_progress_bar=True,
                   gc_after_trial=True)
    best = study.best_params
    print(f"\n[Optuna] best val loss : {study.best_value:.6f}")
    print(f"[Optuna] best params   : {json.dumps(best, indent=2)}")
    return best, study


# ── 10. LAG FEATURE HELPERS FOR RECURSIVE INFERENCE ──────────────────────────
_LAG_MAP = {'lag_1h': 1, 'lag_24h': 24, 'lag_48h': 48,
            'lag_72h': 72, 'lag_168h': 168}

def _compute_lag_feature(col: str, hist: np.ndarray) -> float:
    """Recompute a single past-observed feature from a history buffer."""
    n = len(hist)
    if col in _LAG_MAP:
        sh = _LAG_MAP[col]
        return float(hist[-sh]) if n >= sh else float(hist[-1])
    if col == 'rolling_mean_24h':
        return float(hist[-24:].mean()) if n >= 24 else float(hist.mean())
    if col == 'rolling_std_24h':
        return float(hist[-24:].std())  if n >= 24 else 0.0
    if col == 'rolling_mean_168h':
        return float(hist[-168:].mean()) if n >= 168 else float(hist.mean())
    if col in ('ewm_24h',):
        return float(pd.Series(hist[-72:].astype(float)).ewm(span=24).mean().iloc[-1])
    if col in ('prev_daily_mean', 'how_historical_mean'):
        return float(hist[-24:].mean()) if n >= 24 else float(hist.mean())
    if col == 'prev_daily_max':
        return float(hist[-24:].max()) if n >= 24 else float(hist.max())
    if col == 'resid_vs_how_mean':
        how_m = float(hist[-168:].mean()) if n >= 168 else float(hist.mean())
        return float(hist[-1]) - how_m
    return 0.0


# ── 11. RECURSIVE 168-HOUR FORECASTING ENGINE ─────────────────────────────────
@torch.no_grad()
def direct_forecast_window(model: TFT, cfg: TFTConfig,
                            enc_arr: np.ndarray, dec_arr: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Single forward pass for one (encoder_window, decoder_window) pair.
    Decoder lag features come from the dataset (teacher-forced / pre-computed).
    Returns median predictions [decoder_length] (still in scaled space).
    """
    model.eval()
    q50 = cfg.quantiles.index(0.5) if 0.5 in cfg.quantiles else 1
    enc_x = torch.tensor(enc_arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    dec_x = torch.tensor(dec_arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    preds, interp = model(enc_x, dec_x)
    return preds[0, :, q50].cpu().numpy(), interp


@torch.no_grad()
def recursive_forecast_168(model: TFT, cfg: TFTConfig, fc: dict,
                            df_sc: pd.DataFrame, scalers: dict,
                            abs_start: int) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Truly recursive 168-step forecast.

    At each step h:
      1. Inverse-transform accumulated predictions → raw MWh history buffer
      2. Recompute ALL past-observed lag/rolling features from raw MWh buffer
      3. Scale each feature with its own RobustScaler
      4. Run a full TFT forward pass; extract prediction at decoder position h
      5. Append scaled prediction to the accumulator and repeat

    Due to the causal attention mask, the model's output at position h depends
    only on encoder positions 0..enc_len-1 and decoder positions 0..h, so
    filling decoder positions h+1..dec_len-1 with any value is safe.

    Returns:
      preds_raw   : [168] scaled (target space) median predictions
      preds_mwh   : [168] inverse-transformed MWh predictions
      interp      : attention & variable-weight arrays from the first step
    """
    model.eval()
    enc_len = cfg.encoder_length
    dec_len = cfg.decoder_length
    tgt_col = cfg.target_col
    q50     = cfg.quantiles.index(0.5) if 0.5 in cfg.quantiles else 1
    n_kf    = len(fc['known_future'])
    n_po    = len(fc['past_observed'])

    all_cols = fc['known_future'] + fc['past_observed'] + [tgt_col]
    dec_cols = fc['known_future'] + fc['past_observed']

    # encoder window — fixed real data
    enc_window = df_sc.iloc[abs_start: abs_start + enc_len]
    enc_arr    = enc_window[all_cols].values.astype(np.float32)

    # known-future features for the decoder (time features, always available)
    future_window  = df_sc.iloc[abs_start + enc_len: abs_start + enc_len + dec_len]
    kf_future_sc   = future_window[fc['known_future']].values.astype(np.float32)  # already scaled

    # raw MWh history: real encoder values inverse-transformed
    sc_tgt      = scalers[tgt_col]
    hist_raw    = sc_tgt.inverse_transform(
        enc_window[tgt_col].values.reshape(-1, 1)).ravel().astype(np.float64)

    # individual scalers for past-observed columns
    po_scalers  = {c: scalers[c] for c in fc['past_observed'] if c in scalers}

    preds_raw   = np.zeros(dec_len, dtype=np.float32)   # scaled predictions accumulated
    preds_mwh   = np.zeros(dec_len, dtype=np.float64)   # raw MWh predictions accumulated
    interp_save: dict = {}

    enc_x = torch.tensor(enc_arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    for h in range(dec_len):
        # full raw history = encoder history + predictions so far (in MWh)
        full_hist_raw = np.concatenate([hist_raw, preds_mwh[:h]])

        # build decoder input [dec_len, n_dec]
        dec_input = np.zeros((dec_len, n_kf + n_po), dtype=np.float32)
        dec_input[:, :n_kf] = kf_future_sc   # known future for all positions

        for j, col in enumerate(fc['past_observed']):
            raw_val = _compute_lag_feature(col, full_hist_raw)
            if col in po_scalers:
                sc_val = float(po_scalers[col].transform([[raw_val]])[0, 0])
            else:
                sc_val = raw_val
            # fill position h (and forward — safe due to causal mask)
            dec_input[h:, n_kf + j] = sc_val

        dec_x = torch.tensor(dec_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        pred_all, interp = model(enc_x, dec_x)

        pred_sc = pred_all[0, h, q50].item()
        preds_raw[h]  = pred_sc
        preds_mwh[h]  = float(sc_tgt.inverse_transform([[pred_sc]])[0, 0])

        if h == 0:
            interp_save = {k: v[0].cpu().numpy() for k, v in interp.items()}

    return preds_raw, preds_mwh, interp_save


# ── 12. PAGE-HINKLEY DRIFT DETECTOR ──────────────────────────────────────────
class PageHinkley:
    """Detects upward drift in a stream of values."""
    def __init__(self, delta: float = 0.005, lam: float = 50.0):
        self.delta = delta; self.lam = lam
        self.n = 0; self.cum = 0.; self.minimum = 0.

    def update(self, x: float) -> bool:
        self.n   += 1
        self.cum += x - self.delta
        self.minimum = min(self.minimum, self.cum)
        return (self.cum - self.minimum) > self.lam

    def reset(self):
        self.n = 0; self.cum = 0.; self.minimum = 0.


# ── 13. ELASTIC WEIGHT CONSOLIDATION ─────────────────────────────────────────
class EWC:
    """
    Compute Fisher information approximation after initial training.
    penalty() returns the regularisation term to add during fine-tuning.
    """
    def __init__(self, model: TFT, cfg: TFTConfig, dataset: TFTDataset,
                 n_samples: int = 400):
        self.cfg = cfg
        self.star = {n: p.clone().detach() for n, p in model.named_parameters()}
        self.fisher = self._compute(model, dataset, n_samples)

    def _compute(self, model: TFT, ds: TFTDataset, n_samples: int) -> dict:
        fisher: dict = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        model.eval()
        loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
        seen = 0
        for enc_x, dec_x, tgt in loader:
            if seen >= n_samples: break
            enc_x, dec_x, tgt = enc_x.to(DEVICE), dec_x.to(DEVICE), tgt.to(DEVICE)
            model.zero_grad()
            pred, _ = model(enc_x, dec_x)
            q50 = self.cfg.quantiles.index(0.5) if 0.5 in self.cfg.quantiles else 1
            loss = F.mse_loss(pred[:, :, q50], tgt)
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)
            seen += enc_x.size(0)
        for n in fisher:
            fisher[n] /= max(seen, 1)
        return fisher

    def penalty(self, model: TFT) -> torch.Tensor:
        loss = torch.tensor(0., device=DEVICE)
        for n, p in model.named_parameters():
            loss = loss + (self.fisher[n] * (p - self.star[n]).pow(2)).sum()
        return loss


# ── 14. ONLINE LEARNING MODULE ────────────────────────────────────────────────
class OnlineLearner:
    """
    Monitors prediction errors and fine-tunes the model when concept drift
    is detected via the Page-Hinkley test.

    After initial training:
      1. EWC consolidation is computed on the training set
      2. On each new observation, the PH detector is updated
      3. On drift → fine-tune on last ol_window hours (EWC-regularised)
    """
    def __init__(self, model: TFT, cfg: TFTConfig, fc: dict,
                 scalers: dict, train_dataset: TFTDataset):
        self.model   = model
        self.cfg     = cfg
        self.fc      = fc
        self.scalers = scalers
        self.ph      = PageHinkley(cfg.ph_delta, cfg.ph_lambda)
        self.ewc     = EWC(model, cfg, train_dataset)
        self.drift_count = 0
        print("[OnlineLearner] EWC Fisher matrix computed.")

    def observe(self, error: float, df_recent: Optional[pd.DataFrame] = None) -> bool:
        """Feed one new absolute error; returns True if drift was detected."""
        drifted = self.ph.update(error)
        if drifted and df_recent is not None:
            self.drift_count += 1
            print(f"  [OnlineLearner] drift #{self.drift_count} detected → fine-tuning")
            self._fine_tune(df_recent)
            self.ph.reset()
        return drifted

    def _fine_tune(self, df_recent: pd.DataFrame, epochs: int = 8):
        ft_cfg = copy.copy(self.cfg)
        ft_cfg.max_epochs = epochs
        ft_cfg.lr         = self.cfg.lr * 0.05
        ft_cfg.patience   = epochs

        ds = TFTDataset(df_recent, ft_cfg, self.fc, stride=3)
        if len(ds) < 4: return
        dl = DataLoader(ds, batch_size=min(16, len(ds)), shuffle=True, num_workers=0)
        opt = AdamW(self.model.parameters(), lr=ft_cfg.lr, weight_decay=self.cfg.weight_decay)

        self.model.train()
        for _ in range(epochs):
            for enc_x, dec_x, tgt in dl:
                enc_x, dec_x, tgt = enc_x.to(DEVICE), dec_x.to(DEVICE), tgt.to(DEVICE)
                opt.zero_grad()
                pred, _ = self.model(enc_x, dec_x)
                task_loss = combined_loss(pred, tgt, self.cfg.quantiles)
                ewc_loss  = self.cfg.ewc_lambda * self.ewc.penalty(self.model)
                (task_loss + ewc_loss).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                opt.step()
        self.model.eval()


# ── 15. EXPLAINABILITY MODULE ─────────────────────────────────────────────────
class TFTExplainer:
    """
    Collect and visualise interpretability information from TFT.

    Provides
      • encoder variable importance  (mean VSN weights over time)
      • decoder variable importance
      • temporal attention heatmap   (decoder → encoder)
      • SHAP values via DeepExplainer (if shap available)
      • quantile prediction-interval coverage
    """
    def __init__(self, model: TFT, cfg: TFTConfig, fc: dict, scalers: dict,
                 artifact_dir: str):
        self.model   = model
        self.cfg     = cfg
        self.fc      = fc
        self.scalers = scalers
        self.art_dir = Path(artifact_dir)
        self.art_dir.mkdir(parents=True, exist_ok=True)

        enc_names = fc['known_future'] + fc['past_observed'] + [cfg.target_col]
        dec_names = fc['known_future'] + fc['past_observed']
        self.enc_names = enc_names
        self.dec_names = dec_names

    # ── 15a. Variable importance from VSN weights
    @torch.no_grad()
    def variable_importance(self, loader: DataLoader, n_batches: int = 30) -> dict:
        self.model.eval()
        enc_w_list, dec_w_list = [], []
        for i, (enc_x, dec_x, _) in enumerate(loader):
            if i >= n_batches: break
            enc_x, dec_x = enc_x.to(DEVICE), dec_x.to(DEVICE)
            _, interp = self.model(enc_x, dec_x)
            enc_w_list.append(interp['enc_var_weights'].mean(1).cpu().numpy())  # [B, n_enc]
            dec_w_list.append(interp['dec_var_weights'].mean(1).cpu().numpy())  # [B, n_dec]
        enc_imp = np.concatenate(enc_w_list).mean(0)   # [n_enc]
        dec_imp = np.concatenate(dec_w_list).mean(0)   # [n_dec]
        return {'encoder': dict(zip(self.enc_names, enc_imp)),
                'decoder': dict(zip(self.dec_names, dec_imp))}

    # ── 15b. Attention heatmap for one window
    @torch.no_grad()
    def attention_heatmap(self, enc_arr: np.ndarray, dec_arr: np.ndarray) -> np.ndarray:
        """Returns [T_dec, T_enc] attention weights."""
        self.model.eval()
        enc_x = torch.tensor(enc_arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        dec_x = torch.tensor(dec_arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        _, interp = self.model(enc_x, dec_x)
        return interp['attn_weights'][0].cpu().numpy()

    # ── 15c. SHAP values (DeepExplainer on enc_x flattened)
    def shap_analysis(self, background_loader: DataLoader, test_loader: DataLoader,
                      n_bg: int = 64, n_test: int = 32):
        if not HAS_SHAP:
            print("[XAI] SHAP not available — skipping."); return None

        # Wrap model so SHAP sees a single-input function
        class _EncWrapper(nn.Module):
            def __init__(self, m):
                super().__init__(); self.m = m
            def forward(self, enc_x):
                dec_shape = (enc_x.shape[0], self.m.cfg.decoder_length,
                             enc_x.shape[2] - 1)  # n_dec = n_enc - 1 (no target)
                dec_x = torch.zeros(dec_shape, device=enc_x.device)
                pred, _ = self.m(enc_x, dec_x)
                q50 = self.m.cfg.quantiles.index(0.5) if 0.5 in self.m.cfg.quantiles else 1
                return pred[:, :, q50].mean(1, keepdim=True)

        wrapper = _EncWrapper(self.model).to(DEVICE)
        bg_list, te_list = [], []
        for enc_x, _, _ in background_loader:
            bg_list.append(enc_x);
            if sum(x.shape[0] for x in bg_list) >= n_bg: break
        for enc_x, _, _ in test_loader:
            te_list.append(enc_x);
            if sum(x.shape[0] for x in te_list) >= n_test: break

        bg = torch.cat(bg_list)[:n_bg].to(DEVICE)
        te = torch.cat(te_list)[:n_test].to(DEVICE)

        explainer = _shap.DeepExplainer(wrapper, bg)
        shap_vals = explainer.shap_values(te)   # [n_test, T_enc, n_enc]
        return shap_vals   # shape depends on shap version

    # ── 15d. Quantile coverage
    def quantile_coverage(self, actuals: np.ndarray, preds_q: np.ndarray) -> dict:
        """
        actuals  : [N]
        preds_q  : [N, n_quantiles]
        """
        q_low  = preds_q[:, 0]
        q_high = preds_q[:, -1]
        covered = np.mean((actuals >= q_low) & (actuals <= q_high))
        nominal = self.cfg.quantiles[-1] - self.cfg.quantiles[0]
        return {'nominal_coverage': nominal,
                'empirical_coverage': float(covered),
                'coverage_gap': float(covered - nominal)}


# ── 16. COMPREHENSIVE METRICS ─────────────────────────────────────────────────
def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denom = (np.abs(actual) + np.abs(predicted)) / 2
    return float(np.mean(np.where(denom == 0, 0, np.abs(actual - predicted) / denom)) * 100)

def mape(actual: np.ndarray, predicted: np.ndarray, eps=1e-8) -> float:
    return float(np.mean(np.abs((actual - predicted) / (np.abs(actual) + eps))) * 100)

def compute_metrics(actuals: np.ndarray, preds: np.ndarray, label: str = "") -> dict:
    mae   = mean_absolute_error(actuals, preds)
    rmse  = np.sqrt(mean_squared_error(actuals, preds))
    r2    = r2_score(actuals, preds)
    _mape  = mape(actuals, preds)
    _smape = smape(actuals, preds)
    if label:
        print(f"  [{label}]  MAE={mae:.2f}  RMSE={rmse:.2f}  "
              f"MAPE={_mape:.2f}%  sMAPE={_smape:.2f}%  R²={r2:.4f}")
    return {'mae': mae, 'rmse': rmse, 'mape': _mape, 'smape': _smape, 'r2': r2}

def diebold_mariano_test(e1: np.ndarray, e2: np.ndarray) -> Tuple[float, float]:
    """
    Paired t-test variant of the DM test.
    H0: equal forecast accuracy (zero mean loss differential).
    e1, e2 are per-horizon mean absolute errors.
    """
    d  = e1 - e2           # loss differential: positive = e1 is worse
    n  = len(d)
    if n < 2 or d.std() == 0:
        return 0.0, 1.0
    dm = d.mean() / (d.std(ddof=1) / math.sqrt(n))
    p  = 2 * (1 - stats.t.cdf(abs(dm), df=n - 1))
    return float(dm), float(p)

def naive_seasonal_forecast(df: pd.DataFrame, target_col: str,
                             enc_len: int, dec_len: int) -> np.ndarray:
    """Seasonal-naïve baseline: repeat last 168-hour block."""
    preds = df[target_col].values[enc_len - dec_len: enc_len]
    return np.tile(preds, math.ceil(dec_len / len(preds)))[:dec_len]


# ── 17. EVALUATE ON TEST SET (per-horizon metrics) ────────────────────────────
def evaluate_test_set(model: TFT, cfg: TFTConfig, fc: dict,
                      df_sc: pd.DataFrame, test_df: pd.DataFrame,
                      scalers: dict, eval_step: int = 1) -> dict:
    """
    Runs direct multi-step forecasting for every window in the test set.
    Returns per-horizon MAE (identical evaluation grid to new_model.py).
    """
    model.eval()
    target    = cfg.target_col
    enc_len   = cfg.encoder_length
    dec_len   = cfg.decoder_length
    sc_tgt    = scalers[target]
    q50       = cfg.quantiles.index(0.5) if 0.5 in cfg.quantiles else 1

    all_cols  = fc['known_future'] + fc['past_observed'] + [target]
    dec_cols  = fc['known_future'] + fc['past_observed']

    full_arr  = df_sc[all_cols].values.astype(np.float32)
    dec_arr_f = df_sc[dec_cols].values.astype(np.float32)
    tgt_arr   = df_sc[target].values.astype(np.float32)

    # find where test starts in the full scaled df
    test_start_pos = df_sc.index.get_loc(test_df.index[0])

    horizons          = range(1, dec_len + 1)
    errors_per_horizon: Dict[int, list] = {h: [] for h in horizons}
    naive_errors:       Dict[int, list] = {h: [] for h in horizons}

    q_all_pred: list = []   # for quantile coverage [N*dec_len, nq]
    q_all_true: list = []

    indices = range(test_start_pos,
                    test_start_pos + len(test_df) - enc_len - dec_len,
                    eval_step)

    all_preds_unscaled, all_actuals_unscaled = [], []

    for abs_start in tqdm(indices, desc="Test-set evaluation"):
        if abs_start + enc_len + dec_len > len(df_sc): break

        enc_arr = full_arr[abs_start: abs_start + enc_len]
        d_arr   = dec_arr_f[abs_start + enc_len: abs_start + enc_len + dec_len]
        tgt_sl  = tgt_arr[abs_start + enc_len: abs_start + enc_len + dec_len]

        enc_x = torch.tensor(enc_arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        dec_x = torch.tensor(d_arr,   dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            preds_q, _ = model(enc_x, dec_x)           # [1, dec_len, nq]

        preds_q_np  = preds_q[0].cpu().numpy()         # [dec_len, nq]
        preds_med   = preds_q_np[:, q50]               # [dec_len]

        # inverse transform
        p_unsc = sc_tgt.inverse_transform(preds_med.reshape(-1, 1)).ravel()
        a_unsc = sc_tgt.inverse_transform(tgt_sl.reshape(-1, 1)).ravel()

        all_preds_unscaled.append(p_unsc)
        all_actuals_unscaled.append(a_unsc)

        # inverse-transform each quantile column independently
        n_q = preds_q_np.shape[1]
        q_inv = np.column_stack([
            sc_tgt.inverse_transform(preds_q_np[:, qi].reshape(-1, 1)).ravel()
            for qi in range(n_q)
        ])
        q_all_pred.append(q_inv)
        q_all_true.append(a_unsc)

        # naive seasonal baseline
        naive = naive_seasonal_forecast(
            df_sc.iloc[abs_start: abs_start + enc_len + dec_len],
            target, enc_len, dec_len)
        naive = sc_tgt.inverse_transform(naive.reshape(-1, 1)).ravel()

        for h_idx, (p, a, n) in enumerate(zip(p_unsc, a_unsc, naive)):
            h = h_idx + 1
            errors_per_horizon[h].append(abs(p - a))
            naive_errors[h].append(abs(n - a))

    mean_errors  = {h: np.mean(v) for h, v in errors_per_horizon.items() if v}
    naive_mean   = {h: np.mean(v) for h, v in naive_errors.items() if v}

    all_p = np.concatenate(all_preds_unscaled)
    all_a = np.concatenate(all_actuals_unscaled)
    overall = compute_metrics(all_a, all_p, "Overall test")

    # Diebold-Mariano: compare per-horizon mean errors of TFT vs naïve
    tft_h_mae   = np.array([np.mean(v) for v in errors_per_horizon.values() if v])
    naive_h_mae = np.array([np.mean(v) for v in naive_errors.values()        if v])
    dm_stat, dm_p = diebold_mariano_test(tft_h_mae, naive_h_mae)

    print(f"\n{'═'*60}")
    print("PER-HORIZON MAE (test set, direct multi-step)")
    print(f"{'Horizon (h)':>12} | {'TFT MAE':>10} | {'Naïve MAE':>10} | {'Skill':>8}")
    print("-" * 48)
    for h in range(1, dec_len + 1):
        if h in mean_errors:
            skill = 1 - mean_errors[h] / naive_mean.get(h, 1)
            print(f"{h:>12} | {mean_errors[h]:>10.2f} | {naive_mean.get(h, 0):>10.2f} | {skill:>8.3f}")

    overall_mae = np.mean([v for v in mean_errors.values()])
    print(f"\nOverall 7-Day MAE  : {overall_mae:.4f}")
    print(f"DM stat            : {dm_stat:.3f}   p={dm_p:.4f}")
    print(f"{'═'*60}\n")

    # quantile coverage
    q_pred_all = np.concatenate(q_all_pred, axis=0) if q_all_pred else np.zeros((1, len(cfg.quantiles)))
    q_true_all = np.concatenate(q_all_true)          if q_all_true else np.zeros(1)

    return {
        'mean_errors':  mean_errors,
        'naive_errors': naive_mean,
        'overall':      overall,
        'dm':           {'stat': dm_stat, 'p': dm_p},
        'q_pred':       q_pred_all,
        'q_true':       q_true_all,
        'all_preds':    all_p,
        'all_actuals':  all_a,
    }


# ── 18. VISUALISATION ─────────────────────────────────────────────────────────
def plot_all(results: dict, var_imp: dict, attn_map: np.ndarray,
             enc_names: List[str], dec_names: List[str],
             cfg: TFTConfig, art_dir: str):
    art = Path(art_dir)
    art.mkdir(parents=True, exist_ok=True)
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            plt.style.use('ggplot')

    # ── Fig 1: Per-horizon MAE
    fig, ax = plt.subplots(figsize=(14, 5))
    h_vals = list(results['mean_errors'].keys())
    tft_mae   = [results['mean_errors'][h] for h in h_vals]
    naive_mae = [results['naive_errors'].get(h, 0) for h in h_vals]
    ax.plot(h_vals, tft_mae,   label='TFT (recursive)', lw=2)
    ax.plot(h_vals, naive_mae, label='Seasonal Naïve', lw=1.5, ls='--', color='grey')
    ax.fill_between(h_vals, tft_mae, naive_mae,
                    where=[t < n for t, n in zip(tft_mae, naive_mae)],
                    alpha=0.15, color='green', label='TFT better')
    ax.set_xlabel('Forecast horizon (hours)'); ax.set_ylabel('MAE (MWh)')
    ax.set_title('Per-Horizon MAE — TFT vs Seasonal Naïve'); ax.legend()
    fig.tight_layout(); fig.savefig(art / 'per_horizon_mae.png', dpi=150)
    plt.close(fig)

    # ── Fig 2: Actual vs Predicted (first 2 weeks of test)
    n_show = min(336, len(results['all_actuals']))
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(results['all_actuals'][:n_show], label='Actual', lw=1.5, color='steelblue')
    ax.plot(results['all_preds'][:n_show],   label='TFT Prediction', lw=1.5,
            color='tomato', alpha=0.85)
    ax.set_xlabel('Hour'); ax.set_ylabel('Consumption (MWh)')
    ax.set_title('Actual vs Predicted — First 2 Weeks of Test Set'); ax.legend()
    fig.tight_layout(); fig.savefig(art / 'actual_vs_pred.png', dpi=150)
    plt.close(fig)

    # ── Fig 3: Quantile bands (first 168 hours)
    if results['q_pred'].shape[0] >= 168:
        fig, ax = plt.subplots(figsize=(16, 5))
        n = 168
        q_p = results['q_pred'][:n]
        ax.fill_between(range(n), q_p[:, 0], q_p[:, -1], alpha=0.25,
                        color='orange', label='10-90% interval')
        ax.plot(results['all_actuals'][:n], lw=1.5, color='steelblue', label='Actual')
        ax.plot(q_p[:, len(cfg.quantiles)//2], lw=1.5, color='tomato', label='Median pred')
        ax.set_title('7-Day Forecast with Uncertainty Bands (q10–q90)')
        ax.legend(); fig.tight_layout(); fig.savefig(art / 'quantile_bands.png', dpi=150)
        plt.close(fig)

    # ── Fig 4: Encoder variable importance
    enc_imp = var_imp.get('encoder', {})
    if enc_imp:
        fig, ax = plt.subplots(figsize=(10, max(4, len(enc_imp) * 0.35)))
        names = list(enc_imp.keys()); vals = list(enc_imp.values())
        order = np.argsort(vals)
        ax.barh([names[i] for i in order], [vals[i] for i in order], color='steelblue')
        ax.set_title('Encoder Variable Importance (VSN weights)')
        ax.set_xlabel('Mean selection weight')
        fig.tight_layout(); fig.savefig(art / 'enc_var_importance.png', dpi=150)
        plt.close(fig)

    # ── Fig 5: Decoder variable importance
    dec_imp = var_imp.get('decoder', {})
    if dec_imp:
        fig, ax = plt.subplots(figsize=(10, max(4, len(dec_imp) * 0.35)))
        names = list(dec_imp.keys()); vals = list(dec_imp.values())
        order = np.argsort(vals)
        ax.barh([names[i] for i in order], [vals[i] for i in order], color='darkorange')
        ax.set_title('Decoder Variable Importance (VSN weights)')
        ax.set_xlabel('Mean selection weight')
        fig.tight_layout(); fig.savefig(art / 'dec_var_importance.png', dpi=150)
        plt.close(fig)

    # ── Fig 6: Temporal attention heatmap
    if attn_map is not None and attn_map.size > 0:
        # downsample for readability
        T_dec, T_enc = attn_map.shape
        step_d = max(1, T_dec // 24); step_e = max(1, T_enc // 48)
        am = attn_map[::step_d, ::step_e]
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(am, cmap='Blues', ax=ax, xticklabels=False, yticklabels=False)
        ax.set_xlabel('Encoder timestep'); ax.set_ylabel('Decoder step (hour)')
        ax.set_title('Temporal Self-Attention — Decoder → Encoder (first forecast window)')
        fig.tight_layout(); fig.savefig(art / 'attention_heatmap.png', dpi=150)
        plt.close(fig)

    # ── Fig 7: Error distribution
    errors = results['all_actuals'] - results['all_preds']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].hist(errors, bins=80, color='steelblue', edgecolor='white')
    axes[0].set_title('Residual distribution'); axes[0].set_xlabel('Error (MWh)')
    stats.probplot(errors, plot=axes[1])
    axes[1].set_title('QQ Plot of residuals')
    acf_vals = tsacf(errors, nlags=48, fft=True)
    ci = 1.96 / np.sqrt(len(errors))
    axes[2].bar(range(len(acf_vals)), acf_vals, color='steelblue', width=0.6)
    axes[2].axhline(ci,  ls='--', color='r', lw=1)
    axes[2].axhline(-ci, ls='--', color='r', lw=1)
    axes[2].set_title('Residual ACF (48 lags)'); axes[2].set_xlabel('Lag (h)')
    fig.tight_layout(); fig.savefig(art / 'residual_analysis.png', dpi=150)
    plt.close(fig)

    # ── Fig 8: Skill score per horizon
    skill_scores = [1 - results['mean_errors'].get(h, 0) / max(results['naive_errors'].get(h, 1), 1e-8)
                    for h in range(1, cfg.decoder_length + 1)]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(1, len(skill_scores)+1), skill_scores, color='mediumseagreen', alpha=0.8)
    ax.axhline(0, color='red', ls='--', lw=1)
    ax.set_xlabel('Forecast horizon (h)'); ax.set_ylabel('Skill score (vs Naïve)')
    ax.set_title('Forecast Skill Score per Horizon')
    fig.tight_layout(); fig.savefig(art / 'skill_scores.png', dpi=150)
    plt.close(fig)

    print(f"[Viz] 8 plots saved to {art}/")


# ── 19. MAIN ──────────────────────────────────────────────────────────────────
def main():
    cfg = TFTConfig()
    Path(cfg.artifact_dir).mkdir(parents=True, exist_ok=True)

    # ── 19.1 Data
    print("=" * 70)
    print("STEP 1 — Loading and preprocessing data")
    print("=" * 70)
    df, df_sc, train_df, val_df, test_df, scalers, fc = load_data(cfg)

    # ── 19.2 Optuna hyper-parameter search
    print("\n" + "=" * 70)
    print("STEP 2 — Optuna hyperparameter optimisation")
    print("=" * 70)
    best_params, study = run_optuna(cfg, df_sc, fc)

    # Update cfg with best params
    for k, v in best_params.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    # ── 19.3 Train final model on full train+val
    print("\n" + "=" * 70)
    print("STEP 3 — Training final TFT with best hyperparameters")
    print("=" * 70)
    # train on train split, validate on val split
    tr_ds = TFTDataset(train_df, cfg, fc, stride=1)
    va_ds = TFTDataset(val_df,   cfg, fc, stride=2)
    tr_dl = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    print(f"  train samples={len(tr_ds):,}  val samples={len(va_ds):,}")
    model = TFT(cfg, tr_ds.n_enc, tr_ds.n_dec).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  TFT parameters: {total_params:,}")

    trainer = Trainer(model, cfg, tr_dl, va_dl)
    best_val = trainer.fit(verbose=True)
    print(f"  Final best val loss: {best_val:.6f}")

    # save model
    ckpt_path = Path(cfg.artifact_dir) / "tft_best.pt"
    torch.save({'model_state': model.state_dict(),
                'cfg': cfg.__dict__,
                'feat_cfg': fc}, ckpt_path)
    print(f"  Checkpoint saved: {ckpt_path}")

    # ── 19.4 Per-horizon evaluation on full test set
    print("\n" + "=" * 70)
    print("STEP 4 — Evaluating on test set (all 168-hour horizons)")
    print("=" * 70)
    results = evaluate_test_set(model, cfg, fc, df_sc, test_df, scalers, eval_step=1)

    # ── 19.5 Online learning demonstration
    print("\n" + "=" * 70)
    print("STEP 5 — Online learning (Page-Hinkley + EWC fine-tuning)")
    print("=" * 70)
    learner = OnlineLearner(model, cfg, fc, scalers, tr_ds)

    sc_tgt       = scalers[cfg.target_col]
    test_arr     = test_df[cfg.target_col].values
    test_abs_pos = df_sc.index.get_loc(test_df.index[0])   # int position in df_sc
    n_stream     = min(2000, len(test_arr))
    print(f"  Simulating {n_stream} streaming observations …")
    for i in tqdm(range(0, n_stream - cfg.encoder_length - cfg.decoder_length, 168),
                  desc="Online stream"):
        abs_s = test_abs_pos + i
        _, preds_mwh, _ = recursive_forecast_168(
            model, cfg, fc, df_sc, scalers, abs_s)
        actuals_mwh = sc_tgt.inverse_transform(
            test_arr[i: i + cfg.decoder_length].reshape(-1, 1)).ravel()
        horizon_mae = mean_absolute_error(actuals_mwh, preds_mwh)

        # recent data window for potential fine-tuning (last ol_window rows before current pos)
        ol_pos_start = max(0, abs_s - cfg.ol_window)
        df_recent = df_sc.iloc[ol_pos_start: abs_s + cfg.decoder_length]
        learner.observe(
            horizon_mae,
            df_recent if len(df_recent) > cfg.encoder_length + cfg.decoder_length else None
        )

    print(f"  Online learning complete — drift events: {learner.drift_count}")

    # ── 19.6 Explainability
    print("\n" + "=" * 70)
    print("STEP 6 — Explainability (XAI)")
    print("=" * 70)
    explainer = TFTExplainer(model, cfg, fc, scalers, cfg.artifact_dir)

    # Variable importance over 30 test batches
    te_ds = TFTDataset(test_df, cfg, fc, stride=4)
    te_dl = DataLoader(te_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    var_imp = explainer.variable_importance(te_dl, n_batches=30)
    print("\n  Encoder variable importance (top 10):")
    enc_sorted = sorted(var_imp['encoder'].items(), key=lambda x: -x[1])
    for feat, w in enc_sorted[:10]:
        print(f"    {feat:>25}  {w:.4f}")
    print("\n  Decoder variable importance (top 10):")
    dec_sorted = sorted(var_imp['decoder'].items(), key=lambda x: -x[1])
    for feat, w in dec_sorted[:10]:
        print(f"    {feat:>25}  {w:.4f}")

    # Attention heatmap for first test window
    test_start_pos = df_sc.index.get_loc(test_df.index[0])
    all_cols = fc['known_future'] + fc['past_observed'] + [cfg.target_col]
    dec_cols = fc['known_future'] + fc['past_observed']
    enc_eg = df_sc[all_cols].values[test_start_pos: test_start_pos + cfg.encoder_length].astype(np.float32)
    dec_eg = df_sc[dec_cols].values[test_start_pos + cfg.encoder_length:
                                     test_start_pos + cfg.encoder_length + cfg.decoder_length].astype(np.float32)
    attn_map = explainer.attention_heatmap(enc_eg, dec_eg)

    # SHAP (if available)
    if HAS_SHAP:
        print("\n  Running SHAP DeepExplainer …")
        shap_vals = explainer.shap_analysis(te_dl, te_dl, n_bg=64, n_test=32)
        if shap_vals is not None:
            print(f"  SHAP values shape: {np.array(shap_vals).shape}")

    # Quantile coverage
    cov = explainer.quantile_coverage(results['q_true'], results['q_pred'])
    print(f"\n  Quantile coverage: nominal={cov['nominal_coverage']:.2f}  "
          f"empirical={cov['empirical_coverage']:.3f}  "
          f"gap={cov['coverage_gap']:+.3f}")

    # ── 19.7 Truly recursive showcase (last 168 hours)
    print("\n" + "=" * 70)
    print("STEP 7 — Truly recursive 168-hour showcase forecast")
    print("=" * 70)
    last_window_pos = (len(df_sc)
                       - cfg.encoder_length
                       - cfg.decoder_length
                       - 1)
    _, preds_mwh_rc, rc_interp = recursive_forecast_168(
        model, cfg, fc, df_sc, scalers, last_window_pos)
    actual_rc = sc_tgt.inverse_transform(
        df_sc[cfg.target_col].values[
            last_window_pos + cfg.encoder_length:
            last_window_pos + cfg.encoder_length + cfg.decoder_length
        ].reshape(-1, 1)
    ).ravel()
    rc_metrics = compute_metrics(actual_rc, preds_mwh_rc, "Recursive-168")
    print(f"  Recursive forecast complete.")

    # ── 19.8 Visualisations
    print("\n" + "=" * 70)
    print("STEP 8 — Generating visualisations")
    print("=" * 70)
    plot_all(results, var_imp, attn_map,
             explainer.enc_names, explainer.dec_names, cfg, cfg.artifact_dir)

    # ── 19.9 Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    ov = results['overall']
    print(f"  Test MAE   : {ov['mae']:.2f} MWh")
    print(f"  Test RMSE  : {ov['rmse']:.2f} MWh")
    print(f"  Test MAPE  : {ov['mape']:.2f}%")
    print(f"  Test sMAPE : {ov['smape']:.2f}%")
    print(f"  Test R²    : {ov['r2']:.4f}")
    print(f"  DM test    : stat={results['dm']['stat']:.3f}  p={results['dm']['p']:.4f}")
    print(f"  Coverage   : {cov['empirical_coverage']:.3f} (nominal {cov['nominal_coverage']:.2f})")
    print(f"  Drift events (online): {learner.drift_count}")
    print(f"  Artefacts  : {cfg.artifact_dir}/")
    print("=" * 70)

    # save results
    save_path = Path(cfg.artifact_dir) / "results.json"
    save_data = {
        'overall_metrics': ov,
        'per_horizon_mae': {str(k): float(v) for k, v in results['mean_errors'].items()},
        'recursive_metrics': rc_metrics,
        'quantile_coverage': cov,
        'dm_test': results['dm'],
        'best_params': best_params,
        'total_params': total_params,
    }
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"  Results saved: {save_path}")


if __name__ == '__main__':
    main()
