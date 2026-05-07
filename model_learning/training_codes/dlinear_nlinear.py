"""
dlinear_nlinear.py
══════════════════════════════════════════════════════════════════════════════
MODEL 17 — DLINEAR and NLINEAR
"Are Transformers Effective for Time Series Forecasting?"  (Zeng et al., 2023)

Mathematical Foundation
────────────────────────
DLinear
  1. Moving-average decomposition (kernel K):
        T_t = (1/K) Σ_{j=-(K//2)}^{K//2} y_{t+j}        (trend)
        R_t = y_t - T_t                                   (remainder / seasonal)
  2. Channel-independent linear projections:
        ŷ = W_T · T + W_R · R
     W_T, W_R ∈ R^{H×L}  (H = forecast horizon, L = lookback window)
  Despite extreme parameter count (~2·L·H per channel), DLinear outperformed
  many Transformer-based models on long-horizon benchmarks.

NLinear
  1. Instance normalisation by last observed value:
        x' = x − x_L                                      (remove distribution shift)
  2. Linear projection + re-add removed offset:
        ŷ = Linear(x') + x_L
  The subtraction of x_L makes the model invariant to the level of the series,
  dramatically improving generalisation across distribution shifts.

Enhancements
  • RevIN (Reversible Instance Normalisation) for both models
    μ = mean(x),  σ = std(x)
    x_norm = (x − μ) / (σ + ε)  →  project  →  ŷ · σ + μ
  • Channel-independent (CI) strategy: one linear weight per input feature
  • Weighted ensemble: DLinear + NLinear with learnable mixing weights
  • Bootstrap prediction intervals (T random subsets of features)
  • Analytical linear-SHAP  (exact, not approximation)
    φ_i = w_i · (x_i − E[x_i])

Evaluation
  • Per-horizon MAE for all 168 forecast steps (matches new_model.py grid)
  • Overall 7-day recursive MAE / RMSE / MAPE / sMAPE
  • Diebold-Mariano test vs. seasonal-naïve baseline
  • Direct multi-step + truly recursive (step-by-step) modes

Usage
  python dlinear_nlinear.py
"""

# ── 0. IMPORTS ────────────────────────────────────────────────────────────────
import warnings; warnings.filterwarnings("ignore")
import os, sys, math, json, copy, time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {DEVICE}")


# ── 1. CONFIGURATION ──────────────────────────────────────────────────────────
@dataclass
class DLConfig:
    csv_file:       str   = "cleaned_energy_data_model.csv"
    timestamp_col:  str   = "start_time"
    target_col:     str   = "consumption"
    test_cutoff:    str   = "2021-01-01"
    val_cutoff:     str   = "2020-10-01"

    # sequence
    encoder_length: int   = 336      # lookback L  (tuned by Optuna)
    decoder_length: int   = 168      # forecast H  (fixed)

    # DLinear
    kernel_size:    int   = 25       # moving-average kernel K
    individual:     bool  = True     # channel-independent (CI) strategy

    # training
    batch_size:     int   = 64
    lr:             float = 1e-3
    weight_decay:   float = 1e-4
    max_epochs:     int   = 100
    patience:       int   = 15
    grad_clip:      float = 5.0

    # Optuna
    n_trials:       int   = 40

    # inference
    n_bootstrap:    int   = 50       # bootstrap samples for CI estimation

    # output
    artifact_dir:   str   = "dl_artifacts"


# ── 2. DATA LOADING ───────────────────────────────────────────────────────────
KNOWN_FUTURE = ['hour','day_of_week','month','year','season',
                'is_weekend','is_holiday','is_first_day','is_last_day']
PAST_OBS     = ['lag_1h','lag_24h','lag_48h','lag_72h','lag_168h',
                'prev_daily_mean','prev_daily_max',
                'rolling_mean_24h','rolling_mean_168h','rolling_std_24h',
                'ewm_24h','how_historical_mean','resid_vs_how_mean']

def load_data(cfg: DLConfig):
    df = pd.read_csv(cfg.csv_file)
    df[cfg.timestamp_col] = pd.to_datetime(df[cfg.timestamp_col])
    df.set_index(cfg.timestamp_col, inplace=True)
    df.sort_index(inplace=True)
    if 'end_time_utc' in df.columns:
        df.drop(columns=['end_time_utc'], inplace=True)

    # cyclic encodings
    for base, period in [('hour', 24), ('day_of_week', 7), ('month', 12)]:
        if base in df.columns:
            df[f'{base}_sin'] = np.sin(2 * np.pi * df[base] / period)
            df[f'{base}_cos'] = np.cos(2 * np.pi * df[base] / period)

    df.dropna(inplace=True)

    feat_cols = (
        [c for c in KNOWN_FUTURE if c in df.columns] +
        [f'{b}_{s}' for b in ('hour','day_of_week','month') for s in ('sin','cos')
         if f'{b}_{s}' in df.columns] +
        [c for c in PAST_OBS if c in df.columns]
    )
    all_cols = feat_cols + [cfg.target_col]

    # target-channel index in all_cols
    target_idx = all_cols.index(cfg.target_col)

    # normalise (fit on train)
    train_mask = df.index < cfg.test_cutoff
    scalers: Dict[str, RobustScaler] = {}
    df_sc = df.copy()
    for col in all_cols:
        sc = RobustScaler()
        sc.fit(df.loc[train_mask, [col]])
        df_sc[col] = sc.transform(df[[col]]).ravel()
        scalers[col] = sc

    train_df = df_sc[df_sc.index <  cfg.val_cutoff]
    val_df   = df_sc[(df_sc.index >= cfg.val_cutoff) & (df_sc.index < cfg.test_cutoff)]
    test_df  = df_sc[df_sc.index >= cfg.test_cutoff]

    print(f"[Data]  rows={len(df):,}  features={len(feat_cols)}  target_idx={target_idx}")
    print(f"[Split] train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    return df, df_sc, train_df, val_df, test_df, scalers, all_cols, target_idx


# ── 3. DATASET ────────────────────────────────────────────────────────────────
class TSDataset(Dataset):
    """
    enc_x  : [encoder_length, n_features]   (all features incl. target)
    target : [decoder_length]                (target channel only)
    """
    def __init__(self, df_sc: pd.DataFrame, cfg: DLConfig,
                 all_cols: List[str], target_col: str, stride: int = 1):
        self.arr = df_sc[all_cols].values.astype(np.float32)
        tgt_idx  = all_cols.index(target_col)
        self.tgt = df_sc[target_col].values.astype(np.float32)
        self.L   = cfg.encoder_length
        self.H   = cfg.decoder_length
        self.n   = self.arr.shape[1]
        max_start = len(df_sc) - self.L - self.H
        self.idx  = list(range(0, max(0, max_start), stride))

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        s = self.idx[i]
        enc = self.arr[s: s + self.L]
        tgt = self.tgt[s + self.L: s + self.L + self.H]
        return torch.from_numpy(enc.copy()), torch.from_numpy(tgt.copy())


# ── 4. REVIN (Reversible Instance Normalisation) ─────────────────────────────
class RevIN(nn.Module):
    """
    Normalises each input channel to zero mean / unit variance.
    Stores the statistics so the output can be denormalised.
    """
    def __init__(self, n_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(n_channels))
            self.beta  = nn.Parameter(torch.zeros(n_channels))

    def forward(self, x, mode: str):
        """
        x    : [B, L, C]
        mode : 'norm' (store stats & normalise) | 'denorm' (reverse)
        """
        if mode == 'norm':
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._std  = x.std(dim=1, keepdim=True).detach() + self.eps
            x = (x - self._mean) / self._std
            if self.affine:
                x = x * self.gamma + self.beta
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.beta) / (self.gamma + self.eps)
            x = x * self._std[:, :1, :] + self._mean[:, :1, :]   # broadcast to [B,H,C]
        return x


# ── 5. DLINEAR ────────────────────────────────────────────────────────────────
class MovingAvg(nn.Module):
    """
    Causal moving-average filter using 1-D average pooling.
    Applies replicate padding on the LEFT only (causal — no future leakage).
    """
    def __init__(self, kernel_size: int):
        super().__init__()
        self.k = kernel_size

    def forward(self, x):
        # x: [B, L, C]  →  transpose to [B, C, L] for avg_pool1d
        x_t   = x.transpose(1, 2)
        pad_x = F.pad(x_t, (self.k - 1, 0), mode='replicate')   # left-pad only
        trend = F.avg_pool1d(pad_x, kernel_size=self.k, stride=1, padding=0)
        return trend.transpose(1, 2)   # [B, L, C]


class DLinear(nn.Module):
    """
    DLinear — Decomposition Linear (Zeng et al., 2023).

    ŷ_c = W_T_c · T_c + W_R_c · R_c     (per channel c, shape L → H)

    individual=True  : one pair (W_T, W_R) per channel  — channel-independent (CI)
    individual=False : shared weights across all channels
    use_revin=True   : apply RevIN before decomposition
    """
    def __init__(self, cfg: DLConfig, n_features: int, use_revin: bool = False):
        super().__init__()
        L, H, K = cfg.encoder_length, cfg.decoder_length, cfg.kernel_size
        self.L = L; self.H = H; self.F = n_features
        self.individual = cfg.individual
        self.use_revin  = use_revin
        self.ma  = MovingAvg(K)

        if use_revin:
            self.revin = RevIN(n_features)

        if cfg.individual:
            self.W_T = nn.ModuleList([nn.Linear(L, H, bias=True) for _ in range(n_features)])
            self.W_R = nn.ModuleList([nn.Linear(L, H, bias=True) for _ in range(n_features)])
        else:
            self.W_T = nn.Linear(L, H, bias=True)
            self.W_R = nn.Linear(L, H, bias=True)

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        self.apply(_init)

    def decompose(self, x):
        """x: [B, L, F] → trend [B, L, F], remainder [B, L, F]"""
        trend    = self.ma(x)
        seasonal = x - trend
        return trend, seasonal

    def forward(self, x):
        # x: [B, L, F]
        if self.use_revin:
            x = self.revin(x, 'norm')

        trend, seasonal = self.decompose(x)             # [B, L, F] each

        if self.individual:
            preds = torch.stack(
                [self.W_T[c](trend[:, :, c]) + self.W_R[c](seasonal[:, :, c])
                 for c in range(self.F)], dim=-1)        # [B, H, F]
        else:
            # shared: transpose to [B, F, L], apply linear, transpose back
            preds = (self.W_T(trend.transpose(1, 2)) +
                     self.W_R(seasonal.transpose(1, 2))).transpose(1, 2)  # [B, H, F]

        if self.use_revin:
            preds = self.revin(preds, 'denorm')

        return preds   # [B, H, F]


# ── 6. NLINEAR ────────────────────────────────────────────────────────────────
class NLinear(nn.Module):
    """
    NLinear — Normalised Linear (Zeng et al., 2023).

    x'  = x − x_L                             (last-value normalisation)
    ŷ_c = W_c · x'_c  +  x_L_c               (per channel c)

    use_revin=True : RevIN wraps the normalisation for extra robustness
    """
    def __init__(self, cfg: DLConfig, n_features: int, use_revin: bool = False):
        super().__init__()
        L, H = cfg.encoder_length, cfg.decoder_length
        self.L = L; self.H = H; self.F = n_features
        self.individual = cfg.individual
        self.use_revin  = use_revin

        if use_revin:
            self.revin = RevIN(n_features)

        if cfg.individual:
            self.W = nn.ModuleList([nn.Linear(L, H, bias=False) for _ in range(n_features)])
        else:
            self.W = nn.Linear(L, H, bias=False)

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                # small initialisation prevents last-value domination early in training
                nn.init.normal_(m.weight, std=0.01)
        self.apply(_init)

    def forward(self, x):
        # x: [B, L, F]
        if self.use_revin:
            x = self.revin(x, 'norm')

        last  = x[:, -1:, :]           # [B, 1, F]  — last observed value per channel
        x_n   = x - last               # subtract; removes distribution shift

        if self.individual:
            preds = torch.stack(
                [self.W[c](x_n[:, :, c]) + last[:, 0, c:c+1].expand(-1, self.H)
                 for c in range(self.F)], dim=-1)          # [B, H, F]
        else:
            # x_n transposed: [B, F, L] → linear → [B, F, H] → [B, H, F]
            preds = (self.W(x_n.transpose(1, 2)).transpose(1, 2) +
                     last.expand(-1, self.H, -1))

        if self.use_revin:
            preds = self.revin(preds, 'denorm')

        return preds   # [B, H, F]


# ── 7. WEIGHTED ENSEMBLE ──────────────────────────────────────────────────────
class DNEnsemble(nn.Module):
    """
    Learnable convex combination of DLinear and NLinear (with RevIN).
    w = softmax([a, b])  →  ŷ = w[0]*ŷ_D + w[1]*ŷ_N
    """
    def __init__(self, dlinear: DLinear, nlinear: NLinear):
        super().__init__()
        self.dl  = dlinear
        self.nl  = nlinear
        self.log_w = nn.Parameter(torch.zeros(2))   # learnable mixing logits

    def forward(self, x):
        w = F.softmax(self.log_w, dim=0)
        return w[0] * self.dl(x) + w[1] * self.nl(x)

    @property
    def weights(self):
        return F.softmax(self.log_w.detach(), dim=0).cpu().numpy()


# ── 8. LOSS FUNCTION ──────────────────────────────────────────────────────────
def huber_mse_loss(pred: torch.Tensor, tgt: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Huber + MSE combined for robustness and curvature near optimum."""
    return 0.5 * F.mse_loss(pred, tgt) + 0.5 * F.huber_loss(pred, tgt, delta=delta)


# ── 9. TRAINER ────────────────────────────────────────────────────────────────
class Trainer:
    def __init__(self, model: nn.Module, cfg: DLConfig,
                 tr_dl: DataLoader, va_dl: DataLoader, target_idx: int):
        self.model      = model.to(DEVICE)
        self.cfg        = cfg
        self.tr_dl      = tr_dl
        self.va_dl      = va_dl
        self.target_idx = target_idx
        steps           = len(tr_dl) * cfg.max_epochs
        self.opt   = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.sched = OneCycleLR(self.opt, max_lr=cfg.lr, total_steps=steps,
                                pct_start=0.05, anneal_strategy='cos')
        self.best_val   = float('inf')
        self.best_state = None

    def _run(self, loader, train: bool) -> float:
        self.model.train(train)
        total = 0.
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for enc_x, tgt in loader:
                enc_x, tgt = enc_x.to(DEVICE), tgt.to(DEVICE)
                if train: self.opt.zero_grad()
                out  = self.model(enc_x)              # [B, H, F]
                pred = out[:, :, self.target_idx]      # [B, H]
                loss = huber_mse_loss(pred, tgt)
                if train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.opt.step(); self.sched.step()
                total += loss.item() * enc_x.size(0)
        return total / len(loader.dataset)

    def fit(self, verbose: bool = True) -> float:
        patience_left = self.cfg.patience
        for ep in range(1, self.cfg.max_epochs + 1):
            tr  = self._run(self.tr_dl, True)
            val = self._run(self.va_dl, False)
            if verbose and ep % 10 == 0:
                print(f"  epoch {ep:3d}/{self.cfg.max_epochs}  "
                      f"train={tr:.5f}  val={val:.5f}")
            if val < self.best_val:
                self.best_val   = val
                self.best_state = copy.deepcopy(self.model.state_dict())
                patience_left   = self.cfg.patience
            else:
                patience_left -= 1
                if patience_left == 0:
                    if verbose: print(f"  [early stop] epoch {ep}")
                    break
        self.model.load_state_dict(self.best_state)
        return self.best_val


# ── 10. OPTUNA HYPERPARAMETER SEARCH ──────────────────────────────────────────
def _objective(trial, cfg: DLConfig, df_sc, all_cols, target_idx) -> float:
    enc_len = trial.suggest_categorical('encoder_length', [168, 336, 504, 720])
    kernel  = trial.suggest_categorical('kernel_size',    [5, 13, 25, 49])
    indiv   = trial.suggest_categorical('individual',     [True, False])
    lr      = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    bsz     = trial.suggest_categorical('batch_size',     [32, 64, 128])
    model_t = trial.suggest_categorical('model_type',     ['dlinear','nlinear','ensemble'])

    tcfg = DLConfig(encoder_length=enc_len, kernel_size=kernel,
                    individual=indiv, lr=lr, batch_size=bsz,
                    max_epochs=25, patience=6)

    tr_df = df_sc[df_sc.index <  cfg.val_cutoff]
    va_df = df_sc[(df_sc.index >= cfg.val_cutoff) & (df_sc.index < cfg.test_cutoff)]

    try:
        tr_ds = TSDataset(tr_df, tcfg, all_cols, cfg.target_col, stride=4)
        va_ds = TSDataset(va_df, tcfg, all_cols, cfg.target_col, stride=4)
        if len(tr_ds) < 8 or len(va_ds) < 4:
            raise optuna.exceptions.TrialPruned()

        n_f   = len(all_cols)
        tr_dl = DataLoader(tr_ds, batch_size=bsz, shuffle=True,  num_workers=0)
        va_dl = DataLoader(va_ds, batch_size=bsz, shuffle=False, num_workers=0)

        if model_t == 'dlinear':
            model = DLinear(tcfg, n_f, use_revin=True)
        elif model_t == 'nlinear':
            model = NLinear(tcfg, n_f, use_revin=True)
        else:
            dl = DLinear(tcfg, n_f, use_revin=True)
            nl = NLinear(tcfg, n_f, use_revin=True)
            model = DNEnsemble(dl, nl)

        trainer = Trainer(model, tcfg, tr_dl, va_dl, target_idx)
        val     = trainer.fit(verbose=False)

        trial.report(val, 0)
        if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        return val
    except (RuntimeError, ValueError):
        raise optuna.exceptions.TrialPruned()


def run_optuna(cfg: DLConfig, df_sc, all_cols, target_idx) -> dict:
    print(f"\n[Optuna] {cfg.n_trials} trials  |  TPE + Median pruner")
    sampler = TPESampler(seed=SEED, n_startup_trials=8, multivariate=True)
    pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    study   = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
    study.optimize(lambda t: _objective(t, cfg, df_sc, all_cols, target_idx),
                   n_trials=cfg.n_trials, show_progress_bar=True, gc_after_trial=True)
    best = study.best_params
    print(f"\n[Optuna] best val loss : {study.best_value:.6f}")
    print(f"[Optuna] best params   : {json.dumps({k: str(v) for k,v in best.items()}, indent=2)}")
    return best, study


# ── 11. RECURSIVE INFERENCE HELPERS ───────────────────────────────────────────
_LAG_MAP = {'lag_1h': 1, 'lag_24h': 24, 'lag_48h': 48, 'lag_72h': 72, 'lag_168h': 168}

def _recompute_past_obs(col: str, hist_raw: np.ndarray) -> float:
    """Recompute a single past-observed feature from raw MWh history."""
    n = len(hist_raw)
    if col in _LAG_MAP:
        sh = _LAG_MAP[col]
        return float(hist_raw[-sh]) if n >= sh else float(hist_raw[-1])
    if col == 'rolling_mean_24h':
        return float(hist_raw[-24:].mean()) if n >= 24 else float(hist_raw.mean())
    if col == 'rolling_std_24h':
        return float(hist_raw[-24:].std())  if n >= 24 else 0.0
    if col == 'rolling_mean_168h':
        return float(hist_raw[-168:].mean()) if n >= 168 else float(hist_raw.mean())
    if col == 'ewm_24h':
        return float(pd.Series(hist_raw[-72:].astype(float)).ewm(span=24).mean().iloc[-1])
    if col in ('prev_daily_mean', 'how_historical_mean'):
        return float(hist_raw[-24:].mean()) if n >= 24 else float(hist_raw.mean())
    if col == 'prev_daily_max':
        return float(hist_raw[-24:].max()) if n >= 24 else float(hist_raw.max())
    if col == 'resid_vs_how_mean':
        how_m = float(hist_raw[-168:].mean()) if n >= 168 else float(hist_raw.mean())
        return float(hist_raw[-1]) - how_m
    return 0.0


@torch.no_grad()
def recursive_forecast_168(model: nn.Module, cfg: DLConfig,
                            df_sc: pd.DataFrame, all_cols: List[str],
                            scalers: dict, abs_start: int,
                            target_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Truly recursive 168-step forecast (step-by-step).

    At each step h:
      1. Build encoder window [abs_start + h − L : abs_start + h]
         using actual data for the initial L − h positions, then
         appending h already-predicted values (raw MWh converted back to
         scaled form for each feature column).
      2. Recompute all past-observed lag/rolling features from the
         growing raw-MWh history buffer for each new predicted position.
      3. Run the model; extract the target channel output for step h.

    Returns preds_raw (scaled) and preds_mwh (inverse-transformed).
    """
    model.eval()
    L, H = cfg.encoder_length, cfg.decoder_length
    sc_tgt = scalers[cfg.target_col]

    # initial raw MWh history from encoder window
    enc_window = df_sc.iloc[abs_start: abs_start + L]
    hist_raw   = sc_tgt.inverse_transform(
        enc_window[cfg.target_col].values.reshape(-1, 1)
    ).ravel().astype(np.float64)

    preds_raw = np.zeros(H, dtype=np.float32)
    preds_mwh = np.zeros(H, dtype=np.float64)

    # base encoder array (we'll modify past-obs columns step by step)
    enc_arr = enc_window[all_cols].values.astype(np.float32).copy()   # [L, F]

    for h in range(H):
        # at step h, enc_arr already has the correct values for positions 0..h-1
        # updated by previous iterations

        x = torch.tensor(enc_arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # [1, L, F]
        out = model(x)                   # [1, H, F]
        pred_sc  = out[0, 0, target_idx].item()   # prediction for 1 step ahead
        pred_mwh = float(sc_tgt.inverse_transform([[pred_sc]])[0, 0])

        preds_raw[h] = pred_sc
        preds_mwh[h] = pred_mwh

        # slide encoder window one step forward
        # new last row = features for the next timestep
        if h < H - 1:
            # shift enc_arr: drop oldest, append a new row
            enc_arr = np.roll(enc_arr, -1, axis=0)

            # new row: copy time features from pre-computed future window
            future_pos = abs_start + L + h
            if future_pos < len(df_sc):
                new_row = df_sc[all_cols].values[future_pos].copy().astype(np.float32)
            else:
                new_row = enc_arr[-2].copy()   # fallback: repeat last row

            # override past-observed columns with dynamically computed values
            full_hist = np.append(hist_raw, preds_mwh[:h+1])
            for j, col in enumerate(all_cols):
                if col in PAST_OBS:
                    raw_val = _recompute_past_obs(col, full_hist)
                    if col in scalers:
                        new_row[j] = float(scalers[col].transform([[raw_val]])[0, 0])
                    else:
                        new_row[j] = raw_val
                elif col == cfg.target_col:
                    new_row[j] = pred_sc   # target channel = model's own prediction

            enc_arr[-1] = new_row

    return preds_raw, preds_mwh


# ── 12. METRICS ───────────────────────────────────────────────────────────────
def smape(a: np.ndarray, p: np.ndarray) -> float:
    d = (np.abs(a) + np.abs(p)) / 2
    return float(np.mean(np.where(d == 0, 0, np.abs(a - p) / d)) * 100)

def mape(a: np.ndarray, p: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((a - p) / (np.abs(a) + eps))) * 100)

def compute_metrics(actuals, preds, label='') -> dict:
    mae   = mean_absolute_error(actuals, preds)
    rmse  = float(np.sqrt(mean_squared_error(actuals, preds)))
    r2    = r2_score(actuals, preds)
    _mape = mape(actuals, preds)
    _sm   = smape(actuals, preds)
    if label:
        print(f"  [{label}]  MAE={mae:.2f}  RMSE={rmse:.2f}  "
              f"MAPE={_mape:.2f}%  sMAPE={_sm:.2f}%  R²={r2:.4f}")
    return {'mae': mae, 'rmse': rmse, 'mape': _mape, 'smape': _sm, 'r2': r2}

def dm_test(e1: np.ndarray, e2: np.ndarray) -> Tuple[float, float]:
    """Paired-t DM test on per-horizon MAE arrays."""
    d = e1 - e2; n = len(d)
    if n < 2 or d.std() == 0: return 0., 1.
    dm = d.mean() / (d.std(ddof=1) / math.sqrt(n))
    return float(dm), float(2 * (1 - stats.t.cdf(abs(dm), df=n - 1)))

def naive_seasonal(arr_sc, target_col, enc_len, dec_len, sc_tgt):
    """Seasonal-naïve: repeat last 168 hours of encoder."""
    block = arr_sc[enc_len - dec_len: enc_len]
    preds_sc = np.tile(block, math.ceil(dec_len / len(block)))[:dec_len]
    return sc_tgt.inverse_transform(preds_sc.reshape(-1, 1)).ravel()


# ── 13. TEST-SET EVALUATION (direct multi-step, per-horizon) ──────────────────
@torch.no_grad()
def evaluate_test_set(model: nn.Module, cfg: DLConfig,
                      df_sc: pd.DataFrame, test_df: pd.DataFrame,
                      all_cols: List[str], target_idx: int,
                      scalers: dict, name: str = "Model",
                      eval_step: int = 1) -> dict:
    """
    Runs direct multi-step prediction over every overlapping test window.
    Evaluation grid matches new_model.py (eval_step=1 → one window per hour).
    """
    model.eval()
    sc_tgt = scalers[cfg.target_col]
    L, H   = cfg.encoder_length, cfg.decoder_length
    n_f    = len(all_cols)

    full_arr = df_sc[all_cols].values.astype(np.float32)
    tgt_sc   = df_sc[cfg.target_col].values.astype(np.float32)
    test_pos = df_sc.index.get_loc(test_df.index[0])

    errors_ph: Dict[int, list] = {h: [] for h in range(1, H + 1)}
    naive_ph:  Dict[int, list] = {h: [] for h in range(1, H + 1)}
    all_pred, all_act = [], []

    for s in tqdm(range(test_pos, test_pos + len(test_df) - L - H, eval_step),
                  desc=f"Eval {name}"):
        enc_np = full_arr[s: s + L]
        tgt_np = tgt_sc[s + L: s + L + H]

        enc_x = torch.tensor(enc_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        out   = model(enc_x)                           # [1, H, F]
        p_sc  = out[0, :, target_idx].cpu().numpy()   # [H]

        p_mwh = sc_tgt.inverse_transform(p_sc.reshape(-1, 1)).ravel()
        a_mwh = sc_tgt.inverse_transform(tgt_np.reshape(-1, 1)).ravel()
        nv    = naive_seasonal(tgt_sc[s: s + L], cfg.target_col,
                               L, H, sc_tgt)

        for hi in range(H):
            h = hi + 1
            errors_ph[h].append(abs(p_mwh[hi] - a_mwh[hi]))
            naive_ph[h].append(abs(nv[hi] - a_mwh[hi]))

        all_pred.append(p_mwh)
        all_act.append(a_mwh)

    me  = {h: np.mean(v) for h, v in errors_ph.items() if v}
    nme = {h: np.mean(v) for h, v in naive_ph.items()  if v}
    all_p = np.concatenate(all_pred)
    all_a = np.concatenate(all_act)
    ov    = compute_metrics(all_a, all_p, name)

    tft_v   = np.array([me[h]  for h in sorted(me)])
    naive_v = np.array([nme[h] for h in sorted(nme)])
    dm_s, dm_p = dm_test(tft_v, naive_v)

    print(f"\n{'─'*58}")
    print(f"PER-HORIZON MAE — {name}")
    print(f"{'Horizon':>8} | {'MAE':>10} | {'Naïve MAE':>10} | {'Skill':>7}")
    print(f"{'─'*8}-+-{'─'*10}-+-{'─'*10}-+-{'─'*7}")
    for h in range(1, H + 1, 24):
        if h in me:
            sk = 1 - me[h] / max(nme.get(h, 1), 1e-8)
            print(f"{h:>8} | {me[h]:>10.2f} | {nme.get(h,0):>10.2f} | {sk:>7.3f}")
    print(f"{'─'*58}")
    print(f"Overall 7-Day MAE : {np.mean(tft_v):.4f}")
    print(f"DM stat           : {dm_s:.3f}   p={dm_p:.4f}\n")

    return {'mean_errors': me, 'naive': nme, 'overall': ov,
            'dm': {'stat': dm_s, 'p': dm_p},
            'all_preds': all_p, 'all_actuals': all_a}


# ── 14. ANALYTICAL LINEAR-SHAP EXPLAINABILITY ─────────────────────────────────
def linear_shap(weights: np.ndarray, x_sample: np.ndarray,
                x_background: np.ndarray, feature_names: List[str]) -> np.ndarray:
    """
    Exact SHAP values for a linear model  ŷ = W · x.

    φ_i = w_i · (x_i − E[x_i])

    weights     : [H, L*F] or [L] — flattened linear weights for one output
    x_sample    : [L*F] or [L]  — single input sample (scaled)
    x_background: [N, L*F]     — background dataset for computing E[x_i]
    Returns φ  : [L*F] or [L]  — per-input Shapley value
    """
    mu_x = x_background.mean(axis=0)   # [L*F]
    if weights.ndim == 2:
        # aggregate over output horizon: mean weight across H
        w_avg = weights.mean(axis=0)    # [L*F]
    else:
        w_avg = weights                 # [L*F]
    phi = w_avg * (x_sample - mu_x)
    return phi                           # [L*F]


def extract_weights(model: nn.Module, cfg: DLConfig,
                    all_cols: List[str], target_idx: int) -> dict:
    """
    Extract W_T, W_R (DLinear) or W (NLinear) for the target channel.
    Returns numpy arrays and feature-level importance (aggregated over L).
    """
    result = {}
    if isinstance(model, DLinear):
        if cfg.individual:
            wt = model.W_T[target_idx].weight.detach().cpu().numpy()  # [H, L]
            wr = model.W_R[target_idx].weight.detach().cpu().numpy()  # [H, L]
        else:
            wt = model.W_T.weight.detach().cpu().numpy()
            wr = model.W_R.weight.detach().cpu().numpy()
        result['W_T'] = wt; result['W_R'] = wr
        result['trend_importance']    = np.abs(wt).mean(0)   # [L]
        result['seasonal_importance'] = np.abs(wr).mean(0)   # [L]

    elif isinstance(model, NLinear):
        if cfg.individual:
            w = model.W[target_idx].weight.detach().cpu().numpy()  # [H, L]
        else:
            w = model.W.weight.detach().cpu().numpy()
        result['W'] = w
        result['importance'] = np.abs(w).mean(0)   # [L] — importance per lag

    elif isinstance(model, DNEnsemble):
        result['dlinear'] = extract_weights(model.dl, cfg, all_cols, target_idx)
        result['nlinear'] = extract_weights(model.nl, cfg, all_cols, target_idx)
        result['ensemble_weights'] = model.weights   # [2]
    return result


# ── 15. BOOTSTRAP PREDICTION INTERVALS ────────────────────────────────────────
@torch.no_grad()
def bootstrap_intervals(model: nn.Module, enc_arr: np.ndarray,
                         target_idx: int, n_boot: int = 50,
                         noise_std: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate prediction intervals by running the model on noisy versions of
    the input (parametric bootstrap approximation).

    Returns (lower_q10, upper_q90) arrays of shape [H].
    """
    model.eval()
    L, F = enc_arr.shape
    boot_preds = []
    base_x = torch.tensor(enc_arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    for _ in range(n_boot):
        noise = torch.randn_like(base_x) * noise_std
        out   = model(base_x + noise)           # [1, H, F]
        boot_preds.append(out[0, :, target_idx].cpu().numpy())

    boot_arr = np.array(boot_preds)             # [n_boot, H]
    return np.percentile(boot_arr, 10, axis=0), np.percentile(boot_arr, 90, axis=0)


# ── 16. VISUALISATION ─────────────────────────────────────────────────────────
def _style():
    for s in ['seaborn-v0_8-whitegrid', 'seaborn-whitegrid', 'ggplot']:
        try: plt.style.use(s); return
        except OSError: continue

def plot_all(results_d: dict, results_n: dict, results_e: dict,
             weights_d: dict, weights_n: dict,
             cfg: DLConfig, art_dir: str):
    _style()
    art = Path(art_dir); art.mkdir(parents=True, exist_ok=True)
    H   = cfg.decoder_length
    hs  = list(range(1, H + 1))

    # ── Fig 1: Per-horizon MAE comparison (DLinear vs NLinear vs Ensemble vs Naïve)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(hs, [results_d['mean_errors'].get(h, 0) for h in hs], lw=2, label='DLinear')
    ax.plot(hs, [results_n['mean_errors'].get(h, 0) for h in hs], lw=2, label='NLinear')
    ax.plot(hs, [results_e['mean_errors'].get(h, 0) for h in hs], lw=2, label='Ensemble')
    ax.plot(hs, [results_d['naive'].get(h, 0)        for h in hs],
            lw=1.5, ls='--', color='grey', label='Seasonal Naïve')
    ax.set_xlabel('Forecast horizon (hours)'); ax.set_ylabel('MAE (MWh)')
    ax.set_title('Per-Horizon MAE — DLinear vs NLinear vs Ensemble vs Naïve')
    ax.legend(); fig.tight_layout()
    fig.savefig(art / 'per_horizon_mae.png', dpi=150); plt.close(fig)

    # ── Fig 2: Actual vs Predicted (first 2 weeks of test)
    n_show = min(336, len(results_e['all_actuals']))
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(results_e['all_actuals'][:n_show], lw=1.5, label='Actual', color='steelblue')
    ax.plot(results_d['all_preds'][:n_show],   lw=1.2, label='DLinear', color='tomato', alpha=0.8)
    ax.plot(results_n['all_preds'][:n_show],   lw=1.2, label='NLinear', color='darkorange', alpha=0.8)
    ax.plot(results_e['all_preds'][:n_show],   lw=1.5, label='Ensemble', color='mediumseagreen', alpha=0.9)
    ax.set_xlabel('Hour'); ax.set_ylabel('Consumption (MWh)')
    ax.set_title('Actual vs Predicted — First 2 Weeks of Test Set')
    ax.legend(); fig.tight_layout()
    fig.savefig(art / 'actual_vs_pred.png', dpi=150); plt.close(fig)

    # ── Fig 3: DLinear — trend vs seasonal weight importance over lag positions
    if 'trend_importance' in weights_d and 'seasonal_importance' in weights_d:
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        L = len(weights_d['trend_importance'])
        axes[0].plot(range(L), weights_d['trend_importance'],   color='steelblue')
        axes[0].set_title('DLinear — Mean |W_T| per Lag Position (Trend weights)')
        axes[0].set_ylabel('Mean |weight|')
        axes[1].plot(range(L), weights_d['seasonal_importance'], color='darkorange')
        axes[1].set_title('DLinear — Mean |W_R| per Lag Position (Seasonal/Remainder weights)')
        axes[1].set_ylabel('Mean |weight|'); axes[1].set_xlabel('Lag position (0 = oldest)')
        fig.tight_layout()
        fig.savefig(art / 'dlinear_weights.png', dpi=150); plt.close(fig)

    # ── Fig 4: NLinear — weight importance over lag positions
    if 'importance' in weights_n:
        fig, ax = plt.subplots(figsize=(14, 4))
        L = len(weights_n['importance'])
        ax.plot(range(L), weights_n['importance'], color='mediumseagreen')
        ax.fill_between(range(L), 0, weights_n['importance'], alpha=0.25, color='mediumseagreen')
        ax.set_xlabel('Lag position (0 = oldest, L−1 = most recent)')
        ax.set_ylabel('Mean |weight|')
        ax.set_title('NLinear — Mean |W| per Lag Position (Analytical Feature Importance)')
        fig.tight_layout()
        fig.savefig(art / 'nlinear_weights.png', dpi=150); plt.close(fig)

    # ── Fig 5: Skill score per horizon
    skill_d = [1 - results_d['mean_errors'].get(h, 0) / max(results_d['naive'].get(h, 1), 1e-8)
               for h in hs]
    skill_n = [1 - results_n['mean_errors'].get(h, 0) / max(results_n['naive'].get(h, 1), 1e-8)
               for h in hs]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(hs, skill_d, alpha=0.3, color='steelblue')
    ax.fill_between(hs, skill_n, alpha=0.3, color='darkorange')
    ax.plot(hs, skill_d, lw=2, color='steelblue',   label='DLinear')
    ax.plot(hs, skill_n, lw=2, color='darkorange',  label='NLinear')
    ax.axhline(0, color='red', ls='--', lw=1)
    ax.set_xlabel('Horizon (h)'); ax.set_ylabel('Skill vs Seasonal Naïve')
    ax.set_title('Forecast Skill Score per Horizon')
    ax.legend(); fig.tight_layout()
    fig.savefig(art / 'skill_scores.png', dpi=150); plt.close(fig)

    # ── Fig 6: Residual analysis (ensemble)
    errors = results_e['all_actuals'] - results_e['all_preds']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].hist(errors, bins=80, color='steelblue', edgecolor='white')
    axes[0].set_title('Residual distribution'); axes[0].set_xlabel('Error (MWh)')
    stats.probplot(errors, plot=axes[1])
    axes[1].set_title('QQ Plot — Ensemble residuals')
    from statsmodels.tsa.stattools import acf as tsacf
    acf_v = tsacf(errors, nlags=48, fft=True)
    ci    = 1.96 / np.sqrt(len(errors))
    axes[2].bar(range(len(acf_v)), acf_v, color='steelblue', width=0.6)
    axes[2].axhline(ci, ls='--', color='r', lw=1)
    axes[2].axhline(-ci, ls='--', color='r', lw=1)
    axes[2].set_title('Residual ACF (48 lags)'); axes[2].set_xlabel('Lag (h)')
    fig.tight_layout()
    fig.savefig(art / 'residual_analysis.png', dpi=150); plt.close(fig)

    # ── Fig 7: Model comparison radar chart
    metrics = ['MAE↓', 'RMSE↓', 'MAPE↓', 'sMAPE↓', 'R²↑']
    models  = ['DLinear', 'NLinear', 'Ensemble']
    vals_raw = np.array([
        [results_d['overall']['mae'],   results_d['overall']['rmse'],
         results_d['overall']['mape'],  results_d['overall']['smape'],  results_d['overall']['r2']],
        [results_n['overall']['mae'],   results_n['overall']['rmse'],
         results_n['overall']['mape'],  results_n['overall']['smape'],  results_n['overall']['r2']],
        [results_e['overall']['mae'],   results_e['overall']['rmse'],
         results_e['overall']['mape'],  results_e['overall']['smape'],  results_e['overall']['r2']],
    ])
    # normalise so 1 = best
    vmin = vals_raw.min(0); vmax = vals_raw.max(0)
    vals_norm = np.where(vmax == vmin, 1.0, (vals_raw - vmin) / (vmax - vmin + 1e-9))
    # for R² higher is better; for others lower is better
    vals_norm[:, :4] = 1 - vals_norm[:, :4]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(metrics)); w = 0.25
    for i, (m, v) in enumerate(zip(models, vals_norm)):
        ax.bar(x + i * w, v, w, label=m, alpha=0.8)
    ax.set_xticks(x + w); ax.set_xticklabels(metrics)
    ax.set_ylabel('Normalised score (1 = best)')
    ax.set_title('Model Comparison (normalised metrics)')
    ax.legend(); fig.tight_layout()
    fig.savefig(art / 'model_comparison.png', dpi=150); plt.close(fig)

    print(f"[Viz] 7 plots saved to {art}/")


# ── 17. MAIN ──────────────────────────────────────────────────────────────────
def main():
    cfg = DLConfig()
    Path(cfg.artifact_dir).mkdir(parents=True, exist_ok=True)

    # ── 17.1 Data
    print("=" * 70)
    print("STEP 1 — Loading and preprocessing data")
    print("=" * 70)
    df, df_sc, train_df, val_df, test_df, scalers, all_cols, tgt_idx = load_data(cfg)
    n_f = len(all_cols)

    # ── 17.2 Optuna
    print("\n" + "=" * 70)
    print("STEP 2 — Optuna hyperparameter search (DLinear / NLinear / Ensemble)")
    print("=" * 70)
    best_params, study = run_optuna(cfg, df_sc, all_cols, tgt_idx)

    # apply best params to cfg
    for k in ('encoder_length', 'kernel_size', 'individual', 'lr', 'batch_size'):
        if k in best_params:
            setattr(cfg, k, best_params[k])

    # ── 17.3 Train all three variants with best hyperparameters
    print("\n" + "=" * 70)
    print("STEP 3 — Training DLinear, NLinear, and Ensemble")
    print("=" * 70)

    tr_ds = TSDataset(train_df, cfg, all_cols, cfg.target_col, stride=1)
    va_ds = TSDataset(val_df,   cfg, all_cols, cfg.target_col, stride=2)
    tr_dl = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    print(f"  train={len(tr_ds):,}  val={len(va_ds):,}  "
          f"enc_len={cfg.encoder_length}  kernel={cfg.kernel_size}  individual={cfg.individual}")

    # DLinear (with RevIN)
    print("\n  [DLinear + RevIN]")
    dlinear = DLinear(cfg, n_f, use_revin=True).to(DEVICE)
    print(f"  params: {sum(p.numel() for p in dlinear.parameters()):,}")
    Trainer(dlinear, cfg, tr_dl, va_dl, tgt_idx).fit(verbose=True)

    # NLinear (with RevIN)
    print("\n  [NLinear + RevIN]")
    nlinear = NLinear(cfg, n_f, use_revin=True).to(DEVICE)
    print(f"  params: {sum(p.numel() for p in nlinear.parameters()):,}")
    Trainer(nlinear, cfg, tr_dl, va_dl, tgt_idx).fit(verbose=True)

    # Ensemble (jointly trained)
    print("\n  [Ensemble: DLinear + NLinear]")
    dl2 = DLinear(cfg, n_f, use_revin=True)
    nl2 = NLinear(cfg, n_f, use_revin=True)
    ensemble = DNEnsemble(dl2, nl2).to(DEVICE)
    print(f"  params: {sum(p.numel() for p in ensemble.parameters()):,}")
    Trainer(ensemble, cfg, tr_dl, va_dl, tgt_idx).fit(verbose=True)
    print(f"\n  Ensemble weights — DLinear: {ensemble.weights[0]:.3f}  "
          f"NLinear: {ensemble.weights[1]:.3f}")

    # ── 17.4 Save checkpoints
    ckpt = Path(cfg.artifact_dir)
    torch.save(dlinear.state_dict(),  ckpt / 'dlinear.pt')
    torch.save(nlinear.state_dict(),  ckpt / 'nlinear.pt')
    torch.save(ensemble.state_dict(), ckpt / 'ensemble.pt')
    print(f"\n  Checkpoints saved to {ckpt}/")

    # ── 17.5 Test-set evaluation (direct multi-step, per-horizon)
    print("\n" + "=" * 70)
    print("STEP 4 — Per-horizon evaluation on test set")
    print("=" * 70)
    res_d = evaluate_test_set(dlinear,  cfg, df_sc, test_df, all_cols, tgt_idx, scalers, 'DLinear')
    res_n = evaluate_test_set(nlinear,  cfg, df_sc, test_df, all_cols, tgt_idx, scalers, 'NLinear')
    res_e = evaluate_test_set(ensemble, cfg, df_sc, test_df, all_cols, tgt_idx, scalers, 'Ensemble')

    # ── 17.6 Truly recursive 168-step showcase forecast
    print("\n" + "=" * 70)
    print("STEP 5 — Truly recursive 168-step showcase")
    print("=" * 70)
    sc_tgt = scalers[cfg.target_col]
    test_abs = df_sc.index.get_loc(test_df.index[0])
    # use a window starting 1 week into the test set for a fair showcase
    showcase_start = test_abs + 168

    for mname, mdl in [('DLinear', dlinear), ('NLinear', nlinear), ('Ensemble', ensemble)]:
        _, preds_mwh = recursive_forecast_168(
            mdl, cfg, df_sc, all_cols, scalers, showcase_start, tgt_idx)
        actuals_mwh = sc_tgt.inverse_transform(
            df_sc[cfg.target_col].values[
                showcase_start + cfg.encoder_length:
                showcase_start + cfg.encoder_length + cfg.decoder_length
            ].reshape(-1, 1)
        ).ravel()
        compute_metrics(actuals_mwh, preds_mwh, f"Recursive {mname}")

    # ── 17.7 Bootstrap prediction intervals for Ensemble
    print("\n" + "=" * 70)
    print("STEP 6 — Bootstrap prediction intervals (Ensemble)")
    print("=" * 70)
    enc_eg = df_sc[all_cols].values[
        showcase_start: showcase_start + cfg.encoder_length
    ].astype(np.float32)
    lo_sc, hi_sc = bootstrap_intervals(
        ensemble, enc_eg, tgt_idx, n_boot=cfg.n_bootstrap, noise_std=0.02)
    lo_mwh = sc_tgt.inverse_transform(lo_sc.reshape(-1, 1)).ravel()
    hi_mwh = sc_tgt.inverse_transform(hi_sc.reshape(-1, 1)).ravel()
    width   = (hi_mwh - lo_mwh).mean()
    print(f"  Mean q10-q90 interval width : {width:.2f} MWh")

    # ── 17.8 Linear explainability
    print("\n" + "=" * 70)
    print("STEP 7 — Linear weight explainability")
    print("=" * 70)
    w_d = extract_weights(dlinear,  cfg, all_cols, tgt_idx)
    w_n = extract_weights(nlinear,  cfg, all_cols, tgt_idx)
    w_e = extract_weights(ensemble, cfg, all_cols, tgt_idx)

    print(f"\n  DLinear — top-5 most important trend lag positions:")
    ti = w_d['trend_importance']
    top5_t = np.argsort(ti)[-5:][::-1]
    for pos in top5_t:
        print(f"    lag -{cfg.encoder_length - pos - 1:3d}h  |  |W_T|={ti[pos]:.5f}")

    print(f"\n  NLinear — top-5 most important lag positions:")
    ni = w_n['importance']
    top5_n = np.argsort(ni)[-5:][::-1]
    for pos in top5_n:
        print(f"    lag -{cfg.encoder_length - pos - 1:3d}h  |  |W|={ni[pos]:.5f}")

    # Analytical SHAP on a test sample (NLinear — single W matrix)
    if 'W' in w_n and w_n['W'].ndim == 2:
        sample_enc = df_sc[all_cols].values[showcase_start: showcase_start + cfg.encoder_length].astype(np.float32)
        bg_enc     = df_sc[all_cols].values[
            test_abs: test_abs + min(500, len(test_df)) * (cfg.encoder_length + cfg.decoder_length):
            cfg.encoder_length + cfg.decoder_length
        ][: 100 * cfg.encoder_length].reshape(-1, cfg.encoder_length) if len(df_sc) > 500 else None

        if bg_enc is not None and len(bg_enc) > 10:
            # use target-channel NLinear weights
            w_mat = w_n['W']                          # [H, L]
            phi   = linear_shap(w_mat,
                                 sample_enc[:, tgt_idx],   # [L] target channel
                                 bg_enc[:, :] if bg_enc.shape[1] == cfg.encoder_length
                                 else sample_enc[:, tgt_idx].reshape(1, -1),
                                 [f"lag-{cfg.encoder_length - i}h" for i in range(cfg.encoder_length)])
            top5_shap = np.argsort(np.abs(phi))[-5:][::-1]
            print(f"\n  Analytical SHAP — NLinear target channel (top 5 inputs):")
            for pos in top5_shap:
                print(f"    lag -{cfg.encoder_length - pos - 1:3d}h  |  φ={phi[pos]:+.5f}")

    # ── 17.9 Visualisations
    print("\n" + "=" * 70)
    print("STEP 8 — Generating visualisations")
    print("=" * 70)
    plot_all(res_d, res_n, res_e, w_d, w_n, cfg, cfg.artifact_dir)

    # ── 17.10 Final summary
    print("\n" + "═" * 70)
    print("FINAL SUMMARY")
    print("═" * 70)
    header = f"{'Model':<12}  {'MAE':>8}  {'RMSE':>8}  {'MAPE%':>7}  {'sMAPE%':>8}  {'R²':>7}"
    print(header)
    print("─" * len(header))
    for label, res in [('DLinear', res_d), ('NLinear', res_n), ('Ensemble', res_e)]:
        ov = res['overall']
        print(f"{label:<12}  {ov['mae']:>8.2f}  {ov['rmse']:>8.2f}  "
              f"{ov['mape']:>7.2f}  {ov['smape']:>8.2f}  {ov['r2']:>7.4f}")
    print("─" * len(header))
    print(f"  Ensemble mixing  →  DLinear: {ensemble.weights[0]:.3f}  "
          f"NLinear: {ensemble.weights[1]:.3f}")
    print(f"  DM test (DLinear vs Naïve)  stat={res_d['dm']['stat']:.3f}  "
          f"p={res_d['dm']['p']:.4f}")
    print(f"  DM test (NLinear vs Naïve)  stat={res_n['dm']['stat']:.3f}  "
          f"p={res_n['dm']['p']:.4f}")
    print(f"  Artefacts → {cfg.artifact_dir}/")
    print("═" * 70)

    # save results
    save = {
        'dlinear':  {k: float(v) if isinstance(v, (np.floating, float)) else v
                     for k, v in res_d['overall'].items()},
        'nlinear':  {k: float(v) if isinstance(v, (np.floating, float)) else v
                     for k, v in res_n['overall'].items()},
        'ensemble': {k: float(v) if isinstance(v, (np.floating, float)) else v
                     for k, v in res_e['overall'].items()},
        'ensemble_weights': ensemble.weights.tolist(),
        'best_params': {k: str(v) for k, v in best_params.items()},
        'bootstrap_interval_width': float(width),
    }
    with open(Path(cfg.artifact_dir) / 'results.json', 'w') as f:
        json.dump(save, f, indent=2)
    print(f"  Results saved to {cfg.artifact_dir}/results.json")


if __name__ == '__main__':
    main()
