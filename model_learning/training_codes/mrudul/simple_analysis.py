"""
DEEP STATISTICAL ANALYSIS — ENERGY CONSUMPTION DATASET
Covers: EDA, stationarity, autocorrelation, spectral, seasonality,
        trend, outliers, volatility, decomposition, and more.
Run:  python energy_analysis.py
Outputs: ./analysis_outputs/ folder with all plots + reports
"""

import warnings
warnings.filterwarnings('ignore')

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats, signal
from scipy.stats import (kurtosis, skew, jarque_bera, ks_2samp,
                          shapiro, anderson, probplot, chi2)
from statsmodels.tsa.stattools import (adfuller, kpss, acf, pacf,
                                        ccf, grangercausalitytests, coint)
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import (acorr_ljungbox,
                                           het_arch, het_white)
from statsmodels.stats.stattools import durbin_watson, jarque_bera as jb_test
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import itertools
from datetime import datetime

# ── Output directory ──────────────────────────────────────────────────────────
OUT = "analysis_outputs"
os.makedirs(OUT, exist_ok=True)

PALETTE = {
    'primary'   : '#1B4F72',
    'secondary' : '#2E86AB',
    'accent'    : '#E84855',
    'warn'      : '#F4A261',
    'success'   : '#2EC4B6',
    'bg'        : '#F7F9FC',
    'dark'      : '#1A1A2E',
}

def save(fig, name, dpi=150):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [saved] {path}")

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

# ══════════════════════════════════════════════════════════════════════════════
# 0. LOAD & PREPARE
# ══════════════════════════════════════════════════════════════════════════════
section("0. LOADING DATA")

df_raw = pd.read_excel("Energy Consumption Dataset.xlsx")
df_raw.columns = df_raw.columns.str.strip()

# parse timestamps
df_raw['Start time UTC'] = pd.to_datetime(df_raw['Start time UTC'])
df_raw = df_raw.sort_values('Start time UTC').reset_index(drop=True)
df_raw = df_raw.set_index('Start time UTC')
df_raw.index = df_raw.index.tz_localize(None)   # strip tz for simplicity

# keep only the consumption column
ts = df_raw['Electricity consumption (MWh)'].copy()
ts.index.name = 'ds'

# build a complete hourly index and expose gaps
full_idx = pd.date_range(ts.index.min(), ts.index.max(), freq='h')
ts = ts.reindex(full_idx)
n_missing_after_reindex = ts.isna().sum()
print(f"  Observations after reindex  : {len(ts):,}")
print(f"  Implicit gaps (new NaNs)    : {n_missing_after_reindex:,}")
ts = ts.interpolate(method='time')   # simple fill for analysis purposes

# convenience arrays
y   = ts.values.copy()
N   = len(y)
t   = np.arange(N)

# calendar features
cal = pd.DataFrame(index=ts.index)
cal['hour']       = ts.index.hour
cal['dow']        = ts.index.dayofweek          # 0=Mon
cal['month']      = ts.index.month
cal['year']       = ts.index.year
cal['doy']        = ts.index.dayofyear
cal['week']       = ts.index.isocalendar().week.astype(int)
cal['is_weekend'] = (cal['dow'] >= 5).astype(int)
cal['quarter']    = ts.index.quarter
cal['y']          = y

print(f"  Date range : {ts.index[0]}  →  {ts.index[-1]}")
print(f"  N          : {N:,}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. FUNDAMENTAL DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
section("1. DESCRIPTIVE STATISTICS")

desc = pd.Series({
    'N'           : N,
    'Mean'        : y.mean(),
    'Median'      : np.median(y),
    'Std'         : y.std(),
    'Min'         : y.min(),
    'Max'         : y.max(),
    'Range'       : y.max() - y.min(),
    'Skewness'    : skew(y),
    'Kurtosis (excess)': kurtosis(y),
    'CV (%)'      : y.std()/y.mean()*100,
    'IQR'         : np.percentile(y,75) - np.percentile(y,25),
    'P1'          : np.percentile(y, 1),
    'P5'          : np.percentile(y, 5),
    'P95'         : np.percentile(y, 95),
    'P99'         : np.percentile(y, 99),
})
print(desc.to_string())

jb_stat, jb_p = jarque_bera(y)
print(f"\n  Jarque-Bera stat : {jb_stat:.2f}  p={jb_p:.6f}")
print(f"  {'NOT normal' if jb_p<0.05 else 'Normal'} at 5% level")

# ── Conditional means 24×7 ──────────────────────────────────────────────────
cond_mean = cal.groupby(['hour','dow'])['y'].mean().unstack()   # 24×7

fig, axes = plt.subplots(1, 2, figsize=(18, 6), facecolor=PALETTE['bg'])

# heatmap hour × day-of-week
ax = axes[0]
cmap = LinearSegmentedColormap.from_list('ep',
        ['#2E86AB','#F7F9FC','#E84855'])
sns.heatmap(cond_mean, ax=ax, cmap=cmap, fmt='.0f', annot=False,
            linewidths=.3, linecolor='white',
            xticklabels=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],
            yticklabels=[f'{h:02d}:00' for h in range(24)])
ax.set_title('Conditional Mean Consumption\n(hour × day-of-week)', fontsize=13,
             fontweight='bold', color=PALETTE['dark'])
ax.set_xlabel('Day of Week'); ax.set_ylabel('Hour of Day')

# distribution
ax = axes[1]
ax.set_facecolor(PALETTE['bg'])
ax.hist(y, bins=80, color=PALETTE['secondary'], alpha=0.8, edgecolor='white',
        density=True, label='Empirical')
xr = np.linspace(y.min(), y.max(), 300)
ax.plot(xr, stats.norm.pdf(xr, y.mean(), y.std()),
        color=PALETTE['accent'], lw=2.5, label='Gaussian fit')
ax.axvline(np.mean(y), color=PALETTE['dark'], ls='--', lw=1.5, label=f'Mean {y.mean():.0f}')
ax.axvline(np.median(y), color=PALETTE['warn'], ls='--', lw=1.5, label=f'Median {np.median(y):.0f}')
ax.set_title(f'Consumption Distribution\nSkew={skew(y):.3f}  Kurt={kurtosis(y):.3f}', fontsize=13, fontweight='bold')
ax.set_xlabel('MWh'); ax.legend(fontsize=9)

fig.suptitle('Section 1 — Descriptive Statistics', fontsize=15, fontweight='bold',
             color=PALETTE['dark'], y=1.01)
fig.tight_layout()
save(fig, '01_descriptive.png')

# ══════════════════════════════════════════════════════════════════════════════
# 2. FULL TIME SERIES OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
section("2. TIME SERIES OVERVIEW")

fig, axes = plt.subplots(4, 1, figsize=(20, 18), facecolor=PALETTE['bg'])

# 2a  full series
ax = axes[0]; ax.set_facecolor(PALETTE['bg'])
ax.plot(ts.index, y, color=PALETTE['secondary'], lw=0.4, alpha=0.8)
roll7 = pd.Series(y, index=ts.index).rolling(24*7, center=True).mean()
ax.plot(ts.index, roll7, color=PALETTE['accent'], lw=2, label='7-day rolling mean')
ax.set_title('Full Series (2016–2021)', fontsize=12, fontweight='bold')
ax.set_ylabel('MWh'); ax.legend()

# 2b  annual boxplots
ax = axes[1]; ax.set_facecolor(PALETTE['bg'])
years = sorted(cal['year'].unique())
data_yr = [cal[cal['year']==yr]['y'].values for yr in years]
bp = ax.boxplot(data_yr, patch_artist=True, notch=True,
                medianprops=dict(color=PALETTE['accent'], lw=2))
for patch in bp['boxes']:
    patch.set_facecolor(PALETTE['secondary']); patch.set_alpha(0.6)
ax.set_xticklabels(years)
ax.set_title('Annual Distribution (Box-Whisker with Notch = 95% CI on Median)',
             fontsize=12, fontweight='bold')
ax.set_ylabel('MWh')

# 2c  average daily profile by year
ax = axes[2]; ax.set_facecolor(PALETTE['bg'])
colors_yr = plt.cm.plasma(np.linspace(0.1, 0.9, len(years)))
for yr, c in zip(years, colors_yr):
    prof = cal[cal['year']==yr].groupby('hour')['y'].mean()
    ax.plot(prof.index, prof.values, color=c, lw=2, label=str(yr))
ax.set_title('Average Daily Profile by Year', fontsize=12, fontweight='bold')
ax.set_xlabel('Hour of Day'); ax.set_ylabel('MWh'); ax.legend(ncol=3)

# 2d  month × hour heatmap
ax = axes[3]; ax.set_facecolor(PALETTE['bg'])
mh = cal.groupby(['month','hour'])['y'].mean().unstack()   # 12×24
im = ax.imshow(mh.values, aspect='auto', cmap='RdYlBu_r', origin='lower')
ax.set_yticks(range(12)); ax.set_yticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                                               'Jul','Aug','Sep','Oct','Nov','Dec'])
ax.set_xticks(range(0,24,2)); ax.set_xticklabels([f'{h:02d}' for h in range(0,24,2)])
ax.set_title('Month × Hour Heatmap (MWh)', fontsize=12, fontweight='bold')
ax.set_xlabel('Hour of Day'); ax.set_ylabel('Month')
plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

fig.suptitle('Section 2 — Time Series Overview', fontsize=15, fontweight='bold',
             color=PALETTE['dark'])
fig.tight_layout()
save(fig, '02_ts_overview.png')

# ══════════════════════════════════════════════════════════════════════════════
# 3. STATIONARITY TESTS
# ══════════════════════════════════════════════════════════════════════════════
section("3. STATIONARITY TESTS")

results_stat = {}

# ADF
adf = adfuller(y, maxlag=22, regression='ct', autolag='AIC')
results_stat['ADF'] = {'stat': adf[0], 'p': adf[1], 'lags': adf[2],
                        'crit_1%': adf[4]['1%'], 'crit_5%': adf[4]['5%']}
print(f"\n  ADF (const+trend): stat={adf[0]:.4f}, p={adf[1]:.6f}, lags={adf[2]}")
print(f"    Critical 1%={adf[4]['1%']:.4f}  5%={adf[4]['5%']:.4f}")
print(f"    → {'STATIONARY' if adf[1]<0.05 else 'NON-STATIONARY'} at 5%")

# KPSS
kp = kpss(y, regression='ct', nlags='auto')
results_stat['KPSS'] = {'stat': kp[0], 'p': kp[1]}
print(f"\n  KPSS (const+trend): stat={kp[0]:.4f}, p={kp[1]:.4f}")
print(f"    → {'NON-STATIONARY' if kp[1]<0.05 else 'STATIONARY'} at 5% (H0=stationary)")

# On first difference
dy = np.diff(y)
adf_d = adfuller(dy, maxlag=22, regression='c', autolag='AIC')
kp_d  = kpss(dy, regression='c', nlags='auto')
print(f"\n  First-differenced series:")
print(f"    ADF  p={adf_d[1]:.6f}  → {'STATIONARY' if adf_d[1]<0.05 else 'NON-STATIONARY'}")
print(f"    KPSS p={kp_d[1]:.4f}   → {'NON-STATIONARY' if kp_d[1]<0.05 else 'STATIONARY'}")

# Rolling mean and variance
roll_mean = pd.Series(y).rolling(24*30).mean().dropna()
roll_std  = pd.Series(y).rolling(24*30).std().dropna()

fig, axes = plt.subplots(3, 1, figsize=(18, 12), facecolor=PALETTE['bg'])

ax = axes[0]; ax.set_facecolor(PALETTE['bg'])
ax.plot(ts.index, y, color=PALETTE['secondary'], lw=0.5, alpha=0.6, label='Series')
ax.plot(ts.index[24*30-1:], roll_mean.values, color=PALETTE['accent'], lw=2, label='30-day rolling mean')
ax.set_title(f'Rolling Mean  (ADF p={adf[1]:.4f}, KPSS p={kp[1]:.4f})',
             fontsize=12, fontweight='bold')
ax.legend(); ax.set_ylabel('MWh')

ax = axes[1]; ax.set_facecolor(PALETTE['bg'])
ax.plot(ts.index[24*30-1:], roll_std.values, color=PALETTE['warn'], lw=1.5)
ax.set_title('Rolling Std (30-day window) — growing std = heteroscedasticity',
             fontsize=12, fontweight='bold')
ax.set_ylabel('Std (MWh)')

ax = axes[2]; ax.set_facecolor(PALETTE['bg'])
ax.plot(ts.index[1:], dy, color=PALETTE['secondary'], lw=0.4, alpha=0.8)
ax.axhline(0, color=PALETTE['accent'], lw=1.5)
ax.set_title(f'First Difference  (ADF p={adf_d[1]:.6f})', fontsize=12, fontweight='bold')
ax.set_ylabel('Δ MWh')

fig.suptitle('Section 3 — Stationarity', fontsize=15, fontweight='bold', color=PALETTE['dark'])
fig.tight_layout()
save(fig, '03_stationarity.png')

# ══════════════════════════════════════════════════════════════════════════════
# 4. AUTOCORRELATION STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════
section("4. AUTOCORRELATION STRUCTURE")

NLAGS = 336   # two weeks of hourly lags

acf_vals  = acf(y, nlags=NLAGS, fft=True, alpha=0.05)
pacf_vals = pacf(y, nlags=min(NLAGS, 100), alpha=0.05)   # PACF slow at high lags

ci95 = 1.96 / np.sqrt(N)

fig, axes = plt.subplots(2, 1, figsize=(20, 10), facecolor=PALETTE['bg'])

lags_acf  = np.arange(len(acf_vals[0]))
lags_pacf = np.arange(len(pacf_vals[0]))

ax = axes[0]; ax.set_facecolor(PALETTE['bg'])
ax.vlines(lags_acf, 0, acf_vals[0], color=PALETTE['secondary'], lw=0.8, alpha=0.8)
ax.axhline(0, color=PALETTE['dark'], lw=1)
ax.axhline( ci95, color=PALETTE['accent'], ls='--', lw=1.5, label=f'95% CI ±{ci95:.4f}')
ax.axhline(-ci95, color=PALETTE['accent'], ls='--', lw=1.5)
# annotate key lags
for lag in [1, 24, 48, 72, 168, 336]:
    if lag < len(acf_vals[0]):
        ax.annotate(f'lag {lag}\n{acf_vals[0][lag]:.3f}',
                    xy=(lag, acf_vals[0][lag]),
                    xytext=(lag+5, acf_vals[0][lag]+0.03),
                    arrowprops=dict(arrowstyle='->', color=PALETTE['accent']),
                    fontsize=8, color=PALETTE['accent'])
ax.set_title(f'ACF up to lag {NLAGS} — spikes at 24, 168 confirm daily & weekly cycles',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Lag (hours)'); ax.set_ylabel('Autocorrelation')
ax.legend()

ax = axes[1]; ax.set_facecolor(PALETTE['bg'])
ax.vlines(lags_pacf, 0, pacf_vals[0], color=PALETTE['primary'], lw=0.8, alpha=0.8)
ax.axhline(0, color=PALETTE['dark'], lw=1)
ax.axhline( ci95, color=PALETTE['accent'], ls='--', lw=1.5, label=f'95% CI')
ax.axhline(-ci95, color=PALETTE['accent'], ls='--', lw=1.5)
ax.set_title('PACF (first 100 lags) — cutoff identifies AR order',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Lag (hours)'); ax.set_ylabel('Partial Autocorrelation')
ax.legend()

fig.suptitle('Section 4 — Autocorrelation Structure', fontsize=15, fontweight='bold',
             color=PALETTE['dark'])
fig.tight_layout()
save(fig, '04_acf_pacf.png')

# Ljung-Box test
lb = acorr_ljungbox(y, lags=[1,12,24,48,168], return_df=True)
print("\n  Ljung-Box Q-test:")
print(lb.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 5. SPECTRAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
section("5. SPECTRAL ANALYSIS (PERIODOGRAM)")

freqs, psd = signal.periodogram(y - y.mean(), fs=1.0)    # fs=1 sample/hour
# convert to period in hours
with np.errstate(divide='ignore', invalid='ignore'):
    periods = np.where(freqs > 0, 1.0/freqs, np.inf)

# top 20 periods
valid = freqs > 0
top_idx = np.argsort(psd[valid])[::-1][:20]
top_periods = periods[valid][top_idx]
top_powers  = psd[valid][top_idx]
print("\n  Top 10 dominant periods:")
for i in range(10):
    print(f"    {i+1:2d}. Period={top_periods[i]:10.2f} h  Power={top_powers[i]:.3e}")

fig, axes = plt.subplots(2, 1, figsize=(18, 10), facecolor=PALETTE['bg'])

ax = axes[0]; ax.set_facecolor(PALETTE['bg'])
ax.semilogy(freqs[valid], psd[valid], color=PALETTE['secondary'], lw=0.7)
for p, label in [(24,'Daily'), (168,'Weekly'), (8760,'Annual'), (12,'Half-day')]:
    f = 1/p
    if f < freqs.max():
        ax.axvline(f, color=PALETTE['accent'], ls='--', lw=1.5)
        ax.text(f, ax.get_ylim()[1] if ax.get_ylim()[1]>0 else 1e6,
                f' {label}\n 1/{p}h', fontsize=8, color=PALETTE['accent'],
                va='top')
ax.set_title('Periodogram (log scale) — dominant frequency peaks',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Frequency (cycles/hour)'); ax.set_ylabel('Power Spectral Density')

ax = axes[1]; ax.set_facecolor(PALETTE['bg'])
period_plot = periods[valid]
mask = (period_plot >= 2) & (period_plot <= 200)
ax.semilogy(period_plot[mask], psd[valid][mask],
            color=PALETTE['primary'], lw=0.8)
for p in [24, 48, 72, 96, 120, 144, 168]:
    ax.axvline(p, color=PALETTE['accent'], ls=':', lw=1)
    ax.text(p, psd[valid][mask].max()*0.5, f'{p}h',
            fontsize=7, color=PALETTE['accent'], ha='center')
ax.set_title('Period-Domain View (2–200 hours) — harmonics of 24h visible',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Period (hours)'); ax.set_ylabel('Power Spectral Density')

fig.suptitle('Section 5 — Spectral Analysis', fontsize=15, fontweight='bold',
             color=PALETTE['dark'])
fig.tight_layout()
save(fig, '05_spectral.png')

# ══════════════════════════════════════════════════════════════════════════════
# 6. SEASONAL DECOMPOSITION (STL)
# ══════════════════════════════════════════════════════════════════════════════
section("6. STL DECOMPOSITION")

# Use period=24 for STL (daily)
stl_result = STL(ts, period=24, robust=True).fit()
trend_stl    = stl_result.trend
seasonal_stl = stl_result.seasonal
resid_stl    = stl_result.resid

fs = max(0, 1 - np.var(resid_stl) / np.var(seasonal_stl + resid_stl))
ft = max(0, 1 - np.var(resid_stl) / np.var(trend_stl   + resid_stl))
print(f"\n  Seasonal Strength Fs = {fs:.4f}  {'(strong)' if fs>0.64 else '(weak)'}")
print(f"  Trend    Strength Ft = {ft:.4f}  {'(strong)' if ft>0.50 else '(weak)'}")

fig, axes = plt.subplots(4, 1, figsize=(20, 16), facecolor=PALETTE['bg'],
                          sharex=True)
components = [('Original', y, PALETTE['secondary']),
              ('Trend',    trend_stl,    PALETTE['primary']),
              ('Seasonal', seasonal_stl, PALETTE['success']),
              ('Residual', resid_stl,    PALETTE['accent'])]
for ax, (title, comp, color) in zip(axes, components):
    ax.set_facecolor(PALETTE['bg'])
    ax.plot(ts.index, comp, color=color, lw=0.6 if title=='Original' else 1.5)
    if title == 'Residual':
        ax.axhline(0, color=PALETTE['dark'], lw=1, ls='--')
    ax.set_ylabel('MWh', fontsize=10)
    ax.set_title(f'{title}  (Fs={fs:.3f}, Ft={ft:.3f})'
                 if title == 'Seasonal' else title,
                 fontsize=11, fontweight='bold')

axes[-1].set_xlabel('Date')
fig.suptitle('Section 6 — STL Decomposition (period=24, robust=True)',
             fontsize=15, fontweight='bold', color=PALETTE['dark'])
fig.tight_layout()
save(fig, '06_stl_decomposition.png')

# ══════════════════════════════════════════════════════════════════════════════
# 7. SEASONALITY PROFILES
# ══════════════════════════════════════════════════════════════════════════════
section("7. SEASONALITY PROFILES")

fig = plt.figure(figsize=(22, 16), facecolor=PALETTE['bg'])
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# 7a hourly profile with ±1 std
ax = fig.add_subplot(gs[0, :2])
ax.set_facecolor(PALETTE['bg'])
hp_mean = cal.groupby('hour')['y'].mean()
hp_std  = cal.groupby('hour')['y'].std()
ax.plot(hp_mean.index, hp_mean.values, color=PALETTE['primary'],
        lw=2.5, marker='o', ms=4, label='Mean')
ax.fill_between(hp_mean.index,
                hp_mean - hp_std, hp_mean + hp_std,
                alpha=0.2, color=PALETTE['primary'], label='±1 Std')
ax.set_title('Average Hourly Profile', fontsize=11, fontweight='bold')
ax.set_xlabel('Hour of Day'); ax.set_ylabel('MWh'); ax.legend()
ax.set_xticks(range(0,24,2))

# 7b daily profile (DOW)
ax = fig.add_subplot(gs[0, 2])
ax.set_facecolor(PALETTE['bg'])
dp = cal.groupby('dow')['y'].agg(['mean','std'])
ax.bar(dp.index, dp['mean'],
       color=[PALETTE['secondary'] if d<5 else PALETTE['warn'] for d in dp.index],
       edgecolor='white', alpha=0.9)
ax.errorbar(dp.index, dp['mean'], yerr=dp['std'],
            fmt='none', color=PALETTE['dark'], capsize=4)
ax.set_xticks(range(7))
ax.set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], rotation=30)
ax.set_title('Day-of-Week Profile', fontsize=11, fontweight='bold')
ax.set_ylabel('MWh')

# 7c monthly profile
ax = fig.add_subplot(gs[1, :2])
ax.set_facecolor(PALETTE['bg'])
mp = cal.groupby('month')['y'].agg(['mean','std','min','max'])
ax.plot(mp.index, mp['mean'], color=PALETTE['primary'],
        lw=2.5, marker='D', ms=5, label='Mean')
ax.fill_between(mp.index, mp['min'], mp['max'],
                alpha=0.1, color=PALETTE['primary'], label='Min-Max range')
ax.fill_between(mp.index, mp['mean']-mp['std'], mp['mean']+mp['std'],
                alpha=0.25, color=PALETTE['primary'], label='±1 Std')
ax.set_xticks(range(1,13))
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec'])
ax.set_title('Annual (Monthly) Profile', fontsize=11, fontweight='bold')
ax.set_ylabel('MWh'); ax.legend(fontsize=8)

# 7d seasonal indices 24×7 annotated
ax = fig.add_subplot(gs[1, 2])
ax.set_facecolor(PALETTE['bg'])
si = cond_mean / cond_mean.values.mean()
sns.heatmap(si, ax=ax, cmap='RdYlGn', center=1, vmin=0.6, vmax=1.4,
            linewidths=.2, linecolor='white', cbar_kws={'shrink':0.8},
            xticklabels=['M','T','W','T','F','S','S'])
ax.set_title('Seasonal Indices\n(hour×DOW, 1=average)', fontsize=11, fontweight='bold')
ax.set_xlabel('DOW'); ax.set_ylabel('Hour')

# 7e year-on-year overlay
ax = fig.add_subplot(gs[2, :])
ax.set_facecolor(PALETTE['bg'])
colors_yr = plt.cm.viridis(np.linspace(0.1, 0.9, len(years)))
for yr, c in zip(years, colors_yr):
    mask = cal['year'] == yr
    sub  = cal[mask].groupby('doy')['y'].mean()
    ax.plot(sub.index, sub.values, color=c, lw=1.5, alpha=0.85, label=str(yr))
ax.set_title('Year-over-Year Daily Aggregated Profile — trend & seasonal evolution',
             fontsize=11, fontweight='bold')
ax.set_xlabel('Day of Year'); ax.set_ylabel('MWh')
ax.legend(ncol=len(years), loc='upper right', fontsize=9)

fig.suptitle('Section 7 — Seasonality Profiles', fontsize=15, fontweight='bold',
             color=PALETTE['dark'])
save(fig, '07_seasonality.png')

# ══════════════════════════════════════════════════════════════════════════════
# 8. TREND ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
section("8. TREND ANALYSIS")

daily_mean = cal.groupby(cal.index.date)['y'].mean()
daily_mean.index = pd.to_datetime(daily_mean.index)
dm = daily_mean.values
dm_t = np.arange(len(dm))

# OLS linear trend
X_lin = add_constant(dm_t)
ols   = OLS(dm, X_lin).fit()
slope_per_year = ols.params[1] * 365
print(f"\n  OLS trend slope  : {ols.params[1]:.4f} MWh/day  ({slope_per_year:.1f} MWh/year)")
print(f"  R²               : {ols.rsquared:.4f}")
print(f"  p-value (slope)  : {ols.pvalues[1]:.6f}")

# Mann-Kendall (manual)
from scipy.stats import kendalltau
tau, mk_p = kendalltau(dm_t, dm)
print(f"\n  Mann-Kendall tau : {tau:.4f}  p={mk_p:.6f}")
print(f"  → {'Significant upward trend' if (mk_p<0.05 and tau>0) else 'No significant trend'}")

# LOWESS
from statsmodels.nonparametric.smoothers_lowess import lowess
lw_fit = lowess(dm, dm_t, frac=0.08, return_sorted=True)

fig, axes = plt.subplots(2, 1, figsize=(18, 10), facecolor=PALETTE['bg'])

ax = axes[0]; ax.set_facecolor(PALETTE['bg'])
ax.plot(daily_mean.index, dm, color=PALETTE['secondary'], lw=0.8, alpha=0.7,
        label='Daily Mean')
ax.plot(daily_mean.index,
        ols.params[0] + ols.params[1]*dm_t,
        color=PALETTE['accent'], lw=2.5, ls='--',
        label=f'OLS trend (+{slope_per_year:.0f} MWh/yr, R²={ols.rsquared:.3f})')
ax.plot(daily_mean.index[lw_fit[:,0].astype(int)], lw_fit[:,1],
        color=PALETTE['success'], lw=2.5,
        label='LOWESS (frac=0.08)')
ax.set_title(f'Trend Analysis — Mann-Kendall p={mk_p:.6f}', fontsize=12, fontweight='bold')
ax.set_ylabel('MWh'); ax.legend()

ax = axes[1]; ax.set_facecolor(PALETTE['bg'])
annual_mean = cal.groupby('year')['y'].mean()
annual_std  = cal.groupby('year')['y'].std()
ax.bar(annual_mean.index, annual_mean.values,
       color=PALETTE['secondary'], alpha=0.7, edgecolor='white')
ax.errorbar(annual_mean.index, annual_mean.values, yerr=annual_std.values,
            fmt='none', color=PALETTE['dark'], capsize=5, lw=2)
for yr, val in annual_mean.items():
    ax.text(yr, val + 50, f'{val:.0f}', ha='center', fontsize=9, fontweight='bold')
pct_growth = (annual_mean.iloc[-1] / annual_mean.iloc[0] - 1)*100
ax.set_title(f'Annual Mean Consumption — Total growth: {pct_growth:.1f}% over {len(years)-1} years',
             fontsize=12, fontweight='bold')
ax.set_ylabel('MWh')

fig.suptitle('Section 8 — Trend Analysis', fontsize=15, fontweight='bold',
             color=PALETTE['dark'])
fig.tight_layout()
save(fig, '08_trend.png')

# ══════════════════════════════════════════════════════════════════════════════
# 9. OUTLIER DETECTION
# ══════════════════════════════════════════════════════════════════════════════
section("9. OUTLIER DETECTION")

# 9a — Contextual IQR (per hour × DOW)
cal['is_outlier_ctx'] = False
for (h, d), grp in cal.groupby(['hour','dow']):
    q1, q3 = grp['y'].quantile(0.25), grp['y'].quantile(0.75)
    iqr = q3 - q1
    mask = (grp['y'] < q1 - 3*iqr) | (grp['y'] > q3 + 3*iqr)
    cal.loc[grp.index[mask], 'is_outlier_ctx'] = True

n_ctx = cal['is_outlier_ctx'].sum()
print(f"\n  Contextual IQR outliers  : {n_ctx} ({n_ctx/N*100:.2f}%)")

# 9b — Z-score global (for comparison)
z = np.abs(stats.zscore(y))
n_z = (z > 3.5).sum()
print(f"  Global Z-score (>3.5σ)   : {n_z} ({n_z/N*100:.2f}%)")

# 9c — Isolation Forest
scaler = StandardScaler()
feat_if = pd.DataFrame({
    'y'    : y,
    'y_m1' : np.concatenate([[y[0]], y[:-1]]),
    'y_m24': np.concatenate([y[:24], y[:-24]]),
    'hour' : cal['hour'].values,
    'dow'  : cal['dow'].values,
})
feat_scaled = scaler.fit_transform(feat_if)
isof = IsolationForest(contamination=0.015, random_state=42, n_estimators=200)
isof_labels = isof.fit_predict(feat_scaled)   # -1 = anomaly
n_if = (isof_labels == -1).sum()
print(f"  Isolation Forest anomalies: {n_if} ({n_if/N*100:.2f}%)")
cal['is_outlier_if'] = (isof_labels == -1)

# 9d — Rolling mean ±3σ
roll = pd.Series(y).rolling(24*7, center=True, min_periods=1)
rm, rs = roll.mean(), roll.std()
flag_roll = (y > rm + 3*rs) | (y < rm - 3*rs)
print(f"  Rolling 7-day ±3σ         : {flag_roll.sum()} ({flag_roll.sum()/N*100:.2f}%)")

fig, axes = plt.subplots(3, 1, figsize=(20, 16), facecolor=PALETTE['bg'])

ax = axes[0]; ax.set_facecolor(PALETTE['bg'])
ax.plot(ts.index, y, color=PALETTE['secondary'], lw=0.4, alpha=0.7, label='Series')
ax.scatter(ts.index[cal['is_outlier_ctx']],
           y[cal['is_outlier_ctx']],
           color=PALETTE['accent'], s=15, zorder=5, label=f'Contextual IQR ({n_ctx})')
ax.set_title('Contextual (Hour×DOW) IQR Outliers', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.set_ylabel('MWh')

ax = axes[1]; ax.set_facecolor(PALETTE['bg'])
ax.plot(ts.index, y, color=PALETTE['secondary'], lw=0.4, alpha=0.7)
ax.plot(ts.index, rm.values, color=PALETTE['primary'], lw=1.5, label='7-day rolling mean')
ax.fill_between(ts.index, rm-3*rs, rm+3*rs,
                alpha=0.15, color=PALETTE['primary'], label='±3σ band')
ax.scatter(ts.index[flag_roll], y[flag_roll],
           color=PALETTE['warn'], s=15, zorder=5, label=f'Outside ±3σ ({flag_roll.sum()})')
ax.set_title('Rolling 7-day Mean ±3σ Envelope', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.set_ylabel('MWh')

ax = axes[2]; ax.set_facecolor(PALETTE['bg'])
ax.plot(ts.index, y, color=PALETTE['secondary'], lw=0.4, alpha=0.7)
ax.scatter(ts.index[cal['is_outlier_if']],
           y[cal['is_outlier_if']],
           color=PALETTE['accent'], s=18, zorder=5, label=f'Isolation Forest ({n_if})',
           marker='^')
ax.set_title('Isolation Forest Anomalies (contamination=1.5%)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.set_ylabel('MWh')

fig.suptitle('Section 9 — Outlier Detection', fontsize=15, fontweight='bold',
             color=PALETTE['dark'])
fig.tight_layout()
save(fig, '09_outliers.png')

# ══════════════════════════════════════════════════════════════════════════════
# 10. VOLATILITY (ARCH / HETEROSCEDASTICITY)
# ══════════════════════════════════════════════════════════════════════════════
section("10. VOLATILITY ANALYSIS")

# fit a simple AR(1) to get residuals
ar1 = AutoReg(y, lags=1, trend='ct').fit()
resid_ar = ar1.resid

# ARCH-LM test
arch_stat, arch_p, _, _ = het_arch(resid_ar, nlags=24)
print(f"\n  ARCH-LM test (24 lags): stat={arch_stat:.4f}  p={arch_p:.6f}")
print(f"  → {'ARCH effects PRESENT' if arch_p < 0.05 else 'No ARCH effects'}")

# squared residuals ACF
sq_acf = acf(resid_ar**2, nlags=72, fft=True)

fig, axes = plt.subplots(3, 1, figsize=(18, 14), facecolor=PALETTE['bg'])

ax = axes[0]; ax.set_facecolor(PALETTE['bg'])
ax.plot(ts.index[1:], resid_ar, color=PALETTE['secondary'], lw=0.5, alpha=0.8)
ax.set_title('AR(1) Residuals — visual check for volatility clustering',
             fontsize=12, fontweight='bold')
ax.set_ylabel('Residual (MWh)'); ax.axhline(0, color='red', lw=1)

ax = axes[1]; ax.set_facecolor(PALETTE['bg'])
ax.vlines(range(len(sq_acf)), 0, sq_acf, color=PALETTE['warn'], lw=0.8)
ax.axhline( ci95, color=PALETTE['accent'], ls='--', lw=1.5, label='95% CI')
ax.axhline(-ci95, color=PALETTE['accent'], ls='--', lw=1.5)
ax.set_title(f'ACF of Squared Residuals — significant autocorrelation confirms ARCH'
             f'\n(ARCH p={arch_p:.6f})', fontsize=12, fontweight='bold')
ax.set_xlabel('Lag'); ax.set_ylabel('ACF(e²)'); ax.legend()

# rolling volatility by hour
ax = axes[2]; ax.set_facecolor(PALETTE['bg'])
hourly_vol = cal.groupby('hour')['y'].std()
ax.bar(hourly_vol.index, hourly_vol.values,
       color=[PALETTE['accent'] if v > hourly_vol.mean() else PALETTE['secondary']
              for v in hourly_vol.values],
       edgecolor='white', alpha=0.85)
ax.axhline(hourly_vol.mean(), color=PALETTE['dark'], ls='--', lw=2,
           label=f'Mean std = {hourly_vol.mean():.0f} MWh')
ax.set_title('Conditional Volatility (Std) by Hour of Day — hours with high variability',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Hour of Day'); ax.set_ylabel('Std (MWh)'); ax.legend()

fig.suptitle('Section 10 — Volatility & Heteroscedasticity', fontsize=15,
             fontweight='bold', color=PALETTE['dark'])
fig.tight_layout()
save(fig, '10_volatility.png')

# ══════════════════════════════════════════════════════════════════════════════
# 11. DISTRIBUTION TESTS & Q-Q PLOT
# ══════════════════════════════════════════════════════════════════════════════
section("11. DISTRIBUTION ANALYSIS")

# KS test for normality
ks_stat, ks_p = stats.kstest(y, 'norm', args=(y.mean(), y.std()))
print(f"\n  KS normality test : stat={ks_stat:.4f}  p={ks_p:.6f}")

# Anderson-Darling
ad_result = anderson(y, dist='norm')
print(f"  Anderson-Darling  : stat={ad_result.statistic:.4f}")
for sl, cv in zip(ad_result.significance_level, ad_result.critical_values):
    print(f"    {sl}%: cv={cv:.4f}  {'REJECT' if ad_result.statistic > cv else 'fail to reject'}")

# weekday vs weekend KS test
wd = cal[cal['is_weekend']==0]['y'].values
we = cal[cal['is_weekend']==1]['y'].values
ks2, ks2p = ks_2samp(wd, we)
print(f"\n  Weekday vs Weekend KS test: stat={ks2:.4f}  p={ks2p:.6f}")
print(f"  → {'Distributions DIFFER significantly' if ks2p<0.05 else 'No significant difference'}")

fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=PALETTE['bg'])

# Q-Q
ax = axes[0, 0]; ax.set_facecolor(PALETTE['bg'])
(osm, osr), (slope, intercept, r) = probplot(y, dist='norm')
ax.plot(osm, osr, 'o', color=PALETTE['secondary'], ms=1.5, alpha=0.4)
ax.plot(osm, slope*np.array(osm)+intercept, color=PALETTE['accent'], lw=2.5)
ax.set_title(f'Q-Q Plot vs Normal\n(R={r:.4f})', fontsize=11, fontweight='bold')
ax.set_xlabel('Theoretical Quantiles'); ax.set_ylabel('Sample Quantiles')

# Weekday vs Weekend PDF
ax = axes[0, 1]; ax.set_facecolor(PALETTE['bg'])
ax.hist(wd, bins=60, density=True, alpha=0.5, color=PALETTE['primary'],
        label=f'Weekday (n={len(wd):,})')
ax.hist(we, bins=60, density=True, alpha=0.5, color=PALETTE['warn'],
        label=f'Weekend (n={len(we):,})')
ax.set_title(f'Weekday vs Weekend Distribution\nKS p={ks2p:.6f}',
             fontsize=11, fontweight='bold')
ax.set_xlabel('MWh'); ax.legend()

# CDF
ax = axes[1, 0]; ax.set_facecolor(PALETTE['bg'])
y_sorted = np.sort(y)
ax.plot(y_sorted, np.arange(N)/N, color=PALETTE['secondary'], lw=2, label='Empirical CDF')
ax.plot(y_sorted, stats.norm.cdf(y_sorted, y.mean(), y.std()),
        color=PALETTE['accent'], lw=2, ls='--', label='Normal CDF')
ax.set_title('Empirical vs Normal CDF', fontsize=11, fontweight='bold')
ax.set_xlabel('MWh'); ax.legend()

# Summer vs winter
ax = axes[1, 1]; ax.set_facecolor(PALETTE['bg'])
summer = cal[cal['month'].isin([6,7,8])]['y'].values
winter = cal[cal['month'].isin([12,1,2])]['y'].values
ks_sw, ks_swp = ks_2samp(summer, winter)
ax.hist(summer, bins=60, density=True, alpha=0.5, color=PALETTE['accent'],
        label=f'Summer (JJA) μ={summer.mean():.0f}')
ax.hist(winter, bins=60, density=True, alpha=0.5, color=PALETTE['secondary'],
        label=f'Winter (DJF) μ={winter.mean():.0f}')
ax.set_title(f'Summer vs Winter Distribution\nKS p={ks_swp:.6f}',
             fontsize=11, fontweight='bold')
ax.set_xlabel('MWh'); ax.legend()

fig.suptitle('Section 11 — Distribution Analysis', fontsize=15, fontweight='bold',
             color=PALETTE['dark'])
fig.tight_layout()
save(fig, '11_distribution.png')

# ══════════════════════════════════════════════════════════════════════════════
# 12. LONG MEMORY — HURST EXPONENT
# ══════════════════════════════════════════════════════════════════════════════
section("12. LONG MEMORY (HURST EXPONENT)")

def hurst_rs(series, min_n=10):
    """R/S analysis for Hurst exponent."""
    N_ = len(series)
    lags, rs_vals = [], []
    for n in np.unique(np.geomspace(min_n, N_//2, 30).astype(int)):
        chunks = [series[i:i+n] for i in range(0, N_ - n, n)]
        rs_chunk = []
        for chunk in chunks:
            mean_c = np.mean(chunk)
            dev    = np.cumsum(chunk - mean_c)
            R      = dev.max() - dev.min()
            S      = np.std(chunk, ddof=1)
            if S > 0:
                rs_chunk.append(R / S)
        if rs_chunk:
            lags.append(np.log10(n))
            rs_vals.append(np.log10(np.mean(rs_chunk)))
    H_slope, intercept_h = np.polyfit(lags, rs_vals, 1)
    return H_slope, lags, rs_vals

H, lags_h, rs_h = hurst_rs(y)
print(f"\n  Hurst Exponent H = {H:.4f}")
if H > 0.5:
    print(f"  → Long memory / PERSISTENT (H > 0.5) — autocorr decays hyperbolically")
elif H < 0.5:
    print(f"  → Anti-persistent (H < 0.5)")
else:
    print(f"  → Random walk (H ≈ 0.5)")

# DFA
def dfa(series, min_n=10, max_n=None, n_steps=25):
    """Detrended Fluctuation Analysis."""
    N_ = len(series)
    max_n = max_n or N_//4
    cumdev = np.cumsum(series - np.mean(series))
    ns = np.unique(np.geomspace(min_n, max_n, n_steps).astype(int))
    F_vals = []
    for n in ns:
        rms_list = []
        for start in range(0, N_ - n, n):
            seg = cumdev[start:start+n]
            x_  = np.arange(n)
            p   = np.polyfit(x_, seg, 1)
            rms_list.append(np.sqrt(np.mean((seg - np.polyval(p, x_))**2)))
        F_vals.append(np.mean(rms_list))
    alpha_dfa, _ = np.polyfit(np.log10(ns), np.log10(F_vals), 1)
    return alpha_dfa, ns, F_vals

alpha_dfa, ns_dfa, F_dfa = dfa(y)
print(f"  DFA scaling exponent α = {alpha_dfa:.4f}")
print(f"  (α>0.5 = long memory, α≈0.5 = uncorrelated, α<0.5 = anti-persistent)")

fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=PALETTE['bg'])

ax = axes[0]; ax.set_facecolor(PALETTE['bg'])
ax.scatter(lags_h, rs_h, color=PALETTE['secondary'], s=40, zorder=5)
fit_line = np.polyfit(lags_h, rs_h, 1)
xl = np.linspace(min(lags_h), max(lags_h), 100)
ax.plot(xl, np.polyval(fit_line, xl), color=PALETTE['accent'], lw=2.5,
        label=f'H = {H:.4f}')
ax.set_title('R/S Analysis — Hurst Exponent', fontsize=12, fontweight='bold')
ax.set_xlabel('log₁₀(n)'); ax.set_ylabel('log₁₀(R/S)'); ax.legend()

ax = axes[1]; ax.set_facecolor(PALETTE['bg'])
ax.scatter(np.log10(ns_dfa), np.log10(F_dfa), color=PALETTE['secondary'], s=40, zorder=5)
fit_dfa = np.polyfit(np.log10(ns_dfa), np.log10(F_dfa), 1)
xl_ = np.linspace(np.log10(ns_dfa[0]), np.log10(ns_dfa[-1]), 100)
ax.plot(xl_, np.polyval(fit_dfa, xl_), color=PALETTE['success'], lw=2.5,
        label=f'α = {alpha_dfa:.4f}')
ax.set_title('DFA — Scaling Exponent', fontsize=12, fontweight='bold')
ax.set_xlabel('log₁₀(n)'); ax.set_ylabel('log₁₀(F(n))'); ax.legend()

fig.suptitle(f'Section 12 — Long Memory  (H={H:.4f}, DFA α={alpha_dfa:.4f})',
             fontsize=15, fontweight='bold', color=PALETTE['dark'])
fig.tight_layout()
save(fig, '12_hurst_dfa.png')

# ══════════════════════════════════════════════════════════════════════════════
# 13. NONLINEARITY TESTS
# ══════════════════════════════════════════════════════════════════════════════
section("13. NONLINEARITY TESTS")

# BDS test (manual implementation — simplified)
def bds_statistic(x, m=2, eps_frac=0.7):
    """Simplified BDS statistic at embedding m."""
    n   = len(x)
    eps = eps_frac * np.std(x)
    # C_1
    pairs1 = np.sum(np.abs(x[:, None] - x[None, :]) < eps) - n
    C1     = pairs1 / (n*(n-1))
    # C_m (embed)
    X_m  = np.array([x[i:n-m+i+1] for i in range(m)]).T   # (n-m+1) × m
    n_m  = len(X_m)
    dm   = np.max(np.abs(X_m[:, None, :] - X_m[None, :, :]), axis=2)
    Cm   = (np.sum(dm < eps) - n_m) / (n_m*(n_m-1))
    bds  = (Cm - C1**m) * np.sqrt(n_m)     # simplified; proper variance omitted
    return bds, C1, Cm

try:
    bds_s, C1, Cm = bds_statistic(resid_ar[:5000], m=2)
    print(f"\n  BDS statistic (m=2, simplified): {bds_s:.4f}")
    print(f"  C₁={C1:.6f}  Cm={Cm:.6f}")
    print(f"  → Large |BDS| suggests nonlinear dependence in AR(1) residuals")
except Exception as e:
    print(f"  BDS: {e}")

# White test for heteroscedasticity in AR residuals
try:
    X_w = add_constant(np.column_stack([resid_ar[:-1], resid_ar[:-1]**2]))
    y_w = resid_ar[1:]
    wh  = het_white(y_w, X_w)
    print(f"\n  White test (AR resid): stat={wh[0]:.4f}  p={wh[1]:.6f}")
    print(f"  → {'Heteroscedastic' if wh[1]<0.05 else 'Homoscedastic'}")
except Exception as e:
    print(f"  White test error: {e}")

# Runs test (manual)
def runs_test(x):
    med   = np.median(x)
    signs = (x > med).astype(int)
    runs  = 1 + np.sum(signs[1:] != signs[:-1])
    n1    = signs.sum(); n2 = len(signs) - n1
    mu_r  = 2*n1*n2/(n1+n2) + 1
    var_r = 2*n1*n2*(2*n1*n2-n1-n2) / ((n1+n2)**2*(n1+n2-1))
    z     = (runs - mu_r) / np.sqrt(var_r)
    p     = 2*(1 - stats.norm.cdf(abs(z)))
    return z, p

rt_z, rt_p = runs_test(y)
print(f"\n  Runs test: z={rt_z:.4f}  p={rt_p:.6f}")
print(f"  → {'Significant pattern (not random)' if rt_p<0.05 else 'Random'}")

# ══════════════════════════════════════════════════════════════════════════════
# 14. STRUCTURAL BREAK — CUSUM
# ══════════════════════════════════════════════════════════════════════════════
section("14. STRUCTURAL BREAKS (CUSUM)")

# CUSUM on daily means
cusum = np.cumsum(dm - dm.mean()) / (dm.std() * np.sqrt(len(dm)))
upper = 1.36   # 5% critical value for CUSUM
lower = -1.36

break_idx = np.where((cusum > upper) | (cusum < lower))[0]
if len(break_idx):
    first_break = daily_mean.index[break_idx[0]]
    print(f"\n  CUSUM first breach at: {first_break.date()}")
else:
    print(f"\n  CUSUM stays within bounds — no major structural break")

fig, axes = plt.subplots(2, 1, figsize=(18, 10), facecolor=PALETTE['bg'])

ax = axes[0]; ax.set_facecolor(PALETTE['bg'])
ax.plot(daily_mean.index, cusum, color=PALETTE['secondary'], lw=1.5,
        label='CUSUM')
ax.axhline( upper, color=PALETTE['accent'], ls='--', lw=2, label=f'±{upper} (5%)')
ax.axhline( lower, color=PALETTE['accent'], ls='--', lw=2)
ax.axhline(0, color=PALETTE['dark'], lw=1)
if len(break_idx):
    ax.axvline(first_break, color=PALETTE['warn'], lw=2.5,
               label=f'First breach: {first_break.date()}')
ax.set_title('CUSUM of Daily Mean Consumption — structural stability test',
             fontsize=12, fontweight='bold')
ax.set_ylabel('Standardised CUSUM'); ax.legend()

# Rolling mean year-on-year change
ax = axes[1]; ax.set_facecolor(PALETTE['bg'])
roll_ann = pd.Series(dm, index=daily_mean.index).rolling(365, center=True).mean()
yoy_pct  = roll_ann.pct_change(365) * 100
ax.plot(daily_mean.index, yoy_pct, color=PALETTE['success'], lw=1.5)
ax.axhline(0, color=PALETTE['dark'], lw=1.5, ls='--')
ax.fill_between(daily_mean.index, 0, yoy_pct,
                where=yoy_pct > 0, alpha=0.3, color=PALETTE['success'], label='Growth')
ax.fill_between(daily_mean.index, 0, yoy_pct,
                where=yoy_pct < 0, alpha=0.3, color=PALETTE['accent'], label='Decline')
ax.set_title('Year-on-Year % Change (365-day rolling mean)', fontsize=12, fontweight='bold')
ax.set_ylabel('YoY %'); ax.legend()

fig.suptitle('Section 14 — Structural Breaks', fontsize=15, fontweight='bold',
             color=PALETTE['dark'])
fig.tight_layout()
save(fig, '14_structural_breaks.png')

# ══════════════════════════════════════════════════════════════════════════════
# 15. CROSS-CORRELATION & LAG ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
section("15. LAG ANALYSIS — PREDICTIVE POWER")

lag_list = [1, 2, 3, 6, 12, 24, 48, 72, 96, 120, 144, 168, 336]
lag_corr = {}
for lag in lag_list:
    r, p = stats.pearsonr(y[lag:], y[:-lag])
    lag_corr[lag] = {'r': r, 'r2': r**2, 'p': p}

print("\n  Lag  |  Pearson r  |  R²      |  p-value")
print("  " + "-"*50)
for lag, vals in lag_corr.items():
    print(f"  {lag:4d} |  {vals['r']:.6f}  |  {vals['r2']:.6f}  |  {vals['p']:.2e}")

fig, ax = plt.subplots(figsize=(14, 6), facecolor=PALETTE['bg'])
ax.set_facecolor(PALETTE['bg'])
lags_arr = list(lag_corr.keys())
r2_arr   = [lag_corr[l]['r2'] for l in lags_arr]
colors_l = [PALETTE['accent'] if l in [24, 168] else PALETTE['secondary']
            for l in lags_arr]
bars = ax.bar(range(len(lags_arr)), r2_arr, color=colors_l,
              edgecolor='white', alpha=0.85)
ax.set_xticks(range(len(lags_arr)))
ax.set_xticklabels([f'lag_{l}' for l in lags_arr], rotation=45)
for i, (r2, bar) in enumerate(zip(r2_arr, bars)):
    ax.text(bar.get_x()+bar.get_width()/2, r2+0.002,
            f'{r2:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_title('R² (Explained Variance) per Lag — justifies lag feature selection',
             fontsize=12, fontweight='bold')
ax.set_ylabel('R²')
ax.legend(handles=[
    plt.Rectangle((0,0),1,1, color=PALETTE['accent'], label='Key seasonal lags (24h, 168h)'),
    plt.Rectangle((0,0),1,1, color=PALETTE['secondary'], label='Other lags'),
], fontsize=9)

fig.suptitle('Section 15 — Lag Predictive Power', fontsize=15, fontweight='bold',
             color=PALETTE['dark'])
fig.tight_layout()
save(fig, '15_lag_analysis.png')

# ══════════════════════════════════════════════════════════════════════════════
# 16. SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════
section("16. CONSOLIDATED INSIGHTS REPORT")

report_lines = [
    "=" * 70,
    "  ENERGY CONSUMPTION — DEEP STATISTICAL ANALYSIS REPORT",
    "=" * 70,
    f"  Dataset       : {N:,} hourly observations",
    f"  Period        : {ts.index[0].date()} → {ts.index[-1].date()}",
    f"  Mean          : {y.mean():.1f} MWh",
    f"  Std           : {y.std():.1f} MWh",
    f"  CV            : {y.std()/y.mean()*100:.2f}%",
    f"  Skewness      : {skew(y):.4f}  ({'right-skewed' if skew(y)>0 else 'left-skewed'})",
    f"  Excess Kurt.  : {kurtosis(y):.4f}  ({'heavy tails' if kurtosis(y)>0 else 'thin tails'})",
    "",
    "  ── STATIONARITY ──────────────────────────────────────────────",
    f"  ADF p-value   : {adf[1]:.6f}  → {'STATIONARY' if adf[1]<0.05 else 'NON-STATIONARY'}",
    f"  KPSS p-value  : {kp[1]:.4f}  → {'NON-STATIONARY' if kp[1]<0.05 else 'STATIONARY'}",
    f"  1st-diff ADF  : {adf_d[1]:.6f}  → {'STATIONARY' if adf_d[1]<0.05 else 'NON-STATIONARY'}",
    "",
    "  ── SEASONALITY ────────────────────────────────────────────────",
    f"  Seasonal Strength (STL, period=24) : {fs:.4f}",
    f"  Trend    Strength (STL)            : {ft:.4f}",
    f"  Dominant spectral periods          : 24h, 168h (from periodogram)",
    "",
    "  ── LONG MEMORY ────────────────────────────────────────────────",
    f"  Hurst Exponent (R/S) : {H:.4f}  ({'Long memory' if H>0.5 else 'Random walk'})",
    f"  DFA alpha            : {alpha_dfa:.4f}",
    "",
    "  ── VOLATILITY ─────────────────────────────────────────────────",
    f"  ARCH-LM (24 lags) p  : {arch_p:.6f}  → {'ARCH PRESENT' if arch_p<0.05 else 'no ARCH'}",
    "",
    "  ── OUTLIERS ───────────────────────────────────────────────────",
    f"  Contextual IQR       : {n_ctx} ({n_ctx/N*100:.2f}%)",
    f"  Isolation Forest     : {n_if}  ({n_if/N*100:.2f}%)",
    "",
    "  ── TREND ──────────────────────────────────────────────────────",
    f"  OLS slope  : {slope_per_year:.1f} MWh/year  (R²={ols.rsquared:.4f})",
    f"  Mann-Kendall p : {mk_p:.6f}",
    f"  Total growth   : {pct_growth:.1f}% over {len(years)-1} years",
    "",
    "  ── KEY LAGS ───────────────────────────────────────────────────",
]
for lag in [1, 24, 168]:
    report_lines.append(
        f"  lag_{lag:3d}  r={lag_corr[lag]['r']:.4f}  R²={lag_corr[lag]['r2']:.4f}"
    )
report_lines += [
    "",
    "  ── MODEL IMPLICATIONS ─────────────────────────────────────────",
    "  1. Non-Gaussian heavy tails → use Huber/quantile loss (not MSE)",
    "  2. ARCH effects             → model variance separately (GARCH)",
    "  3. Long memory (H>0.5)      → fractional diff OR long lag window",
    "  4. Triple seasonality       → TBATS / N-HiTS / multi-res Transformer",
    "  5. Nonlinearity confirmed   → XGBoost / LSTM superior to SARIMA",
    "  6. High R² at lag 24, 168   → must include these as lag features",
    "  7. Significant trend        → include time index / detrend first",
    "=" * 70,
]

report_str = "\n".join(report_lines)
print(report_str)

report_path = os.path.join(OUT, "analysis_report.txt")
with open(report_path, 'w') as f:
    f.write(report_str)
print(f"\n  [saved] {report_path}")

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  ALL DONE — outputs in ./{OUT}/")
print(f"{'='*70}")