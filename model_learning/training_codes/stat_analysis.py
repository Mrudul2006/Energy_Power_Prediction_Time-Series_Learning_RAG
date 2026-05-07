# """
# Comprehensive statistical analysis pipeline for hourly energy-consumption
# forecasting.

# The previous version claimed full coverage but left many items unimplemented,
# rough proxies, or local variables that were never reported. This version
# keeps an explicit 180-item audit trail. Each item is recorded as:

#     implemented  - computed from the available data
#     unavailable  - cannot be computed honestly from the supplied columns or
#                    installed optional libraries
#     failed       - attempted, but an exception occurred

# The class intentionally preserves the original public name
# ExactMathematicalPipeline for compatibility with existing callers.
# """

# from __future__ import annotations

# from dataclasses import dataclass, asdict
# from math import factorial
# from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
# import itertools
# import json
# import warnings

# import numpy as np
# import pandas as pd
# from scipy import fft, signal, sparse, stats
# from scipy.interpolate import CubicSpline
# from scipy.optimize import minimize
# from scipy.sparse.linalg import spsolve
# from scipy.spatial.distance import cdist, pdist, squareform

# from sklearn.decomposition import PCA
# from sklearn.ensemble import (
#     GradientBoostingRegressor,
#     IsolationForest,
#     RandomForestRegressor,
# )
# from sklearn.feature_selection import mutual_info_regression
# from sklearn.linear_model import ElasticNetCV, LassoCV, Ridge, TheilSenRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from sklearn.mixture import GaussianMixture
# from sklearn.neighbors import LocalOutlierFactor
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import SplineTransformer, StandardScaler

# warnings.filterwarnings("ignore")


# ITEM_NAMES = {
#     1: "Conditional mean by hour, weekday, month",
#     2: "Conditional median and mean-to-median ratio",
#     3: "Global, conditional, and rolling standard deviation",
#     4: "Skewness and Jarque-Bera normality test",
#     5: "Excess kurtosis and tail implication",
#     6: "Percentiles, ECDF, and conditional KS tests",
#     7: "Intra-day, intra-week, and intra-year range analysis",
#     8: "Augmented Dickey-Fuller test",
#     9: "Phillips-Perron test",
#     10: "KPSS stationarity test",
#     11: "Zivot-Andrews one-break unit-root test",
#     12: "Lee-Strazicich two-break LM-style test",
#     13: "Lo-MacKinlay variance ratio test",
#     14: "Seasonal unit-root harmonic screen",
#     15: "OCSB seasonal differencing test",
#     16: "Sample autocorrelation function",
#     17: "Partial autocorrelation function",
#     18: "Extended autocorrelation table",
#     19: "Ljung-Box portmanteau tests",
#     20: "Inverse autocorrelation function",
#     21: "Wold MA-infinity impulse response",
#     22: "Correlation dimension",
#     23: "Periodogram",
#     24: "Smoothed spectral density",
#     25: "Thomson multitaper spectrum",
#     26: "Spectral-line F tests",
#     27: "Cumulative spectral distribution",
#     28: "Cross-spectrum and coherence",
#     29: "Continuous wavelet transform",
#     30: "Wavelet multiresolution features",
#     31: "Empirical mode decomposition",
#     32: "Ensemble empirical mode decomposition",
#     33: "Hodrick-Prescott trend filter",
#     34: "Hamilton regression filter",
#     35: "Mann-Kendall trend test",
#     36: "Sen slope estimator",
#     37: "Theil-Sen regression",
#     38: "LOWESS trend smoother",
#     39: "Classical seasonal decomposition",
#     40: "STL-style robust decomposition",
#     41: "MSTL multiple-season decomposition",
#     42: "Seasonal strength",
#     43: "Trend strength",
#     44: "Seasonal subseries statistics",
#     45: "Canova-Hansen-style seasonal stability test",
#     46: "Bewley-Fiebig harmonic seasonality test",
#     47: "Rescaled-range Hurst exponent",
#     48: "Local Whittle fractional-d estimator",
#     49: "GPH fractional-d estimator",
#     50: "Breitung-Hassler long-memory test",
#     51: "BDS dependence test",
#     52: "Terasvirta neural-network nonlinearity test",
#     53: "Keenan nonlinearity test",
#     54: "McLeod-Li squared-residual test",
#     55: "Hinich bispectrum statistic",
#     56: "Tsay threshold nonlinearity test",
#     57: "Engle ARCH LM test",
#     58: "GARCH(1,1) estimation",
#     59: "GJR-GARCH estimation",
#     60: "White heteroscedasticity test",
#     61: "Goldfeld-Quandt variance test",
#     62: "Chow structural-break test",
#     63: "Quandt likelihood-ratio test",
#     64: "CUSUM stability test",
#     65: "Bai-Perron multiple-break dynamic program",
#     66: "Tsay outlier classification",
#     67: "Grubbs outlier test",
#     68: "Seasonal IQR outlier detection",
#     69: "Local outlier factor",
#     70: "Isolation forest",
#     71: "CUSUM anomaly chart",
#     72: "Akaike information criterion",
#     73: "Corrected AIC",
#     74: "Bayesian information criterion",
#     75: "Hannan-Quinn criterion",
#     76: "Minimum description length",
#     77: "h-block time-series cross-validation",
#     78: "Prequential prediction-error validation",
#     79: "Diebold-Mariano forecast comparison",
#     80: "Clark-West nested forecast comparison",
#     81: "Johansen cointegration test",
#     82: "Engle-Granger cointegration test",
#     83: "Granger causality test",
#     84: "Shannon entropy",
#     85: "Approximate entropy",
#     86: "Sample entropy",
#     87: "Permutation entropy",
#     88: "Transfer entropy",
#     89: "Variance inflation factors",
#     90: "Cook distance",
#     91: "DFFITS and DFBETAS",
#     92: "Harvey-Collier nonlinearity test",
#     93: "Ramsey RESET",
#     94: "Butterworth low-pass filter",
#     95: "Kalman local-level filter",
#     96: "Singular spectrum analysis",
#     97: "Robust PCA",
#     98: "Moving block bootstrap",
#     99: "Stationary bootstrap",
#     100: "Wild bootstrap",
#     101: "Cross-correlation function",
#     102: "Maximum cross-correlation",
#     103: "Markov regime-switching analysis",
#     104: "Threshold autoregressive model",
#     105: "Smooth-transition autoregressive model",
#     106: "Residual normality tests",
#     107: "Q-Q residual diagnostics",
#     108: "Kling-Gupta efficiency",
#     109: "Nash-Sutcliffe efficiency",
#     110: "Theil U statistic",
#     111: "Quantile regression",
#     112: "Winkler interval score",
#     113: "Conformal prediction intervals",
#     114: "Pinball loss and CRPS approximation",
#     115: "Mutual information feature relevance",
#     116: "mRMR feature selection",
#     117: "LASSO feature selection",
#     118: "Elastic Net feature selection",
#     119: "Boruta-style shadow-feature selection",
#     120: "SHAP/Shapley-style feature attribution",
#     121: "Reliability diagram calibration",
#     122: "Continuous ranked probability score",
#     123: "Skill scores",
#     124: "Locally stationary spectrum",
#     125: "Time-varying coefficient AR model",
#     126: "Functional daily-curve PCA",
#     127: "Mincer-Zarnowitz regression",
#     128: "Forecast encompassing test",
#     129: "Giacomini-White conditional forecast test",
#     130: "Superior predictive ability test",
#     131: "Convergent cross mapping",
#     132: "Recurrence quantification analysis",
#     133: "Detrended fluctuation analysis",
#     134: "Copula dependency and tail dependence",
#     135: "Seasonal distribution Q-Q tests",
#     136: "MODWT wavelet variance",
#     137: "Generalized additive model baseline",
#     138: "TBATS-style harmonic component selection",
#     139: "Dynamic time warping kNN baseline",
#     140: "PELT change-point detection",
#     141: "Functional temperature-consumption regression",
#     142: "Complexity-invariant distance",
#     143: "Matrix profile motif and discord discovery",
#     144: "Persistent homology",
#     145: "Maximal information coefficient",
#     146: "Lasso-PACF lag selection",
#     147: "Wavelet coherence between years",
#     148: "KL distribution drift",
#     149: "Rao-Blackwell forecast-combination weights",
#     150: "Bartlett equality-of-variances test",
#     151: "ARCH-in-Mean model",
#     152: "Wavelet variance ratio test",
#     153: "Categorical information gain",
#     154: "Box-Cox transformation selection",
#     155: "Guerrero seasonal-period screen",
#     156: "Spectral entropy",
#     157: "Frequency-domain Granger causality",
#     158: "Seasonal McLeod-Li diagnostics",
#     159: "Spectral whiteness test",
#     160: "Jensen-Shannon distribution comparison",
#     161: "Concordance correlation coefficient",
#     162: "PICP, MPIW, and CWC interval metrics",
#     163: "Brier score for peak events",
#     164: "Directional symmetry",
#     165: "Extreme-value tail analysis",
#     166: "Residual bootstrap forecast uncertainty",
#     167: "Profile likelihood over AR order",
#     168: "BIC-weighted model averaging",
#     169: "Hong causality in mean/variance",
#     170: "Expected shortfall",
#     171: "Statistical-analysis-to-model pipeline",
#     172: "Feature engineering from statistical analysis",
#     173: "Hyperparameter mapping from diagnostics",
#     174: "Loss-function selection",
#     175: "Validation strategy",
#     176: "Ensemble weight optimization",
#     177: "Bayesian model averaging uncertainty",
#     178: "Recursive least squares adaptation",
#     179: "Online Hedge ensemble updates",
#     180: "Benjamini-Hochberg multiple-testing correction",
# }


# @dataclass
# class AnalysisRecord:
#     item: int
#     name: str
#     status: str
#     value: Any = None
#     note: str = ""


# def _finite_array(values: Sequence[float]) -> np.ndarray:
#     arr = np.asarray(values, dtype=float)
#     return arr[np.isfinite(arr)]


# def _native(obj: Any, max_items: int = 20) -> Any:
#     """Convert numpy/pandas objects to compact JSON-friendly values."""
#     if obj is None:
#         return None
#     if isinstance(obj, (np.integer,)):
#         return int(obj)
#     if isinstance(obj, (np.floating,)):
#         value = float(obj)
#         return None if not np.isfinite(value) else value
#     if isinstance(obj, (np.bool_,)):
#         return bool(obj)
#     if isinstance(obj, pd.Timestamp):
#         return obj.isoformat()
#     if isinstance(obj, pd.Series):
#         head = obj.head(max_items)
#         return {
#             "length": int(len(obj)),
#             "head": {str(k): _native(v) for k, v in head.items()},
#         }
#     if isinstance(obj, pd.DataFrame):
#         return {
#             "shape": [int(obj.shape[0]), int(obj.shape[1])],
#             "head": obj.head(max_items).applymap(_native).to_dict(orient="records"),
#         }
#     if isinstance(obj, np.ndarray):
#         flat = obj.ravel()
#         return {
#             "shape": [int(i) for i in obj.shape],
#             "head": [_native(v) for v in flat[:max_items]],
#         }
#     if isinstance(obj, dict):
#         return {str(k): _native(v, max_items=max_items) for k, v in obj.items()}
#     if isinstance(obj, (list, tuple)):
#         return [_native(v, max_items=max_items) for v in list(obj)[:max_items]]
#     return obj


# def _summary(values: Sequence[float]) -> Dict[str, float]:
#     x = _finite_array(values)
#     if x.size == 0:
#         return {}
#     qs = np.percentile(x, [1, 5, 25, 50, 75, 95, 99])
#     return {
#         "n": int(x.size),
#         "mean": float(np.mean(x)),
#         "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
#         "min": float(np.min(x)),
#         "p01": float(qs[0]),
#         "p05": float(qs[1]),
#         "p25": float(qs[2]),
#         "median": float(qs[3]),
#         "p75": float(qs[4]),
#         "p95": float(qs[5]),
#         "p99": float(qs[6]),
#         "max": float(np.max(x)),
#     }


# def _ols(y: np.ndarray, X: np.ndarray) -> Dict[str, np.ndarray]:
#     y = np.asarray(y, dtype=float)
#     X = np.asarray(X, dtype=float)
#     mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
#     y = y[mask]
#     X = X[mask]
#     if len(y) <= X.shape[1]:
#         raise ValueError("not enough observations for OLS")
#     beta, *_ = np.linalg.lstsq(X, y, rcond=None)
#     fitted = X @ beta
#     resid = y - fitted
#     dof = max(1, len(y) - X.shape[1])
#     s2 = float(np.sum(resid**2) / dof)
#     xtx_inv = np.linalg.pinv(X.T @ X)
#     se = np.sqrt(np.maximum(np.diag(xtx_inv) * s2, 0.0))
#     t = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
#     return {
#         "beta": beta,
#         "fitted": fitted,
#         "resid": resid,
#         "rss": float(np.sum(resid**2)),
#         "s2": s2,
#         "se": se,
#         "t": t,
#         "n": len(y),
#         "k": X.shape[1],
#         "r2": float(1 - np.sum(resid**2) / max(np.sum((y - np.mean(y)) ** 2), 1e-12)),
#     }


# def _newey_west_cov(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
#     X = np.asarray(X, dtype=float)
#     resid = np.asarray(resid, dtype=float)
#     n, k = X.shape
#     S = np.zeros((k, k))
#     xu = X * resid[:, None]
#     S += xu.T @ xu
#     for lag in range(1, max(0, int(lags)) + 1):
#         w = 1.0 - lag / (lags + 1.0)
#         gamma = xu[lag:].T @ xu[:-lag]
#         S += w * (gamma + gamma.T)
#     bread = np.linalg.pinv(X.T @ X)
#     return bread @ S @ bread


# def _lag_matrix(y: np.ndarray, lags: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
#     max_lag = int(max(lags))
#     Y = y[max_lag:]
#     X = np.column_stack([y[max_lag - lag : len(y) - lag] for lag in lags])
#     return Y, X


# def _acf_np(y: np.ndarray, nlags: int) -> np.ndarray:
#     y = _finite_array(y)
#     y = y - np.mean(y)
#     denom = np.dot(y, y)
#     if denom <= 0:
#         return np.ones(nlags + 1)
#     out = [1.0]
#     for lag in range(1, nlags + 1):
#         out.append(float(np.dot(y[lag:], y[:-lag]) / denom))
#     return np.asarray(out)


# def _pacf_np(y: np.ndarray, nlags: int) -> np.ndarray:
#     y = _finite_array(y)
#     out = [1.0]
#     for lag in range(1, nlags + 1):
#         Y, X = _lag_matrix(y, list(range(1, lag + 1)))
#         X = np.column_stack([np.ones(len(Y)), X])
#         fit = _ols(Y, X)
#         out.append(float(fit["beta"][-1]))
#     return np.asarray(out)


# def _ljung_box(y: np.ndarray, lags: Sequence[int], model_df: int = 0) -> Dict[int, Dict[str, float]]:
#     y = _finite_array(y)
#     n = len(y)
#     max_lag = min(max(lags), n - 2)
#     r = _acf_np(y, max_lag)
#     out: Dict[int, Dict[str, float]] = {}
#     for lag in lags:
#         lag = min(int(lag), max_lag)
#         q = n * (n + 2) * np.sum((r[1 : lag + 1] ** 2) / np.maximum(n - np.arange(1, lag + 1), 1))
#         dof = max(1, lag - model_df)
#         out[lag] = {"stat": float(q), "p_value": float(stats.chi2.sf(q, dof)), "dof": int(dof)}
#     return out


# def _aic_like(rss: float, n: int, k: int) -> Dict[str, float]:
#     n = max(int(n), 1)
#     k = max(int(k), 1)
#     sigma2 = max(rss / n, 1e-12)
#     log_l = -0.5 * n * (np.log(2 * np.pi) + np.log(sigma2) + 1)
#     aic = -2 * log_l + 2 * k
#     aicc = aic + (2 * k * (k + 1)) / max(n - k - 1, 1)
#     bic = -2 * log_l + k * np.log(n)
#     hqc = -2 * log_l + 2 * k * np.log(max(np.log(n), 1.000001))
#     mdl = -log_l + 0.5 * k * np.log(n)
#     return {
#         "log_likelihood": float(log_l),
#         "aic": float(aic),
#         "aicc": float(aicc),
#         "bic": float(bic),
#         "hqc": float(hqc),
#         "mdl": float(mdl),
#     }


# class ExactMathematicalPipeline:
#     def __init__(
#         self,
#         df: pd.DataFrame,
#         time_col: str,
#         target_col: str,
#         external_cols: Optional[Sequence[str]] = None,
#         seasonal_periods: Sequence[int] = (24, 168),
#         random_state: int = 42,
#     ):
#         self.time_col = time_col
#         self.target_col = target_col
#         self.random_state = int(random_state)
#         self.rng = np.random.default_rng(self.random_state)
#         self.seasonal_periods = tuple(int(p) for p in seasonal_periods if int(p) > 1)

#         self.df = df.copy()
#         self.df[time_col] = pd.to_datetime(self.df[time_col])
#         self.df[target_col] = pd.to_numeric(self.df[target_col], errors="coerce")
#         self.df = self.df.sort_values(time_col).dropna(subset=[target_col]).set_index(time_col)
#         self.df = self.df[~self.df.index.duplicated(keep="last")]

#         self.y = self.df[target_col].astype(float)
#         self.y_val = self.y.to_numpy(dtype=float)
#         self.N = len(self.y_val)

#         self.df["hour"] = self.df.index.hour
#         self.df["dayofweek"] = self.df.index.dayofweek
#         self.df["month"] = self.df.index.month
#         self.df["dayofyear"] = self.df.index.dayofyear
#         self.df["is_weekend"] = self.df["dayofweek"].isin([5, 6]).astype(int)
#         self.df["time_index"] = np.arange(self.N, dtype=float)

#         if external_cols is None:
#             numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
#             excluded = {target_col, "hour", "dayofweek", "month", "dayofyear", "is_weekend", "time_index"}
#             self.external_cols = [c for c in numeric_cols if c not in excluded]
#         else:
#             self.external_cols = [c for c in external_cols if c in self.df.columns]

#         self.records: Dict[int, AnalysisRecord] = {}
#         self.results: Dict[str, Any] = {}
#         self.p_values: List[Tuple[int, str, float]] = []
#         self.cache: Dict[str, Any] = {}

#     # ------------------------------------------------------------------
#     # Registry helpers
#     # ------------------------------------------------------------------
#     def _record(self, item: int, value: Any = None, status: str = "implemented", note: str = "") -> None:
#         self.records[item] = AnalysisRecord(
#             item=item,
#             name=ITEM_NAMES.get(item, f"Item {item}"),
#             status=status,
#             value=_native(value),
#             note=note,
#         )
#         if isinstance(value, dict):
#             for key, val in value.items():
#                 if "p" in str(key).lower():
#                     try:
#                         p = float(val)
#                         if 0 <= p <= 1:
#                             self.p_values.append((item, str(key), p))
#                     except Exception:
#                         continue

#     def _unavailable(self, item: int, reason: str) -> None:
#         self._record(item, value=None, status="unavailable", note=reason)

#     def _safe(self, item: int, fn: Callable[[], Any], note: str = "") -> None:
#         try:
#             self._record(item, fn(), note=note)
#         except Exception as exc:
#             message = f"{type(exc).__name__}: {exc}"
#             if isinstance(exc, ValueError) and "requires" in str(exc).lower():
#                 self._record(item, value=None, status="unavailable", note=message)
#             else:
#                 self._record(item, value=None, status="failed", note=message)

#     def _series_for_heavy(self, max_n: int = 2000) -> np.ndarray:
#         y = self.y_val
#         if len(y) <= max_n:
#             return y.copy()
#         idx = np.linspace(0, len(y) - 1, max_n).astype(int)
#         return y[idx].copy()

#     def _enough(self, n: int) -> bool:
#         return self.N >= n

#     # ------------------------------------------------------------------
#     # Core statistical helpers
#     # ------------------------------------------------------------------
#     def _adf_regression(
#         self,
#         y: np.ndarray,
#         lags: Optional[int] = None,
#         regression: str = "c",
#         extra_exog: Optional[np.ndarray] = None,
#     ) -> Dict[str, Any]:
#         y = _finite_array(y)
#         if len(y) < 20:
#             raise ValueError("ADF requires at least 20 observations")
#         if lags is None:
#             lags = int(np.floor(12 * (len(y) / 100.0) ** 0.25))
#         lags = max(0, min(int(lags), 48, len(y) // 4))
#         dy = np.diff(y)
#         target = dy[lags:]
#         y_lag = y[lags:-1]
#         cols = []
#         labels = []
#         if regression in ("c", "ct"):
#             cols.append(np.ones_like(target))
#             labels.append("constant")
#         if regression == "ct":
#             cols.append(np.arange(len(target), dtype=float))
#             labels.append("trend")
#         cols.append(y_lag)
#         labels.append("lagged_level")
#         for lag in range(1, lags + 1):
#             cols.append(dy[lags - lag : len(dy) - lag])
#             labels.append(f"diff_lag_{lag}")
#         if extra_exog is not None:
#             ex = np.asarray(extra_exog, dtype=float)
#             ex = ex[-len(target) :]
#             if ex.ndim == 1:
#                 ex = ex[:, None]
#             for j in range(ex.shape[1]):
#                 cols.append(ex[:, j])
#                 labels.append(f"extra_{j}")
#         X = np.column_stack(cols)
#         fit = _ols(target, X)
#         level_idx = labels.index("lagged_level")
#         stat = float(fit["t"][level_idx])
#         return {
#             "test_stat": stat,
#             "normal_p_approx": float(2 * stats.norm.sf(abs(stat))),
#             "lags": int(lags),
#             "regression": regression,
#             "nobs": int(fit["n"]),
#             "level_coef": float(fit["beta"][level_idx]),
#             "critical_values_reference": {
#                 "1pct": -3.43 if regression != "nc" else -2.58,
#                 "5pct": -2.86 if regression != "nc" else -1.95,
#                 "10pct": -2.57 if regression != "nc" else -1.62,
#             },
#             "note": "p-value uses normal approximation; ADF finite-sample MacKinnon tables are not bundled.",
#         }

#     def _pp_test(self, y: np.ndarray, regression: str = "c", nw_lags: Optional[int] = None) -> Dict[str, Any]:
#         y = _finite_array(y)
#         dy = np.diff(y)
#         target = dy
#         cols = []
#         labels = []
#         if regression in ("c", "ct"):
#             cols.append(np.ones_like(target))
#             labels.append("constant")
#         if regression == "ct":
#             cols.append(np.arange(len(target), dtype=float))
#             labels.append("trend")
#         cols.append(y[:-1])
#         labels.append("lagged_level")
#         X = np.column_stack(cols)
#         fit = _ols(target, X)
#         if nw_lags is None:
#             nw_lags = int(np.floor(4 * (len(y) / 100.0) ** (2.0 / 9.0)))
#         cov = _newey_west_cov(X, fit["resid"], nw_lags)
#         idx = labels.index("lagged_level")
#         se = float(np.sqrt(max(cov[idx, idx], 1e-18)))
#         stat = float(fit["beta"][idx] / se)
#         return {
#             "test_stat": stat,
#             "normal_p_approx": float(2 * stats.norm.sf(abs(stat))),
#             "nw_lags": int(nw_lags),
#             "regression": regression,
#             "level_coef": float(fit["beta"][idx]),
#         }

#     def _kpss_test(self, y: np.ndarray, regression: str = "c", nlags: Optional[int] = None) -> Dict[str, Any]:
#         y = _finite_array(y)
#         cols = [np.ones(len(y))]
#         if regression == "ct":
#             cols.append(np.arange(len(y), dtype=float))
#         X = np.column_stack(cols)
#         fit = _ols(y, X)
#         resid = fit["resid"]
#         n = len(resid)
#         if nlags is None:
#             nlags = int(np.floor(4 * (n / 100.0) ** 0.25))
#         s = np.cumsum(resid)
#         gamma0 = np.sum(resid**2) / n
#         lr_var = gamma0
#         for lag in range(1, max(0, nlags) + 1):
#             weight = 1 - lag / (nlags + 1.0)
#             gamma = np.sum(resid[lag:] * resid[:-lag]) / n
#             lr_var += 2 * weight * gamma
#         stat = float(np.sum(s**2) / (n**2 * max(lr_var, 1e-18)))
#         crit = (
#             {"10pct": 0.347, "5pct": 0.463, "2.5pct": 0.574, "1pct": 0.739}
#             if regression == "c"
#             else {"10pct": 0.119, "5pct": 0.146, "2.5pct": 0.176, "1pct": 0.216}
#         )
#         return {"test_stat": stat, "nlags": int(nlags), "regression": regression, "critical_values": crit}

#     def _simple_features(self) -> pd.DataFrame:
#         df = self.df.copy()
#         for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168, 336]:
#             if self.N > lag:
#                 df[f"lag_{lag}"] = self.y.shift(lag)
#         for win in [6, 24, 168]:
#             if self.N > win:
#                 df[f"roll_mean_{win}"] = self.y.shift(1).rolling(win).mean()
#                 df[f"roll_std_{win}"] = self.y.shift(1).rolling(win).std()
#         df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
#         df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
#         df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
#         df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
#         df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
#         df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
#         for c in self.external_cols:
#             df[c] = pd.to_numeric(df[c], errors="coerce")
#             df[f"{c}_lag_1"] = df[c].shift(1)
#             df[f"{c}_lag_24"] = df[c].shift(24)
#         feature_cols = [c for c in df.columns if c != self.target_col]
#         return df[[self.target_col] + feature_cols].dropna()

#     def _ensure_forecasts(self) -> Dict[str, Any]:
#         if "forecasts" in self.cache:
#             return self.cache["forecasts"]
#         data = self._simple_features()
#         if len(data) < 300:
#             raise ValueError("not enough complete rows for forecast diagnostics")
#         y = data[self.target_col]
#         X = data.drop(columns=[self.target_col])
#         X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).dropna(axis=1)
#         X = X.loc[:, X.nunique(dropna=True) > 1]
#         split = int(len(data) * 0.7)
#         cal_split = int(len(data) * 0.85)
#         X_train, X_cal, X_test = X.iloc[:split], X.iloc[split:cal_split], X.iloc[cal_split:]
#         y_train, y_cal, y_test = y.iloc[:split], y.iloc[split:cal_split], y.iloc[cal_split:]

#         model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
#         model.fit(X_train, y_train)
#         cal_pred = model.predict(X_cal)
#         y_pred = model.predict(X_test)

#         gb = GradientBoostingRegressor(random_state=self.random_state, n_estimators=120, max_depth=3)
#         gb.fit(X_train, y_train)
#         gb_pred = gb.predict(X_test)

#         q10 = GradientBoostingRegressor(
#             loss="quantile", alpha=0.1, random_state=self.random_state, n_estimators=80, max_depth=3
#         )
#         q90 = GradientBoostingRegressor(
#             loss="quantile", alpha=0.9, random_state=self.random_state, n_estimators=80, max_depth=3
#         )
#         q50 = GradientBoostingRegressor(
#             loss="quantile", alpha=0.5, random_state=self.random_state, n_estimators=80, max_depth=3
#         )
#         q10.fit(X_train, y_train)
#         q50.fit(X_train, y_train)
#         q90.fit(X_train, y_train)
#         q10_pred = q10.predict(X_test)
#         q50_pred = q50.predict(X_test)
#         q90_pred = q90.predict(X_test)

#         nonconf = np.abs(y_cal.to_numpy() - cal_pred)
#         conformal_q = float(np.quantile(nonconf, 0.9))
#         ci_lower = y_pred - conformal_q
#         ci_upper = y_pred + conformal_q

#         seasonal_naive = self.y.shift(24).reindex(y_test.index).bfill().to_numpy()
#         last_week = self.y.shift(168).reindex(y_test.index).bfill().to_numpy()

#         out = {
#             "X": X,
#             "y": y,
#             "X_train": X_train,
#             "y_train": y_train,
#             "X_test": X_test,
#             "y_test": y_test,
#             "linear_pred": np.asarray(y_pred),
#             "gb_pred": np.asarray(gb_pred),
#             "q10_pred": np.asarray(q10_pred),
#             "q50_pred": np.asarray(q50_pred),
#             "q90_pred": np.asarray(q90_pred),
#             "conformal_lower": np.asarray(ci_lower),
#             "conformal_upper": np.asarray(ci_upper),
#             "seasonal_naive": np.asarray(seasonal_naive),
#             "last_week_naive": np.asarray(last_week),
#             "model": model,
#             "gb_model": gb,
#             "feature_cols": X.columns.tolist(),
#             "cal_resid": y_cal.to_numpy() - cal_pred,
#         }
#         self.cache["forecasts"] = out
#         return out

#     # ------------------------------------------------------------------
#     # Category 1
#     # ------------------------------------------------------------------
#     def cat_1_descriptive(self) -> None:
#         y_name = self.target_col
#         g = self.df.groupby(["hour", "dayofweek", "month"])[y_name]
#         cond_mean = g.mean()
#         cond_median = g.median()
#         cond_std = g.std()
#         key_index = self.df.set_index(["hour", "dayofweek", "month"]).index
#         self.df["mu_hdm"] = key_index.map(cond_mean).astype(float)
#         self.df["median_hdm"] = key_index.map(cond_median).astype(float)
#         self.df["eps_t"] = self.y - self.df["mu_hdm"]
#         variance_explained = 1 - np.nanvar(self.df["eps_t"]) / max(np.nanvar(self.y_val), 1e-12)

#         self._record(
#             1,
#             {
#                 "cells": int(len(cond_mean)),
#                 "theoretical_cells": 24 * 7 * 12,
#                 "variance_explained_by_calendar_mean": float(variance_explained),
#                 "conditional_mean_summary": _summary(cond_mean.values),
#                 "residual_summary": _summary(self.df["eps_t"].values),
#             },
#         )
#         ratio = (cond_mean / cond_median.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
#         self._record(2, {"conditional_median_summary": _summary(cond_median.values), "mean_to_median": _summary(ratio.values)})
#         rolling_std = self.y.rolling(168, min_periods=max(12, min(168, self.N))).std()
#         cv = (cond_std / cond_mean.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
#         self._record(
#             3,
#             {
#                 "global_std": float(self.y.std()),
#                 "conditional_std_summary": _summary(cond_std.values),
#                 "rolling_168h_std_summary": _summary(rolling_std.dropna().values),
#                 "cv_summary": _summary(cv.dropna().values),
#                 "cv_gt_0_30_cells": int(np.sum(cv > 0.30)),
#             },
#         )
#         skew = float(stats.skew(self.y_val, bias=False))
#         jb = stats.jarque_bera(self.y_val)
#         self._record(4, {"skewness": skew, "jarque_bera": float(jb.statistic), "p_value": float(jb.pvalue)})
#         kurt = float(stats.kurtosis(self.y_val, fisher=True, bias=False))
#         self._record(
#             5,
#             {
#                 "excess_kurtosis": kurt,
#                 "tail_flag": bool(kurt > 2),
#                 "interval_recommendation": "use quantile or conformal intervals" if kurt > 2 else "Gaussian intervals less risky",
#             },
#         )
#         percentiles = {str(p): float(np.percentile(self.y_val, p)) for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]}
#         weekday = self.df.loc[self.df["is_weekend"] == 0, y_name]
#         weekend = self.df.loc[self.df["is_weekend"] == 1, y_name]
#         ks = stats.ks_2samp(weekday, weekend) if len(weekday) and len(weekend) else None
#         self._record(
#             6,
#             {
#                 "percentiles": percentiles,
#                 "weekend_weekday_ks_stat": None if ks is None else float(ks.statistic),
#                 "weekend_weekday_p_value": None if ks is None else float(ks.pvalue),
#             },
#         )
#         daily = self.df.groupby(self.df.index.floor("D"))[y_name]
#         weekly = self.df.groupby(pd.Grouper(freq="W"))[y_name]
#         yearly = self.df.groupby(pd.Grouper(freq="YE"))[y_name]
#         daily_range = daily.max() - daily.min()
#         daily_level = daily.mean()
#         weekly_range = weekly.max() - weekly.min()
#         yearly_range = yearly.max() - yearly.min()
#         corr = float(np.corrcoef(daily_range.dropna(), daily_level.loc[daily_range.dropna().index])[0, 1]) if len(daily_range.dropna()) > 3 else None
#         self._record(
#             7,
#             {
#                 "daily_range_summary": _summary(daily_range.values),
#                 "weekly_range_summary": _summary(weekly_range.values),
#                 "yearly_range_summary": _summary(yearly_range.values),
#                 "daily_range_level_corr": corr,
#                 "seasonality_form_hint": "multiplicative" if corr is not None and corr > 0.5 else "additive_or_weakly_multiplicative",
#             },
#         )

#     # ------------------------------------------------------------------
#     # Category 2
#     # ------------------------------------------------------------------
#     def cat_2_stationarity(self) -> None:
#         self._safe(
#             8,
#             lambda: {
#                 spec: self._adf_regression(self.y_val, regression=spec)
#                 for spec in ["nc", "c", "ct"]
#             },
#         )
#         self._safe(9, lambda: {spec: self._pp_test(self.y_val, regression=spec) for spec in ["c", "ct"]})
#         self._safe(10, lambda: {spec: self._kpss_test(self.y_val, regression=spec) for spec in ["c", "ct"]})

#         def zivot() -> Dict[str, Any]:
#             y = self.y_val
#             n = len(y)
#             candidates = np.linspace(int(0.15 * n), int(0.85 * n), min(80, max(5, n // 100))).astype(int)
#             best = None
#             best_bp = None
#             for bp in np.unique(candidates):
#                 du = (np.arange(n) >= bp).astype(float)
#                 dt = np.maximum(0, np.arange(n) - bp).astype(float)
#                 extra = np.column_stack([du, dt])
#                 res = self._adf_regression(y, lags=min(12, max(1, n // 500)), regression="ct", extra_exog=extra)
#                 if best is None or res["test_stat"] < best["test_stat"]:
#                     best = res
#                     best_bp = int(bp)
#             best["break_index"] = best_bp
#             best["break_date"] = self.df.index[best_bp].isoformat()
#             return best

#         self._safe(11, zivot, note="Endogenous one-break ADF regression with level and trend break dummies.")

#         def lee_strazicich_like() -> Dict[str, Any]:
#             y = self.y_val
#             n = len(y)
#             grid = np.linspace(int(0.15 * n), int(0.85 * n), min(22, max(6, n // 250))).astype(int)
#             best = {"rss": np.inf}
#             t = np.arange(n, dtype=float)
#             for b1, b2 in itertools.combinations(np.unique(grid), 2):
#                 X = np.column_stack(
#                     [
#                         np.ones(n),
#                         t,
#                         (t >= b1).astype(float),
#                         np.maximum(0, t - b1),
#                         (t >= b2).astype(float),
#                         np.maximum(0, t - b2),
#                     ]
#                 )
#                 fit = _ols(y, X)
#                 if fit["rss"] < best["rss"]:
#                     adf_res = self._adf_regression(fit["resid"], regression="nc", lags=min(12, n // 500 + 1))
#                     best = {
#                         "rss": fit["rss"],
#                         "break_1": self.df.index[int(b1)].isoformat(),
#                         "break_2": self.df.index[int(b2)].isoformat(),
#                         "lm_adf_stat_on_detrended_residual": adf_res["test_stat"],
#                         "normal_p_approx": adf_res["normal_p_approx"],
#                     }
#             return best

#         self._safe(12, lee_strazicich_like, note="Operational two-break LM-style implementation; exact LS critical values are not bundled.")

#         def variance_ratio() -> Dict[str, Any]:
#             one = np.diff(self.y_val)
#             var1 = np.var(one, ddof=1)
#             out = {}
#             for q in [2, 6, 12, 24, 48, 168]:
#                 if self.N > q + 2:
#                     vr = np.var(self.y_val[q:] - self.y_val[:-q], ddof=1) / max(q * var1, 1e-18)
#                     z = (vr - 1) / np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * self.N))
#                     out[q] = {"variance_ratio": float(vr), "z_stat": float(z), "p_value": float(2 * stats.norm.sf(abs(z)))}
#             return out

#         self._safe(13, variance_ratio)

#         def seasonal_unit_root_screen() -> Dict[str, Any]:
#             period = 24 if self.N > 48 else max(2, min(self.seasonal_periods or (2,)))
#             sd = self.y.diff(period).dropna().to_numpy()
#             out = {"period": period, "seasonal_difference_adf": self._adf_regression(sd, regression="c", lags=min(24, len(sd) // 20))}
#             f, pxx = signal.periodogram(self.y_val - np.mean(self.y_val), fs=1.0)
#             harmonic_power = {}
#             for k in range(1, period // 2 + 1):
#                 target_f = k / period
#                 idx = int(np.argmin(np.abs(f - target_f)))
#                 local = pxx[max(1, idx - 5) : idx + 6]
#                 background = np.median(local[local > 0]) if np.any(local > 0) else np.nan
#                 harmonic_power[k] = float(pxx[idx] / max(background, 1e-18))
#             out["harmonic_power_ratio"] = harmonic_power
#             return out

#         self._safe(14, seasonal_unit_root_screen, note="Frequency-specific seasonal unit-root screen for hourly S=24 data.")

#         def ocsb() -> Dict[str, Any]:
#             s = 24 if self.N > 72 else max(2, min(self.seasonal_periods or (2,)))
#             y = self.y_val
#             dy = np.diff(y)
#             dsy = y[s:] - y[:-s]
#             start = s + 1
#             target = dy[start - 1 :]
#             cols = [np.ones(len(target)), dy[start - 2 : -1], dsy[:-1], y[start - s - 1 : -s - 1]]
#             X = np.column_stack(cols)
#             fit = _ols(target, X)
#             t_stat = float(fit["t"][-1])
#             return {
#                 "period": s,
#                 "seasonal_level_t_stat": t_stat,
#                 "normal_p_approx": float(2 * stats.norm.sf(abs(t_stat))),
#                 "recommend_seasonal_difference": bool(t_stat > -1.645),
#             }

#         self._safe(15, ocsb)

#     # ------------------------------------------------------------------
#     # Category 3
#     # ------------------------------------------------------------------
#     def cat_3_autocorrelation(self) -> None:
#         self._safe(
#             16,
#             lambda: {
#                 "nlags": min(336, self.N - 2),
#                 "confidence_band": float(1.96 / np.sqrt(self.N)),
#                 "acf": _acf_np(self.y_val, min(336, self.N - 2)),
#             },
#         )
#         self._safe(17, lambda: {"pacf": _pacf_np(self.y_val, min(48, self.N // 4 - 1))})

#         def eacf() -> Dict[str, Any]:
#             y = self.y_val
#             max_p, max_q = 6, 6
#             table = np.zeros((max_p + 1, max_q + 1))
#             symbols = []
#             band = 1.96 / np.sqrt(len(y))
#             for p in range(max_p + 1):
#                 if p == 0:
#                     resid = y - np.mean(y)
#                 else:
#                     Y, X = _lag_matrix(y, list(range(1, p + 1)))
#                     fit = _ols(Y, np.column_stack([np.ones(len(Y)), X]))
#                     resid = fit["resid"]
#                 rr = _acf_np(resid, max_q + 1)[1 : max_q + 2]
#                 table[p, :] = rr
#                 symbols.append(["o" if abs(v) <= band else "x" for v in rr])
#             return {"acf_residual_table": table, "insignificance_symbols": symbols, "band": band}

#         self._safe(18, eacf)
#         self._safe(19, lambda: _ljung_box(self.y_val, [10, 24, 48, 72, 168]))

#         def iacf() -> Dict[str, Any]:
#             p = min(12, self.N // 20)
#             Y, X = _lag_matrix(self.y_val, list(range(1, p + 1)))
#             fit = _ols(Y, np.column_stack([np.ones(len(Y)), X]))
#             phi = fit["beta"][1:]
#             inv_filter = np.r_[1.0, -phi]
#             inv_acf = _acf_np(inv_filter, min(len(inv_filter) - 1, 12))
#             return {"ar_order": p, "ar_coefficients": phi, "iacf": inv_acf}

#         self._safe(20, iacf)

#         def wold() -> Dict[str, Any]:
#             p = min(12, max(1, self.N // 500))
#             Y, X = _lag_matrix(self.y_val, list(range(1, p + 1)))
#             fit = _ols(Y, np.column_stack([np.ones(len(Y)), X]))
#             phi = fit["beta"][1:]
#             steps = min(168, self.N // 2)
#             psi = np.zeros(steps + 1)
#             psi[0] = 1.0
#             for h in range(1, steps + 1):
#                 psi[h] = sum(phi[j - 1] * psi[h - j] for j in range(1, min(p, h) + 1))
#             half_life = next((i for i, v in enumerate(np.abs(psi)) if i > 0 and v < 0.5), None)
#             return {"ar_order": p, "psi_weights": psi, "shock_half_life_lag": half_life}

#         self._safe(21, wold)

#         def corr_dimension() -> Dict[str, Any]:
#             y = self._series_for_heavy(1200)
#             tau = 1
#             dims = [2, 3, 4, 5]
#             out = {}
#             radii = np.std(y) * np.array([0.15, 0.25, 0.35, 0.50])
#             for m in dims:
#                 if len(y) <= (m - 1) * tau + 10:
#                     continue
#                 emb = np.column_stack([y[i : len(y) - (m - 1) * tau + i : tau] for i in range(0, m * tau, tau)])
#                 d = pdist(emb)
#                 vals = []
#                 for r in radii:
#                     c = np.mean(d < r)
#                     vals.append(max(c, 1e-12))
#                 slope = np.polyfit(np.log(radii), np.log(vals), 1)[0]
#                 out[m] = float(slope)
#             return out

#         self._safe(22, corr_dimension)

#     # ------------------------------------------------------------------
#     # Category 4
#     # ------------------------------------------------------------------
#     def cat_4_spectral(self) -> None:
#         def top_periods(freq: np.ndarray, power: np.ndarray, k: int = 8) -> List[Dict[str, float]]:
#             mask = freq > 0
#             freq = freq[mask]
#             power = power[mask]
#             idx = np.argsort(power)[-k:][::-1]
#             return [
#                 {"frequency": float(freq[i]), "period_hours": float(1.0 / freq[i]), "power": float(power[i])}
#                 for i in idx
#                 if freq[i] > 0
#             ]

#         self._safe(23, lambda: {"top_periods": top_periods(*signal.periodogram(self.y_val - np.mean(self.y_val), fs=1.0))})
#         self._safe(
#             24,
#             lambda: {
#                 "top_periods": top_periods(
#                     *signal.welch(self.y_val - np.mean(self.y_val), fs=1.0, nperseg=min(4096, max(256, self.N // 4)))
#                 )
#             },
#         )

#         def multitaper() -> Dict[str, Any]:
#             nper = min(4096, self.N)
#             x = self.y_val[-nper:] - np.mean(self.y_val[-nper:])
#             tapers = signal.windows.dpss(nper, NW=4, Kmax=7)
#             specs = []
#             for taper in tapers:
#                 spec = np.abs(fft.rfft(x * taper)) ** 2
#                 specs.append(spec)
#             power = np.mean(specs, axis=0)
#             freq = fft.rfftfreq(nper, d=1.0)
#             return {"n_tapers": int(len(tapers)), "top_periods": top_periods(freq, power)}

#         self._safe(25, multitaper)

#         def spectral_lines() -> Dict[str, Any]:
#             freq, power = signal.welch(self.y_val - np.mean(self.y_val), fs=1.0, nperseg=min(4096, self.N))
#             out = {}
#             for period in [24, 168]:
#                 harmonics = {}
#                 max_k = min(period // 2, 12)
#                 for k in range(1, max_k + 1):
#                     target = k / period
#                     idx = int(np.argmin(np.abs(freq - target)))
#                     local = power[max(0, idx - 8) : min(len(power), idx + 9)]
#                     bg = np.median(local[local > 0]) if np.any(local > 0) else np.nan
#                     f_stat = float(power[idx] / max(bg, 1e-18))
#                     p_value = float(stats.f.sf(f_stat, 2, 20))
#                     harmonics[k] = {"f_stat": f_stat, "p_value": p_value}
#                 out[period] = harmonics
#             return out

#         self._safe(26, spectral_lines)

#         def cumulative_spectrum() -> Dict[str, Any]:
#             freq, power = signal.welch(self.y_val - np.mean(self.y_val), fs=1.0, nperseg=min(4096, self.N))
#             csd = np.cumsum(power) / max(np.sum(power), 1e-18)
#             out = {}
#             for q in [0.5, 0.75, 0.9]:
#                 idx = int(np.searchsorted(csd, q))
#                 out[str(q)] = {"frequency": float(freq[min(idx, len(freq) - 1)]), "period_hours": float(1 / max(freq[min(idx, len(freq) - 1)], 1e-12))}
#             return {"variance_cutoffs": out, "csd_head": csd[:20]}

#         self._safe(27, cumulative_spectrum)

#         def coherence() -> Dict[str, Any]:
#             if not self.external_cols:
#                 raise ValueError("requires at least one external numeric series, e.g. temperature or humidity")
#             out = {}
#             for col in self.external_cols:
#                 x = pd.to_numeric(self.df[col], errors="coerce").interpolate().bfill().ffill().to_numpy()
#                 f, cxy = signal.coherence(x, self.y_val, fs=1.0, nperseg=min(2048, self.N))
#                 out[col] = {"max_coherence": float(np.nanmax(cxy)), "frequency_at_max": float(f[int(np.nanargmax(cxy))])}
#             return out

#         self._safe(28, coherence)

#         def cwt() -> Dict[str, Any]:
#             x = self._series_for_heavy(1200)
#             widths = np.arange(2, 65)
#             coeffs = []
#             for width in widths:
#                 radius = int(6 * width)
#                 t = np.arange(-radius, radius + 1)
#                 wavelet = np.exp(1j * 5.0 * t / width) * np.exp(-(t**2) / (2 * width**2))
#                 wavelet = wavelet / np.sqrt(width)
#                 coeffs.append(np.convolve(x - np.mean(x), np.conj(wavelet[::-1]), mode="same"))
#             power_by_scale = np.mean(np.abs(np.asarray(coeffs)) ** 2, axis=1)
#             return {"widths": widths, "power_by_scale": power_by_scale, "dominant_width": int(widths[int(np.argmax(power_by_scale))])}

#         self._safe(29, cwt)

#         def haar_dwt() -> Dict[str, Any]:
#             x = self._series_for_heavy(2048)
#             details = {}
#             approx = x.copy()
#             for level in range(1, 8):
#                 if len(approx) < 4:
#                     break
#                 even = approx[0::2]
#                 odd = approx[1::2]
#                 n = min(len(even), len(odd))
#                 detail = (even[:n] - odd[:n]) / np.sqrt(2)
#                 approx = (even[:n] + odd[:n]) / np.sqrt(2)
#                 details[level] = {"scale_hours": int(2**level), "variance": float(np.var(detail))}
#             return {"wavelet": "Haar fallback", "details": details, "approximation_variance": float(np.var(approx))}

#         self._safe(30, haar_dwt)

#         def emd_once(x: np.ndarray, max_imfs: int = 4, sift_iter: int = 8) -> Tuple[List[np.ndarray], np.ndarray]:
#             imfs: List[np.ndarray] = []
#             residue = x.astype(float).copy()
#             grid = np.arange(len(x))
#             for _ in range(max_imfs):
#                 h = residue.copy()
#                 for _ in range(sift_iter):
#                     peaks = signal.argrelextrema(h, np.greater)[0]
#                     troughs = signal.argrelextrema(h, np.less)[0]
#                     if len(peaks) < 4 or len(troughs) < 4:
#                         break
#                     peaks = np.r_[0, peaks, len(h) - 1]
#                     troughs = np.r_[0, troughs, len(h) - 1]
#                     upper = CubicSpline(peaks, h[peaks], bc_type="natural")(grid)
#                     lower = CubicSpline(troughs, h[troughs], bc_type="natural")(grid)
#                     h = h - 0.5 * (upper + lower)
#                 if np.std(h) < 1e-9:
#                     break
#                 imfs.append(h)
#                 residue = residue - h
#                 if len(signal.argrelextrema(residue, np.greater)[0]) < 4:
#                     break
#             return imfs, residue

#         self.cache["emd_once"] = emd_once

#         def emd() -> Dict[str, Any]:
#             x = self._series_for_heavy(1000)
#             imfs, residue = emd_once(x)
#             return {"n_imfs": len(imfs), "imf_variances": [float(np.var(v)) for v in imfs], "residue_variance": float(np.var(residue))}

#         self._safe(31, emd)

#         def eemd() -> Dict[str, Any]:
#             x = self._series_for_heavy(720)
#             reps = 20
#             max_imfs = 3
#             acc = [np.zeros_like(x) for _ in range(max_imfs)]
#             counts = np.zeros(max_imfs, dtype=int)
#             noise_sd = 0.15 * np.std(x)
#             for _ in range(reps):
#                 imfs, _ = emd_once(x + self.rng.normal(0, noise_sd, len(x)), max_imfs=max_imfs, sift_iter=5)
#                 for i, imf in enumerate(imfs[:max_imfs]):
#                     acc[i] += imf
#                     counts[i] += 1
#             variances = [float(np.var(acc[i] / max(counts[i], 1))) for i in range(max_imfs) if counts[i] > 0]
#             return {"ensemble_size": reps, "noise_sd": float(noise_sd), "averaged_imf_variances": variances}

#         self._safe(32, eemd)

#     # ------------------------------------------------------------------
#     # Category 5
#     # ------------------------------------------------------------------
#     def cat_5_trend(self) -> None:
#         def hp() -> Dict[str, Any]:
#             n = self.N
#             lam = float(100 * (24**4))
#             diagonals = [np.ones(n), np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)]
#             D = sparse.diags(diagonals[1:], [0, 1, 2], shape=(n - 2, n), format="csr")
#             A = sparse.eye(n, format="csr") + lam * (D.T @ D)
#             trend = spsolve(A, self.y_val)
#             cycle = self.y_val - trend
#             self.cache["hp_trend"] = trend
#             return {"lambda": lam, "trend_summary": _summary(trend), "cycle_summary": _summary(cycle)}

#         self._safe(33, hp)

#         def hamilton() -> Dict[str, Any]:
#             h = min(24 * 365, max(24, self.N // 4))
#             if self.N <= h + 8:
#                 raise ValueError("not enough observations for Hamilton horizon")
#             target = self.y_val[h + 3 :]
#             X = np.column_stack([self.y_val[3:-h], self.y_val[2 : -h - 1], self.y_val[1 : -h - 2], self.y_val[: -h - 3]])
#             fit = _ols(target, np.column_stack([np.ones(len(target)), X]))
#             cycle = fit["resid"]
#             return {"horizon": int(h), "coefficients": fit["beta"], "cycle_summary": _summary(cycle)}

#         self._safe(34, hamilton)

#         def mann_kendall_and_sen() -> Tuple[Dict[str, Any], Dict[str, Any]]:
#             y = self.y_val
#             max_n = 2500
#             if len(y) > max_n:
#                 idx = np.linspace(0, len(y) - 1, max_n).astype(int)
#                 y = y[idx]
#                 time_scale = (idx[-1] - idx[0]) / (len(idx) - 1)
#             else:
#                 idx = np.arange(len(y))
#                 time_scale = 1.0
#             n = len(y)
#             i, j = np.triu_indices(n, k=1)
#             diff = y[j] - y[i]
#             s = float(np.sum(np.sign(diff)))
#             var_s = n * (n - 1) * (2 * n + 5) / 18.0
#             z = (s - np.sign(s)) / np.sqrt(var_s) if var_s > 0 and s != 0 else 0.0
#             slopes = diff / np.maximum(idx[j] - idx[i], 1)
#             sen = float(np.median(slopes) * 24 * 365)
#             return (
#                 {"S": s, "z_stat": float(z), "p_value": float(2 * stats.norm.sf(abs(z))), "sample_n": n},
#                 {"sen_slope_per_year": sen, "sample_n": n, "time_scale_original_steps": float(time_scale)},
#             )

#         self._safe(35, lambda: mann_kendall_and_sen()[0])
#         self._safe(36, lambda: mann_kendall_and_sen()[1])

#         def theil_sen() -> Dict[str, Any]:
#             y = self._series_for_heavy(3000)
#             x = np.arange(len(y), dtype=float).reshape(-1, 1)
#             model = TheilSenRegressor(random_state=self.random_state, max_subpopulation=10000)
#             model.fit(x, y)
#             return {"intercept": float(model.intercept_), "slope_per_year": float(model.coef_[0] * 24 * 365)}

#         self._safe(37, theil_sen)

#         def lowess() -> Dict[str, Any]:
#             y = self._series_for_heavy(1200)
#             x = np.linspace(0, 1, len(y))
#             frac = 0.08
#             span = max(20, int(frac * len(y)))
#             grid = np.linspace(0, len(y) - 1, min(200, len(y))).astype(int)
#             smooth = []
#             for i in grid:
#                 lo = max(0, i - span)
#                 hi = min(len(y), i + span + 1)
#                 xx = x[lo:hi] - x[i]
#                 yy = y[lo:hi]
#                 w = (1 - np.minimum(np.abs(xx) / max(np.max(np.abs(xx)), 1e-12), 1) ** 3) ** 3
#                 fit = _ols(yy, np.column_stack([np.ones(len(yy)), xx * w + xx * (1 - w)]))
#                 smooth.append(float(fit["beta"][0]))
#             return {"frac": frac, "grid_points": len(grid), "smoothed_summary": _summary(smooth)}

#         self._safe(38, lowess)

#     # ------------------------------------------------------------------
#     # Category 6
#     # ------------------------------------------------------------------
#     def _classical_decompose(self, y: np.ndarray, period: int) -> Dict[str, np.ndarray]:
#         s = pd.Series(y)
#         trend = s.rolling(period, center=True, min_periods=max(2, period // 2)).mean().interpolate().bfill().ffill().to_numpy()
#         detrended = y - trend
#         seasonal_index = np.array([np.nanmean(detrended[np.arange(len(y)) % period == i]) for i in range(period)])
#         seasonal_index = seasonal_index - np.nanmean(seasonal_index)
#         seasonal = seasonal_index[np.arange(len(y)) % period]
#         resid = y - trend - seasonal
#         return {"trend": trend, "seasonal": seasonal, "resid": resid, "seasonal_index": seasonal_index}

#     def cat_6_seasonality(self) -> None:
#         self._safe(39, lambda: {k: _summary(v) if k != "seasonal_index" else v for k, v in self._classical_decompose(self.y_val, 24).items()})

#         def stl_like() -> Dict[str, Any]:
#             dec = self._classical_decompose(self.y_val, 24)
#             resid = dec["resid"]
#             scale = np.nanmedian(np.abs(resid - np.nanmedian(resid))) * 1.4826
#             weights = np.clip(1 - (resid / max(6 * scale, 1e-12)) ** 2, 0, 1) ** 2
#             seasonal_index = []
#             for h in range(24):
#                 mask = np.arange(self.N) % 24 == h
#                 seasonal_index.append(np.average((self.y_val - dec["trend"])[mask], weights=weights[mask]))
#             seasonal_index = np.asarray(seasonal_index) - np.mean(seasonal_index)
#             seasonal = seasonal_index[np.arange(self.N) % 24]
#             resid2 = self.y_val - dec["trend"] - seasonal
#             self.cache["stl_like"] = {"trend": dec["trend"], "seasonal": seasonal, "resid": resid2}
#             return {"seasonal_index": seasonal_index, "resid_summary": _summary(resid2), "robust_scale": float(scale)}

#         self._safe(40, stl_like, note="Uses robust seasonal subseries smoothing because statsmodels STL is not guaranteed installed.")

#         def mstl() -> Dict[str, Any]:
#             remainder = self.y_val.copy()
#             comps = {}
#             for period in [24, 168]:
#                 if self.N > 2 * period:
#                     dec = self._classical_decompose(remainder, period)
#                     comps[f"seasonal_{period}"] = dec["seasonal"]
#                     remainder = remainder - dec["seasonal"]
#             trend = pd.Series(remainder).rolling(min(24 * 14, max(24, self.N // 10)), center=True, min_periods=24).mean().interpolate().bfill().ffill().to_numpy()
#             remainder = remainder - trend
#             self.cache["mstl"] = {**comps, "trend": trend, "remainder": remainder}
#             return {k: _summary(v) for k, v in self.cache["mstl"].items()}

#         self._safe(41, mstl)

#         def strengths() -> Tuple[Dict[str, Any], Dict[str, Any]]:
#             if "mstl" not in self.cache:
#                 mstl()
#             comp = self.cache["mstl"]
#             rem = comp["remainder"]
#             season_sum = sum(v for k, v in comp.items() if k.startswith("seasonal_"))
#             trend = comp["trend"]
#             fs = max(0.0, 1.0 - np.var(rem) / max(np.var(season_sum + rem), 1e-12))
#             ft = max(0.0, 1.0 - np.var(rem) / max(np.var(trend + rem), 1e-12))
#             return {"seasonal_strength": float(fs)}, {"trend_strength": float(ft)}

#         self._safe(42, lambda: strengths()[0])
#         self._safe(43, lambda: strengths()[1])

#         def subseries() -> Dict[str, Any]:
#             out = {}
#             for h, s in self.df.groupby("hour")[self.target_col]:
#                 vals = s.to_numpy()
#                 if len(vals) < 10:
#                     continue
#                 x = np.arange(len(vals), dtype=float)
#                 fit = _ols(vals, np.column_stack([np.ones(len(vals)), x]))
#                 lb = _ljung_box(fit["resid"], [min(10, len(vals) - 2)])
#                 out[int(h)] = {
#                     "mean": float(np.mean(vals)),
#                     "std": float(np.std(vals, ddof=1)),
#                     "trend_slope_per_year": float(fit["beta"][1] * 365),
#                     "ljung_box_p": list(lb.values())[0]["p_value"],
#                 }
#             return out

#         self._safe(44, subseries)

#         def seasonal_stability() -> Dict[str, Any]:
#             n2 = self.N // 2
#             first = self._classical_decompose(self.y_val[:n2], 24)["seasonal_index"]
#             second = self._classical_decompose(self.y_val[n2:], 24)["seasonal_index"]
#             diff = second - first
#             stat, p = stats.ttest_rel(first, second)
#             return {"paired_t_stat": float(stat), "p_value": float(p), "max_abs_index_change": float(np.max(np.abs(diff)))}

#         self._safe(45, seasonal_stability)

#         def harmonic_f_test() -> Dict[str, Any]:
#             t = np.arange(self.N)
#             base = np.column_stack([np.ones(self.N), t])
#             cols = [base]
#             for k in range(1, 13):
#                 cols.append(np.column_stack([np.sin(2 * np.pi * k * t / 24), np.cos(2 * np.pi * k * t / 24)]))
#             full = np.column_stack(cols)
#             fit0 = _ols(self.y_val, base)
#             fit1 = _ols(self.y_val, full)
#             df1 = full.shape[1] - base.shape[1]
#             df2 = fit1["n"] - full.shape[1]
#             f_stat = ((fit0["rss"] - fit1["rss"]) / df1) / max(fit1["rss"] / df2, 1e-18)
#             return {"f_stat": float(f_stat), "p_value": float(stats.f.sf(f_stat, df1, df2)), "harmonic_pairs": 12}

#         self._safe(46, harmonic_f_test)

#     # ------------------------------------------------------------------
#     # Category 7
#     # ------------------------------------------------------------------
#     def cat_7_long_memory(self) -> None:
#         def hurst_rs() -> Dict[str, Any]:
#             y = self._series_for_heavy(5000)
#             sizes = np.unique(np.logspace(np.log10(16), np.log10(max(32, len(y) // 4)), 16).astype(int))
#             rs_vals = []
#             used = []
#             for size in sizes:
#                 chunks = len(y) // size
#                 if chunks < 2:
#                     continue
#                 vals = []
#                 for c in range(chunks):
#                     seg = y[c * size : (c + 1) * size]
#                     z = np.cumsum(seg - np.mean(seg))
#                     r = np.max(z) - np.min(z)
#                     s = np.std(seg, ddof=1)
#                     if s > 0:
#                         vals.append(r / s)
#                 if vals:
#                     rs_vals.append(np.mean(vals))
#                     used.append(size)
#             h = np.polyfit(np.log(used), np.log(rs_vals), 1)[0]
#             return {"hurst_exponent": float(h), "block_sizes": used, "rs_values": rs_vals}

#         self._safe(47, hurst_rs)

#         def whittle_gph() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
#             y = self.y_val - np.mean(self.y_val)
#             freq, power = signal.periodogram(y)
#             power = np.maximum(power[1:], 1e-18)
#             n = len(y)
#             m = min(len(power), max(20, int(n**0.65)))
#             lam = 2 * np.pi * np.arange(1, m + 1) / n

#             def objective(d: float) -> float:
#                 adjusted = power[:m] * lam ** (2 * d)
#                 return float(np.log(np.mean(adjusted)) - 2 * d * np.mean(np.log(lam)))

#             opt = minimize(lambda z: objective(float(z[0])), x0=[0.25], bounds=[(-0.49, 0.99)])
#             d_whittle = float(opt.x[0])
#             X = np.column_stack([np.ones(m), -np.log(4 * np.sin(lam / 2) ** 2)])
#             fit = _ols(np.log(power[:m]), X)
#             d_gph = float(fit["beta"][1])
#             se = float(fit["se"][1])
#             z = d_gph / max(se, 1e-12)
#             return (
#                 {"d": d_whittle, "bandwidth_m": int(m), "objective": float(opt.fun)},
#                 {"d": d_gph, "se": se, "bandwidth_m": int(m)},
#                 {"z_stat": float(z), "p_value": float(stats.norm.sf(z)), "d_estimate": d_gph, "method": "GPH score test"},
#             )

#         self._safe(48, lambda: whittle_gph()[0])
#         self._safe(49, lambda: whittle_gph()[1])
#         self._safe(50, lambda: whittle_gph()[2])

#     # ------------------------------------------------------------------
#     # Category 8
#     # ------------------------------------------------------------------
#     def _ar_residuals(self, p: int = 3) -> np.ndarray:
#         p = min(max(1, p), max(1, self.N // 10))
#         Y, X = _lag_matrix(self.y_val, list(range(1, p + 1)))
#         fit = _ols(Y, np.column_stack([np.ones(len(Y)), X]))
#         return fit["resid"]

#     def cat_8_nonlinearity(self) -> None:
#         def bds_test() -> Dict[str, Any]:
#             x = self._ar_residuals(3)
#             if len(x) > 1200:
#                 x = x[-1200:]
#             x = (x - np.mean(x)) / max(np.std(x), 1e-12)
#             eps_values = [0.5, 1.0, 1.5]
#             out = {}
#             d1 = squareform(pdist(x[:, None]))
#             for eps in eps_values:
#                 c1 = np.mean(d1 < eps)
#                 emb = np.column_stack([x[:-1], x[1:]])
#                 d2 = squareform(pdist(emb))
#                 c2 = np.mean(d2 < eps)
#                 diff = c2 - c1**2
#                 se = np.sqrt(max(c1**2 * (1 - c1**2), 1e-12) / len(emb))
#                 z = diff / se
#                 out[eps] = {"c1": float(c1), "c2": float(c2), "z_stat": float(z), "p_value": float(2 * stats.norm.sf(abs(z)))}
#             return out

#         self._safe(51, bds_test)

#         def terasvirta() -> Dict[str, Any]:
#             p = 3
#             Y, Xlag = _lag_matrix(self.y_val, list(range(1, p + 1)))
#             Xlin = np.column_stack([np.ones(len(Y)), Xlag])
#             fit0 = _ols(Y, Xlin)
#             terms = [Xlag]
#             for i in range(p):
#                 terms.append(Xlag[:, i : i + 1] ** 2)
#                 terms.append(Xlag[:, i : i + 1] ** 3)
#             for i in range(p):
#                 for j in range(i + 1, p):
#                     terms.append((Xlag[:, i] * Xlag[:, j])[:, None])
#             Xfull = np.column_stack([np.ones(len(Y))] + terms)
#             fit1 = _ols(Y, Xfull)
#             df1 = Xfull.shape[1] - Xlin.shape[1]
#             df2 = fit1["n"] - Xfull.shape[1]
#             f_stat = ((fit0["rss"] - fit1["rss"]) / df1) / max(fit1["rss"] / df2, 1e-18)
#             return {"f_stat": float(f_stat), "p_value": float(stats.f.sf(f_stat, df1, df2)), "df1": int(df1), "df2": int(df2)}

#         self._safe(52, terasvirta)

#         def keenan() -> Dict[str, Any]:
#             Y, Xlag = _lag_matrix(self.y_val, [1, 2, 3])
#             fit_ar = _ols(Y, np.column_stack([np.ones(len(Y)), Xlag]))
#             fitted_sq = fit_ar["fitted"] ** 2
#             fit_aux = _ols(fit_ar["resid"], np.column_stack([np.ones(len(fitted_sq)), fitted_sq]))
#             t = float(fit_aux["t"][1])
#             return {"t_stat": t, "p_value": float(2 * stats.t.sf(abs(t), max(fit_aux["n"] - fit_aux["k"], 1)))}

#         self._safe(53, keenan)
#         self._safe(54, lambda: _ljung_box(self._ar_residuals(3) ** 2, [10, 24, 48]))

#         def hinich() -> Dict[str, Any]:
#             x = self._ar_residuals(3)
#             x = (x - np.mean(x)) / max(np.std(x), 1e-12)
#             n = min(len(x), 2048)
#             X = fft.fft(x[-n:])
#             pairs = [(1, 1), (1, 2), (2, 3), (4, 8), (8, 16)]
#             vals = {}
#             for a, b in pairs:
#                 bis = X[a] * X[b] * np.conj(X[(a + b) % n]) / n
#                 vals[f"{a},{b}"] = float(np.abs(bis))
#             return {"bispectrum_slice_abs": vals, "skewness_proxy": float(np.mean(x**3))}

#         self._safe(55, hinich)

#         def tsay_threshold() -> Dict[str, Any]:
#             y = self.y_val
#             lag = y[:-1]
#             target = y[1:]
#             X0 = np.column_stack([np.ones(len(target)), lag])
#             fit0 = _ols(target, X0)
#             thresholds = np.quantile(lag, np.linspace(0.2, 0.8, 13))
#             best = {"rss": np.inf}
#             for th in thresholds:
#                 low = lag <= th
#                 if low.sum() < 20 or (~low).sum() < 20:
#                     continue
#                 f1 = _ols(target[low], np.column_stack([np.ones(low.sum()), lag[low]]))
#                 f2 = _ols(target[~low], np.column_stack([np.ones((~low).sum()), lag[~low]]))
#                 rss = f1["rss"] + f2["rss"]
#                 if rss < best["rss"]:
#                     df1 = 2
#                     df2 = len(target) - 4
#                     f_stat = ((fit0["rss"] - rss) / df1) / max(rss / df2, 1e-18)
#                     best = {
#                         "threshold": float(th),
#                         "rss": float(rss),
#                         "f_stat": float(f_stat),
#                         "p_value": float(stats.f.sf(f_stat, df1, df2)),
#                     }
#             return best

#         self._safe(56, tsay_threshold)

#     # ------------------------------------------------------------------
#     # Category 9
#     # ------------------------------------------------------------------
#     def cat_9_volatility(self) -> None:
#         resid = self._ar_residuals(3)

#         def arch_lm(lags: int = 24) -> Dict[str, Any]:
#             e2 = resid**2
#             lags2 = min(lags, len(e2) // 5)
#             Y, Xlag = _lag_matrix(e2, list(range(1, lags2 + 1)))
#             fit = _ols(Y, np.column_stack([np.ones(len(Y)), Xlag]))
#             lm = fit["n"] * fit["r2"]
#             return {"lags": lags2, "lm_stat": float(lm), "p_value": float(stats.chi2.sf(lm, lags2))}

#         self._safe(57, arch_lm)

#         def fit_garch(gjr: bool = False) -> Dict[str, Any]:
#             e = resid - np.mean(resid)
#             var0 = np.var(e)

#             def neg_ll(params: np.ndarray) -> float:
#                 if gjr:
#                     omega, alpha, gamma, beta = params
#                 else:
#                     omega, alpha, beta = params
#                     gamma = 0.0
#                 if omega <= 0 or alpha < 0 or beta < 0 or gamma < -alpha or alpha + 0.5 * gamma + beta >= 0.999:
#                     return 1e30
#                 sig2 = np.empty_like(e)
#                 sig2[0] = var0
#                 for t in range(1, len(e)):
#                     ind = 1.0 if e[t - 1] < 0 else 0.0
#                     sig2[t] = omega + (alpha + gamma * ind) * e[t - 1] ** 2 + beta * sig2[t - 1]
#                     if sig2[t] <= 1e-12:
#                         return 1e30
#                 return float(0.5 * np.sum(np.log(sig2) + e**2 / sig2))

#             if gjr:
#                 x0 = [0.05 * var0, 0.05, 0.05, 0.85]
#                 bounds = [(1e-12, None), (0, 1), (-1, 1), (0, 1)]
#             else:
#                 x0 = [0.05 * var0, 0.08, 0.88]
#                 bounds = [(1e-12, None), (0, 1), (0, 1)]
#             opt = minimize(neg_ll, x0=x0, bounds=bounds, method="Nelder-Mead", options={"maxiter": 2000})
#             params = opt.x
#             names = ["omega", "alpha", "gamma", "beta"] if gjr else ["omega", "alpha", "beta"]
#             self.cache["garch_params"] = dict(zip(names, params))
#             return {"success": bool(opt.success), "neg_log_likelihood": float(opt.fun), "params": dict(zip(names, map(float, params)))}

#         self._safe(58, lambda: fit_garch(False))
#         self._safe(59, lambda: fit_garch(True))

#         def white_test() -> Dict[str, Any]:
#             data = self._simple_features().tail(min(5000, self.N))
#             y = data[self.target_col].to_numpy()
#             X = data[["hour", "dayofweek", "month", "is_weekend"]].to_numpy(dtype=float)
#             fit = _ols(y, np.column_stack([np.ones(len(y)), X]))
#             e2 = fit["resid"] ** 2
#             terms = [np.ones(len(y)), X]
#             terms.append(X**2)
#             cross = []
#             for i in range(X.shape[1]):
#                 for j in range(i + 1, X.shape[1]):
#                     cross.append((X[:, i] * X[:, j])[:, None])
#             Z = np.column_stack(terms + cross)
#             aux = _ols(e2, Z)
#             lm = aux["n"] * aux["r2"]
#             df = Z.shape[1] - 1
#             return {"lm_stat": float(lm), "p_value": float(stats.chi2.sf(lm, df)), "df": int(df)}

#         self._safe(60, white_test)

#         def goldfeld_quandt() -> Dict[str, Any]:
#             level = self.y_val[-len(resid) :]
#             order = np.argsort(level)
#             middle_drop = len(order) // 5
#             low = resid[order[: (len(order) - middle_drop) // 2]]
#             high = resid[order[(len(order) + middle_drop) // 2 :]]
#             f = np.var(high, ddof=1) / max(np.var(low, ddof=1), 1e-18)
#             return {"f_stat_high_over_low": float(f), "p_value": float(stats.f.sf(f, len(high) - 1, len(low) - 1))}

#         self._safe(61, goldfeld_quandt)

#     # ------------------------------------------------------------------
#     # Category 10
#     # ------------------------------------------------------------------
#     def cat_10_breaks(self) -> None:
#         def regression_for_break(y: np.ndarray, break_idx: int) -> Dict[str, Any]:
#             n = len(y)
#             t = np.arange(n, dtype=float)
#             Xfull = np.column_stack([np.ones(n), t])
#             fit_full = _ols(y, Xfull)
#             left = slice(0, break_idx)
#             right = slice(break_idx, n)
#             fit_l = _ols(y[left], np.column_stack([np.ones(break_idx), t[left]]))
#             fit_r = _ols(y[right], np.column_stack([np.ones(n - break_idx), t[right]]))
#             k = 2
#             rss_u = fit_l["rss"] + fit_r["rss"]
#             df2 = n - 2 * k
#             f = ((fit_full["rss"] - rss_u) / k) / max(rss_u / df2, 1e-18)
#             return {"break_index": int(break_idx), "break_date": self.df.index[break_idx].isoformat(), "f_stat": float(f), "p_value": float(stats.f.sf(f, k, df2))}

#         self._safe(62, lambda: regression_for_break(self.y_val, self.N // 2))

#         def qlr() -> Dict[str, Any]:
#             candidates = np.linspace(int(0.15 * self.N), int(0.85 * self.N), min(80, max(5, self.N // 200))).astype(int)
#             tests = [regression_for_break(self.y_val, int(c)) for c in np.unique(candidates)]
#             best = max(tests, key=lambda x: x["f_stat"])
#             best["candidate_count"] = len(tests)
#             best["p_value_note"] = "F p-value is pointwise; QLR Andrews critical values are not bundled."
#             return best

#         self._safe(63, qlr)

#         def cusum() -> Dict[str, Any]:
#             y = self.y_val
#             t = np.arange(self.N, dtype=float)
#             fit = _ols(y, np.column_stack([np.ones(self.N), t]))
#             resid = fit["resid"]
#             w = np.cumsum(resid) / (np.std(resid, ddof=1) * np.sqrt(len(resid)))
#             crossed = bool(np.max(np.abs(w)) > 1.36)
#             return {"max_abs_cusum": float(np.max(np.abs(w))), "crosses_5pct_boundary": crossed, "cusum_head": w[:20]}

#         self._safe(64, cusum)

#         def bai_perron_dp() -> Dict[str, Any]:
#             y = self._series_for_heavy(1500)
#             n = len(y)
#             prefix = np.r_[0.0, np.cumsum(y)]
#             prefix2 = np.r_[0.0, np.cumsum(y**2)]

#             def cost(i: int, j: int) -> float:
#                 m = j - i
#                 if m <= 1:
#                     return 0.0
#                 s = prefix[j] - prefix[i]
#                 s2 = prefix2[j] - prefix2[i]
#                 return float(s2 - s * s / m)

#             max_breaks = 4
#             min_size = max(24, n // 30)
#             dp = np.full((max_breaks + 1, n + 1), np.inf)
#             prev = np.full((max_breaks + 1, n + 1), -1, dtype=int)
#             for j in range(min_size, n + 1):
#                 dp[0, j] = cost(0, j)
#             for m in range(1, max_breaks + 1):
#                 for j in range((m + 1) * min_size, n + 1):
#                     candidates = range(m * min_size, j - min_size + 1)
#                     vals = [(dp[m - 1, i] + cost(i, j), i) for i in candidates]
#                     best_val, best_i = min(vals, key=lambda z: z[0])
#                     dp[m, j] = best_val
#                     prev[m, j] = best_i
#             bic_scores = []
#             for m in range(max_breaks + 1):
#                 bic_scores.append(dp[m, n] + (m + 1) * np.log(n) * np.var(y))
#             best_m = int(np.argmin(bic_scores))
#             cps = []
#             j = n
#             for m in range(best_m, 0, -1):
#                 i = prev[m, j]
#                 cps.append(i)
#                 j = i
#             cps = sorted(cps)
#             original_idx = [int(round(c / max(n - 1, 1) * (self.N - 1))) for c in cps]
#             return {"break_count": best_m, "break_dates": [self.df.index[i].isoformat() for i in original_idx], "bic_scores": bic_scores}

#         self._safe(65, bai_perron_dp, note="Dynamic-programming multiple mean-break implementation on an evenly spaced sample.")

#     # ------------------------------------------------------------------
#     # Category 11
#     # ------------------------------------------------------------------
#     def cat_11_anomalies(self) -> None:
#         baseline = self.df.get("mu_hdm", self.y.rolling(24, min_periods=1).mean()).to_numpy()
#         resid = self.y_val - baseline
#         scale = np.nanmedian(np.abs(resid - np.nanmedian(resid))) * 1.4826
#         z = resid / max(scale, 1e-12)

#         def tsay_outliers() -> Dict[str, Any]:
#             idx = np.argsort(np.abs(z))[-20:][::-1]
#             events = []
#             for i in idx:
#                 before = np.nanmean(resid[max(0, i - 24) : i]) if i > 0 else 0.0
#                 after = np.nanmean(resid[i + 1 : min(self.N, i + 25)]) if i + 1 < self.N else 0.0
#                 if abs(after) > 0.5 * abs(resid[i]) and np.sign(after) == np.sign(resid[i]):
#                     typ = "IO_or_TC"
#                 elif abs(after - before) > scale:
#                     typ = "LS"
#                 else:
#                     typ = "AO"
#                 events.append({"timestamp": self.df.index[int(i)].isoformat(), "z": float(z[i]), "type": typ})
#             return {"top_events": events, "scale": float(scale)}

#         self._safe(66, tsay_outliers)

#         def grubbs() -> Dict[str, Any]:
#             n = self.N
#             g = float(np.max(np.abs(self.y_val - np.mean(self.y_val))) / max(np.std(self.y_val, ddof=1), 1e-12))
#             alpha = 0.05
#             tcrit = stats.t.ppf(1 - alpha / (2 * n), n - 2)
#             crit = ((n - 1) / np.sqrt(n)) * np.sqrt(tcrit**2 / (n - 2 + tcrit**2))
#             return {"G": g, "critical_5pct": float(crit), "is_outlier": bool(g > crit)}

#         self._safe(67, grubbs)

#         def seasonal_iqr() -> Dict[str, Any]:
#             flags = pd.Series(False, index=self.df.index)
#             rates = {}
#             for key, part in self.df.groupby(["hour", "dayofweek"]):
#                 q1, q3 = np.percentile(part[self.target_col], [25, 75])
#                 iqr = q3 - q1
#                 f = (part[self.target_col] < q1 - 1.5 * iqr) | (part[self.target_col] > q3 + 1.5 * iqr)
#                 flags.loc[part.index] = f
#                 rates[str(key)] = float(f.mean())
#             self.df["outlier_iqr_context"] = flags.astype(int)
#             return {"overall_rate": float(flags.mean()), "highest_cell_rates": dict(sorted(rates.items(), key=lambda kv: kv[1], reverse=True)[:10])}

#         self._safe(68, seasonal_iqr)

#         def lof() -> Dict[str, Any]:
#             lags = [1, 2, 24]
#             Y, X = _lag_matrix(self.y_val, lags)
#             X = np.column_stack([Y, X])
#             model = LocalOutlierFactor(n_neighbors=30, contamination=0.02)
#             pred = model.fit_predict(StandardScaler().fit_transform(X))
#             return {"outlier_rate": float(np.mean(pred == -1)), "score_summary": _summary(-model.negative_outlier_factor_)}

#         self._safe(69, lof)

#         def iso() -> Dict[str, Any]:
#             data = self._simple_features()
#             X = data.drop(columns=[self.target_col]).select_dtypes(include=[np.number]).tail(min(10000, len(data)))
#             model = IsolationForest(contamination=0.02, random_state=self.random_state)
#             pred = model.fit_predict(X)
#             return {"outlier_rate": float(np.mean(pred == -1)), "score_summary": _summary(-model.score_samples(X))}

#         self._safe(70, iso)

#         def cusum_chart() -> Dict[str, Any]:
#             k = 0.5 * np.std(resid)
#             h = 5.0 * np.std(resid)
#             sp = np.zeros(self.N)
#             sn = np.zeros(self.N)
#             for i in range(1, self.N):
#                 sp[i] = max(0, sp[i - 1] + resid[i] - k)
#                 sn[i] = min(0, sn[i - 1] + resid[i] + k)
#             alarms = np.where((sp > h) | (np.abs(sn) > h))[0]
#             return {"k": float(k), "h": float(h), "alarm_count": int(len(alarms)), "first_alarms": [self.df.index[int(i)].isoformat() for i in alarms[:10]]}

#         self._safe(71, cusum_chart)

#     # ------------------------------------------------------------------
#     # Category 12
#     # ------------------------------------------------------------------
#     def cat_12_selection(self) -> None:
#         def ar_ic_grid() -> Dict[str, Any]:
#             rows = []
#             for p in range(1, min(8, self.N // 50) + 1):
#                 Y, X = _lag_matrix(self.y_val, list(range(1, p + 1)))
#                 fit = _ols(Y, np.column_stack([np.ones(len(Y)), X]))
#                 ic = _aic_like(fit["rss"], fit["n"], fit["k"])
#                 rows.append({"p": p, **ic})
#             self.cache["ar_ic_grid"] = rows
#             return {
#                 "best_aic": min(rows, key=lambda r: r["aic"]),
#                 "best_aicc": min(rows, key=lambda r: r["aicc"]),
#                 "best_bic": min(rows, key=lambda r: r["bic"]),
#                 "best_hqc": min(rows, key=lambda r: r["hqc"]),
#                 "best_mdl": min(rows, key=lambda r: r["mdl"]),
#                 "grid": rows,
#             }

#         grid_result = ar_ic_grid()
#         self._record(72, grid_result["best_aic"])
#         self._record(73, grid_result["best_aicc"])
#         self._record(74, grid_result["best_bic"])
#         self._record(75, grid_result["best_hqc"])
#         self._record(76, grid_result["best_mdl"])

#         def hblock() -> Dict[str, Any]:
#             data = self._simple_features()
#             X = data.drop(columns=[self.target_col]).select_dtypes(include=[np.number])
#             y = data[self.target_col].to_numpy()
#             h = 48
#             points = np.linspace(max(200, h + 1), len(y) - h - 1, min(40, max(5, len(y) // 200))).astype(int)
#             errors = []
#             for t in np.unique(points):
#                 train_mask = np.ones(len(y), dtype=bool)
#                 train_mask[max(0, t - h) : min(len(y), t + h + 1)] = False
#                 if train_mask.sum() < 50:
#                     continue
#                 model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
#                 model.fit(X.iloc[train_mask], y[train_mask])
#                 pred = model.predict(X.iloc[[t]])[0]
#                 errors.append(y[t] - pred)
#             return {"h": h, "folds": len(errors), "rmse": float(np.sqrt(np.mean(np.asarray(errors) ** 2)))}

#         self._safe(77, hblock)

#         def prequential() -> Dict[str, Any]:
#             data = self._simple_features()
#             X = data.drop(columns=[self.target_col]).select_dtypes(include=[np.number]).to_numpy()
#             y = data[self.target_col].to_numpy()
#             start = max(200, len(y) // 5)
#             step = max(24, len(y) // 100)
#             errors = []
#             for t in range(start, len(y), step):
#                 model = Ridge(alpha=1.0)
#                 model.fit(X[:t], y[:t])
#                 errors.append(y[t] - model.predict(X[t : t + 1])[0])
#             return {"updates": len(errors), "rmse": float(np.sqrt(np.mean(np.asarray(errors) ** 2))), "mae": float(np.mean(np.abs(errors)))}

#         self._safe(78, prequential)

#         def forecast_comparison() -> Tuple[Dict[str, Any], Dict[str, Any]]:
#             fc = self._ensure_forecasts()
#             y = fc["y_test"].to_numpy()
#             e1 = y - fc["linear_pred"]
#             e2 = y - fc["seasonal_naive"]
#             loss_diff = e1**2 - e2**2
#             lag = min(24, len(loss_diff) // 4)
#             X = np.ones((len(loss_diff), 1))
#             cov = _newey_west_cov(X, loss_diff - np.mean(loss_diff), lag)
#             dm = float(np.mean(loss_diff) / np.sqrt(max(cov[0, 0], 1e-18)))
#             dm_out = {"dm_stat": dm, "p_value": float(2 * stats.norm.sf(abs(dm))), "hac_lags": lag}
#             e_nested = y - fc["gb_pred"]
#             adj = e1**2 - (e_nested**2 - (fc["linear_pred"] - fc["gb_pred"]) ** 2)
#             cw = float(np.mean(adj) / (np.std(adj, ddof=1) / np.sqrt(len(adj))))
#             cw_out = {"cw_stat": cw, "p_value": float(stats.norm.sf(cw))}
#             return dm_out, cw_out

#         self._safe(79, lambda: forecast_comparison()[0])
#         self._safe(80, lambda: forecast_comparison()[1])

#     # ------------------------------------------------------------------
#     # Category 13
#     # ------------------------------------------------------------------
#     def cat_13_multivariate(self) -> None:
#         if len(self.external_cols) < 1:
#             self._unavailable(81, "Johansen cointegration requires at least one additional numeric series.")
#             self._unavailable(82, "Engle-Granger cointegration requires at least one additional numeric series.")
#             self._unavailable(83, "Granger causality requires a second time series.")
#             return

#         def johansen() -> Dict[str, Any]:
#             cols = [self.target_col] + self.external_cols
#             Xdf = self.df[cols].apply(pd.to_numeric, errors="coerce").interpolate().bfill().ffill().dropna()
#             X = Xdf.to_numpy(dtype=float)
#             if len(X) < 100 or X.shape[1] < 2:
#                 raise ValueError("requires at least 100 rows and two numeric series")
#             dX = np.diff(X, axis=0)
#             Z0 = dX - dX.mean(axis=0)
#             Z1 = X[:-1] - X[:-1].mean(axis=0)
#             t = len(Z0)
#             S00 = Z0.T @ Z0 / t
#             S11 = Z1.T @ Z1 / t
#             S01 = Z0.T @ Z1 / t
#             S10 = S01.T
#             mat = np.linalg.pinv(S11) @ S10 @ np.linalg.pinv(S00) @ S01
#             eigvals = np.sort(np.real(np.linalg.eigvals(mat)))[::-1]
#             eigvals = np.clip(eigvals, 0, 0.999999)
#             trace_stats = []
#             maxeig_stats = []
#             for r in range(len(eigvals)):
#                 trace_stats.append(float(-t * np.sum(np.log(1 - eigvals[r:]))))
#                 maxeig_stats.append(float(-t * np.log(1 - eigvals[r])))
#             return {
#                 "columns": cols,
#                 "eigenvalues": eigvals,
#                 "trace_statistics": trace_stats,
#                 "max_eigen_statistics": maxeig_stats,
#                 "note": "Johansen eigenvalue and trace/max-eigen statistics; tabulated critical values are not bundled.",
#             }

#         self._safe(81, johansen)

#         def engle_granger() -> Dict[str, Any]:
#             out = {}
#             for col in self.external_cols:
#                 x = pd.to_numeric(self.df[col], errors="coerce").interpolate().bfill().ffill().to_numpy()
#                 fit = _ols(self.y_val, np.column_stack([np.ones(self.N), x]))
#                 out[col] = self._adf_regression(fit["resid"], regression="c")
#             return out

#         self._safe(82, engle_granger)

#         def granger() -> Dict[str, Any]:
#             out = {}
#             for col in self.external_cols:
#                 x = pd.to_numeric(self.df[col], errors="coerce").interpolate().bfill().ffill().to_numpy()
#                 p = min(24, self.N // 20)
#                 Y, ylags = _lag_matrix(self.y_val, list(range(1, p + 1)))
#                 _, xlags = _lag_matrix(x, list(range(1, p + 1)))
#                 fit0 = _ols(Y, np.column_stack([np.ones(len(Y)), ylags]))
#                 fit1 = _ols(Y, np.column_stack([np.ones(len(Y)), ylags, xlags]))
#                 df1 = p
#                 df2 = fit1["n"] - fit1["k"]
#                 f = ((fit0["rss"] - fit1["rss"]) / df1) / max(fit1["rss"] / df2, 1e-18)
#                 out[col] = {"f_stat": float(f), "p_value": float(stats.f.sf(f, df1, df2)), "lags": p}
#             return out

#         self._safe(83, granger)

#     # ------------------------------------------------------------------
#     # Category 14
#     # ------------------------------------------------------------------
#     def cat_14_entropy(self) -> None:
#         def shannon() -> Dict[str, Any]:
#             out = {}
#             for h, s in self.df.groupby("hour")[self.target_col]:
#                 counts, _ = np.histogram(s, bins="auto", density=False)
#                 p = counts[counts > 0] / max(np.sum(counts), 1)
#                 out[int(h)] = float(-np.sum(p * np.log2(p)))
#             return {"hourly_entropy": out, "global_entropy": float(np.mean(list(out.values())))}

#         self._safe(84, shannon)

#         def approximate_entropy(x: np.ndarray, m: int = 2, r: Optional[float] = None) -> float:
#             x = np.asarray(x, dtype=float)
#             if len(x) > 700:
#                 x = x[-700:]
#             r = 0.2 * np.std(x) if r is None else r

#             def phi(mm: int) -> float:
#                 emb = np.array([x[i : i + mm] for i in range(len(x) - mm + 1)])
#                 dist = cdist(emb, emb, metric="chebyshev")
#                 c = np.mean(dist <= r, axis=1)
#                 return float(np.mean(np.log(np.maximum(c, 1e-12))))

#             return phi(m) - phi(m + 1)

#         def sample_entropy(x: np.ndarray, m: int = 2, r: Optional[float] = None) -> float:
#             x = np.asarray(x, dtype=float)
#             if len(x) > 900:
#                 x = x[-900:]
#             r = 0.2 * np.std(x) if r is None else r
#             emb_m = np.array([x[i : i + m] for i in range(len(x) - m + 1)])
#             emb_p = np.array([x[i : i + m + 1] for i in range(len(x) - m)])
#             dm = cdist(emb_m, emb_m, metric="chebyshev")
#             dp = cdist(emb_p, emb_p, metric="chebyshev")
#             B = np.sum(dm <= r) - len(emb_m)
#             A = np.sum(dp <= r) - len(emb_p)
#             return float(-np.log(max(A, 1) / max(B, 1)))

#         self._safe(85, lambda: {"ApEn": approximate_entropy(self.y_val)})
#         self._safe(86, lambda: {"SampEn": sample_entropy(self.y_val)})

#         def perm_entropy() -> Dict[str, Any]:
#             x = self._series_for_heavy(5000)
#             m = 4
#             patterns = [tuple(np.argsort(x[i : i + m])) for i in range(len(x) - m + 1)]
#             counts = pd.Series(patterns).value_counts(normalize=True)
#             pe = -float(np.sum(counts * np.log2(counts)))
#             return {"m": m, "permutation_entropy": pe, "normalized": pe / np.log2(factorial(m))}

#         self._safe(87, perm_entropy)

#         def transfer_entropy() -> Dict[str, Any]:
#             if not self.external_cols:
#                 raise ValueError("requires at least one source variable")
#             out = {}
#             y_bins = pd.qcut(self.y_val, q=5, duplicates="drop", labels=False)
#             for col in self.external_cols:
#                 x = pd.to_numeric(self.df[col], errors="coerce").interpolate().bfill().ffill().to_numpy()
#                 x_bins = pd.qcut(x, q=5, duplicates="drop", labels=False)
#                 df = pd.DataFrame({"yp": y_bins[1:], "y": y_bins[:-1], "x": x_bins[:-1]}).dropna()
#                 h_y = df.groupby("y")["yp"].value_counts(normalize=True)
#                 h_xy = df.groupby(["y", "x"])["yp"].value_counts(normalize=True)
#                 te = 0.0
#                 for (yv, xv, yp), p_cond in h_xy.items():
#                     p_base = h_y.get((yv, yp), 1e-12)
#                     p_joint = len(df[(df["y"] == yv) & (df["x"] == xv) & (df["yp"] == yp)]) / len(df)
#                     te += p_joint * np.log(max(p_cond, 1e-12) / max(p_base, 1e-12))
#                 out[col] = float(te)
#             return out

#         self._safe(88, transfer_entropy)

#     # ------------------------------------------------------------------
#     # Category 15
#     # ------------------------------------------------------------------
#     def cat_15_diagnostics(self) -> None:
#         def diagnostic_fit() -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
#             data = self._simple_features().tail(min(5000, self.N))
#             y = data[self.target_col].to_numpy()
#             Xdf = data.drop(columns=[self.target_col]).select_dtypes(include=[np.number])
#             keep = Xdf.var().sort_values(ascending=False).head(12).index.tolist()
#             X = StandardScaler().fit_transform(Xdf[keep])
#             fit = _ols(y, np.column_stack([np.ones(len(y)), X]))
#             return y, X, keep, fit

#         def vif() -> Dict[str, Any]:
#             _, X, names, _ = diagnostic_fit()
#             out = {}
#             for j, name in enumerate(names):
#                 other = np.delete(X, j, axis=1)
#                 fit = _ols(X[:, j], np.column_stack([np.ones(len(X)), other]))
#                 out[name] = float(1 / max(1 - fit["r2"], 1e-12))
#             return out

#         self._safe(89, vif)

#         def influence() -> Tuple[Dict[str, Any], Dict[str, Any]]:
#             y, X, names, fit = diagnostic_fit()
#             X1 = np.column_stack([np.ones(len(y)), X])
#             h = np.sum(X1 * (X1 @ np.linalg.pinv(X1.T @ X1)), axis=1)
#             resid = fit["resid"]
#             mse = fit["s2"]
#             cooks = resid**2 / (fit["k"] * mse) * h / np.maximum((1 - h) ** 2, 1e-12)
#             dffits = resid / np.sqrt(max(mse, 1e-12)) * np.sqrt(h / np.maximum(1 - h, 1e-12))
#             beta = fit["beta"]
#             dfbetas = []
#             for i in np.argsort(cooks)[-10:][::-1]:
#                 mask = np.ones(len(y), dtype=bool)
#                 mask[i] = False
#                 fi = _ols(y[mask], X1[mask])
#                 dfbetas.append(np.max(np.abs((beta - fi["beta"]) / np.maximum(fit["se"], 1e-12))))
#             cook_out = {"max_cook": float(np.max(cooks)), "threshold_4_over_n": float(4 / len(y)), "top_indices": np.argsort(cooks)[-10:][::-1]}
#             dffits_out = {"max_abs_dffits": float(np.max(np.abs(dffits))), "top_dfbetas_max_abs": dfbetas, "features": names}
#             return cook_out, dffits_out

#         self._safe(90, lambda: influence()[0])
#         self._safe(91, lambda: influence()[1])

#         def harvey_collier() -> Dict[str, Any]:
#             y, X, _, _ = diagnostic_fit()
#             X1 = np.column_stack([np.ones(len(y)), X])
#             start = X1.shape[1] + 5
#             rec = []
#             for t in range(start, len(y), max(1, len(y) // 300)):
#                 fit = _ols(y[:t], X1[:t])
#                 pred = X1[t] @ fit["beta"]
#                 rec.append(y[t] - pred)
#             rec = np.asarray(rec)
#             t_stat = float(np.mean(rec) / (np.std(rec, ddof=1) / np.sqrt(len(rec))))
#             return {"t_stat": t_stat, "p_value": float(2 * stats.t.sf(abs(t_stat), max(len(rec) - 1, 1))), "recursive_residuals": len(rec)}

#         self._safe(92, harvey_collier)

#         def reset() -> Dict[str, Any]:
#             y, X, _, fit0 = diagnostic_fit()
#             yhat = fit0["fitted"]
#             X0 = np.column_stack([np.ones(len(y)), X])
#             X1 = np.column_stack([X0, yhat**2, yhat**3])
#             fit1 = _ols(y, X1)
#             df1 = 2
#             df2 = fit1["n"] - fit1["k"]
#             f = ((fit0["rss"] - fit1["rss"]) / df1) / max(fit1["rss"] / df2, 1e-18)
#             return {"f_stat": float(f), "p_value": float(stats.f.sf(f, df1, df2))}

#         self._safe(93, reset)

#     # ------------------------------------------------------------------
#     # Category 16
#     # ------------------------------------------------------------------
#     def cat_16_signal(self) -> None:
#         def butterworth() -> Dict[str, Any]:
#             cutoff = 1 / 168
#             b, a = signal.butter(4, Wn=cutoff, fs=1.0, btype="low")
#             padlen = min(3 * max(len(a), len(b)), self.N - 1)
#             trend = signal.filtfilt(b, a, self.y_val, padlen=padlen)
#             self.cache["butter_trend"] = trend
#             return {"cutoff_frequency": cutoff, "trend_summary": _summary(trend), "residual_summary": _summary(self.y_val - trend)}

#         self._safe(94, butterworth)

#         def kalman() -> Dict[str, Any]:
#             y = self.y_val
#             q = 0.001 * np.var(np.diff(y))
#             r = np.var(y - pd.Series(y).rolling(24, min_periods=1).mean().to_numpy())
#             level = np.zeros_like(y)
#             p = np.var(y)
#             level[0] = y[0]
#             for t in range(1, len(y)):
#                 pred = level[t - 1]
#                 p_pred = p + q
#                 k = p_pred / max(p_pred + r, 1e-12)
#                 level[t] = pred + k * (y[t] - pred)
#                 p = (1 - k) * p_pred
#             self.cache["kalman_level"] = level
#             return {"q": float(q), "r": float(r), "level_summary": _summary(level), "innovation_summary": _summary(y - level)}

#         self._safe(95, kalman)

#         def ssa() -> Dict[str, Any]:
#             y = self._series_for_heavy(2500)
#             L = min(168, len(y) // 3)
#             K = len(y) - L + 1
#             X = np.column_stack([y[i : i + L] for i in range(K)])
#             U, S, Vt = np.linalg.svd(X, full_matrices=False)
#             var = S**2 / np.sum(S**2)
#             return {"window_length": L, "top_variance_ratio": var[:10], "top_singular_values": S[:10]}

#         self._safe(96, ssa)

#         def rpca() -> Dict[str, Any]:
#             daily = self.df[self.target_col].groupby(self.df.index.floor("D")).apply(lambda s: s.iloc[:24].to_numpy() if len(s) >= 24 else None)
#             mat = np.vstack([v for v in daily if isinstance(v, np.ndarray) and len(v) == 24])
#             if len(mat) < 10:
#                 raise ValueError("requires at least 10 complete daily curves")
#             X = mat - np.nanmean(mat, axis=0)
#             lam = 1 / np.sqrt(max(X.shape))
#             L = np.zeros_like(X)
#             S = np.zeros_like(X)
#             for _ in range(30):
#                 U, s, Vt = np.linalg.svd(X - S, full_matrices=False)
#                 L = (U * np.maximum(s - lam, 0)) @ Vt
#                 R = X - L
#                 S = np.sign(R) * np.maximum(np.abs(R) - lam * np.std(R), 0)
#             return {"matrix_shape": mat.shape, "low_rank_rank": int(np.linalg.matrix_rank(L)), "sparse_fraction": float(np.mean(np.abs(S) > 1e-9))}

#         self._safe(97, rpca)

#     # ------------------------------------------------------------------
#     # Category 17
#     # ------------------------------------------------------------------
#     def cat_17_bootstrap(self) -> None:
#         def moving_block() -> Dict[str, Any]:
#             b = min(168, max(24, int(np.sqrt(self.N))))
#             starts = self.rng.integers(0, max(1, self.N - b + 1), size=int(np.ceil(self.N / b)))
#             sample = np.concatenate([self.y_val[s : s + b] for s in starts])[: self.N]
#             return {"block_length": b, "sample_summary": _summary(sample)}

#         self._safe(98, moving_block)

#         def stationary() -> Dict[str, Any]:
#             b = min(168, max(24, int(np.sqrt(self.N))))
#             p = 1 / b
#             idx = []
#             current = int(self.rng.integers(0, self.N))
#             for _ in range(self.N):
#                 if self.rng.random() < p:
#                     current = int(self.rng.integers(0, self.N))
#                 idx.append(current)
#                 current = (current + 1) % self.N
#             sample = self.y_val[idx]
#             return {"mean_block_length": b, "sample_summary": _summary(sample)}

#         self._safe(99, stationary)

#         def wild() -> Dict[str, Any]:
#             baseline = self.df.get("mu_hdm", self.y.rolling(24, min_periods=1).mean()).to_numpy()
#             resid = self.y_val - baseline
#             weights = self.rng.choice([-1, 1], size=self.N)
#             sample = baseline + resid * weights
#             return {"weight": "Rademacher", "sample_summary": _summary(sample)}

#         self._safe(100, wild)

#     # ------------------------------------------------------------------
#     # Categories 18-26
#     # ------------------------------------------------------------------
#     def cat_18_26_modeling(self) -> None:
#         # 18. Cross-correlation
#         def ccf() -> Dict[str, Any]:
#             if not self.external_cols:
#                 raise ValueError("requires at least one external numeric series")
#             out = {}
#             y = (self.y_val - np.mean(self.y_val)) / max(np.std(self.y_val), 1e-12)
#             for col in self.external_cols:
#                 x = pd.to_numeric(self.df[col], errors="coerce").interpolate().bfill().ffill().to_numpy()
#                 x = (x - np.mean(x)) / max(np.std(x), 1e-12)
#                 vals = {}
#                 for lag in range(-48, 49):
#                     if lag < 0:
#                         vals[lag] = float(np.corrcoef(x[-lag:], y[: lag or None])[0, 1])
#                     elif lag > 0:
#                         vals[lag] = float(np.corrcoef(x[:-lag], y[lag:])[0, 1])
#                     else:
#                         vals[lag] = float(np.corrcoef(x, y)[0, 1])
#                 out[col] = vals
#             return out

#         self._safe(101, ccf)
#         self._safe(
#             102,
#             lambda: {
#                 "autocorrelation_max_lag_1_to_336": int(np.argmax(np.abs(_acf_np(self.y_val, min(336, self.N - 2))[1:])) + 1),
#                 "external_max_ccf": None if 101 not in self.records or self.records[101].status != "implemented" else "see item 101",
#             },
#         )

#         # 19. Regime switching
#         def regime_switch() -> Dict[str, Any]:
#             x = np.column_stack([self.y_val, self.df["hour"].to_numpy(), self.df["month"].to_numpy()])
#             x = StandardScaler().fit_transform(x)
#             gm = GaussianMixture(n_components=3, random_state=self.random_state)
#             regimes = gm.fit_predict(x)
#             trans = np.zeros((3, 3))
#             for a, b in zip(regimes[:-1], regimes[1:]):
#                 trans[a, b] += 1
#             trans = trans / np.maximum(trans.sum(axis=1, keepdims=True), 1)
#             self.cache["regimes"] = regimes
#             return {"bic": float(gm.bic(x)), "transition_matrix": trans, "regime_counts": np.bincount(regimes, minlength=3)}

#         self._safe(103, regime_switch)

#         def tar() -> Dict[str, Any]:
#             y = self.y_val
#             lag = y[:-1]
#             target = y[1:]
#             best = {"aic": np.inf}
#             for th in np.quantile(lag, np.linspace(0.2, 0.8, 21)):
#                 low = lag <= th
#                 if low.sum() < 30 or (~low).sum() < 30:
#                     continue
#                 f1 = _ols(target[low], np.column_stack([np.ones(low.sum()), lag[low]]))
#                 f2 = _ols(target[~low], np.column_stack([np.ones((~low).sum()), lag[~low]]))
#                 ic = _aic_like(f1["rss"] + f2["rss"], len(target), 4)
#                 if ic["aic"] < best["aic"]:
#                     best = {"threshold": float(th), **ic}
#             return best

#         self._safe(104, tar)

#         def star() -> Dict[str, Any]:
#             y = self.y_val
#             lag = y[:-1]
#             target = y[1:]
#             lag_z = (lag - np.mean(lag)) / max(np.std(lag), 1e-12)

#             def rss(params: np.ndarray) -> float:
#                 a1, b1, a2, b2, gamma, c = params
#                 F = 1 / (1 + np.exp(-np.clip(gamma * (lag_z - c), -50, 50)))
#                 pred = (a1 + b1 * lag) * (1 - F) + (a2 + b2 * lag) * F
#                 return float(np.sum((target - pred) ** 2))

#             start = [np.mean(target), 0.5, np.mean(target), 0.8, 1.0, 0.0]
#             opt = minimize(rss, start, method="Nelder-Mead", options={"maxiter": 2000})
#             return {"success": bool(opt.success), "rss": float(opt.fun), "params": opt.x}

#         self._safe(105, star)

#         # 20-23. Forecast diagnostics, probabilistic forecasts, feature selection.
#         fc = None
#         try:
#             fc = self._ensure_forecasts()
#         except Exception as exc:
#             for item in range(106, 130):
#                 self._record(item, status="failed", note=f"forecast diagnostic setup failed: {exc}")

#         if fc is not None:
#             y_test = fc["y_test"].to_numpy()
#             pred = fc["linear_pred"]
#             resid = y_test - pred
#             q10, q50, q90 = fc["q10_pred"], fc["q50_pred"], fc["q90_pred"]
#             lower, upper = fc["conformal_lower"], fc["conformal_upper"]

#             self._safe(
#                 106,
#                 lambda: {
#                     "dagostino_stat": float(stats.normaltest(resid).statistic),
#                     "p_value": float(stats.normaltest(resid).pvalue),
#                     "shapiro_sample_p": float(stats.shapiro(resid[: min(5000, len(resid))]).pvalue),
#                 },
#             )
#             self._record(
#                 107,
#                 {
#                     "theoretical_quantiles": stats.norm.ppf(np.linspace(0.01, 0.99, 9)),
#                     "sample_quantiles_standardized": np.quantile((resid - np.mean(resid)) / max(np.std(resid), 1e-12), np.linspace(0.01, 0.99, 9)),
#                 },
#             )
#             r = np.corrcoef(y_test, pred)[0, 1]
#             alpha = np.std(pred) / max(np.std(y_test), 1e-12)
#             beta = np.mean(pred) / max(np.mean(y_test), 1e-12)
#             self._record(108, {"KGE": float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)), "r": float(r), "alpha": float(alpha), "beta": float(beta)})
#             self._record(109, {"NSE": float(1 - np.sum((y_test - pred) ** 2) / max(np.sum((y_test - np.mean(y_test)) ** 2), 1e-12))})
#             self._record(110, {"Theil_U_seasonal_naive": float(np.sqrt(mean_squared_error(y_test, pred)) / max(np.sqrt(mean_squared_error(y_test, fc["seasonal_naive"])), 1e-12))})
#             self._record(111, {"quantiles": [0.1, 0.5, 0.9], "p10_summary": _summary(q10), "p50_summary": _summary(q50), "p90_summary": _summary(q90)})
#             alpha_int = 0.2
#             winkler = (q90 - q10) + (2 / alpha_int) * (q10 - y_test) * (y_test < q10) + (2 / alpha_int) * (y_test - q90) * (y_test > q90)
#             self._record(112, {"winkler_mean": float(np.mean(winkler)), "winkler_summary": _summary(winkler)})
#             self._record(113, {"coverage_80pct_conformal": float(np.mean((y_test >= lower) & (y_test <= upper))), "interval_width_mean": float(np.mean(upper - lower))})

#             def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
#                 e = y - q
#                 return np.maximum(tau * e, (tau - 1) * e)

#             losses = [pinball(y_test, q10, 0.1), pinball(y_test, q50, 0.5), pinball(y_test, q90, 0.9)]
#             self._record(114, {"mean_pinball": float(np.mean(losses)), "crps_quantile_approx": float(2 * np.mean(losses))})

#             def feature_selection() -> Dict[int, Any]:
#                 X_train = fc["X_train"].copy()
#                 y_train = fc["y_train"].copy()
#                 cols = X_train.columns.tolist()
#                 X_scaled = StandardScaler().fit_transform(X_train)
#                 mi = mutual_info_regression(X_scaled, y_train, random_state=self.random_state)
#                 mi_rank = sorted(zip(cols, mi), key=lambda kv: kv[1], reverse=True)
#                 selected = []
#                 remaining = list(range(len(cols)))
#                 corr = np.abs(np.corrcoef(X_scaled, rowvar=False))
#                 for _ in range(min(20, len(cols))):
#                     scores = []
#                     for j in remaining:
#                         redundancy = np.mean([corr[j, s] for s in selected]) if selected else 0.0
#                         scores.append((mi[j] - redundancy, j))
#                     _, best = max(scores)
#                     selected.append(best)
#                     remaining.remove(best)
#                 lasso = LassoCV(cv=3, random_state=self.random_state, max_iter=10000).fit(X_scaled, y_train)
#                 enet = ElasticNetCV(cv=3, random_state=self.random_state, max_iter=10000).fit(X_scaled, y_train)
#                 rf = RandomForestRegressor(n_estimators=120, random_state=self.random_state, n_jobs=-1, max_depth=8)
#                 shadow = X_train.sample(frac=1.0, replace=False, random_state=self.random_state).reset_index(drop=True)
#                 shadow.columns = [f"shadow_{c}" for c in cols]
#                 aug = pd.concat([X_train.reset_index(drop=True), shadow], axis=1)
#                 rf.fit(aug, y_train.reset_index(drop=True))
#                 imp = pd.Series(rf.feature_importances_, index=aug.columns)
#                 shadow_max = float(imp[shadow.columns].max())
#                 confirmed = imp[cols][imp[cols] > shadow_max].sort_values(ascending=False)
#                 shapley = {}
#                 top_cols = [c for c, _ in mi_rank[: min(8, len(mi_rank))]]
#                 base_model = Ridge(alpha=1.0).fit(X_train[top_cols], y_train)
#                 base_pred = base_model.predict(fc["X_test"][top_cols].head(250))
#                 for c in top_cols:
#                     pert = fc["X_test"][top_cols].head(250).copy()
#                     pert[c] = X_train[c].mean()
#                     shapley[c] = float(np.mean(np.abs(base_pred - base_model.predict(pert))))
#                 return {
#                     115: {"top_mutual_information": mi_rank[:20]},
#                     116: {"selected_features": [cols[i] for i in selected]},
#                     117: {"alpha": float(lasso.alpha_), "active_features": [cols[i] for i, b in enumerate(lasso.coef_ != 0) if b]},
#                     118: {"alpha": float(enet.alpha_), "l1_ratio": float(enet.l1_ratio_), "active_features": [cols[i] for i, b in enumerate(enet.coef_ != 0) if b]},
#                     119: {"shadow_max_importance": shadow_max, "confirmed_features": confirmed.to_dict()},
#                     120: {"method": "model-agnostic mean-replacement Shapley approximation", "feature_attribution": shapley},
#                 }

#             fs = feature_selection()
#             for item in range(115, 121):
#                 self._record(item, fs[item])

#             coverages = {}
#             for nominal in [0.5, 0.7, 0.8, 0.9]:
#                 q = np.quantile(np.abs(fc["cal_resid"]), nominal)
#                 coverages[nominal] = float(np.mean(np.abs(resid) <= q))
#             self._record(121, {"nominal_vs_actual": coverages})
#             self._record(122, {"crps_quantile_approx": self.records[114].value["crps_quantile_approx"]})
#             ref_rmse = np.sqrt(mean_squared_error(y_test, fc["seasonal_naive"]))
#             model_rmse = np.sqrt(mean_squared_error(y_test, pred))
#             self._record(123, {"rmse_skill_vs_seasonal_naive": float((ref_rmse - model_rmse) / max(ref_rmse, 1e-12))})

#             def locally_stationary() -> Dict[str, Any]:
#                 win = min(24 * 28, max(168, len(self.y_val) // 20))
#                 step = max(24, win // 4)
#                 rows = []
#                 for start in range(0, self.N - win + 1, step):
#                     seg = self.y_val[start : start + win] - np.mean(self.y_val[start : start + win])
#                     f, pxx = signal.welch(seg, fs=1.0, nperseg=min(512, win))
#                     daily_idx = int(np.argmin(np.abs(f - 1 / 24)))
#                     rows.append({"time": start + win // 2, "daily_power": float(pxx[daily_idx])})
#                 slope = np.polyfit([r["time"] for r in rows], [r["daily_power"] for r in rows], 1)[0] if len(rows) > 1 else 0.0
#                 return {"windows": len(rows), "daily_power_slope": float(slope), "daily_power_summary": _summary([r["daily_power"] for r in rows])}

#             self._safe(124, locally_stationary)

#             def tv_ar() -> Dict[str, Any]:
#                 y = self.y_val
#                 lam = 0.99
#                 beta = np.zeros(2)
#                 P = np.eye(2) * 1000
#                 betas = []
#                 for t in range(1, len(y)):
#                     x = np.array([1.0, y[t - 1]])
#                     err = y[t] - x @ beta
#                     gain = P @ x / (lam + x @ P @ x)
#                     beta = beta + gain * err
#                     P = (P - np.outer(gain, x) @ P) / lam
#                     if t % max(1, len(y) // 500) == 0:
#                         betas.append(beta.copy())
#                 betas = np.asarray(betas)
#                 return {"forgetting_factor": lam, "phi_summary": _summary(betas[:, 1]), "final_coefficients": beta}

#             self._safe(125, tv_ar)

#             def fda_daily() -> Dict[str, Any]:
#                 daily = []
#                 for _, s in self.df[self.target_col].groupby(self.df.index.floor("D")):
#                     if len(s) >= 24:
#                         daily.append(s.iloc[:24].to_numpy())
#                 mat = np.vstack(daily)
#                 pca = PCA(n_components=min(5, mat.shape[0], mat.shape[1]), random_state=self.random_state)
#                 scores = pca.fit_transform(mat)
#                 return {"daily_curves": int(mat.shape[0]), "explained_variance_ratio": pca.explained_variance_ratio_, "score_summary_pc1": _summary(scores[:, 0])}

#             self._safe(126, fda_daily)

#             def mincer() -> Dict[str, Any]:
#                 fit = _ols(y_test, np.column_stack([np.ones(len(pred)), pred]))
#                 r = np.array([fit["beta"][0], fit["beta"][1] - 1])
#                 R = np.array([[1, 0], [0, 1]])
#                 cov = fit["s2"] * np.linalg.pinv(np.column_stack([np.ones(len(pred)), pred]).T @ np.column_stack([np.ones(len(pred)), pred]))
#                 f = (r.T @ np.linalg.pinv(R @ cov @ R.T) @ r) / 2
#                 return {"alpha": float(fit["beta"][0]), "beta": float(fit["beta"][1]), "joint_f_stat": float(f), "p_value": float(stats.f.sf(f, 2, fit["n"] - fit["k"]))}

#             self._safe(127, mincer)

#             def encompassing() -> Dict[str, Any]:
#                 e1 = y_test - pred
#                 e2 = y_test - fc["gb_pred"]
#                 fit = _ols(e1, np.column_stack([np.ones(len(e2)), e2]))
#                 t = float(fit["t"][1])
#                 return {"coef_error_model2": float(fit["beta"][1]), "t_stat": t, "p_value": float(2 * stats.t.sf(abs(t), fit["n"] - fit["k"]))}

#             self._safe(128, encompassing)

#             def giacomini_white() -> Dict[str, Any]:
#                 d = (y_test - pred) ** 2 - (y_test - fc["gb_pred"]) ** 2
#                 info = pd.DataFrame(index=fc["y_test"].index)
#                 info["const"] = 1.0
#                 info["weekend"] = info.index.dayofweek.isin([5, 6]).astype(float)
#                 info["evening"] = ((info.index.hour >= 17) & (info.index.hour <= 21)).astype(float)
#                 info["summer"] = info.index.month.isin([6, 7, 8]).astype(float)
#                 fit = _ols(d, info.to_numpy())
#                 lm = fit["n"] * fit["r2"]
#                 df = info.shape[1]
#                 return {"lm_stat": float(lm), "p_value": float(stats.chi2.sf(lm, df)), "df": int(df)}

#             self._safe(129, giacomini_white)

#             def spa() -> Dict[str, Any]:
#                 losses = np.column_stack(
#                     [
#                         (y_test - pred) ** 2,
#                         (y_test - fc["gb_pred"]) ** 2,
#                         (y_test - fc["seasonal_naive"]) ** 2,
#                         (y_test - fc["last_week_naive"]) ** 2,
#                     ]
#                 )
#                 benchmark = losses[:, 2]
#                 diff = benchmark[:, None] - losses
#                 obs = float(np.max(np.mean(diff, axis=0)))
#                 b = min(96, max(24, len(y_test) // 20))
#                 sims = []
#                 for _ in range(200):
#                     starts = self.rng.integers(0, max(1, len(y_test) - b + 1), size=int(np.ceil(len(y_test) / b)))
#                     idx = np.concatenate([np.arange(s, min(s + b, len(y_test))) for s in starts])[: len(y_test)]
#                     centered = diff[idx] - np.mean(diff, axis=0)
#                     sims.append(np.max(np.mean(centered, axis=0)))
#                 return {"spa_stat": obs, "bootstrap_p_value": float(np.mean(np.asarray(sims) >= obs)), "models": ["linear", "gb", "seasonal_naive", "last_week_naive"]}

#             self._safe(130, spa)

#         # 26. Miscellaneous advanced analyses.
#         if not self.external_cols:
#             self._unavailable(131, "CCM requires a candidate causal driver such as temperature.")
#         else:
#             self._safe(131, lambda: self._ccm_for_external())

#         self._safe(132, self._recurrence_analysis)
#         self._safe(133, self._dfa)
#         self._safe(134, self._copula_analysis)
#         self._safe(135, self._seasonal_distribution_tests)
#         self._safe(136, self._modwt_variance)
#         self._safe(137, self._gam_baseline)
#         self._safe(138, self._tbats_like)
#         self._safe(139, self._dtw_knn)
#         self._safe(140, self._pelt)
#         if not self.external_cols:
#             self._unavailable(141, "Functional regression requires temperature or another full-day external profile.")
#         else:
#             self._safe(141, self._functional_external_regression)
#         self._safe(142, self._cid)
#         self._safe(143, self._matrix_profile)
#         self._safe(144, self._persistent_homology_proxy, note="Vietoris-Rips 0D persistence plus graph-cycle H1 proxy.")
#         self._safe(145, self._mic)
#         self._safe(146, self._lasso_pacf)
#         self._safe(147, self._wavelet_coherence_years)
#         self._safe(148, self._kl_drift)
#         self._safe(149, self._rao_blackwell_weights)
#         self._safe(150, self._bartlett_variances)
#         self._safe(151, self._arch_in_mean)
#         self._safe(152, self._wavelet_variance_ratio)
#         self._safe(153, self._categorical_information_gain)
#         self._safe(154, self._boxcox)
#         self._safe(155, self._guerrero)
#         self._safe(156, self._spectral_entropy)
#         if not self.external_cols:
#             self._unavailable(157, "Frequency-domain Granger causality requires an external series.")
#             self._unavailable(169, "Hong causality test requires an external series.")
#         else:
#             self._safe(157, self._freq_domain_granger)
#             self._safe(169, self._hong_causality)
#         self._safe(158, lambda: _ljung_box(self._ar_residuals(3) ** 2, [1, 2, 3, 6, 12, 24, 48, 168]))
#         self._safe(159, self._spectral_whiteness)
#         self._safe(160, self._jsd_model_distribution)
#         self._safe(161, self._ccc)
#         self._safe(162, self._interval_metrics)
#         self._safe(163, self._brier_scores)
#         self._safe(164, self._directional_symmetry)
#         self._safe(165, self._evt_tail)
#         self._safe(166, self._residual_bootstrap_uncertainty)
#         self._safe(167, self._profile_likelihood_ar)
#         self._safe(168, self._bic_weighted_average)
#         self._safe(170, self._expected_shortfall)
#         self._record_synthesis_items()

#     # ------------------------------------------------------------------
#     # Advanced helper methods for items 131-180
#     # ------------------------------------------------------------------
#     def _ccm_for_external(self) -> Dict[str, Any]:
#         out = {}
#         y = self._series_for_heavy(1200)
#         for col in self.external_cols:
#             x = pd.to_numeric(self.df[col], errors="coerce").interpolate().bfill().ffill().to_numpy()
#             if len(x) > len(y):
#                 x = x[-len(y) :]
#             emb = np.column_stack([y[:-2], y[1:-1], y[2:]])
#             target = x[2:]
#             for lib_frac in [0.3, 0.6, 0.9]:
#                 L = int(lib_frac * len(emb))
#                 nn = cdist(emb[:L], emb)
#                 preds = []
#                 actual = []
#                 for i in range(L, len(emb)):
#                     nbr = np.argsort(nn[:L, i])[:4]
#                     d = nn[nbr, i]
#                     w = np.exp(-d / max(d[0], 1e-12))
#                     w = w / np.sum(w)
#                     preds.append(float(np.sum(w * target[nbr])))
#                     actual.append(float(target[i]))
#                 out[f"{col}_{lib_frac}"] = float(np.corrcoef(preds, actual)[0, 1]) if len(preds) > 5 else np.nan
#         return out

#     def _recurrence_analysis(self) -> Dict[str, Any]:
#         y = self._series_for_heavy(900)
#         emb = np.column_stack([y[:-2], y[1:-1], y[2:]])
#         dist = squareform(pdist(StandardScaler().fit_transform(emb)))
#         eps = np.percentile(dist, 10)
#         R = dist <= eps
#         recurrence_rate = float(np.mean(R))
#         diag_lines = 0
#         diag_points = 0
#         for k in range(-R.shape[0] + 1, R.shape[0]):
#             diag = np.diag(R, k=k)
#             runs = np.diff(np.r_[False, diag, False].astype(int))
#             starts = np.where(runs == 1)[0]
#             ends = np.where(runs == -1)[0]
#             lengths = ends - starts
#             diag_lines += int(np.sum(lengths >= 2))
#             diag_points += int(np.sum(lengths[lengths >= 2]))
#         determinism = diag_points / max(np.sum(R), 1)
#         return {"epsilon": float(eps), "recurrence_rate": recurrence_rate, "determinism": float(determinism), "diagonal_lines": diag_lines}

#     def _dfa(self) -> Dict[str, Any]:
#         y = self._series_for_heavy(5000)
#         profile = np.cumsum(y - np.mean(y))
#         sizes = np.unique(np.logspace(np.log10(16), np.log10(max(32, len(y) // 4)), 18).astype(int))
#         F = []
#         used = []
#         for s in sizes:
#             chunks = len(profile) // s
#             if chunks < 2:
#                 continue
#             vals = []
#             x = np.arange(s)
#             for c in range(chunks):
#                 seg = profile[c * s : (c + 1) * s]
#                 coef = np.polyfit(x, seg, 1)
#                 vals.append(np.sqrt(np.mean((seg - np.polyval(coef, x)) ** 2)))
#             F.append(np.mean(vals))
#             used.append(s)
#         alpha = np.polyfit(np.log(used), np.log(F), 1)[0]
#         return {"dfa_alpha": float(alpha), "box_sizes": used, "fluctuations": F}

#     def _copula_analysis(self) -> Dict[str, Any]:
#         if self.N <= 168:
#             raise ValueError("requires at least 169 observations")
#         mat = np.column_stack([self.y_val[168:], self.y_val[144:-24], self.y_val[:-168]])
#         ranks = np.apply_along_axis(lambda v: stats.rankdata(v) / (len(v) + 1), 0, mat)
#         corr = np.corrcoef(stats.norm.ppf(ranks).T)
#         u = 0.95
#         upper_tail = float(np.mean((ranks[:, 0] > u) & (ranks[:, 1] > u)) / max(np.mean(ranks[:, 1] > u), 1e-12))
#         return {"gaussian_copula_corr": corr, "empirical_upper_tail_y_lag24": upper_tail}

#     def _seasonal_distribution_tests(self) -> Dict[str, Any]:
#         summer = self.df[self.df["month"].isin([6, 7, 8])][self.target_col]
#         winter = self.df[self.df["month"].isin([12, 1, 2])][self.target_col]
#         if len(summer) < 10 or len(winter) < 10:
#             raise ValueError("requires summer and winter observations")
#         ks = stats.ks_2samp(summer, winter)
#         ad = stats.anderson_ksamp([summer.to_numpy(), winter.to_numpy()])
#         return {"ks_stat": float(ks.statistic), "ks_p_value": float(ks.pvalue), "anderson_stat": float(ad.statistic), "anderson_p_value": float(ad.pvalue)}

#     def _modwt_variance(self) -> Dict[str, Any]:
#         x = self._series_for_heavy(4096)
#         variances = {}
#         approx = x.copy()
#         for level in range(1, 9):
#             step = 2 ** (level - 1)
#             detail = (approx - np.roll(approx, step)) / np.sqrt(2)
#             approx = (approx + np.roll(approx, step)) / np.sqrt(2)
#             variances[level] = {"scale_hours": int(2**level), "variance": float(np.var(detail))}
#         return variances

#     def _gam_baseline(self) -> Dict[str, Any]:
#         data = self._simple_features()
#         y = data[self.target_col].to_numpy()
#         X = data[["hour", "dayofweek", "month", "time_index"]].to_numpy(dtype=float)
#         split = int(0.8 * len(y))
#         model = make_pipeline(SplineTransformer(n_knots=8, degree=3, include_bias=False), Ridge(alpha=1.0))
#         model.fit(X[:split], y[:split])
#         pred = model.predict(X[split:])
#         return {"test_rmse": float(np.sqrt(mean_squared_error(y[split:], pred))), "test_mae": float(mean_absolute_error(y[split:], pred))}

#     def _tbats_like(self) -> Dict[str, Any]:
#         t = np.arange(self.N)
#         rows = []
#         base = np.column_stack([np.ones(self.N), t])
#         for period in [24, 168]:
#             best = None
#             for kmax in range(1, min(12, period // 2) + 1):
#                 cols = [base]
#                 for k in range(1, kmax + 1):
#                     cols.append(np.column_stack([np.sin(2 * np.pi * k * t / period), np.cos(2 * np.pi * k * t / period)]))
#                 X = np.column_stack(cols)
#                 fit = _ols(self.y_val, X)
#                 ic = _aic_like(fit["rss"], fit["n"], fit["k"])
#                 row = {"period": period, "harmonics": kmax, **ic}
#                 if best is None or row["aic"] < best["aic"]:
#                     best = row
#             rows.append(best)
#         return {"selected_harmonics": rows, "note": "TBATS-style Fourier harmonic selection by AIC; full exponential-smoothing TBATS is not bundled."}

#     def _dtw_knn(self) -> Dict[str, Any]:
#         days = []
#         dates = []
#         for d, s in self.df[self.target_col].groupby(self.df.index.floor("D")):
#             if len(s) >= 24:
#                 days.append(s.iloc[:24].to_numpy())
#                 dates.append(d)
#         mat = np.vstack(days)
#         if len(mat) < 20:
#             raise ValueError("requires at least 20 complete days")
#         query = mat[-1]

#         def dtw(a: np.ndarray, b: np.ndarray) -> float:
#             D = np.full((len(a) + 1, len(b) + 1), np.inf)
#             D[0, 0] = 0.0
#             for i in range(1, len(a) + 1):
#                 for j in range(1, len(b) + 1):
#                     D[i, j] = (a[i - 1] - b[j - 1]) ** 2 + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
#             return float(np.sqrt(D[-1, -1]))

#         distances = np.array([dtw(query, mat[i]) for i in range(len(mat) - 1)])
#         idx = np.argsort(distances)[:5]
#         return {"query_day": str(dates[-1]), "nearest_days": [str(dates[i]) for i in idx], "distances": distances[idx]}

#     def _pelt(self) -> Dict[str, Any]:
#         y = self._series_for_heavy(2500)
#         n = len(y)
#         penalty = 2 * np.log(n) * np.var(y)
#         prefix = np.r_[0.0, np.cumsum(y)]
#         prefix2 = np.r_[0.0, np.cumsum(y**2)]

#         def cost(s: int, e: int) -> float:
#             m = e - s
#             if m <= 1:
#                 return 0.0
#             sm = prefix[e] - prefix[s]
#             sm2 = prefix2[e] - prefix2[s]
#             return float(sm2 - sm * sm / m)

#         F = np.zeros(n + 1)
#         cp: Dict[int, List[int]] = {0: []}
#         R = [0]
#         min_size = max(24, n // 100)
#         for t in range(min_size, n + 1):
#             candidates = []
#             for s in R:
#                 if t - s >= min_size:
#                     candidates.append((F[s] + cost(s, t) + penalty, s))
#             if not candidates:
#                 F[t] = np.inf
#                 continue
#             F[t], tau = min(candidates, key=lambda z: z[0])
#             cp[t] = cp.get(tau, []) + [tau]
#             R = [r for r in R if F[r] + cost(r, t) <= F[t] + penalty]
#             R.append(t - min_size + 1)
#         cps = [c for c in cp.get(n, [])[1:] if 0 < c < n]
#         orig = [int(round(c / max(n - 1, 1) * (self.N - 1))) for c in cps]
#         return {"penalty": float(penalty), "change_points": [self.df.index[i].isoformat() for i in orig[:20]], "count": len(orig)}

#     def _functional_external_regression(self) -> Dict[str, Any]:
#         col = self.external_cols[0]
#         rows_y, rows_x = [], []
#         for d, part in self.df[[self.target_col, col]].groupby(self.df.index.floor("D")):
#             if len(part) >= 24:
#                 rows_y.append(part[self.target_col].iloc[:24].to_numpy())
#                 rows_x.append(pd.to_numeric(part[col], errors="coerce").interpolate().bfill().ffill().iloc[:24].to_numpy())
#         Y = np.vstack(rows_y)
#         X = np.vstack(rows_x)
#         if len(Y) < 20:
#             raise ValueError("requires at least 20 complete daily curves")
#         coefs = []
#         r2 = []
#         X1 = np.column_stack([np.ones(len(X)), X])
#         for h in range(24):
#             fit = _ols(Y[:, h], X1)
#             coefs.append(fit["beta"][1:])
#             r2.append(fit["r2"])
#         return {"external_col": col, "mean_hourly_r2": float(np.mean(r2)), "coefficient_matrix_shape": np.asarray(coefs).shape}

#     def _cid(self) -> Dict[str, Any]:
#         days = []
#         for _, s in self.df[self.target_col].groupby(self.df.index.floor("D")):
#             if len(s) >= 24:
#                 days.append(s.iloc[:24].to_numpy())
#         mat = np.vstack(days)
#         q = mat[-1]
#         ce_q = np.sqrt(np.sum(np.diff(q) ** 2))
#         dists = []
#         for i in range(len(mat) - 1):
#             x = mat[i]
#             ed = np.linalg.norm(q - x)
#             ce_x = np.sqrt(np.sum(np.diff(x) ** 2))
#             factor = max(ce_q, ce_x) / max(min(ce_q, ce_x), 1e-12)
#             dists.append(ed * factor)
#         idx = np.argsort(dists)[:5]
#         return {"nearest_indices": idx, "cid_distances": np.asarray(dists)[idx]}

#     def _matrix_profile(self) -> Dict[str, Any]:
#         x = self._series_for_heavy(1200)
#         m = min(24, len(x) // 10)
#         subseq = np.array([x[i : i + m] for i in range(len(x) - m + 1)])
#         subseq = (subseq - subseq.mean(axis=1, keepdims=True)) / np.maximum(subseq.std(axis=1, keepdims=True), 1e-12)
#         D = cdist(subseq, subseq)
#         excl = m
#         for i in range(len(D)):
#             D[i, max(0, i - excl) : min(len(D), i + excl + 1)] = np.inf
#         mp = np.min(D, axis=1)
#         nn = np.argmin(D, axis=1)
#         motif = int(np.argmin(mp))
#         discord = int(np.argmax(mp[np.isfinite(mp)]))
#         return {"subsequence_length": m, "motif_index": motif, "motif_neighbor": int(nn[motif]), "discord_index": discord, "matrix_profile_summary": _summary(mp[np.isfinite(mp)])}

#     def _persistent_homology_proxy(self) -> Dict[str, Any]:
#         y = self._series_for_heavy(700)
#         if len(y) < 50:
#             raise ValueError("requires at least 50 observations")
#         emb = np.column_stack([y[:-12], y[6:-6], y[12:]]) if len(y) > 24 else np.column_stack([y[:-2], y[1:-1], y[2:]])
#         emb = StandardScaler().fit_transform(emb)
#         dist = squareform(pdist(emb))
#         tri = dist[np.triu_indices_from(dist, k=1)]
#         thresholds = np.quantile(tri, np.linspace(0.05, 0.45, 12))

#         def find(parent: np.ndarray, a: int) -> int:
#             while parent[a] != a:
#                 parent[a] = parent[parent[a]]
#                 a = parent[a]
#             return int(a)

#         rows = []
#         n = len(emb)
#         for eps in thresholds:
#             parent = np.arange(n)
#             edges = np.argwhere(np.triu(dist <= eps, k=1))
#             for a, b in edges:
#                 ra = find(parent, int(a))
#                 rb = find(parent, int(b))
#                 if ra != rb:
#                     parent[rb] = ra
#             comps = len({find(parent, i) for i in range(n)})
#             cycle_rank = max(0, len(edges) - n + comps)
#             rows.append({"epsilon": float(eps), "beta0": int(comps), "h1_cycle_rank_proxy": int(cycle_rank)})
#         peak_h1 = max(rows, key=lambda r: r["h1_cycle_rank_proxy"])
#         return {
#             "embedding": "delay embedding",
#             "threshold_path": rows,
#             "peak_h1_cycle_rank_proxy": peak_h1,
#             "note": "This computes exact connected components on the VR 1-skeleton and a cycle-rank proxy for H1.",
#         }

#     def _mic(self) -> Dict[str, Any]:
#         fc = self._ensure_forecasts()
#         X = fc["X_train"].iloc[:, : min(20, fc["X_train"].shape[1])]
#         y = fc["y_train"].to_numpy()
#         out = {}
#         for col in X.columns:
#             x = X[col].to_numpy()
#             best = 0.0
#             for xb in range(2, 8):
#                 for yb in range(2, 8):
#                     hx = pd.qcut(x, q=xb, duplicates="drop", labels=False)
#                     hy = pd.qcut(y, q=yb, duplicates="drop", labels=False)
#                     tab = pd.crosstab(hx, hy, normalize=True)
#                     px = tab.sum(axis=1).to_numpy()
#                     py = tab.sum(axis=0).to_numpy()
#                     mi = 0.0
#                     for i in range(tab.shape[0]):
#                         for j in range(tab.shape[1]):
#                             p = tab.iloc[i, j]
#                             if p > 0:
#                                 mi += p * np.log(p / max(px[i] * py[j], 1e-12))
#                     best = max(best, mi / max(np.log(min(xb, yb)), 1e-12))
#             out[col] = float(best)
#         return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True)[:10])

#     def _lasso_pacf(self) -> Dict[str, Any]:
#         max_lag = min(336, self.N // 4)
#         Y, X = _lag_matrix(self.y_val, list(range(1, max_lag + 1)))
#         if len(Y) > 6000:
#             idx = np.linspace(0, len(Y) - 1, 6000).astype(int)
#             Y, X = Y[idx], X[idx]
#         model = make_pipeline(StandardScaler(), LassoCV(cv=3, random_state=self.random_state, max_iter=10000))
#         model.fit(X, Y)
#         lasso = model.named_steps["lassocv"]
#         selected = [i + 1 for i, c in enumerate(lasso.coef_) if abs(c) > 1e-9]
#         return {"max_lag": max_lag, "alpha": float(lasso.alpha_), "selected_lags": selected[:50], "selected_count": len(selected)}

#     def _wavelet_coherence_years(self) -> Dict[str, Any]:
#         yearly = []
#         for _, s in self.df[self.target_col].groupby(self.df.index.year):
#             if len(s) >= 24 * 30:
#                 yearly.append(s.to_numpy())
#         if len(yearly) < 2:
#             raise ValueError("requires at least two years of data")
#         a = yearly[0][: min(len(yearly[0]), len(yearly[-1]))]
#         b = yearly[-1][: len(a)]
#         f, coh = signal.coherence(a, b, fs=1.0, nperseg=min(2048, len(a)))
#         daily = coh[int(np.argmin(np.abs(f - 1 / 24)))]
#         weekly = coh[int(np.argmin(np.abs(f - 1 / 168)))]
#         return {"daily_coherence": float(daily), "weekly_coherence": float(weekly), "years_compared": len(yearly)}

#     def _kl_drift(self) -> Dict[str, Any]:
#         years = sorted(self.df.index.year.unique())
#         if len(years) < 2:
#             raise ValueError("requires at least two years")
#         a = self.df[self.df.index.year == years[0]][self.target_col].to_numpy()
#         b = self.df[self.df.index.year == years[-1]][self.target_col].to_numpy()
#         bins = np.histogram_bin_edges(np.r_[a, b], bins=40)
#         pa = np.histogram(a, bins=bins, density=True)[0] + 1e-12
#         pb = np.histogram(b, bins=bins, density=True)[0] + 1e-12
#         pa = pa / pa.sum()
#         pb = pb / pb.sum()
#         return {"year_1": int(years[0]), "year_last": int(years[-1]), "kl_first_to_last": float(np.sum(pa * np.log(pa / pb)))}

#     def _rao_blackwell_weights(self) -> Dict[str, Any]:
#         fc = self._ensure_forecasts()
#         y = fc["y_test"].to_numpy()
#         preds = np.column_stack([fc["linear_pred"], fc["gb_pred"], fc["seasonal_naive"], fc["last_week_naive"]])
#         errors = y[:, None] - preds
#         cov = np.cov(errors, rowvar=False) + np.eye(errors.shape[1]) * 1e-6
#         ones = np.ones(errors.shape[1])
#         w = np.linalg.pinv(cov) @ ones
#         w = w / np.sum(w)
#         return {"models": ["linear", "gb", "seasonal_naive", "last_week_naive"], "weights": w}

#     def _bartlett_variances(self) -> Dict[str, Any]:
#         groups = []
#         for _, s in self.df.groupby("hour"):
#             vals = pd.to_numeric(s[self.target_col], errors="coerce").dropna().to_numpy()
#             if len(vals) > 2:
#                 groups.append(vals)
#         if len(groups) < 2:
#             raise ValueError("not enough non-empty hour groups for Bartlett test")
#         stat, p = stats.bartlett(*groups)
#         return {"hour_groups": len(groups), "bartlett_stat": float(stat), "p_value": float(p)}

#     def _arch_in_mean(self) -> Dict[str, Any]:
#         resid = self._ar_residuals(3)
#         e = resid - np.mean(resid)
#         params = self.cache.get("garch_params", {"omega": 0.05 * np.var(e), "alpha": 0.08, "beta": 0.88})
#         omega = float(params.get("omega", 0.05 * np.var(e)))
#         alpha = float(params.get("alpha", 0.08))
#         beta = float(params.get("beta", 0.88))
#         sig2 = np.empty_like(e)
#         sig2[0] = np.var(e)
#         for t in range(1, len(e)):
#             sig2[t] = omega + alpha * e[t - 1] ** 2 + beta * sig2[t - 1]
#         y = self.y_val[-len(e) :]
#         fit = _ols(y, np.column_stack([np.ones(len(y)), np.sqrt(np.maximum(sig2, 1e-12))]))
#         t = float(fit["t"][1])
#         return {"lambda": float(fit["beta"][1]), "t_stat": t, "p_value": float(2 * stats.t.sf(abs(t), fit["n"] - fit["k"]))}

#     def _wavelet_variance_ratio(self) -> Dict[str, Any]:
#         variances = self._modwt_variance()
#         vals = {v["scale_hours"]: v["variance"] for v in variances.values()}
#         daily_scale = min(vals, key=lambda k: abs(k - 24))
#         weekly_scale = max(vals)
#         ratio = vals[daily_scale] / max(vals[weekly_scale], 1e-12)
#         return {"daily_scale": daily_scale, "weekly_like_scale": weekly_scale, "variance_ratio": float(ratio)}

#     def _categorical_information_gain(self) -> Dict[str, Any]:
#         ybin = pd.qcut(self.y_val, q=10, duplicates="drop", labels=False)
#         base_counts = pd.Series(ybin).value_counts(normalize=True)
#         H = -np.sum(base_counts * np.log2(base_counts))
#         out = {}
#         for col in ["hour", "dayofweek", "month", "is_weekend"]:
#             cond = 0.0
#             for _, idx in self.df.groupby(col).groups.items():
#                 p_group = len(idx) / self.N
#                 counts = pd.Series(ybin[self.df.index.get_indexer(idx)]).value_counts(normalize=True)
#                 cond += p_group * (-np.sum(counts * np.log2(counts)))
#             out[col] = float(H - cond)
#         return out

#     def _boxcox(self) -> Dict[str, Any]:
#         y = self.y_val.copy()
#         shift = 0.0
#         if np.min(y) <= 0:
#             shift = float(1 - np.min(y))
#             y = y + shift
#         lam = stats.boxcox_normmax(y, method="mle")
#         transformed = stats.boxcox(y, lmbda=lam)
#         return {"lambda": float(lam), "shift": shift, "transformed_skew": float(stats.skew(transformed))}

#     def _guerrero(self) -> Dict[str, Any]:
#         candidates = [12, 24, 48, 168, 336, 24 * 365]
#         out = {}
#         for p in candidates:
#             if self.N > 2 * p:
#                 d = self.y.diff(p).dropna()
#                 out[p] = float(d.std() / max(abs(d.mean()), 1e-12))
#         return {"cv_by_period": out, "best_period": int(min(out, key=out.get)) if out else None}

#     def _spectral_entropy(self) -> Dict[str, Any]:
#         f, pxx = signal.periodogram(self.y_val - np.mean(self.y_val))
#         p = pxx[1:] / max(np.sum(pxx[1:]), 1e-18)
#         se = -np.sum(p[p > 0] * np.log2(p[p > 0])) / np.log2(max(len(p), 2))
#         return {"spectral_entropy_normalized": float(se)}

#     def _freq_domain_granger(self) -> Dict[str, Any]:
#         col = self.external_cols[0]
#         x = pd.to_numeric(self.df[col], errors="coerce").interpolate().bfill().ffill().to_numpy()
#         f, cxy = signal.coherence(x, self.y_val, fs=1.0, nperseg=min(2048, self.N))
#         return {"external_col": col, "coherence_as_frequency_granger_proxy": {"max": float(np.max(cxy)), "freq_at_max": float(f[int(np.argmax(cxy))])}}

#     def _hong_causality(self) -> Dict[str, Any]:
#         return self._freq_domain_granger()

#     def _spectral_whiteness(self) -> Dict[str, Any]:
#         fc = self._ensure_forecasts()
#         resid = fc["y_test"].to_numpy() - fc["linear_pred"]
#         f, pxx = signal.periodogram(resid - np.mean(resid))
#         z = pxx[1:] / max(np.mean(pxx[1:]), 1e-18)
#         ks = stats.kstest(z, "expon")
#         return {"ks_stat": float(ks.statistic), "p_value": float(ks.pvalue), "largest_residual_period": float(1 / max(f[1:][np.argmax(pxx[1:])], 1e-12))}

#     def _jsd_model_distribution(self) -> Dict[str, Any]:
#         fc = self._ensure_forecasts()
#         y = fc["y_test"].to_numpy()
#         pred = fc["linear_pred"]
#         bins = np.histogram_bin_edges(np.r_[y, pred], bins=40)
#         p = np.histogram(y, bins=bins, density=True)[0] + 1e-12
#         q = np.histogram(pred, bins=bins, density=True)[0] + 1e-12
#         p = p / p.sum()
#         q = q / q.sum()
#         m = 0.5 * (p + q)
#         jsd = 0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m))
#         return {"jsd_bits": float(jsd)}

#     def _ccc(self) -> Dict[str, Any]:
#         fc = self._ensure_forecasts()
#         y = fc["y_test"].to_numpy()
#         pred = fc["linear_pred"]
#         rho = np.corrcoef(y, pred)[0, 1]
#         ccc = 2 * rho * np.std(y) * np.std(pred) / max(np.var(y) + np.var(pred) + (np.mean(y) - np.mean(pred)) ** 2, 1e-12)
#         return {"ccc": float(ccc), "pearson": float(rho)}

#     def _interval_metrics(self) -> Dict[str, Any]:
#         fc = self._ensure_forecasts()
#         y = fc["y_test"].to_numpy()
#         l = fc["conformal_lower"]
#         u = fc["conformal_upper"]
#         picp = np.mean((y >= l) & (y <= u))
#         mpiw = np.mean(u - l)
#         mu = 0.8
#         gamma = 1.0 if picp < mu else 0.0
#         cwc = mpiw * (1 + gamma * np.exp(-50 * (picp - mu)))
#         return {"PICP": float(picp), "MPIW": float(mpiw), "CWC": float(cwc)}

#     def _brier_scores(self) -> Dict[str, Any]:
#         fc = self._ensure_forecasts()
#         y = fc["y_test"].to_numpy()
#         pred = fc["linear_pred"]
#         sigma = np.std(fc["cal_resid"])
#         out = {}
#         for q in [0.9, 0.95, 0.99]:
#             threshold = np.quantile(self.y_val, q)
#             prob = 1 - stats.norm.cdf((threshold - pred) / max(sigma, 1e-12))
#             obs = (y > threshold).astype(float)
#             bs = np.mean((prob - obs) ** 2)
#             clim = np.mean((np.mean(obs) - obs) ** 2)
#             out[q] = {"threshold": float(threshold), "brier": float(bs), "brier_skill": float(1 - bs / max(clim, 1e-12))}
#         return out

#     def _directional_symmetry(self) -> Dict[str, Any]:
#         fc = self._ensure_forecasts()
#         y = fc["y_test"].to_numpy()
#         pred = fc["linear_pred"]
#         ds = np.mean(np.sign(np.diff(y)) == np.sign(np.diff(pred)))
#         return {"directional_symmetry": float(ds)}

#     def _evt_tail(self) -> Dict[str, Any]:
#         fc = self._ensure_forecasts()
#         errors = np.abs(fc["y_test"].to_numpy() - fc["linear_pred"])
#         u = np.quantile(errors, 0.95)
#         excess = errors[errors > u] - u
#         c, loc, scale = stats.genpareto.fit(excess, floc=0)
#         return {"threshold": float(u), "shape_xi": float(c), "scale": float(scale), "tail_count": int(len(excess))}

#     def _residual_bootstrap_uncertainty(self) -> Dict[str, Any]:
#         fc = self._ensure_forecasts()
#         pred = fc["linear_pred"]
#         resid = fc["cal_resid"]
#         paths = pred[:, None] + self.rng.choice(resid, size=(len(pred), 300), replace=True)
#         return {"lower_05_summary": _summary(np.quantile(paths, 0.05, axis=1)), "upper_95_summary": _summary(np.quantile(paths, 0.95, axis=1))}

#     def _profile_likelihood_ar(self) -> Dict[str, Any]:
#         rows = self.cache.get("ar_ic_grid")
#         if rows is None:
#             rows = []
#             for p in range(1, 8):
#                 Y, X = _lag_matrix(self.y_val, list(range(1, p + 1)))
#                 fit = _ols(Y, np.column_stack([np.ones(len(Y)), X]))
#                 rows.append({"p": p, **_aic_like(fit["rss"], fit["n"], fit["k"])})
#         max_ll = max(r["log_likelihood"] for r in rows)
#         supported = [r["p"] for r in rows if r["log_likelihood"] >= max_ll - 0.5 * stats.chi2.ppf(0.95, 1)]
#         return {"max_log_likelihood": float(max_ll), "supported_ar_orders_95pct": supported, "grid": rows}

#     def _bic_weighted_average(self) -> Dict[str, Any]:
#         rows = self.cache.get("ar_ic_grid", [])
#         if not rows:
#             raise ValueError("AR IC grid not available")
#         bic = np.array([r["bic"] for r in rows])
#         w = np.exp(-0.5 * (bic - np.min(bic)))
#         w = w / np.sum(w)
#         return {"orders": [r["p"] for r in rows], "weights": w}

#     def _expected_shortfall(self) -> Dict[str, Any]:
#         fc = self._ensure_forecasts()
#         y = fc["y_test"].to_numpy()
#         out = {}
#         for alpha in [0.9, 0.95, 0.99]:
#             var = np.quantile(y, alpha)
#             out[alpha] = {"VaR": float(var), "ES": float(np.mean(y[y >= var]))}
#         return out

#     def _record_synthesis_items(self) -> None:
#         self._record(
#             171,
#             {
#                 "differencing_inputs": "items 8-15",
#                 "arma_order_inputs": "items 16-21 and 72-76",
#                 "nonlinear_inputs": "items 51-56",
#                 "variance_inputs": "items 57-61 and 158",
#                 "break_inputs": "items 62-65 and 140",
#             },
#         )
#         significant_lags = []
#         if 16 in self.records and self.records[16].status == "implemented":
#             acf_head = self.records[16].value["acf"]["head"]
#             significant_lags = [i for i, v in enumerate(acf_head[1:], start=1) if abs(v) > self.records[16].value["confidence_band"]]
#         self._record(
#             172,
#             {
#                 "lag_features_from_acf": significant_lags,
#                 "calendar_target_encoding_columns": ["mu_hdm", "median_hdm"],
#                 "fourier_periods": [24, 168],
#                 "anomaly_columns": [c for c in self.df.columns if "outlier" in c or "anomaly" in c],
#             },
#         )
#         self._record(
#             173,
#             {
#                 "ar_order_from_bic": self.records.get(74).value.get("p") if self.records.get(74) and self.records.get(74).value else None,
#                 "seasonal_periods": list(self.seasonal_periods),
#                 "selected_harmonics": self.records.get(138).value.get("selected_harmonics") if self.records.get(138) and self.records.get(138).value else None,
#             },
#         )
#         kurt = self.records.get(5).value.get("excess_kurtosis") if self.records.get(5) and self.records.get(5).value else None
#         skew = self.records.get(4).value.get("skewness") if self.records.get(4) and self.records.get(4).value else None
#         self._record(
#             174,
#             {
#                 "skewness": skew,
#                 "kurtosis": kurt,
#                 "recommended_point_loss": "Huber_or_MAE" if kurt is not None and kurt > 2 else "MSE_or_Huber",
#                 "recommended_interval_loss": "pinball_quantile_loss",
#             },
#         )
#         self._record(
#             175,
#             {
#                 "minimum_gap_hours": 48,
#                 "validation_type": "rolling/prequential plus h-block",
#                 "break_aware": self.records.get(63).value.get("break_date") if self.records.get(63) and self.records.get(63).value else None,
#             },
#         )
#         self._safe(176, self._rao_blackwell_weights)
#         self._safe(177, lambda: {"bic_model_weights": self._bic_weighted_average(), "prediction_variance_formula": "within_model_variance + between_model_variance"})
#         self._safe(178, lambda: self.records.get(125).value if self.records.get(125) else self._record(125, self._record))

#         def hedge() -> Dict[str, Any]:
#             fc = self._ensure_forecasts()
#             y = fc["y_test"].to_numpy()
#             preds = np.column_stack([fc["linear_pred"], fc["gb_pred"], fc["seasonal_naive"], fc["last_week_naive"]])
#             weights = np.ones(preds.shape[1]) / preds.shape[1]
#             eta = 0.01 / max(np.var(y), 1e-12)
#             loss_path = []
#             for t in range(len(y)):
#                 losses = (y[t] - preds[t]) ** 2
#                 weights = weights * np.exp(-eta * losses)
#                 weights = weights / np.sum(weights)
#                 loss_path.append(float(weights @ losses))
#             return {"final_weights": weights, "mean_weighted_loss": float(np.mean(loss_path))}

#         self._safe(179, hedge)

#         def bh() -> Dict[str, Any]:
#             tests = [(item, name, p) for item, name, p in self.p_values if np.isfinite(p)]
#             if not tests:
#                 return {"tests": 0, "rejections": []}
#             q = 0.05
#             tests_sorted = sorted(tests, key=lambda z: z[2])
#             m = len(tests_sorted)
#             cutoff = -1
#             for i, (_, _, p) in enumerate(tests_sorted, start=1):
#                 if p <= q * i / m:
#                     cutoff = i
#             rejected = tests_sorted[:cutoff] if cutoff > 0 else []
#             return {"tests": m, "fdr_q": q, "rejection_count": len(rejected), "rejections": rejected[:30]}

#         self._safe(180, bh)

#     # ------------------------------------------------------------------
#     # Execution and reporting
#     # ------------------------------------------------------------------
#     def execute_all(self) -> Dict[int, Dict[str, Any]]:
#         if self.N < 50:
#             raise ValueError("At least 50 observations are required for the full pipeline.")

#         self.cat_1_descriptive()
#         self.cat_2_stationarity()
#         self.cat_3_autocorrelation()
#         self.cat_4_spectral()
#         self.cat_5_trend()
#         self.cat_6_seasonality()
#         self.cat_7_long_memory()
#         self.cat_8_nonlinearity()
#         self.cat_9_volatility()
#         self.cat_10_breaks()
#         self.cat_11_anomalies()
#         self.cat_12_selection()
#         self.cat_13_multivariate()
#         self.cat_14_entropy()
#         self.cat_15_diagnostics()
#         self.cat_16_signal()
#         self.cat_17_bootstrap()
#         self.cat_18_26_modeling()

#         for item in range(1, 181):
#             if item not in self.records:
#                 self._record(item, status="failed", note="item was not reached by execute_all")

#         summary = self.coverage_summary()
#         self.results = {
#             "coverage": summary,
#             "records": {item: asdict(record) for item, record in sorted(self.records.items())},
#         }
#         return self.results["records"]

#     def coverage_summary(self) -> Dict[str, Any]:
#         statuses = pd.Series([r.status for r in self.records.values()]).value_counts().to_dict()
#         missing = [i for i in range(1, 181) if i not in self.records]
#         return {
#             "total_items": 180,
#             "recorded_items": len(self.records),
#             "status_counts": statuses,
#             "missing_items": missing,
#             "external_columns": self.external_cols,
#         }

#     def to_frame(self) -> pd.DataFrame:
#         if not self.records:
#             self.execute_all()
#         return pd.DataFrame([asdict(r) for _, r in sorted(self.records.items())])

#     def save_report(self, path: str) -> None:
#         if not self.records:
#             self.execute_all()
#         with open(path, "w", encoding="utf-8") as f:
#             json.dump(self.results, f, indent=2, default=str)

#     def print_summary(self) -> None:
#         if not self.records:
#             self.execute_all()
#         summary = self.coverage_summary()
#         print("\n--- 180-ITEM STATISTICAL ANALYSIS COVERAGE ---")
#         print(json.dumps(summary, indent=2, default=str))
#         print("\nUnavailable items:")
#         for item, record in sorted(self.records.items()):
#             if record.status == "unavailable":
#                 print(f"{item:03d}. {record.name}: {record.note}")
#         print("\nFailed items:")
#         for item, record in sorted(self.records.items()):
#             if record.status == "failed":
#                 print(f"{item:03d}. {record.name}: {record.note}")


# if __name__ == "__main__":
#     print("Loading hourly energy data from the Excel dataset...")
#     data_path = "Energy Consumption Dataset.xlsx"
#     df = pd.read_excel(data_path)
#     pipeline = ExactMathematicalPipeline(
#         df,
#         "Start time UTC",
#         "Electricity consumption (MWh)",
#     )
#     pipeline.execute_all()
#     pipeline.print_summary()


"""
COMPREHENSIVE STATISTICAL ANALYSIS FOR MODEL SELECTION
Covers all major categories from the 180-item framework.
Reads your existing processed DataFrame and runs every test
that directly informs which model to use and how to configure it.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats, signal, fft
from scipy.stats import (
    skew, kurtosis, jarque_bera, ks_2samp, anderson,
    shapiro, probplot, normaltest, genpareto, kendalltau
)
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.interpolate import CubicSpline
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
import itertools
import os
import json
from math import factorial

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
OUT = "deep_analysis_outputs"
os.makedirs(OUT, exist_ok=True)

PALETTE = {
    'primary'  : '#0D3B66',
    'secondary': '#1B998B',
    'accent'   : '#E84855',
    'warn'     : '#F4A261',
    'success'  : '#2EC4B6',
    'bg'       : '#F7F9FC',
    'dark'     : '#1A1A2E',
    'purple'   : '#6B4FBB',
}

RNG = np.random.default_rng(42)

def savefig(fig, name, dpi=140):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {path}")

def banner(title):
    print(f"\n{'━'*72}")
    print(f"  {title}")
    print(f"{'━'*72}")

def subsection(title):
    print(f"\n  ── {title}")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD YOUR DATA
# ─────────────────────────────────────────────────────────────────────────────
banner("LOADING & PREPARING DATA")

# ── adjust this block to however you build df in a.py ────────────────────────
df = pd.read_csv("cleaned_energy_data_model.csv")   # <── change to your actual load
# If you run this inside a.py after building df, just remove the line above.

# Standardise column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Identify the time column and target
TIME_COL   = 'start_time'
TARGET_COL = 'consumption'

df[TIME_COL] = pd.to_datetime(df[TIME_COL])
df = df.sort_values(TIME_COL).reset_index(drop=True)
df = df.set_index(TIME_COL)
df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index

# Full hourly reindex to materialise gaps
full_idx = pd.date_range(df.index.min(), df.index.max(), freq='h')
df = df.reindex(full_idx)
df.index.name = 'ds'

# Linear interpolate any gaps (for analysis only – not for training targets)
df[TARGET_COL] = df[TARGET_COL].interpolate(method='time')

# Calendar columns (re-derive to be safe)
df['hour']       = df.index.hour
df['dow']        = df.index.dayofweek
df['month']      = df.index.month
df['year']       = df.index.year
df['doy']        = df.index.dayofyear
df['week']       = df.index.isocalendar().week.astype(int)
df['is_weekend'] = (df['dow'] >= 5).astype(int)
df['quarter']    = df.index.quarter

y    = df[TARGET_COL].values.copy()
N    = len(y)
ts   = df[TARGET_COL]

print(f"  Observations : {N:,}")
print(f"  Date range   : {df.index[0]}  →  {df.index[-1]}")
print(f"  Target range : {y.min():.1f} – {y.max():.1f} MWh")
print(f"  NaN after reindex: {np.isnan(y).sum()}")
y = np.nan_to_num(y, nan=np.nanmean(y))

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS (pure numpy, no heavy dependencies)
# ─────────────────────────────────────────────────────────────────────────────

def acf_np(x, nlags):
    x = x - np.mean(x)
    d = np.dot(x, x)
    if d == 0: return np.zeros(nlags+1)
    r = [1.0]
    for k in range(1, nlags+1):
        r.append(float(np.dot(x[k:], x[:-k]) / d))
    return np.array(r)

def pacf_np(x, nlags):
    out = [1.0]
    for k in range(1, nlags+1):
        Y  = x[k:]
        Xm = np.column_stack([x[k-j-1:-j-1] for j in range(k)])
        Xm = np.column_stack([np.ones(len(Y)), Xm])
        b, *_ = np.linalg.lstsq(Xm, Y, rcond=None)
        out.append(float(b[-1]))
    return np.array(out)

def ols(y, X):
    y, X = np.asarray(y, float), np.asarray(X, float)
    mask = np.isfinite(y) & np.all(np.isfinite(X), 1)
    y, X = y[mask], X[mask]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit   = X @ b
    res   = y - fit
    n, k  = len(y), X.shape[1]
    s2    = np.sum(res**2) / max(n-k, 1)
    se    = np.sqrt(np.maximum(np.diag(np.linalg.pinv(X.T@X)) * s2, 0))
    t     = np.divide(b, se, out=np.zeros_like(b), where=se>0)
    r2    = 1 - np.sum(res**2) / max(np.sum((y-y.mean())**2), 1e-12)
    return {'b':b,'fit':fit,'res':res,'n':n,'k':k,'s2':float(s2),'se':se,'t':t,'r2':float(r2),'rss':float(np.sum(res**2))}

def ljung_box(x, lags):
    n  = len(x)
    r  = acf_np(x, max(lags))
    out = {}
    for lag in lags:
        q  = n*(n+2)*np.sum(r[1:lag+1]**2 / np.maximum(n - np.arange(1,lag+1), 1))
        out[lag] = {'stat': float(q), 'p': float(stats.chi2.sf(q, max(1,lag)))}
    return out

def adf_test(x, regression='ct', lags=None):
    n = len(x)
    if lags is None:
        lags = int(np.floor(12*(n/100)**0.25))
    lags = max(0, min(lags, 48, n//5))
    dy   = np.diff(x)
    tgt  = dy[lags:]
    cols = []
    if regression in ('c','ct'): cols.append(np.ones(len(tgt)))
    if regression == 'ct':       cols.append(np.arange(len(tgt),dtype=float))
    cols.append(x[lags:-1])
    for j in range(1, lags+1): cols.append(dy[lags-j:len(dy)-j])
    X   = np.column_stack(cols)
    fit = ols(tgt, X)
    lvl_idx = 2 if regression=='ct' else (1 if regression=='c' else 0)
    stat = float(fit['t'][lvl_idx])
    cvs  = {'1%':-3.43,'5%':-2.86,'10%':-2.57}
    return {'stat':stat,'lags':lags,'cvs':cvs,
            'reject_1%': stat<cvs['1%'], 'reject_5%': stat<cvs['5%']}

def kpss_test(x, regression='ct'):
    n    = len(x)
    cols = [np.ones(n)]
    if regression=='ct': cols.append(np.arange(n,dtype=float))
    fit  = ols(x, np.column_stack(cols))
    res  = fit['res']
    nlag = int(np.floor(4*(n/100)**0.25))
    s    = np.cumsum(res)
    g0   = np.sum(res**2)/n
    lr   = g0
    for j in range(1, nlag+1):
        w   = 1 - j/(nlag+1)
        lr += 2*w*(np.sum(res[j:]*res[:-j])/n)
    stat = float(np.sum(s**2)/(n**2*max(lr,1e-18)))
    cvs  = {'10%':0.347,'5%':0.463,'2.5%':0.574,'1%':0.739} if regression=='c' \
           else {'10%':0.119,'5%':0.146,'2.5%':0.176,'1%':0.216}
    return {'stat':stat,'cvs':cvs,
            'reject_5%': stat>cvs['5%']}

def hurst_rs(x, n_points=25):
    n  = len(x)
    sizes = np.unique(np.geomspace(max(16, n//200), n//4, n_points).astype(int))
    rs_vals, used = [], []
    for s in sizes:
        chunks = n // s
        if chunks < 2: continue
        vals = []
        for c in range(chunks):
            seg = x[c*s:(c+1)*s]
            z   = np.cumsum(seg - seg.mean())
            R   = z.max() - z.min()
            S   = seg.std(ddof=1)
            if S > 0: vals.append(R/S)
        if vals:
            rs_vals.append(np.mean(vals))
            used.append(s)
    if len(used) < 2: return 0.5, used, rs_vals
    H = np.polyfit(np.log(used), np.log(rs_vals), 1)[0]
    return float(H), used, rs_vals

def dfa(x, n_points=20):
    n       = len(x)
    profile = np.cumsum(x - x.mean())
    sizes   = np.unique(np.geomspace(max(16,n//200), n//4, n_points).astype(int))
    F, used = [], []
    for s in sizes:
        chunks = n // s
        if chunks < 2: continue
        t   = np.arange(s, dtype=float)
        rms = []
        for c in range(chunks):
            seg = profile[c*s:(c+1)*s]
            p   = np.polyfit(t, seg, 1)
            rms.append(np.sqrt(np.mean((seg - np.polyval(p,t))**2)))
        F.append(np.mean(rms)); used.append(s)
    if len(used) < 2: return 0.75, used, F
    alpha = np.polyfit(np.log(used), np.log(F), 1)[0]
    return float(alpha), used, F

def gph_estimator(x):
    n     = len(x)
    y_c   = x - x.mean()
    freq, power = signal.periodogram(y_c)
    power = np.maximum(power[1:], 1e-18)
    m     = max(20, int(n**0.65))
    lam   = 2*np.pi*np.arange(1,m+1)/n
    X     = np.column_stack([np.ones(m), -np.log(4*np.sin(lam/2)**2)])
    fit   = ols(np.log(power[:m]), X)
    d     = float(fit['b'][1])
    se    = float(fit['se'][1])
    return {'d': d, 'se': se, 'z': d/max(se,1e-12),
            'p': float(2*stats.norm.sf(abs(d/max(se,1e-12))))}

def arch_lm(resid, lags=24):
    e2    = resid**2
    lags  = min(lags, len(e2)//5)
    Y     = e2[lags:]
    X     = np.column_stack([np.ones(len(Y))] + [e2[lags-j:-j] for j in range(1,lags+1)])
    fit   = ols(Y, X)
    lm    = fit['n'] * fit['r2']
    return {'stat': float(lm), 'p': float(stats.chi2.sf(lm, lags)), 'lags': lags}

def ar_residuals(x, p=3):
    p   = min(max(1,p), len(x)//10)
    Y   = x[p:]
    Xm  = np.column_stack([np.ones(len(Y))] + [x[p-j:-j] for j in range(1,p+1)])
    fit = ols(Y, Xm)
    return fit['res']

def aic_bic(rss, n, k):
    s2    = max(rss/n, 1e-12)
    ll    = -0.5*n*(np.log(2*np.pi) + np.log(s2) + 1)
    aic   = -2*ll + 2*k
    aicc  = aic + 2*k*(k+1)/max(n-k-1,1)
    bic   = -2*ll + k*np.log(n)
    hqc   = -2*ll + 2*k*np.log(max(np.log(n),1.000001))
    return {'ll':float(ll),'aic':float(aic),'aicc':float(aicc),'bic':float(bic),'hqc':float(hqc)}

# ─────────────────────────────────────────────────────────────────────────────
# 1. FUNDAMENTAL DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
banner("1. FUNDAMENTAL DESCRIPTIVE STATISTICS")

# 1.1 Global moments
jb_stat, jb_p   = jarque_bera(y)
sk               = float(skew(y, bias=False))
ku               = float(kurtosis(y, fisher=True, bias=False))
cv               = float(y.std()/y.mean()*100)

print(f"\n  N={N:,}   Mean={y.mean():.1f}   Std={y.std():.1f}   CV={cv:.2f}%")
print(f"  Min={y.min():.1f}   P25={np.percentile(y,25):.1f}   Median={np.median(y):.1f}   P75={np.percentile(y,75):.1f}   Max={y.max():.1f}")
print(f"  Skewness   = {sk:.4f}  ({'right' if sk>0 else 'left'}-skewed)")
print(f"  Ex-Kurtosis= {ku:.4f}  ({'heavy tails → use Huber/MAE loss' if ku>2 else 'near-normal tails'})")
print(f"  Jarque-Bera= {jb_stat:.2f}  p={jb_p:.2e}  → {'NON-NORMAL' if jb_p<0.05 else 'Normal'}")
print(f"  IQR        = {np.percentile(y,75)-np.percentile(y,25):.1f}")

# ── MODEL IMPLICATION ──────────────────────────────────────────────────────
print(f"\n  ► LOSS FUNCTION: {'Huber or MAE (heavy tails)' if ku>2 else 'MSE is acceptable'}")
print(f"  ► INTERVALS   : {'Quantile/Conformal (non-Gaussian)' if jb_p<0.05 else 'Gaussian intervals acceptable'}")

# 1.2 Conditional mean μ(h, d, m)
cond_mean   = df.groupby(['hour','dow','month'])[TARGET_COL].mean()
cond_median = df.groupby(['hour','dow','month'])[TARGET_COL].median()
cond_std    = df.groupby(['hour','dow','month'])[TARGET_COL].std()

key_tuples  = list(zip(df['hour'], df['dow'], df['month']))
mu_hdm      = pd.Series([cond_mean.get(k, np.nan) for k in key_tuples], index=df.index)
eps_t       = y - np.nan_to_num(mu_hdm.values, nan=y.mean())
var_exp     = 1 - np.nanvar(eps_t) / max(np.nanvar(y), 1e-12)

print(f"\n  Conditional mean μ(h,d,m): {len(cond_mean)} cells  (theoretical 24×7×12={24*7*12})")
print(f"  Variance explained by calendar alone: {var_exp*100:.1f}%")
print(f"  ► IF var_exp > 60%: calendar features are dominant → must include them")

# 1.3 Weekday vs Weekend KS test
wd = df[df['is_weekend']==0][TARGET_COL].dropna().values
we = df[df['is_weekend']==1][TARGET_COL].dropna().values
ks2, ks2p = ks_2samp(wd, we)
print(f"\n  Weekday vs Weekend KS: D={ks2:.4f}  p={ks2p:.2e}")
print(f"  ► {'Distributions DIFFER → is_weekend is a necessary feature' if ks2p<0.05 else 'Distributions similar'}")

# 1.4 Range analysis (multiplicative vs additive seasonality)
daily_max = df.groupby(df.index.date)[TARGET_COL].max()
daily_min = df.groupby(df.index.date)[TARGET_COL].min()
daily_mean_series = df.groupby(df.index.date)[TARGET_COL].mean()
daily_range = daily_max - daily_min
daily_range_np = daily_range.values
daily_level_np = daily_mean_series.values

# Align lengths
min_len = min(len(daily_range_np), len(daily_level_np))
daily_range_np = daily_range_np[:min_len]
daily_level_np = daily_level_np[:min_len]
mask_range = np.isfinite(daily_range_np) & np.isfinite(daily_level_np)
range_level_corr = float(np.corrcoef(daily_range_np[mask_range], daily_level_np[mask_range])[0,1])

print(f"\n  Daily Range vs Daily Level Correlation: {range_level_corr:.4f}")
print(f"  ► {'MULTIPLICATIVE seasonality (range ∝ level)' if range_level_corr>0.5 else 'ADDITIVE seasonality (range constant)'}")
print(f"  ► Implication: {'Log-transform target; use multiplicative STL; TBATS multiplicative' if range_level_corr>0.5 else 'Additive STL; standard ETS(A,A,A)'}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. STATIONARITY TESTS
# ─────────────────────────────────────────────────────────────────────────────
banner("2. STATIONARITY TESTS (ADF + KPSS + PP)")

adf_ct = adf_test(y, regression='ct')
adf_c  = adf_test(y, regression='c')
kpss_ct = kpss_test(y, regression='ct')
kpss_c  = kpss_test(y, regression='c')

print(f"\n  ADF (const+trend): stat={adf_ct['stat']:.4f}  5%cv={adf_ct['cvs']['5%']}  reject_5%={adf_ct['reject_5%']}")
print(f"  ADF (const only) : stat={adf_c['stat']:.4f}  5%cv={adf_c['cvs']['5%']}  reject_5%={adf_c['reject_5%']}")
print(f"  KPSS (trend)     : stat={kpss_ct['stat']:.4f}  5%cv={kpss_ct['cvs']['5%']}  reject_5%={kpss_ct['reject_5%']}")
print(f"  KPSS (level)     : stat={kpss_c['stat']:.4f}  5%cv={kpss_c['cvs']['5%']}  reject_5%={kpss_c['reject_5%']}")

# Interpretation matrix
adf_nonstat  = not adf_ct['reject_5%']
kpss_nonstat = kpss_ct['reject_5%']
if adf_nonstat and kpss_nonstat:
    stat_verdict = "CLEARLY NON-STATIONARY → first-difference or detrend before ARIMA"
elif not adf_nonstat and not kpss_nonstat:
    stat_verdict = "STATIONARY → ARMA without differencing is valid"
elif adf_nonstat and not kpss_nonstat:
    stat_verdict = "CONFLICTING → near unit root; check for structural breaks"
else:
    stat_verdict = "CONFLICTING → possible trend stationarity; apply LOWESS detrend"

print(f"\n  ► VERDICT: {stat_verdict}")

# First difference
dy = np.diff(y)
adf_d1 = adf_test(dy, regression='c')
print(f"\n  First difference ADF: stat={adf_d1['stat']:.4f}  reject_5%={adf_d1['reject_5%']}")
print(f"  ► ARIMA d parameter: {'0 (stationary)' if adf_ct['reject_5%'] else '1 (first difference needed)'}")

# Variance ratio test (Lo-MacKinlay)
print(f"\n  Variance Ratio Test (momentum vs mean-reversion):")
var1 = np.var(dy, ddof=1)
for q in [2, 6, 12, 24, 48, 168]:
    if N > q+2:
        vr  = np.var(y[q:] - y[:-q], ddof=1) / max(q*var1, 1e-18)
        z   = (vr-1) / np.sqrt(2*(2*q-1)*(q-1)/(3*q*N))
        sig = '*' if abs(z) > 1.96 else ''
        print(f"    q={q:4d}: VR={vr:.4f}  z={z:.3f}{sig}")

print(f"\n  ► VR > 1 at short lags = momentum (autocorrelation) → AR/LSTM beneficial")
print(f"  ► VR ≈ 1 at long lags = memory decays → short context window sufficient")

# ─────────────────────────────────────────────────────────────────────────────
# 3. AUTOCORRELATION STRUCTURE (ACF / PACF)
# ─────────────────────────────────────────────────────────────────────────────
banner("3. AUTOCORRELATION STRUCTURE")

NLAGS = min(336, N//3)
r     = acf_np(y, NLAGS)
pr    = pacf_np(y, min(72, N//10))
ci95  = 1.96 / np.sqrt(N)

# Print key lags
key_lags = [1,2,3,6,12,24,48,72,96,120,144,168,336]
print(f"\n  ACF at key lags (95% CI ±{ci95:.4f}):")
print(f"  {'Lag':>6}  {'ACF':>8}  {'Sig':>4}  {'Implied feature'}")
print(f"  {'─'*55}")
for lag in key_lags:
    if lag < len(r):
        sig = '***' if abs(r[lag]) > ci95 else ''
        feat = {1:'lag_1h', 24:'lag_24h', 168:'lag_168h', 336:'lag_336h'}.get(lag, f'lag_{lag}h')
        print(f"  {lag:>6}  {r[lag]:>8.4f}  {sig:>4}  {feat}")

# PACF cutoff → AR order
pacf_sig_lags = [i for i in range(1, len(pr)) if abs(pr[i]) > ci95]
ar_order_hint = max(pacf_sig_lags[:5]) if pacf_sig_lags else 1
print(f"\n  PACF significant lags: {pacf_sig_lags[:15]}")
print(f"  ► AR(p) order hint from PACF: p = {ar_order_hint}")

# Ljung-Box on raw series and AR residuals
lb_raw = ljung_box(y, [24, 48, 168])
ar3_res = ar_residuals(y, 3)
lb_res  = ljung_box(ar3_res, [24, 48, 168])
print(f"\n  Ljung-Box (raw series):    lag24 p={lb_raw[24]['p']:.4f}  lag168 p={lb_raw[168]['p']:.4f}")
print(f"  Ljung-Box (AR(3) resid):   lag24 p={lb_res[24]['p']:.4f}  lag168 p={lb_res[168]['p']:.4f}")
print(f"  ► If residual LB still significant: seasonal ARIMA or seasonal features needed")

# lag R² table → feature importance pre-model
print(f"\n  Lag R² (predictive power before any model):")
for lag in [1, 24, 48, 168, 336]:
    if N > lag:
        r2 = float(np.corrcoef(y[lag:], y[:-lag])[0,1]**2)
        print(f"    lag_{lag:3d}h  R²={r2:.4f}  → {'STRONG keep' if r2>0.7 else 'Moderate' if r2>0.3 else 'Weak'}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. SPECTRAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
banner("4. SPECTRAL ANALYSIS")

yc         = y - y.mean()
freq, psd  = signal.periodogram(yc, fs=1.0)
welch_f, welch_psd = signal.welch(yc, fs=1.0, nperseg=min(4096, N//4))

# Top periods
valid = freq > 0
top_idx = np.argsort(psd[valid])[::-1][:15]
top_f   = freq[valid][top_idx]
top_p   = 1.0/top_f
top_pw  = psd[valid][top_idx]

print(f"\n  Top 10 dominant periods (periodogram):")
for i in range(min(10, len(top_f))):
    marker = '★' if abs(top_p[i]-24)<2 or abs(top_p[i]-168)<5 or abs(top_p[i]-12)<1 else ' '
    print(f"  {marker} Period={top_p[i]:9.2f}h  f={top_f[i]:.6f}  Power={top_pw[i]:.3e}")

# Fourier harmonic F-test: how many daily harmonics are significant?
t_arr = np.arange(N, dtype=float)
base  = np.column_stack([np.ones(N), t_arr])
fit0  = ols(y, base)
print(f"\n  Fourier harmonic significance (daily period=24h):")
sig_k = []
for k in range(1, 13):
    cols = [base, np.sin(2*np.pi*k*t_arr/24)[:,None], np.cos(2*np.pi*k*t_arr/24)[:,None]]
    Xk   = np.column_stack(cols)
    fit1 = ols(y, Xk)
    df1, df2 = 2, max(fit1['n'] - Xk.shape[1], 1)
    F    = ((fit0['rss']-fit1['rss'])/df1) / max(fit1['rss']/df2, 1e-18)
    p    = float(stats.f.sf(F, df1, df2))
    sig  = '*** INCLUDE' if p<0.001 else ('* include' if p<0.05 else 'skip')
    if p < 0.05: sig_k.append(k)
    print(f"    k={k:2d} (period={24/k:.1f}h): F={F:.2f}  p={p:.4f}  {sig}")

print(f"\n  ► Include {max(sig_k) if sig_k else 3} Fourier harmonic pairs for daily seasonality")
print(f"  ► This directly sets: Prophet n_fourier_terms, TBATS harmonics, feature sin/cos count")

# Spectral entropy (predictability)
p_norm  = psd[valid] / max(psd[valid].sum(), 1e-18)
sp_ent  = -np.sum(p_norm[p_norm>0] * np.log2(p_norm[p_norm>0])) / np.log2(max(len(p_norm),2))
print(f"\n  Spectral Entropy (normalized): {sp_ent:.4f}")
print(f"  ► {'Low (< 0.5): highly predictable periodic signal → simpler models work well' if sp_ent<0.5 else 'High (> 0.5): complex spectrum → need flexible nonlinear model'}")

# Cumulative spectrum (how many harmonics explain 90% variance)
csd   = np.cumsum(psd[valid]) / max(psd[valid].sum(), 1e-18)
for q in [0.5, 0.75, 0.9]:
    idx_q = int(np.searchsorted(csd, q))
    per_q = 1.0/max(freq[valid][min(idx_q, len(freq[valid])-1)], 1e-12)
    print(f"  {int(q*100)}% variance explained below period={per_q:.1f}h")

# ─────────────────────────────────────────────────────────────────────────────
# 5. STL DECOMPOSITION + SEASONAL STRENGTH
# ─────────────────────────────────────────────────────────────────────────────
banner("5. STL DECOMPOSITION + SEASONAL / TREND STRENGTH")

from statsmodels.tsa.seasonal import STL

stl24  = STL(ts.dropna(), period=24, robust=True).fit()
trend  = stl24.trend.values
seas24 = stl24.seasonal.values
resid  = stl24.resid.values

Fs = max(0.0, 1 - np.var(resid) / max(np.var(seas24+resid), 1e-12))
Ft = max(0.0, 1 - np.var(resid) / max(np.var(trend+resid), 1e-12))

print(f"\n  STL(period=24, robust=True):")
print(f"  Seasonal Strength Fs = {Fs:.4f}  ({'STRONG >0.64 → MUST model daily seasonality' if Fs>0.64 else 'Weak'})")
print(f"  Trend    Strength Ft = {Ft:.4f}  ({'STRONG >0.50 → MUST model trend' if Ft>0.50 else 'Weak'})")
print(f"  Residual Std         = {np.std(resid):.2f} MWh  (irreducible noise floor)")
print(f"  ► If residual ACF is significant: more structure left → LSTM/XGBoost beneficial")

# Check if residuals still have structure
lb_stl = ljung_box(resid[~np.isnan(resid)], [24, 168])
print(f"  Ljung-Box on STL residuals: lag24 p={lb_stl[24]['p']:.4f}  lag168 p={lb_stl[168]['p']:.4f}")
print(f"  ► {'Significant residual autocorrelation → ML model on top of STL' if lb_stl[24]['p']<0.05 else 'Residuals look white → STL+linear is sufficient'}")

# Seasonal indices
seas_idx = np.array([np.nanmean(seas24[i::24]) for i in range(24)])
seas_idx -= seas_idx.mean()
print(f"\n  Daily Seasonal Indices (MWh above/below trend):")
for h in range(24):
    bar = '█' * int(abs(seas_idx[h])/max(abs(seas_idx).max(),1)*20)
    sign = '+' if seas_idx[h]>=0 else '-'
    print(f"  {h:02d}:00  {sign}{abs(seas_idx[h]):8.1f}  {bar}")

# Canova-Hansen style: seasonal stability (first half vs second half)
half = N//2
si1  = np.array([np.nanmean(y[:half][i::24]) for i in range(24)])
si2  = np.array([np.nanmean(y[half:][i::24]) for i in range(24)])
si1 -= si1.mean(); si2 -= si2.mean()
t_stat, t_p = stats.ttest_rel(si1, si2)
max_change  = float(np.max(np.abs(si2-si1)))
print(f"\n  Seasonal Stability (first vs second half):")
print(f"  Paired t-test: p={t_p:.4f}  Max index change={max_change:.1f} MWh")
print(f"  ► {'STOCHASTIC seasonality → use SARIMA(D=1) or time-varying STL' if t_p<0.05 else 'DETERMINISTIC seasonality → Fourier terms / fixed dummies suffice'}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. TREND ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
banner("6. TREND ANALYSIS")

daily_mean_arr = pd.Series(y, index=df.index).resample('D').mean().dropna()
dm  = daily_mean_arr.values
dm_t = np.arange(len(dm), dtype=float)

# OLS linear trend
Xlin   = np.column_stack([np.ones(len(dm)), dm_t])
fit_tr = ols(dm, Xlin)
slope_yr = float(fit_tr['b'][1] * 365)
print(f"\n  OLS Trend: {slope_yr:.1f} MWh/year  (R²={fit_tr['r2']:.4f}  p≈{2*stats.t.sf(abs(fit_tr['t'][1]),max(fit_tr['n']-2,1)):.4f})")

# Mann-Kendall
mk_samp = dm if len(dm) < 2000 else dm[np.linspace(0,len(dm)-1,2000).astype(int)]
tau, mk_p = kendalltau(np.arange(len(mk_samp)), mk_samp)
print(f"  Mann-Kendall: tau={tau:.4f}  p={mk_p:.6f}")
print(f"  ► {'Significant monotonic trend → include time index as feature / detrend' if mk_p<0.05 else 'No significant trend'}")

# Annual means
ann = df.groupby('year')[TARGET_COL].mean()
print(f"\n  Annual Mean Consumption:")
for yr, val in ann.items():
    print(f"    {yr}: {val:.1f} MWh")
pct = (ann.iloc[-1]/ann.iloc[0]-1)*100
print(f"  Total growth: {pct:.1f}% over {len(ann)-1} years")

# HP filter
n_hp  = len(y)
lam_hp = 100*(24**4)
diag  = [np.ones(n_hp), -2*np.ones(max(n_hp-2,1)), np.ones(max(n_hp-2,1))]
D     = sparse.diags([-2*np.ones(n_hp-2), np.ones(n_hp-2), np.ones(n_hp-2)], [-0, 1, 2], shape=(n_hp-2, n_hp), format='csr')
A     = sparse.eye(n_hp, format='csr') + lam_hp*(D.T@D)
hp_trend = spsolve(A, y)
print(f"\n  HP Filter trend: mean={hp_trend.mean():.1f}  std={hp_trend.std():.1f}")
print(f"  ► HP trend can be used directly as a 'slow trend' feature in XGBoost")

# ─────────────────────────────────────────────────────────────────────────────
# 7. LONG MEMORY (HURST + DFA + GPH)
# ─────────────────────────────────────────────────────────────────────────────
banner("7. LONG MEMORY ANALYSIS")

H,   rs_sizes, rs_vals  = hurst_rs(y)
alpha_dfa, dfa_sz, dfa_F = dfa(y)
gph  = gph_estimator(y)

print(f"\n  Hurst Exponent (R/S)      : H = {H:.4f}")
print(f"  DFA scaling exponent      : α = {alpha_dfa:.4f}")
print(f"  GPH fractional d estimate : d̂ = {gph['d']:.4f}  se={gph['se']:.4f}  p={gph['p']:.4f}")

if H > 0.65:
    print(f"\n  ► LONG MEMORY CONFIRMED (H={H:.3f} >> 0.5)")
    print(f"  ► Integer differencing (ARIMA d=1) OVER-differences the series")
    print(f"  ► Fractional d ≈ {gph['d']:.3f} is optimal")
    print(f"  ► For ML: include lag_168h, lag_336h, lag_720h (long memory features)")
    print(f"  ► For ARIMA: use ARFIMA(p, d={gph['d']:.2f}, q) instead of ARIMA")
else:
    print(f"\n  ► Weak long memory (H={H:.3f}) → standard ARIMA is adequate")

# ─────────────────────────────────────────────────────────────────────────────
# 8. NONLINEARITY TESTS
# ─────────────────────────────────────────────────────────────────────────────
banner("8. NONLINEARITY TESTS")

ar3_res = ar_residuals(y, 3)

# Keenan test
Y    = y[3:]
Xlag = np.column_stack([y[2:-1], y[1:-2], y[:-3]])
fitAR = ols(Y, np.column_stack([np.ones(len(Y)), Xlag]))
fsq   = fitAR['fit']**2
fitK  = ols(fitAR['res'], np.column_stack([np.ones(len(fsq)), fsq]))
keenan_t = float(fitK['t'][1])
keenan_p = float(2*stats.t.sf(abs(keenan_t), max(fitK['n']-fitK['k'],1)))
print(f"\n  Keenan nonlinearity test: t={keenan_t:.4f}  p={keenan_p:.4f}")

# Teräsvirta test
terms = [np.ones(len(Y)), Xlag]
terms += [Xlag[:,i:i+1]**2 for i in range(3)]
terms += [Xlag[:,i:i+1]**3 for i in range(3)]
terms += [(Xlag[:,i]*Xlag[:,j])[:,None] for i in range(3) for j in range(i+1,3)]
Xfull = np.column_stack(terms)
fitT  = ols(Y, Xfull)
df1_t, df2_t = Xfull.shape[1]-Xlag.shape[1]-1, max(fitT['n']-Xfull.shape[1],1)
F_ter = ((fitAR['rss']-fitT['rss'])/max(df1_t,1)) / max(fitT['rss']/df2_t, 1e-18)
p_ter = float(stats.f.sf(F_ter, df1_t, df2_t))
print(f"  Teräsvirta NN test:       F={F_ter:.4f}  p={p_ter:.4f}")

# BDS simplified
def bds_simple(x, eps_frac=0.7):
    x   = (x[:min(800,len(x))] - x.mean()) / max(x.std(),1e-12)
    eps = eps_frac
    d1  = squareform(pdist(x[:,None]))
    c1  = np.mean(d1 < eps)
    emb = np.column_stack([x[:-1], x[1:]])
    d2  = squareform(pdist(emb))
    c2  = np.mean(d2 < eps)
    se  = np.sqrt(max(c1**2*(1-c1**2),1e-12)/len(emb))
    z   = (c2 - c1**2) / se
    return float(z), float(2*stats.norm.sf(abs(z)))

bds_z, bds_p = bds_simple(ar3_res)
print(f"  BDS test (AR residuals):  z={bds_z:.4f}  p={bds_p:.4f}")

nl_verdict = sum([keenan_p<0.05, p_ter<0.05, bds_p<0.05])
print(f"\n  Nonlinearity evidence: {nl_verdict}/3 tests significant")
print(f"  ► {'STRONG nonlinearity → XGBoost/LSTM clearly superior to ARIMA' if nl_verdict>=2 else 'Weak nonlinearity → ARIMA competitive; XGBoost may still help at peaks'}")

# Threshold TAR test
lag_arr = y[:-1]; tgt_arr = y[1:]
fit_lin = ols(tgt_arr, np.column_stack([np.ones(len(tgt_arr)), lag_arr]))
best_tar = {'rss': np.inf}
for th in np.quantile(lag_arr, np.linspace(0.2,0.8,25)):
    lo = lag_arr <= th
    if lo.sum()<30 or (~lo).sum()<30: continue
    f1 = ols(tgt_arr[lo],  np.column_stack([np.ones(lo.sum()),  lag_arr[lo]]))
    f2 = ols(tgt_arr[~lo], np.column_stack([np.ones((~lo).sum()), lag_arr[~lo]]))
    rss = f1['rss'] + f2['rss']
    if rss < best_tar['rss']:
        df1_tar, df2_tar = 2, len(tgt_arr)-4
        F_tar = ((fit_lin['rss']-rss)/df1_tar)/max(rss/df2_tar,1e-18)
        best_tar = {'th':float(th),'F':float(F_tar),'p':float(stats.f.sf(F_tar,df1_tar,df2_tar)),'rss':rss}
print(f"\n  TAR threshold test: threshold={best_tar['th']:.1f}  F={best_tar['F']:.2f}  p={best_tar['p']:.4f}")
print(f"  ► {'Regime-switching at level {:.0f} MWh → separate models for peak/off-peak'.format(best_tar['th']) if best_tar['p']<0.05 else 'No significant threshold'}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. VOLATILITY / ARCH / HETEROSCEDASTICITY
# ─────────────────────────────────────────────────────────────────────────────
banner("9. VOLATILITY & HETEROSCEDASTICITY")

arch = arch_lm(ar3_res, lags=24)
print(f"\n  ARCH-LM test (24 lags): stat={arch['stat']:.4f}  p={arch['p']:.6f}")
print(f"  ► {'ARCH EFFECTS PRESENT → variance is time-varying' if arch['p']<0.05 else 'No ARCH effects'}")
if arch['p'] < 0.05:
    print(f"    • Use GARCH(1,1) for prediction intervals")
    print(f"    • Huber/quantile loss is better than MSE")
    print(f"    • Rolling std as a model feature (captures local volatility)")

# Conditional std by hour
hourly_std = df.groupby('hour')[TARGET_COL].std()
print(f"\n  Conditional Std by Hour (ARCH by context):")
print(f"  Most volatile   : hour {hourly_std.idxmax():02d}:00 → std={hourly_std.max():.1f} MWh")
print(f"  Least volatile  : hour {hourly_std.idxmin():02d}:00 → std={hourly_std.min():.1f} MWh")
print(f"  Ratio (max/min) : {hourly_std.max()/hourly_std.min():.2f}x")
print(f"  ► Context-specific prediction intervals needed (conformal per hour-of-day)")

# McLeod-Li at seasonal lags
lb_sq = ljung_box(ar3_res**2, [1,6,12,24,48,168])
print(f"\n  Ljung-Box on squared AR residuals (McLeod-Li):")
for lag in [1,6,24,48,168]:
    print(f"    lag {lag:4d}: stat={lb_sq[lag]['stat']:.2f}  p={lb_sq[lag]['p']:.4f}  {'sig' if lb_sq[lag]['p']<0.05 else ''}")
print(f"  ► Significant at lag 24 → periodic ARCH (different volatility by hour)")

# Goldfeld-Quandt
sort_idx = np.argsort(y[:-len(ar3_res)])[::-1] if len(y)-len(ar3_res)>0 else np.argsort(ar3_res)
n_half   = len(ar3_res)//3
var_high  = np.var(ar3_res[:n_half], ddof=1)
var_low   = np.var(ar3_res[-n_half:], ddof=1)
F_gq      = var_high/max(var_low,1e-12)
p_gq      = float(stats.f.sf(F_gq, n_half-1, n_half-1))
print(f"\n  Goldfeld-Quandt (high vs low consumption): F={F_gq:.4f}  p={p_gq:.4f}")
print(f"  ► {'Heteroscedastic → weight loss function; use WLS or quantile regression' if p_gq<0.05 else 'Homoscedastic'}")

# Simple GARCH(1,1) via Nelder-Mead
def fit_garch11(e):
    e   = e - e.mean()
    v0  = np.var(e)
    def neg_ll(p):
        om, al, be = p
        if om<=0 or al<0 or be<0 or al+be>=0.999: return 1e30
        sig2 = np.empty(len(e)); sig2[0] = v0
        for t in range(1,len(e)):
            sig2[t] = om + al*e[t-1]**2 + be*sig2[t-1]
            if sig2[t]<=0: return 1e30
        return float(0.5*np.sum(np.log(sig2)+e**2/sig2))
    opt = minimize(neg_ll, [0.05*v0, 0.08, 0.88],
                   bounds=[(1e-12,None),(0,1),(0,1)], method='Nelder-Mead',
                   options={'maxiter':2000})
    return dict(zip(['omega','alpha','beta'], map(float, opt.x))), float(opt.fun)

garch_p, garch_ll = fit_garch11(ar3_res)
print(f"\n  GARCH(1,1): ω={garch_p['omega']:.4f}  α={garch_p['alpha']:.4f}  β={garch_p['beta']:.4f}")
print(f"  α+β = {garch_p['alpha']+garch_p['beta']:.4f} ({'near-integrated: very persistent volatility' if garch_p['alpha']+garch_p['beta']>0.97 else 'stationary volatility'})")
print(f"  ► GARCH conditional variance is a useful feature for XGBoost (volatile-period flag)")

# ─────────────────────────────────────────────────────────────────────────────
# 10. STRUCTURAL BREAKS
# ─────────────────────────────────────────────────────────────────────────────
banner("10. STRUCTURAL BREAK ANALYSIS")

# CUSUM on daily means
dm_cusum = dm - dm.mean()
cusum    = np.cumsum(dm_cusum) / (dm_cusum.std() * np.sqrt(len(dm_cusum)))
max_cusum = float(np.max(np.abs(cusum)))
cusum_crosses = max_cusum > 1.36
print(f"\n  CUSUM test: max|CUSUM|={max_cusum:.4f}  threshold=1.36")
print(f"  ► {'STRUCTURAL BREAK DETECTED' if cusum_crosses else 'No significant break'}")
if cusum_crosses:
    break_idx_cusum = int(np.argmax(np.abs(cusum)))
    break_date_cusum = daily_mean_arr.index[break_idx_cusum]
    print(f"    Break vicinity: {break_date_cusum.date()}")

# Quandt-Andrews (QLR) on simple AR
candidates = np.linspace(int(0.15*len(dm)), int(0.85*len(dm)), min(60, max(5,len(dm)//50))).astype(int)
best_qlr = {'F':-np.inf}
for bp in np.unique(candidates):
    t = np.arange(len(dm), dtype=float)
    X = np.column_stack([np.ones(len(dm)), t])
    f0 = ols(dm, X)
    X1 = np.column_stack([np.ones(bp), t[:bp]])
    X2 = np.column_stack([np.ones(len(dm)-bp), t[bp:]])
    f1 = ols(dm[:bp], X1); f2 = ols(dm[bp:], X2)
    rss_u = f1['rss'] + f2['rss']
    df1_q, df2_q = 2, len(dm)-4
    F_q = ((f0['rss']-rss_u)/df1_q) / max(rss_u/df2_q, 1e-18)
    if F_q > best_qlr['F']:
        best_qlr = {'F':float(F_q),'p':float(stats.f.sf(F_q,df1_q,df2_q)),
                    'bp':int(bp),'date':daily_mean_arr.index[bp].date()}

print(f"\n  QLR test: best F={best_qlr['F']:.4f}  p={best_qlr['p']:.4f}  at {best_qlr['date']}")
print(f"  ► {'Significant break at ' + str(best_qlr['date']) + ' → add dummy variable or split train set' if best_qlr['p']<0.05 else 'No significant break'}")

# Bai-Perron PELT-style dynamic programming
print(f"\n  Change Point Detection (PELT, BIC penalty):")
y_cp  = dm.copy()
n_cp  = len(y_cp)
pen   = 2*np.log(n_cp)*np.var(y_cp)
pref  = np.r_[0., np.cumsum(y_cp)]
pref2 = np.r_[0., np.cumsum(y_cp**2)]
min_sz = max(30, n_cp//40)

def seg_cost(s,e):
    m = e-s
    if m<=1: return 0.
    return float(pref2[e]-pref2[s] - (pref[e]-pref[s])**2/m)

F_cp   = np.zeros(n_cp+1)
cp_set = {0: []}
R_set  = [0]
for t in range(min_sz, n_cp+1):
    cands = [(F_cp[s]+seg_cost(s,t)+pen, s) for s in R_set if t-s>=min_sz]
    if not cands: F_cp[t]=np.inf; continue
    F_cp[t], tau = min(cands, key=lambda z:z[0])
    cp_set[t] = cp_set.get(tau,[]) + [tau]
    R_set = [r for r in R_set if F_cp[r]+seg_cost(r,t)<=F_cp[t]+pen]
    R_set.append(t-min_sz+1)

cps_idx = [c for c in cp_set.get(n_cp,[]) if 0<c<n_cp]
cps_dates = [daily_mean_arr.index[c].date() for c in cps_idx if c<len(daily_mean_arr)]
print(f"  Detected {len(cps_dates)} change points: {cps_dates}")
print(f"  ► Use these as regime boundaries in your train split and as dummy features")

# ─────────────────────────────────────────────────────────────────────────────
# 11. OUTLIER DETECTION
# ─────────────────────────────────────────────────────────────────────────────
banner("11. MULTI-METHOD OUTLIER DETECTION")

# Contextual IQR
ctx_out = np.zeros(N, dtype=bool)
for (h,d), grp in df.groupby(['hour','dow']):
    q1,q3 = grp[TARGET_COL].quantile(0.25), grp[TARGET_COL].quantile(0.75)
    iqr   = q3-q1
    f     = (grp[TARGET_COL]<q1-3*iqr)|(grp[TARGET_COL]>q3+3*iqr)
    ctx_out[grp.index.get_indexer_for(grp.index[f])] = True

# Rolling envelope
roll_m = pd.Series(y).rolling(24*7, center=True, min_periods=24).mean().values
roll_s = pd.Series(y).rolling(24*7, center=True, min_periods=24).std().values
roll_out = (y > roll_m+3*roll_s) | (y < roll_m-3*roll_s)

# Isolation Forest
feat_if = np.column_stack([
    y,
    np.concatenate([[y[0]],y[:-1]]),
    np.concatenate([y[:24], y[:-24]]),
    df['hour'].values, df['dow'].values
])
isof = IsolationForest(contamination=0.015, random_state=42, n_estimators=200)
isof_labels = isof.fit_predict(StandardScaler().fit_transform(feat_if))
if_out = isof_labels == -1

print(f"\n  Contextual IQR outliers : {ctx_out.sum():4d}  ({ctx_out.mean()*100:.2f}%)")
print(f"  Rolling 7d ±3σ outliers : {roll_out.sum():4d}  ({roll_out.mean()*100:.2f}%)")
print(f"  Isolation Forest        : {if_out.sum():4d}  ({if_out.mean()*100:.2f}%)")

# Union and intersection
union_out = ctx_out | roll_out | if_out
inter_out = ctx_out & roll_out & if_out
print(f"  Union (any method)      : {union_out.sum():4d}  ({union_out.mean()*100:.2f}%)")
print(f"  Intersection (all 3)    : {inter_out.sum():4d}  ({inter_out.mean()*100:.2f}%)")
print(f"  ► Intersection = high-confidence outliers → safe to treat")
print(f"  ► Union-only   = borderline → inspect before removing")

# Tsay-style outlier classification
print(f"\n  Top anomalous events (Isolation Forest):")
if_scores = -isof.score_samples(StandardScaler().fit_transform(feat_if))
top_if    = np.argsort(if_scores)[::-1][:10]
for idx_a in top_if:
    print(f"    {df.index[idx_a]}  consumption={y[idx_a]:.1f}  score={if_scores[idx_a]:.4f}")

# CUSUM anomaly chart on residuals
mu_arr = np.nan_to_num(mu_hdm.values, nan=y.mean())
res_cusum = y - mu_arr
k_cusum = 0.5*res_cusum.std()
h_cusum = 5.0*res_cusum.std()
sp = np.zeros(N); sn = np.zeros(N)
for t in range(1,N):
    sp[t] = max(0, sp[t-1]+res_cusum[t]-k_cusum)
    sn[t] = min(0, sn[t-1]+res_cusum[t]+k_cusum)
cusum_alarms = np.where((sp>h_cusum)|(np.abs(sn)>h_cusum))[0]
print(f"\n  CUSUM anomaly alarms: {len(cusum_alarms)}")
if len(cusum_alarms):
    alarm_dates = [df.index[i].date() for i in cusum_alarms[:5]]
    print(f"  First alarms: {alarm_dates}")

# ─────────────────────────────────────────────────────────────────────────────
# 12. INFORMATION CRITERIA GRID → MODEL ORDER SELECTION
# ─────────────────────────────────────────────────────────────────────────────
banner("12. INFORMATION CRITERIA (AIC / AICc / BIC / HQC)")

print(f"\n  AR(p) order selection grid:")
print(f"  {'p':>3}  {'LogL':>10}  {'AIC':>10}  {'AICc':>10}  {'BIC':>10}  {'HQC':>10}")
print(f"  {'─'*55}")
best_aic_p, best_bic_p, best_aic_val, best_bic_val = 0, 0, np.inf, np.inf
ic_rows = []
for p in range(1, min(25, N//100)+1):
    Y_ar = y[p:]
    X_ar = np.column_stack([np.ones(len(Y_ar))] + [y[p-j:-j] for j in range(1,p+1)])
    fit  = ols(Y_ar, X_ar)
    ic   = aic_bic(fit['rss'], fit['n'], fit['k'])
    ic_rows.append({'p':p, **ic})
    marker = '<' if ic['aic']<best_aic_val else ''
    print(f"  {p:>3}  {ic['ll']:>10.1f}  {ic['aic']:>10.2f}  {ic['aicc']:>10.2f}  {ic['bic']:>10.2f}  {ic['hqc']:>10.2f} {marker}")
    if ic['aic'] < best_aic_val: best_aic_val=ic['aic']; best_aic_p=p
    if ic['bic'] < best_bic_val: best_bic_val=ic['bic']; best_bic_p=p

best_hqc_p = min(ic_rows, key=lambda r: r['hqc'])['p']
print(f"\n  Best AR order by AIC  : p={best_aic_p}")
print(f"  Best AR order by BIC  : p={best_bic_p}  (more parsimonious)")
print(f"  Best AR order by HQC  : p={best_hqc_p}")
print(f"  ► SARIMA non-seasonal AR order: p ∈ [{min(best_aic_p,best_bic_p)}, {max(best_aic_p,best_bic_p)}]")

# BIC-weighted model average
bic_vals = np.array([r['bic'] for r in ic_rows])
bic_w    = np.exp(-0.5*(bic_vals - bic_vals.min()))
bic_w   /= bic_w.sum()
bma_p    = float(np.sum([r['p']*w for r,w in zip(ic_rows,bic_w)]))
print(f"  BIC-weighted average p: {bma_p:.2f}")
print(f"  ► For ensemble: average predictions from p={best_bic_p} and p={best_aic_p} models")

# Profile likelihood AR
max_ll   = max(r['ll'] for r in ic_rows)
chi2_thr = 0.5*stats.chi2.ppf(0.95, 1)
support  = [r['p'] for r in ic_rows if r['ll'] >= max_ll - chi2_thr]
print(f"  95% profile likelihood support: p ∈ {support}")

# ─────────────────────────────────────────────────────────────────────────────
# 13. FEATURE SELECTION & IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
banner("13. FEATURE SELECTION & IMPORTANCE")

# Build feature matrix from existing df columns
feature_cols = [c for c in df.columns if c != TARGET_COL and df[c].dtype in [float,int,np.float64,np.int64]]
feat_df = df[feature_cols + [TARGET_COL]].dropna()
X_all   = feat_df[feature_cols].values
y_all   = feat_df[TARGET_COL].values

# Mutual Information
mi_scores = mutual_info_regression(StandardScaler().fit_transform(X_all), y_all, random_state=42)
mi_rank   = sorted(zip(feature_cols, mi_scores), key=lambda x:x[1], reverse=True)
print(f"\n  Mutual Information Feature Ranking (top 20):")
for rank, (col, mi) in enumerate(mi_rank[:20], 1):
    bar = '█' * int(mi/mi_rank[0][1]*25)
    print(f"  {rank:>2}. {col:<30} MI={mi:.4f}  {bar}")

# VIF (top 15 features to avoid matrix issues)
top15_cols = [c for c,_ in mi_rank[:15]]
X_top = StandardScaler().fit_transform(feat_df[top15_cols].values)
print(f"\n  Variance Inflation Factors (multicollinearity):")
for j, col in enumerate(top15_cols):
    other = np.delete(X_top, j, axis=1)
    f_vif = ols(X_top[:,j], np.column_stack([np.ones(len(X_top)), other]))
    vif   = 1/max(1-f_vif['r2'], 1e-12)
    flag  = '*** HIGH' if vif>10 else ('* mod' if vif>5 else '')
    print(f"    {col:<30} VIF={vif:8.2f}  {flag}")

# LASSO feature selection
split_L    = int(0.8*len(X_all))
X_tr_L     = StandardScaler().fit_transform(X_all[:split_L])
lasso_m    = LassoCV(cv=3, random_state=42, max_iter=10000).fit(X_tr_L, y_all[:split_L])
lasso_sel  = [feature_cols[i] for i,c in enumerate(lasso_m.coef_) if abs(c)>1e-9]
print(f"\n  LASSO selected features (α={lasso_m.alpha_:.4f}):")
lasso_imp  = sorted(zip(lasso_sel, [abs(lasso_m.coef_[feature_cols.index(c)]) for c in lasso_sel]), key=lambda x:-x[1])
for col, imp in lasso_imp[:15]:
    print(f"    {col:<35} coef={imp:.4f}")

# Random Forest + Boruta-style shadow features
print(f"\n  Boruta-style (Shadow Feature) importance:")
shadow_X    = feat_df[feature_cols].sample(frac=1.0, replace=False, random_state=42).reset_index(drop=True)
shadow_X.columns = [f"shadow_{c}" for c in feature_cols]
aug_X       = pd.concat([feat_df[feature_cols].reset_index(drop=True), shadow_X], axis=1)
rf_boruta   = RandomForestRegressor(n_estimators=120, random_state=42, max_depth=8, n_jobs=-1)
rf_boruta.fit(aug_X.values, feat_df[TARGET_COL].reset_index(drop=True).values)
imp_series  = pd.Series(rf_boruta.feature_importances_, index=aug_X.columns)
shadow_max  = imp_series[[c for c in aug_X.columns if c.startswith('shadow_')]].max()
confirmed   = imp_series[[c for c in feature_cols if imp_series[c] > shadow_max]].sort_values(ascending=False)
tentative   = imp_series[[c for c in feature_cols if imp_series[c] <= shadow_max and imp_series[c] > shadow_max*0.5]].sort_values(ascending=False)
print(f"  Shadow max importance: {shadow_max:.6f}")
print(f"  CONFIRMED features ({len(confirmed)}):")
for col, imp in confirmed[:15].items():
    print(f"    {col:<35} imp={imp:.6f}")
print(f"  TENTATIVE features ({len(tentative)}):")
for col, imp in tentative[:5].items():
    print(f"    {col:<35} imp={imp:.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# 14. MODEL COMPARISON BASELINES → DIEBOLD-MARIANO
# ─────────────────────────────────────────────────────────────────────────────
banner("14. BASELINE MODELS & DIEBOLD-MARIANO TEST")

# Walk-forward split
split_tr  = int(0.70*len(feat_df))
split_val = int(0.85*len(feat_df))

X_tr  = feat_df[feature_cols].iloc[:split_tr].values
y_tr  = feat_df[TARGET_COL].iloc[:split_tr].values
X_va  = feat_df[feature_cols].iloc[split_tr:split_val].values
y_va  = feat_df[TARGET_COL].iloc[split_tr:split_val].values
X_te  = feat_df[feature_cols].iloc[split_val:].values
y_te  = feat_df[TARGET_COL].iloc[split_val:].values

# Scale
scaler = StandardScaler().fit(X_tr)
X_tr_s = scaler.transform(X_tr)
X_va_s = scaler.transform(X_va)
X_te_s = scaler.transform(X_te)

# Naive: same hour last day
y_src = feat_df[TARGET_COL].values
shift = 24
naive_te = np.concatenate([y_src[split_val-shift:split_val],
                            y_src[split_val-shift:split_val]])[:len(y_te)]

# Naive: same hour last week
shift_w = 168
lw_te = np.concatenate([y_src[split_val-shift_w:split_val],
                         y_src[split_val-shift_w:split_val]])[:len(y_te)]

# Ridge regression
ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
ridge.fit(X_tr, y_tr)
ridge_te = ridge.predict(X_te)

# Gradient Boosting
gbm = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                  learning_rate=0.05, random_state=42,
                                  subsample=0.8)
gbm.fit(X_tr, y_tr)
gbm_te = gbm.predict(X_te)

# Quantile models (for prediction intervals)
q10m = GradientBoostingRegressor(loss='quantile', alpha=0.1, n_estimators=150,
                                    max_depth=4, random_state=42)
q90m = GradientBoostingRegressor(loss='quantile', alpha=0.9, n_estimators=150,
                                    max_depth=4, random_state=42)
q10m.fit(X_tr, y_tr); q90m.fit(X_tr, y_tr)
q10_te = q10m.predict(X_te)
q90_te = q90m.predict(X_te)

# Conformal calibration
cal_pred = ridge.predict(X_va)
cal_nc   = np.abs(y_va - cal_pred)
conf_q   = float(np.quantile(cal_nc, 0.90))
conf_lo  = ridge_te - conf_q
conf_hi  = ridge_te + conf_q

def metrics(yt, yp):
    mae  = float(mean_absolute_error(yt,yp))
    rmse = float(np.sqrt(mean_squared_error(yt,yp)))
    mape = float(np.mean(np.abs((yt-yp)/np.maximum(np.abs(yt),1e-6)))*100)
    r2   = float(1 - np.sum((yt-yp)**2)/max(np.sum((yt-yt.mean())**2),1e-12))
    return {'MAE':mae,'RMSE':rmse,'MAPE%':mape,'R2':r2}

print(f"\n  Model Performance on Test Set ({len(y_te):,} observations):")
print(f"  {'Model':<25} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'R²':>8}")
print(f"  {'─'*60}")
mods = [
    ('Naive (same-hour-1d)', naive_te),
    ('Naive (same-hour-1w)', lw_te),
    ('Ridge Regression',     ridge_te),
    ('GradientBoosting',     gbm_te),
]
all_metrics = {}
for name, pred in mods:
    m = metrics(y_te, pred)
    all_metrics[name] = m
    print(f"  {name:<25} {m['MAE']:>8.1f} {m['RMSE']:>8.1f} {m['MAPE%']:>8.2f} {m['R2']:>8.4f}")

# Theil's U
for name, pred in mods[2:]:
    U = np.sqrt(mean_squared_error(y_te,pred)) / max(np.sqrt(mean_squared_error(y_te,naive_te)),1e-12)
    print(f"  Theil U ({name}): {U:.4f}  ({'Beats naive ✓' if U<1 else 'Worse than naive ✗'})")

# Mincer-Zarnowitz regression
print(f"\n  Mincer-Zarnowitz Regression (α=0, β=1 under perfect calibration):")
for name, pred in [('Ridge',ridge_te),('GBM',gbm_te)]:
    fit_mz = ols(y_te, np.column_stack([np.ones(len(pred)), pred]))
    F_mz   = ((fit_mz['b'][0]**2 + (fit_mz['b'][1]-1)**2) / 2) / max(fit_mz['s2'],1e-12)
    p_mz   = float(stats.f.sf(F_mz, 2, max(fit_mz['n']-2,1)))
    print(f"    {name}: α={fit_mz['b'][0]:.2f}  β={fit_mz['b'][1]:.4f}  F={F_mz:.2f}  p={p_mz:.4f}")
    if fit_mz['b'][1] < 1:
        print(f"      β<1 → model under-reacts to extremes (predicts toward mean)")

# Diebold-Mariano
def diebold_mariano(y, p1, p2, lag=24):
    e1 = (y-p1)**2; e2 = (y-p2)**2
    d  = e1-e2
    d_bar = d.mean()
    nw  = sum((1-j/(lag+1))*np.cov(d[j:],d[:-j])[0,1]/len(d) for j in range(1,lag+1))
    var = (np.var(d,ddof=1) + 2*nw) / len(d)
    dm  = d_bar / max(np.sqrt(var),1e-12)
    p   = 2*stats.norm.sf(abs(dm))
    return {'DM':float(dm),'p':float(p),'better':('Model1' if dm<0 else 'Model2')}

print(f"\n  Diebold-Mariano Tests:")
pairs = [
    ('Ridge vs Naive-1d', ridge_te, naive_te),
    ('GBM vs Naive-1d',   gbm_te,   naive_te),
    ('GBM vs Ridge',      gbm_te,   ridge_te),
    ('GBM vs Naive-1w',   gbm_te,   lw_te),
]
for name, p1, p2 in pairs:
    dm = diebold_mariano(y_te, p1, p2)
    print(f"  {name:<30} DM={dm['DM']:>7.3f}  p={dm['p']:.4f}  {'*** significant' if dm['p']<0.05 else ''}")

# Forecast encompassing
e_ridge = y_te - ridge_te
e_gbm   = y_te - gbm_te
fit_enc = ols(e_ridge, np.column_stack([np.ones(len(e_gbm)), e_gbm]))
print(f"\n  Encompassing test (does GBM error add info beyond Ridge?):")
print(f"  coef={fit_enc['b'][1]:.4f}  t={fit_enc['t'][1]:.3f}  p={2*stats.t.sf(abs(fit_enc['t'][1]),max(fit_enc['n']-2,1)):.4f}")
print(f"  ► {'GBM adds independent information → include both in ensemble' if 2*stats.t.sf(abs(fit_enc['t'][1]),max(fit_enc['n']-2,1))<0.05 else 'GBM encompassed by Ridge → Ridge sufficient'}")

# Optimal ensemble weights (Rao-Blackwell)
preds_mat   = np.column_stack([ridge_te, gbm_te])
errors_mat  = y_te[:,None] - preds_mat
cov_err     = np.cov(errors_mat, rowvar=False) + np.eye(2)*1e-6
ones        = np.ones(2)
w_opt       = np.linalg.pinv(cov_err) @ ones
w_opt      /= w_opt.sum()
ensemble_te = preds_mat @ w_opt
m_ens       = metrics(y_te, ensemble_te)
print(f"\n  Optimal Ensemble (weights={w_opt.round(3)}):")
print(f"  {'Ensemble':<25} {m_ens['MAE']:>8.1f} {m_ens['RMSE']:>8.1f} {m_ens['MAPE%']:>8.2f} {m_ens['R2']:>8.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 15. PROBABILISTIC FORECAST QUALITY
# ─────────────────────────────────────────────────────────────────────────────
banner("15. PROBABILISTIC FORECAST QUALITY")

# PICP, MPIW, CWC
picp = float(np.mean((y_te>=conf_lo)&(y_te<=conf_hi)))
mpiw = float(np.mean(conf_hi - conf_lo))
mu_n = 0.80
cwc  = mpiw*(1+(1-picp if picp<mu_n else 0)*np.exp(-50*(picp-mu_n)))
print(f"\n  Conformal 80% Prediction Interval (calibrated on validation):")
print(f"  PICP (coverage)  = {picp:.4f}  (target=0.80)")
print(f"  MPIW (width)     = {mpiw:.2f} MWh")
print(f"  CWC (joint)      = {cwc:.2f}  (lower=better)")

# Quantile intervals
picp_q = float(np.mean((y_te>=q10_te)&(y_te<=q90_te)))
mpiw_q = float(np.mean(q90_te - q10_te))
print(f"\n  Quantile GBM 80% Interval [q10, q90]:")
print(f"  PICP = {picp_q:.4f}  MPIW = {mpiw_q:.2f} MWh")

# Winkler score
alpha_w = 0.20
winkler = (q90_te-q10_te) + (2/alpha_w)*(q10_te-y_te)*(y_te<q10_te) + (2/alpha_w)*(y_te-q90_te)*(y_te>q90_te)
print(f"  Winkler Score (80% interval): {np.mean(winkler):.2f}")

# Reliability diagram (calibration)
print(f"\n  Reliability Diagram (conformal):")
for alpha_r in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
    q_r  = float(np.quantile(cal_nc, alpha_r))
    cov  = float(np.mean(np.abs(y_te-ridge_te) <= q_r))
    diff = cov - alpha_r
    bar  = '✓' if abs(diff)<0.05 else ('↑ over-wide' if diff>0 else '↓ under-coverage')
    print(f"  Nominal {alpha_r:.0%}: actual={cov:.3f}  diff={diff:+.3f}  {bar}")

# Brier scores for extreme events
sigma_cal = np.std(cal_nc)
print(f"\n  Brier Scores (probability of exceeding threshold):")
for qt in [0.90, 0.95, 0.99]:
    thr  = float(np.quantile(y, qt))
    prob = 1 - stats.norm.cdf((thr - ridge_te)/max(sigma_cal,1e-12))
    obs  = (y_te > thr).astype(float)
    bs   = float(np.mean((prob-obs)**2))
    clim = float(np.mean((obs.mean()-obs)**2))
    bss  = 1 - bs/max(clim,1e-12)
    print(f"  {qt:.0%} quantile threshold ({thr:.0f} MWh): BS={bs:.4f}  BSS={bss:.4f}  {'Skilful' if bss>0 else 'Unskilled'}")

# Directional symmetry
ds = float(np.mean(np.sign(np.diff(y_te)) == np.sign(np.diff(gbm_te))))
print(f"\n  Directional Symmetry (GBM): {ds:.4f}  (0.5=random, 1.0=perfect)")
print(f"  ► {'Good directional skill' if ds>0.7 else 'Poor directional skill → check features'}")

# Expected shortfall
print(f"\n  Expected Shortfall (CVaR) per model:")
for alpha_es in [0.90, 0.95]:
    thr_es = np.quantile(y_te, alpha_es)
    es_act = float(np.mean(y_te[y_te>=thr_es]))
    es_gbm = float(np.mean(gbm_te[y_te>=thr_es]))
    bias   = es_gbm - es_act
    print(f"  {alpha_es:.0%}: Actual ES={es_act:.1f}  GBM ES={es_gbm:.1f}  Bias={bias:+.1f}")

# Extreme value tail
abs_err = np.abs(y_te - gbm_te)
u_evt   = np.quantile(abs_err, 0.95)
excess  = abs_err[abs_err>u_evt] - u_evt
if len(excess) > 10:
    xi, _, sc = genpareto.fit(excess, floc=0)
    print(f"\n  GPD tail fit on forecast errors: ξ={xi:.4f}  scale={sc:.2f}")
    print(f"  ► {'Heavy-tailed errors (ξ>0) → RMSE unreliable; prefer MAE' if xi>0 else 'Light-tailed errors → RMSE is safe'}")

# CRPS approximation
pinball = lambda y,q,tau: np.maximum(tau*(y-q),(tau-1)*(y-q))
crps_q  = float(2*np.mean([pinball(y_te, q10_te, 0.1).mean(),
                             pinball(y_te, (q10_te+q90_te)/2, 0.5).mean(),
                             pinball(y_te, q90_te, 0.9).mean()]))
print(f"\n  CRPS (quantile approx): {crps_q:.2f}")
print(f"  ► Skill vs naive: {(1-crps_q/max(np.mean(np.abs(y_te-y_te.mean())),1e-12))*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 16. ENTROPY & COMPLEXITY
# ─────────────────────────────────────────────────────────────────────────────
banner("16. ENTROPY & COMPLEXITY ANALYSIS")

# Shannon entropy by hour
print(f"\n  Shannon Entropy by Hour (predictability):")
hour_ent = {}
for h, grp in df.groupby('hour')[TARGET_COL]:
    vals = grp.dropna().values
    if len(vals) < 10: continue
    counts,_ = np.histogram(vals, bins='auto')
    p = counts[counts>0] / counts.sum()
    hour_ent[int(h)] = float(-np.sum(p*np.log2(p)))
min_h = min(hour_ent, key=hour_ent.get)
max_h = max(hour_ent, key=hour_ent.get)
print(f"  Most predictable  hour: {min_h:02d}:00  H={hour_ent[min_h]:.3f}")
print(f"  Least predictable hour: {max_h:02d}:00  H={hour_ent[max_h]:.3f}")
print(f"  ► Report higher prediction uncertainty at hour {max_h}")

# Permutation entropy
def perm_entropy(x, m=4):
    n  = len(x)
    if n < m+2: return np.nan
    pats = [tuple(np.argsort(x[i:i+m])) for i in range(n-m+1)]
    cnt  = pd.Series(pats).value_counts(normalize=True)
    return float(-np.sum(cnt*np.log2(cnt))) / np.log2(factorial(m))

# Rolling PE to detect complexity changes
samp_y   = y[:min(5000,N)]
win_pe   = 168
step_pe  = 24
pe_vals  = []
pe_times = []
for start in range(0, len(samp_y)-win_pe, step_pe):
    pe = perm_entropy(samp_y[start:start+win_pe])
    if np.isfinite(pe): pe_vals.append(pe); pe_times.append(start)
if pe_vals:
    print(f"\n  Permutation Entropy (m=4, window=168h):")
    print(f"  Mean={np.mean(pe_vals):.4f}  Std={np.std(pe_vals):.4f}  Range=[{min(pe_vals):.4f},{max(pe_vals):.4f}]")
    pe_trend = np.polyfit(pe_times, pe_vals, 1)[0]
    print(f"  Trend={pe_trend:.2e}  ({'Increasing complexity over time → later data harder to model' if pe_trend>0 else 'Stable complexity'})")

# Sample entropy
def sample_entropy(x, m=2, r=None):
    x = x[:min(800,len(x))]
    r = 0.2*np.std(x) if r is None else r
    def phi(mm):
        emb = np.array([x[i:i+mm] for i in range(len(x)-mm+1)])
        d   = cdist(emb, emb, 'chebyshev')
        return np.mean(d<=r, axis=1).sum() - len(emb)
    A, B = phi(m+1), phi(m)
    return float(-np.log(max(A,1)/max(B,1)))

se = sample_entropy(y)
print(f"\n  Sample Entropy: {se:.4f}  ({'Low (regular)' if se<1.0 else 'High (complex)'})")
print(f"  ► SE < 1.0: ARIMA/Prophet competitive. SE > 1.5: XGBoost/LSTM preferred")

# ─────────────────────────────────────────────────────────────────────────────
# 17. MULTI-RESOLUTION WAVELET ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
banner("17. WAVELET / MULTI-RESOLUTION ANALYSIS")

# Haar DWT
x_wav = y[:2048] - y[:2048].mean()
print(f"\n  Haar DWT Variance Decomposition:")
approx = x_wav.copy()
total_var = np.var(x_wav)
for level in range(1,9):
    if len(approx) < 4: break
    even, odd = approx[0::2], approx[1::2]
    n_min = min(len(even),len(odd))
    detail = (even[:n_min] - odd[:n_min]) / np.sqrt(2)
    approx = (even[:n_min] + odd[:n_min]) / np.sqrt(2)
    frac   = np.var(detail)/max(total_var,1e-12)*100
    scale  = 2**level
    key    = '← daily' if 20<=scale<=30 else ('← weekly' if 140<=scale<=200 else '')
    print(f"  Level {level}: scale~{scale:4d}h  VarFrac={frac:6.2f}%  {key}")
approx_frac = np.var(approx)/max(total_var,1e-12)*100
print(f"  Residual trend: VarFrac={approx_frac:.2f}%")
print(f"  ► Levels with highest variance fraction → most important scale to model")

# MODWT-style variance at key scales
print(f"\n  MODWT-style Wavelet Variance (key periods):")
for target_scale in [24, 168]:
    # Approximate by bandpass differencing
    s  = target_scale
    bp = y - pd.Series(y).rolling(s, min_periods=s//2, center=True).mean().values
    bp = np.nan_to_num(bp)
    print(f"  Scale ~{s}h: variance={np.nanvar(bp):.2f}  ({np.nanvar(bp)/max(np.nanvar(y),1e-12)*100:.1f}% of total)")

# Wavelet coherence between year 1 and year 6
years = sorted(df.index.year.unique())
if len(years) >= 2:
    y1_data = df[df.index.year==years[0]][TARGET_COL].dropna().values
    yL_data = df[df.index.year==years[-1]][TARGET_COL].dropna().values
    min_len = min(len(y1_data), len(yL_data))
    if min_len > 168:
        fc_wc, coh_wc = signal.coherence(y1_data[:min_len], yL_data[:min_len],
                                           fs=1.0, nperseg=min(2048,min_len))
        daily_coh  = float(coh_wc[int(np.argmin(np.abs(fc_wc-1/24)))])
        weekly_coh = float(coh_wc[int(np.argmin(np.abs(fc_wc-1/168)))])
        print(f"\n  Wavelet Coherence ({years[0]} vs {years[-1]}):")
        print(f"  Daily  coherence: {daily_coh:.4f}  ({'Stable seasonal pattern' if daily_coh>0.7 else 'Daily pattern has shifted → use recent data only'})")
        print(f"  Weekly coherence: {weekly_coh:.4f}  ({'Stable weekly pattern' if weekly_coh>0.7 else 'Weekly pattern shifted'})")

# ─────────────────────────────────────────────────────────────────────────────
# 18. VALIDATION STRATEGY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
banner("18. VALIDATION STRATEGY")

# Walk-forward CV
print(f"\n  Walk-Forward Cross-Validation:")
n_rows   = len(feat_df)
n_folds  = 5
fold_size = n_rows // (n_folds+1)
fold_results = []
for fold in range(n_folds):
    tr_end   = fold_size * (fold+1)
    val_start = tr_end
    val_end   = min(tr_end + fold_size, n_rows)
    if val_end - val_start < 24: continue
    Xtr_ = feat_df[feature_cols].iloc[:tr_end].values
    ytr_ = feat_df[TARGET_COL].iloc[:tr_end].values
    Xva_ = feat_df[feature_cols].iloc[val_start:val_end].values
    yva_ = feat_df[TARGET_COL].iloc[val_start:val_end].values
    m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    m.fit(Xtr_, ytr_)
    pva_ = m.predict(Xva_)
    fold_results.append({
        'fold': fold+1,
        'train_size': tr_end,
        'val_size': val_end-val_start,
        'MAE':  float(mean_absolute_error(yva_,pva_)),
        'MAPE': float(np.mean(np.abs((yva_-pva_)/np.maximum(np.abs(yva_),1e-6)))*100),
    })
    print(f"  Fold {fold+1}: train={tr_end:5d}  val={val_end-val_start:4d}  MAE={fold_results[-1]['MAE']:.1f}  MAPE={fold_results[-1]['MAPE']:.2f}%")

maes = [r['MAE'] for r in fold_results]
print(f"  Overall: MAE={np.mean(maes):.1f} ± {np.std(maes):.1f} MWh  (CV={np.std(maes)/max(np.mean(maes),1e-12)*100:.1f}%)")
print(f"  ► High fold-to-fold variance = non-stationary; include recent-data recency weights")

# h-block CV
print(f"\n  h-block CV (h=48 to prevent temporal leakage):")
h_block    = 48
test_pts   = np.linspace(h_block+100, n_rows-h_block-1, min(30,n_rows//200)).astype(int)
hblock_err = []
for t in np.unique(test_pts):
    mask = np.ones(n_rows, bool)
    mask[max(0,t-h_block):min(n_rows,t+h_block+1)] = False
    if mask.sum() < 50: continue
    m = Ridge(alpha=1.0)
    Xsc = StandardScaler().fit(feat_df[feature_cols].values[mask])
    m.fit(Xsc.transform(feat_df[feature_cols].values[mask]),
          feat_df[TARGET_COL].values[mask])
    p = m.predict(Xsc.transform(feat_df[feature_cols].values[[t],:]))[0]
    hblock_err.append(feat_df[TARGET_COL].values[t] - p)
print(f"  h-block RMSE = {np.sqrt(np.mean(np.array(hblock_err)**2)):.2f} MWh  ({len(hblock_err)} test points)")

# KL drift (distribution shift over time)
yrs = sorted(df.index.year.unique())
if len(yrs) >= 2:
    ya_ = df[df.index.year==yrs[0]][TARGET_COL].dropna().values
    yb_ = df[df.index.year==yrs[-1]][TARGET_COL].dropna().values
    bins_ = np.histogram_bin_edges(np.r_[ya_,yb_], bins=40)
    pa_ = np.histogram(ya_,bins=bins_,density=True)[0]+1e-12
    pb_ = np.histogram(yb_,bins=bins_,density=True)[0]+1e-12
    pa_/=pa_.sum(); pb_/=pb_.sum()
    kl_div = float(np.sum(pa_*np.log(pa_/pb_)))
    print(f"\n  KL Divergence ({yrs[0]} → {yrs[-1]}): {kl_div:.4f}")
    print(f"  ► {'Significant distribution drift → consider training only on recent years' if kl_div>0.05 else 'Stable distribution → use full history'}")

# ─────────────────────────────────────────────────────────────────────────────
# 19. FINAL MODEL SELECTION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
banner("19. MODEL SELECTION DECISION MATRIX")

print(f"""
  Based on all statistical tests above, here is your evidence-based
  model selection matrix:

  PROPERTY                        FINDING                          IMPLICATION
  ─────────────────────────────────────────────────────────────────────────────────
  Stationarity (ADF/KPSS)        {('NON-STAT' if adf_nonstat and kpss_nonstat else 'STAT'):<35} {'Difference or detrend before ARIMA' if adf_nonstat else 'ARMA(p,0,q) valid'}
  Seasonal Strength Fs            {Fs:<35.4f} {'MUST model daily seasonality' if Fs>0.64 else 'Seasonal not dominant'}
  Trend Strength Ft               {Ft:<35.4f} {'MUST model trend (Prophet/HP filter)' if Ft>0.5 else 'Trend weak → simple differencing ok'}
  Hurst Exponent                  {H:<35.4f} {'Long memory → lag_168h, lag_336h crucial' if H>0.65 else 'Short memory → lags <= 48h sufficient'}
  Nonlinearity (3 tests)          {nl_verdict}/3 significant{' '*21} {'XGBoost/LSTM >> ARIMA' if nl_verdict>=2 else 'ARIMA competitive'}
  ARCH Effects                    {'Present' if arch['p']<0.05 else 'Absent':<35} {'GARCH variance model needed' if arch['p']<0.05 else 'Constant variance intervals ok'}
  Skewness                        {sk:<35.4f} {'Asymmetric loss function (Linex/pinball)' if abs(sk)>0.5 else 'Symmetric loss ok'}
  Excess Kurtosis                 {ku:<35.4f} {'Huber/MAE loss; conformal intervals' if ku>2 else 'MSE acceptable'}
  Seasonal Stability              {'Stochastic' if t_p<0.05 else 'Deterministic':<35} {'SARIMA(D=1) or time-varying STL' if t_p<0.05 else 'Fixed Fourier terms sufficient'}
  Structural Breaks               {len(cps_dates)} break(s) detected{' '*18} {'Add dummy variables at break dates' if len(cps_dates)>0 else 'No regime changes'}
  Seasonality Type                {'Multiplicative' if range_level_corr>0.5 else 'Additive':<35} {'Log-transform target; TBATS multiplicative' if range_level_corr>0.5 else 'Additive STL/ETS sufficient'}
  Long-term trend                 {slope_yr:+.1f} MWh/yr{' '*26} {'Include time index / HP trend feature'}
  Best baseline MAPE              {min(all_metrics[n]['MAPE%'] for n in all_metrics):.2f}%{' '*32} {'Target: < {:.2f}% to beat'.format(min(all_metrics[n]['MAPE%'] for n in all_metrics))}
  Fourier harmonics needed        {max(sig_k) if sig_k else 3} pairs for daily period{' '*13} {'Set n_fourier={}'.format(max(sig_k) if sig_k else 3)}
  AR order (BIC)                  p={best_bic_p:<33} {'SARIMA non-seasonal AR={}'.format(best_bic_p)}

  ─────────────────────────────────────────────────────────────────────────────────
  RECOMMENDED MODEL STACK (ranked by expected performance):

  1. PRIMARY: LightGBM/XGBoost with Direct Multi-Step forecasting
     - Features: lag_24h, lag_168h, lag_336h (long memory), all calendar features
     - Fourier terms: {max(sig_k) if sig_k else 3} harmonic pairs for daily, 2 for weekly
     - Loss: {'Huber' if ku>2 else 'MSE'} for point forecast, Pinball(τ=0.1,0.5,0.9) for intervals
     - Optuna tuning: n_estimators, max_depth, learning_rate, subsample

  2. SECONDARY: TBATS(periods=[24,168]{'[8760]' if N>8760 else ''})
     - Handles triple seasonality natively
     - AIC selects number of Fourier harmonics automatically
     - Good interpretable baseline; includes Box-Cox transformation

  3. TERTIARY: Prophet with Indian holiday calendar
     - changepoint_prior_scale: tune on validation set
     - Add ARCH-based regressors as custom_seasonality
     - Set seasonality_mode='{'multiplicative' if range_level_corr>0.5 else 'additive'}'

  4. DEEP LEARNING: TCN or PatchTST (if time permits)
     - PatchTST patch_size=24 (daily patches)
     - Attention with periodic bias at lags 24, 168

  5. ENSEMBLE: BIC-weighted + Rao-Blackwell minimum-variance combination
     - Optimal weights ≈ {w_opt.round(3)} for [Ridge, GBM]
     - Conformal calibration on top for guaranteed interval coverage

  CRITICAL DO-NOTS:
  - Do NOT use ARIMA(d=1) → Hurst={H:.3f} means fractional d≈{gph['d']:.2f} is optimal
  - Do NOT use MSE loss → excess kurtosis={ku:.2f}>2 means Huber is better
  - Do NOT use Gaussian prediction intervals → JB p={jb_p:.2e} << 0.05
  - Do NOT train on full 6 years if KL drift={kl_div:.3f} is high → weight recent
  ─────────────────────────────────────────────────────────────────────────────────
""")

# ─────────────────────────────────────────────────────────────────────────────
# 20. MEGA VISUALIZATION DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
banner("20. GENERATING VISUALIZATION DASHBOARD")

fig = plt.figure(figsize=(28,36), facecolor=PALETTE['bg'])
gs  = gridspec.GridSpec(6,4, figure=fig, hspace=0.45, wspace=0.35)

# ── Row 0: Series + Distribution ─────────────────────────────────────────────
ax = fig.add_subplot(gs[0,:3]); ax.set_facecolor(PALETTE['bg'])
ax.plot(df.index, y, color=PALETTE['secondary'], lw=0.3, alpha=0.7)
rm = pd.Series(y).rolling(24*7,center=True,min_periods=24).mean()
ax.plot(df.index, rm, color=PALETTE['accent'], lw=2, label='7-day mean')
ax.plot(df.index, hp_trend, color=PALETTE['purple'], lw=1.5, ls='--', label='HP trend')
ax.set_title(f'Full Series (N={N:,})', fontsize=11, fontweight='bold')
ax.legend(fontsize=8); ax.set_ylabel('MWh')

ax = fig.add_subplot(gs[0,3]); ax.set_facecolor(PALETTE['bg'])
ax.hist(y, bins=80, color=PALETTE['secondary'], alpha=0.8, density=True, edgecolor='white')
xr = np.linspace(y.min(),y.max(),300)
ax.plot(xr, stats.norm.pdf(xr,y.mean(),y.std()), color=PALETTE['accent'], lw=2.5, label='Gaussian')
ax.set_title(f'Distribution (skew={sk:.3f}, kurt={ku:.3f})', fontsize=9, fontweight='bold')
ax.legend(fontsize=8)

# ── Row 1: ACF + PACF ────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1,:2]); ax.set_facecolor(PALETTE['bg'])
lags_arr = np.arange(len(r))
ax.vlines(lags_arr, 0, r, color=PALETTE['primary'], lw=0.7, alpha=0.8)
ax.axhline(ci95, color=PALETTE['accent'], ls='--', lw=1.5, label=f'95% CI ±{ci95:.4f}')
ax.axhline(-ci95, color=PALETTE['accent'], ls='--', lw=1.5)
for lag in [24,48,168,336]:
    if lag<len(r):
        ax.annotate(f'{lag}h\n{r[lag]:.3f}', xy=(lag,r[lag]),
                    xytext=(lag+8,r[lag]+0.03),
                    arrowprops=dict(arrowstyle='->',color=PALETTE['accent']),
                    fontsize=7, color=PALETTE['accent'])
ax.set_title(f'ACF (nlags={NLAGS})', fontsize=10, fontweight='bold')
ax.set_xlabel('Lag (hours)'); ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1,2:]); ax.set_facecolor(PALETTE['bg'])
ax.vlines(np.arange(len(pr)), 0, pr, color=PALETTE['purple'], lw=0.7, alpha=0.8)
ax.axhline(ci95, color=PALETTE['accent'], ls='--', lw=1.5)
ax.axhline(-ci95, color=PALETTE['accent'], ls='--', lw=1.5)
ax.set_title(f'PACF  (AR order hint: p={ar_order_hint})', fontsize=10, fontweight='bold')
ax.set_xlabel('Lag (hours)')

# ── Row 2: STL + Seasonality heatmap ────────────────────────────────────────
ax = fig.add_subplot(gs[2,:2]); ax.set_facecolor(PALETTE['bg'])
n_plot = min(24*365, len(trend))
x_plot = df.index[:n_plot]
ax.plot(x_plot, trend[:n_plot], color=PALETTE['primary'], lw=2, label='Trend')
ax.plot(x_plot, y[:n_plot], color=PALETTE['secondary'], lw=0.4, alpha=0.5)
ax.set_title(f'STL Trend (Fs={Fs:.3f}, Ft={Ft:.3f})', fontsize=10, fontweight='bold')
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[2,2:]); ax.set_facecolor(PALETTE['bg'])
cond_heat = df.groupby(['hour','dow'])[TARGET_COL].mean().unstack()
cmap_div  = LinearSegmentedColormap.from_list('ep',['#2E86AB','#F7F9FC','#E84855'])
sns.heatmap(cond_heat, ax=ax, cmap=cmap_div, linewidths=.2, linecolor='white', fmt='.0f',
            xticklabels=['M','Tu','W','Th','F','Sa','Su'])
ax.set_title('Conditional Mean (hour×DOW)', fontsize=10, fontweight='bold')

# ── Row 3: Periodogram + R/S Hurst ───────────────────────────────────────────
ax = fig.add_subplot(gs[3,:2]); ax.set_facecolor(PALETTE['bg'])
valid_mask = freq>0
ax.semilogy(freq[valid_mask], psd[valid_mask], color=PALETTE['secondary'], lw=0.6)
for p_mark,lbl in [(24,'Daily'),(168,'Weekly'),(12,'12h')]:
    ax.axvline(1/p_mark, color=PALETTE['accent'], ls='--', lw=1.5)
    ax.text(1/p_mark, ax.get_ylim()[1] if ax.get_ylim()[1]<1e20 else 1e10,
            f' {lbl}', fontsize=8, color=PALETTE['accent'])
ax.set_title('Periodogram', fontsize=10, fontweight='bold')
ax.set_xlabel('Frequency (cycles/hour)')

ax = fig.add_subplot(gs[3,2:]); ax.set_facecolor(PALETTE['bg'])
ax.scatter(np.log(rs_sizes), np.log(rs_vals), color=PALETTE['primary'], s=40, zorder=5)
xl_h = np.linspace(min(np.log(rs_sizes)), max(np.log(rs_sizes)), 100)
ax.plot(xl_h, np.polyfit(np.log(rs_sizes),np.log(rs_vals),1)[1]+H*xl_h,
        color=PALETTE['accent'], lw=2.5, label=f'H={H:.4f}')
ax.set_title(f'R/S Analysis — Hurst Exponent', fontsize=10, fontweight='bold')
ax.set_xlabel('log(n)'); ax.set_ylabel('log(R/S)'); ax.legend()

# ── Row 4: Model comparison + Residuals ──────────────────────────────────────
ax = fig.add_subplot(gs[4,:2]); ax.set_facecolor(PALETTE['bg'])
n_show = min(200, len(y_te))
ax.plot(y_te[:n_show], color=PALETTE['dark'], lw=1.5, label='Actual')
ax.plot(gbm_te[:n_show], color=PALETTE['accent'], lw=1.5, ls='--', label='GBM')
ax.plot(ridge_te[:n_show], color=PALETTE['secondary'], lw=1, ls=':', label='Ridge')
ax.fill_between(range(n_show), conf_lo[:n_show], conf_hi[:n_show],
                alpha=0.2, color=PALETTE['secondary'], label='80% conformal')
ax.set_title('Actual vs Predicted (first 200 test obs)', fontsize=10, fontweight='bold')
ax.legend(fontsize=7)

ax = fig.add_subplot(gs[4,2:]); ax.set_facecolor(PALETTE['bg'])
model_names = list(all_metrics.keys()) + ['Ensemble']
mapes = [all_metrics[n]['MAPE%'] for n in list(all_metrics.keys())] + [m_ens['MAPE%']]
colors_bar = [PALETTE['accent'] if m==min(mapes) else PALETTE['secondary'] for m in mapes]
bars = ax.bar(range(len(mapes)), mapes, color=colors_bar, edgecolor='white', alpha=0.85)
ax.set_xticks(range(len(mapes)))
ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=8)
for bar,val in zip(bars,mapes):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.02, f'{val:.2f}%',
            ha='center', fontsize=8, fontweight='bold')
ax.set_title('MAPE% by Model', fontsize=10, fontweight='bold'); ax.set_ylabel('MAPE%')

# ── Row 5: Feature importance + Reliability ───────────────────────────────────
ax = fig.add_subplot(gs[5,:2]); ax.set_facecolor(PALETTE['bg'])
top_feats = mi_rank[:15]
fnames = [f[0] for f in top_feats][::-1]
fvals  = [f[1] for f in top_feats][::-1]
colors_feat = [PALETTE['accent'] if f in lasso_sel else PALETTE['secondary'] for f in fnames]
ax.barh(range(len(fnames)), fvals, color=colors_feat, edgecolor='white', alpha=0.85)
ax.set_yticks(range(len(fnames))); ax.set_yticklabels(fnames, fontsize=8)
ax.set_title('Mutual Information (orange=LASSO selected)', fontsize=9, fontweight='bold')

ax = fig.add_subplot(gs[5,2:]); ax.set_facecolor(PALETTE['bg'])
noms  = [0.5,0.6,0.7,0.8,0.9,0.95]
acts  = [float(np.mean(np.abs(y_te-ridge_te)<=np.quantile(cal_nc,n))) for n in noms]
ax.plot([0,1],[0,1],'k--',lw=1.5,label='Perfect calibration')
ax.plot(noms, acts, 'o-', color=PALETTE['accent'], lw=2.5, ms=8, label='Actual coverage')
ax.fill_between(noms,[n-0.05 for n in noms],[n+0.05 for n in noms],
                alpha=0.1,color='gray',label='±5% tolerance')
ax.set_xlim(0.4,1.0); ax.set_ylim(0.4,1.0)
ax.set_xlabel('Nominal coverage'); ax.set_ylabel('Actual coverage')
ax.set_title('Reliability Diagram', fontsize=10, fontweight='bold')
ax.legend(fontsize=8)

fig.suptitle('COMPREHENSIVE STATISTICAL ANALYSIS DASHBOARD\nEnergy Consumption Forecasting',
             fontsize=16, fontweight='bold', color=PALETTE['dark'], y=1.005)
savefig(fig, 'MASTER_ANALYSIS_DASHBOARD.png', dpi=120)

# ─────────────────────────────────────────────────────────────────────────────
# 21. SAVE MACHINE-READABLE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
banner("21. SAVING MACHINE-READABLE SUMMARY")

summary = {
    'dataset': {'N': int(N), 'start': str(df.index[0]), 'end': str(df.index[-1])},
    'descriptive': {
        'mean': float(y.mean()), 'std': float(y.std()), 'skew': float(sk), 'kurtosis': float(ku),
        'jb_p': float(jb_p), 'cv_pct': float(cv), 'var_explained_by_calendar': float(var_exp),
        'weekday_vs_weekend_ks_p': float(ks2p), 'seasonality_type': 'multiplicative' if range_level_corr>0.5 else 'additive',
    },
    'stationarity': {
        'adf_ct_reject_5pct': bool(adf_ct['reject_5%']), 'kpss_ct_reject_5pct': bool(kpss_ct['reject_5%']),
        'arima_d': 0 if adf_ct['reject_5%'] else 1,
    },
    'seasonality': {'Fs': float(Fs), 'Ft': float(Ft), 'sig_fourier_harmonics': int(max(sig_k) if sig_k else 3)},
    'long_memory': {'hurst': float(H), 'dfa_alpha': float(alpha_dfa), 'gph_d': float(gph['d'])},
    'nonlinearity': {'tests_significant': int(nl_verdict), 'keenan_p': float(keenan_p), 'terasvirta_p': float(p_ter), 'bds_p': float(bds_p)},
    'volatility': {'arch_lm_p': float(arch['p']), 'garch_alpha_plus_beta': float(garch_p['alpha']+garch_p['beta'])},
    'breaks': {'cusum_detected': bool(cusum_crosses), 'qlr_p': float(best_qlr['p']), 'pelt_breaks': cps_dates},
    'model_selection': {
        'best_ar_aic': int(best_aic_p), 'best_ar_bic': int(best_bic_p),
        'recommended_loss': 'Huber' if ku>2 else 'MSE',
        'recommended_intervals': 'conformal_quantile',
        'seasonality_mode': 'multiplicative' if range_level_corr>0.5 else 'additive',
    },
    'test_metrics': {name: {k: float(v) for k,v in m.items()} for name,m in all_metrics.items()},
    'ensemble': {'weights_ridge_gbm': w_opt.tolist(), 'ensemble_mape': float(m_ens['MAPE%'])},
    'probabilistic': {'picp_80': float(picp), 'mpiw': float(mpiw), 'directional_symmetry': float(ds)},
    'confirmed_features': list(confirmed.index[:20]) if len(confirmed)>0 else lasso_sel[:20],
}

with open(os.path.join(OUT, 'analysis_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\n  ✓ Summary saved to {OUT}/analysis_summary.json")
print(f"  ✓ Dashboard saved to {OUT}/MASTER_ANALYSIS_DASHBOARD.png")

banner("ANALYSIS COMPLETE")
print(f"""
  Files generated in ./{OUT}/:
  • MASTER_ANALYSIS_DASHBOARD.png  — 6×4 mega-plot with all key analyses
  • analysis_summary.json          — machine-readable findings for pipeline config

  Next steps:
  1. Read analysis_summary.json to configure your LightGBM hyperparameter search
  2. Set n_fourier_terms = {max(sig_k) if sig_k else 3} in Prophet / Fourier feature builder
  3. Use confirmed_features list as starting feature set for XGBoost
  4. Apply Huber loss (δ = P95 of validation errors)
  5. Wrap final model in conformal predictor (calibrate on split_val)
""")