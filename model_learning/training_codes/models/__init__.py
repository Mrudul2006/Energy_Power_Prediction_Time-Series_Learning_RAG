"""
Forecasting model implementations for hourly energy-consumption data.

Each model from the comparison note lives in its own module. Heavy optional
libraries are imported inside the relevant class methods so importing this
package remains cheap and robust.
"""

from .common import ForecastResult, StandardScaler1D, ensure_series, future_index

MODEL_MODULES = {
    "sarima": "models.sarima",
    "arfima": "models.arfima",
    "prophet": "models.prophet_model",
    "ets": "models.ets",
    "tbats": "models.tbats_model",
    "boosted_trees": "models.boosted_trees",
    "lstm": "models.lstm",
    "tcn": "models.tcn",
    "nbeats": "models.nbeats",
    "nhits": "models.nhits",
    "tft": "models.tft",
    "patchtst": "models.patchtst",
    "itransformer": "models.itransformer",
    "timesnet": "models.timesnet",
    "mamba": "models.mamba",
    "timesfm": "models.timesfm",
    "dlinear": "models.dlinear",
    "conformal": "models.conformal",
}

__all__ = [
    "ForecastResult",
    "MODEL_MODULES",
    "StandardScaler1D",
    "ensure_series",
    "future_index",
]
