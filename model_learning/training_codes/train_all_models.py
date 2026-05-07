import os
import time
import traceback
from typing import Callable, Dict

import numpy as np
import pandas as pd

from models.arfima import ARFIMAForecaster
from models.boosted_trees import BoostedTreesForecaster
from models.common import calendar_features, ensure_dir, ensure_series
from models.conformal import ConformalForecaster
from models.dlinear import LinearBaselineForecaster, LinearBaselineConfig
from models.ets import ETSForecaster
from models.itransformer import iTransformerConfig, iTransformerForecaster
from models.lstm import LSTMForecaster
from models.mamba import MambaForecaster
from models.nbeats import NBEATSForecaster
from models.nhits import NHITSForecaster
from models.patchtst import PatchTSTForecaster
from models.prophet_model import ProphetForecaster
from models.sarima import SARIMAConfig, SARIMAForecaster
from models.tcn import TCNForecaster
from models.tft import TFTConfig, TFTForecaster
from models.tbats_model import TBATSForecaster
from models.timesfm import TimesFMForecaster, TimesFMConfig
from models.timesnet import TimesNetForecaster
from models.neural_common import TorchTrainConfig

DATA_PATH = "Energy Consumption Dataset.xlsx"
TIMESTAMP_COL = "Start time UTC"
TARGET_COL = "Electricity consumption (MWh)"
SAVE_ROOT = "save/models"

EPOCHS = 5
BATCH_SIZE = 64


def load_energy_dataset(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    df = df.dropna(subset=[TIMESTAMP_COL, TARGET_COL]).sort_values(TIMESTAMP_COL)
    return df


def build_time_features(index: pd.Index) -> pd.DataFrame:
    return calendar_features(index, include_india_holidays=True)


def save_model(model, name: str) -> None:
    out_dir = os.path.join(SAVE_ROOT, name)
    ensure_dir(out_dir)
    model.save(out_dir)


def run_step(name: str, fn: Callable[[], None]) -> None:
    start = time.time()
    print(f"\n--- [{name}] start ---")
    try:
        fn()
    except KeyboardInterrupt:
        print(f"[{name}] interrupted by user")
    except Exception as exc:
        print(f"[{name}] failed: {exc}")
        traceback.print_exc()
    else:
        elapsed = time.time() - start
        print(f"[{name}] done in {elapsed:.1f}s")


def main() -> None:
    df = load_energy_dataset(DATA_PATH)
    y = ensure_series(df, target_col=TARGET_COL, timestamp_col=TIMESTAMP_COL)
    features = build_time_features(y.index)
    features = features.reindex(y.index).ffill().fillna(0.0)

    print("Training statistical models...")
    sarima_cfg = SARIMAConfig(fit_kwargs={"disp": False, "maxiter": 100, "method": "powell"})
    run_step(
        "sarima",
        lambda: save_model(SARIMAForecaster(config=sarima_cfg).fit(y, exog=features), "sarima"),
    )
    run_step("arfima", lambda: save_model(ARFIMAForecaster().fit(y), "arfima"))
    run_step("ets", lambda: save_model(ETSForecaster().fit(y), "ets"))
    run_step("tbats", lambda: save_model(TBATSForecaster().fit(y), "tbats"))
    run_step("prophet", lambda: save_model(ProphetForecaster().fit(y, regressors=features), "prophet"))
    run_step("boosted_trees", lambda: save_model(BoostedTreesForecaster().fit(y), "boosted_trees"))

    def train_conformal() -> None:
        split_idx = int(len(y) * 0.8)
        train_y = y.iloc[:split_idx]
        cal_y = y.iloc[split_idx:]
        train_feat = features.iloc[:split_idx]
        cal_feat = features.iloc[split_idx:]
        base = SARIMAForecaster(config=sarima_cfg).fit(train_y, exog=train_feat)
        pred = base.forecast(len(cal_y), future_exog=cal_feat)
        conformal = ConformalForecaster(base).calibrate(cal_y.to_numpy(dtype=float), pred.mean)
        save_model(conformal, "conformal")

    run_step("conformal", train_conformal)

    print("Training neural models...")
    train_config = TorchTrainConfig(epochs=EPOCHS, batch_size=BATCH_SIZE)
    run_step("lstm", lambda: save_model(LSTMForecaster(train_config=train_config).fit(y), "lstm"))
    run_step("tcn", lambda: save_model(TCNForecaster(train_config=train_config).fit(y), "tcn"))
    run_step("nbeats", lambda: save_model(NBEATSForecaster(train_config=train_config).fit(y), "nbeats"))
    run_step("nhits", lambda: save_model(NHITSForecaster(train_config=train_config).fit(y), "nhits"))
    run_step("patchtst", lambda: save_model(PatchTSTForecaster(train_config=train_config).fit(y), "patchtst"))
    run_step("timesnet", lambda: save_model(TimesNetForecaster(train_config=train_config).fit(y), "timesnet"))
    run_step("mamba", lambda: save_model(MambaForecaster(train_config=train_config).fit(y), "mamba"))

    def train_dlinear() -> None:
        dlinear_cfg = LinearBaselineConfig(model_type="dlinear")
        save_model(LinearBaselineForecaster(model_config=dlinear_cfg, train_config=train_config).fit(y), "dlinear")

    def train_nlinear() -> None:
        nlinear_cfg = LinearBaselineConfig(model_type="nlinear")
        save_model(LinearBaselineForecaster(model_config=nlinear_cfg, train_config=train_config).fit(y), "nlinear")

    run_step("dlinear", train_dlinear)
    run_step("nlinear", train_nlinear)

    def train_tft() -> None:
        tft_cfg = TFTConfig(past_input_dim=1, future_input_dim=features.shape[1])
        save_model(TFTForecaster(model_config=tft_cfg, train_config=train_config).fit(y, future_features=features), "tft")

    def train_itransformer() -> None:
        itr_cfg = iTransformerConfig(num_variables=1 + features.shape[1])
        save_model(iTransformerForecaster(model_config=itr_cfg, train_config=train_config).fit(y, features=features), "itransformer")

    run_step("tft", train_tft)
    run_step("itransformer", train_itransformer)
    run_step(
        "timesfm",
        lambda: save_model(TimesFMForecaster(config=TimesFMConfig(), pretrained_model=None, train_config=train_config).fit(y), "timesfm"),
    )

    print("All models attempted. Outputs saved under:", SAVE_ROOT)


if __name__ == "__main__":
    main()
