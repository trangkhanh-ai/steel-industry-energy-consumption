"""Local next-hour inference from measured history, without future labels."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from .preprocessing import BASE_FEATURES, FREQUENCY, HORIZON_STEPS, LONGEST_LAG_STEPS, sha256_file

REQUIRED_HISTORY_ROWS = LONGEST_LAG_STEPS + 1
FEATURE_CONTRACT = "numpy_observed_windows_v1"


def features_from_valid_history(times, usage) -> pd.DataFrame:
    """Shared batch/online arithmetic on already validated observations.

    Independent window reductions avoid pandas' accumulated rolling roundoff.
    Warmup rows remain NaN; callers must require at least 673 observations.
    """
    usage = np.asarray(usage, dtype=float)
    times = pd.DatetimeIndex(times)
    features = {}
    for lag in (0, 1, 4, 96, 672):
        values = np.full(len(usage), np.nan)
        if len(usage) > lag:
            values[lag:] = usage[:len(usage)-lag] if lag else usage
        features[f"usage_lag_{lag}"] = values
    for name, width, reduction in (
        ("usage_sum_last_1h", 4, "sum"), ("usage_mean_last_4h", 16, "mean"),
        ("usage_std_last_4h", 16, "std"), ("usage_mean_last_24h", 96, "mean"),
        ("usage_max_last_24h", 96, "max"),
    ):
        values = np.full(len(usage), np.nan)
        if len(usage) >= width:
            windows = np.lib.stride_tricks.sliding_window_view(usage, width)
            values[width-1:] = getattr(windows, reduction)(axis=1)
        features[name] = values
    start = times + FREQUENCY
    hour, weekday = start.hour.to_numpy() + start.minute.to_numpy()/60, start.dayofweek.to_numpy()
    features.update(forecast_hour_sin=np.sin(2*np.pi*hour/24), forecast_hour_cos=np.cos(2*np.pi*hour/24),
                    forecast_weekday_sin=np.sin(2*np.pi*weekday/7), forecast_weekday_cos=np.cos(2*np.pi*weekday/7),
                    forecast_is_weekend=(weekday >= 5).astype(int))
    return pd.DataFrame(features, columns=list(BASE_FEATURES))


def features_at_issue(history: pd.DataFrame, issue_time) -> pd.DataFrame:
    """Validate measured input and produce exactly one ordered feature row.

    Input must end at issue_time, be sorted, and contain at least 673 consecutive
    observations. No sorting, interpolation, filling, clipping or future slicing
    is silently applied. Only observation_time and Usage_kWh are consumed.
    """
    if not {"observation_time", "Usage_kWh"}.issubset(history.columns):
        raise ValueError("Required columns: observation_time, Usage_kWh")
    if len(history) < REQUIRED_HISTORY_ROWS:
        raise ValueError("Need at least 673 consecutive measured rows through issue_time")
    t = pd.Timestamp(issue_time)
    times = pd.to_datetime(history.observation_time, errors="raise")
    if pd.isna(t) or t.tzinfo is not None or times.dt.tz is not None:
        raise ValueError("Use valid timezone-naive timestamps matching the dataset")
    if times.isna().any() or not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError("History timestamps must be present, unique and sorted")
    if not times.eq(times.dt.floor("15min")).all() or not times.diff().dropna().eq(FREQUENCY).all():
        raise ValueError("History must be a complete aligned 15-minute grid")
    if times.iloc[-1] != t:
        raise ValueError("History must end exactly at issue_time; future or stale rows are not accepted")
    usage = pd.to_numeric(history.Usage_kWh, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(usage).all() or (usage < 0).any():
        raise ValueError("Usage_kWh must contain finite nonnegative measurements")
    return features_from_valid_history(times.iloc[-REQUIRED_HISTORY_ROWS:], usage[-REQUIRED_HISTORY_ROWS:]).tail(1).reset_index(drop=True)


class SteelForecaster:
    """Load one trusted local run once; reuse for sequential forecast requests."""
    def __init__(self, run_dir: Path):
        run_dir = Path(run_dir)
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("feature_contract") != FEATURE_CONTRACT:
            raise ValueError("Model has no compatible window-feature contract; run scripts.package_steel_model")
        if manifest["versions"]["sklearn"] != sklearn.__version__:
            raise ValueError("scikit-learn version differs from the model training run")
        self.configuration = manifest["selected_configuration"]
        self.model_sha256 = sha256_file(run_dir / "selected_model.joblib")
        if self.model_sha256 != manifest["model_sha256"]:
            raise ValueError("Model file checksum differs from its manifest")
        # joblib deserialization is for the team's own trusted local model only.
        self.model = joblib.load(run_dir / "selected_model.joblib")
        if list(self.model.feature_names_in_) != list(BASE_FEATURES):
            raise ValueError("Model feature schema does not match inference")
        frozen = json.loads((run_dir / "frozen_config.json").read_text(encoding="utf-8"))
        parameters = frozen["parameters"][self.configuration]
        if any(self.model.get_params()[key] != value for key, value in parameters.items()):
            raise ValueError("Selected model parameters disagree with the frozen run")

    def predict(self, history: pd.DataFrame, issue_time) -> dict:
        x = features_at_issue(history, issue_time)
        with threadpool_limits(limits=1):
            prediction = float(self.model.predict(x)[0])
        if not np.isfinite(prediction) or prediction < 0:
            raise ValueError("Model returned invalid energy; do not display an operational forecast")
        t = pd.Timestamp(issue_time)
        return {"observation_time": t.isoformat(), "forecast_start": (t+FREQUENCY).isoformat(),
                "forecast_end": (t+HORIZON_STEPS*FREQUENCY).isoformat(),
                "predicted_next_60m_kWh": prediction, "model_configuration": self.configuration,
                "model_sha256": self.model_sha256, "status": "forecast_only_no_alert_policy"}
