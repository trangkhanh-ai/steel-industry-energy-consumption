"""Gate 7 of the team plan: real-data baselines and two fixed ML candidates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .preprocessing import FREQUENCY, HORIZON_STEPS


def make_candidates() -> dict:
    """Use library defaults, with deterministic HGB and no random holdout.

    This is an initial comparison, not a hyperparameter search. Disabling
    automatic early stopping keeps all training rows in chronological train.
    """
    return {
        "ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            random_state=0, early_stopping=False,
        ),
    }


def historical_baselines(history: pd.Series, origins: pd.DatetimeIndex) -> pd.DataFrame:
    """Forecast the next four intervals using only observations available at t.

    A daily forecast sums t-95 .. t-92, not the hour ending at t-96.
    The same convention holds for the corresponding hour one week before.
    """
    if not history.index.is_unique or not history.index.is_monotonic_increasing:
        raise ValueError("History must have unique, sorted timestamps")
    if not history.index.to_series().diff().dropna().eq(FREQUENCY).all():
        raise ValueError("History must use a complete 15-minute grid")
    if not np.isfinite(history.to_numpy()).all() or history.lt(0).any():
        raise ValueError("History must contain finite, nonnegative measured energy")
    predictions = {"last_hour": history.rolling(HORIZON_STEPS).sum()}
    for name, lag in (("previous_day", 96), ("previous_week", 672)):
        predictions[name] = sum(
            history.shift(lag - step) for step in range(1, HORIZON_STEPS + 1)
        )
    result = pd.DataFrame(predictions).reindex(origins)
    if result.isna().any().any():
        raise ValueError("Insufficient observed history for requested forecasts")
    return result


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.ndim != 1 or not actual.size:
        raise ValueError("Expected equally sized, nonempty one-dimensional arrays")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual values and predictions must be finite")
    error = predicted - actual
    return {
        "rows": int(actual.size),
        "mae_kWh": float(np.abs(error).mean()),
        "rmse_kWh": float(np.sqrt(np.square(error).mean())),
        "bias_pred_minus_actual_kWh": float(error.mean()),
    }
