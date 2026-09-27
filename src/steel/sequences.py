"""Lazy, point-in-time sequence windows for a later deep-learning comparison.

This module has no neural-network dependency. It returns past-only NumPy arrays
from the same forecast origins and targets used by the tabular ML models, so a
future PyTorch/Keras model can be compared without redefining the problem.
Normalization must be fitted on train only by the training code.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .preprocessing import FREQUENCY, TARGET


class SteelSequenceView:
    """Indexable historical windows ending at each forecast issue time."""

    def __init__(
        self,
        sorted_raw: pd.DataFrame,
        supervised_split: pd.DataFrame,
        *,
        lookback: int = 96,
        columns: Sequence[str] = ("Usage_kWh",),
    ) -> None:
        if not isinstance(lookback, int) or lookback < 1:
            raise ValueError("lookback must be a positive integer")
        if not columns or any(column not in sorted_raw.columns for column in columns):
            raise ValueError("Sequence columns must exist in the raw data")
        if TARGET not in supervised_split or "observation_time" not in supervised_split:
            raise ValueError("Supervised split needs target and observation_time")
        raw_times = pd.DatetimeIndex(pd.to_datetime(sorted_raw["observation_time"]))
        issue_times = pd.DatetimeIndex(pd.to_datetime(supervised_split["observation_time"]))
        if not raw_times.is_monotonic_increasing or not raw_times.is_unique:
            raise ValueError("Raw timestamps must be unique and sorted")
        if raw_times.hasnans or not raw_times.to_series().diff().dropna().eq(FREQUENCY).all():
            raise ValueError("Sequence history must have a complete 15-minute grid")
        positions = raw_times.get_indexer(issue_times)
        if (positions < lookback - 1).any():
            raise ValueError("A sequence origin has insufficient past history")
        if not np.isfinite(supervised_split[TARGET].to_numpy(dtype=float)).all():
            raise ValueError("Sequence targets must be finite")
        self._values = sorted_raw[list(columns)].to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(self._values).all():
            raise ValueError("Sequence measurements must be finite")
        self._positions = positions
        self._targets = supervised_split[TARGET].to_numpy(dtype=np.float32, copy=True)
        self.lookback = lookback
        self.columns = tuple(columns)
        self.observation_times = issue_times

    def __len__(self) -> int:
        return len(self._positions)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.float32]:
        position = int(self._positions[index])
        start = position - self.lookback + 1
        # The final measurement is exactly the one known at issue time.
        return self._values[start : position + 1].copy(), self._targets[index]
