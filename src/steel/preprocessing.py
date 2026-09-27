"""Validated, leakage-aware preprocessing for UCI Steel Industry Energy Consumption.

Every tabular row represents a forecast issued after the measurement at
``observation_time`` is available. The target is the sum of the next four
15-minute Usage_kWh values. Only measurements at or before the issue time may
appear in model features.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_URL = "https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption"
EXPECTED_SHA256 = "9b1cee6f9cb9cd9df2b95814ca90a9a2ff15b7f5f1fba0fae3c643e82072eacc"
DATE_FORMAT = "%d/%m/%Y %H:%M"
FREQUENCY = pd.Timedelta(minutes=15)
HORIZON_STEPS = 4
LONGEST_LAG_STEPS = 672
TARGET = "target_next_60m_kWh"

EXPECTED_COLUMNS = (
    "date",
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
    "WeekStatus",
    "Day_of_week",
    "Load_Type",
)

NUMERIC_COLUMNS = EXPECTED_COLUMNS[1:8]
SENSOR_SOURCE_COLUMNS = (
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
)
OPTIONAL_SENSOR_FEATURES = (
    "known_lagging_reactive_kVarh",
    "known_leading_reactive_kVarh",
    "known_lagging_power_factor",
    "known_leading_power_factor",
)
BASE_FEATURES = (
    "usage_lag_0",
    "usage_lag_1",
    "usage_lag_4",
    "usage_lag_96",
    "usage_lag_672",
    "usage_sum_last_1h",
    "usage_mean_last_4h",
    "usage_std_last_4h",
    "usage_mean_last_24h",
    "usage_max_last_24h",
    "forecast_hour_sin",
    "forecast_hour_cos",
    "forecast_weekday_sin",
    "forecast_weekday_cos",
    "forecast_is_weekend",
)

SPLIT_STARTS = {
    "train": pd.Timestamp("2018-01-01"),
    "validation": pd.Timestamp("2018-09-01"),
    "calibration": pd.Timestamp("2018-10-01"),
    "test": pd.Timestamp("2018-11-01"),
    "end": pd.Timestamp("2019-01-01"),
}
EXPECTED_SPLIT_ROWS = {
    "train": 22652,
    "validation": 2876,
    "calibration": 2972,
    "test": 5852,
}


def sha256_file(path: Path) -> str:
    """Hash file bytes without changing the original source."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_and_sort_raw(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject invalid meter data and sort by the declared timestamp.

    We intentionally do not impute, cap peaks, or infer a corrected timestamp
    for the source's end-of-day 00:00 row. Its declared date/NSM/day agree, so
    the timestamp field is the explicit ordering contract for this prototype.
    """
    if list(frame.columns) != list(EXPECTED_COLUMNS):
        raise ValueError("Steel CSV columns do not match the official source schema")
    if frame.empty:
        raise ValueError("Steel CSV is empty")

    try:
        timestamps = pd.to_datetime(frame["date"], format=DATE_FORMAT, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid Steel date; expected DD/MM/YYYY HH:MM") from exc
    if timestamps.isna().any() or timestamps.duplicated().any():
        raise ValueError("Missing or duplicate Steel timestamps")

    sorted_frame = frame.copy()
    sorted_frame.insert(0, "observation_time", timestamps)
    sorted_frame = sorted_frame.sort_values("observation_time", kind="stable").reset_index(drop=True)
    for column in NUMERIC_COLUMNS:
        try:
            sorted_frame[column] = pd.to_numeric(sorted_frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Non-numeric values in {column}") from exc
    if sorted_frame.isna().any().any():
        raise ValueError("Steel source contains missing values")
    if not np.isfinite(sorted_frame[list(NUMERIC_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError("Steel source contains non-finite measurements")

    intervals = sorted_frame["observation_time"].diff().dropna()
    if not intervals.eq(FREQUENCY).all():
        raise ValueError("Steel timestamps are not a complete 15-minute grid")
    if (sorted_frame["Usage_kWh"] < 0).any():
        raise ValueError("Usage_kWh cannot be negative")
    for column in SENSOR_SOURCE_COLUMNS[:2]:
        if (sorted_frame[column] < 0).any():
            raise ValueError(f"{column} cannot be negative")
    for column in SENSOR_SOURCE_COLUMNS[2:]:
        if not sorted_frame[column].between(0, 100).all():
            raise ValueError(f"{column} must be in [0, 100]")
    if (sorted_frame["CO2(tCO2)"] < 0).any():
        raise ValueError("CO2(tCO2) cannot be negative")

    ts = sorted_frame["observation_time"].dt
    seconds_from_midnight = ts.hour * 3600 + ts.minute * 60
    if not sorted_frame["NSM"].eq(seconds_from_midnight).all():
        raise ValueError("NSM disagrees with date")
    if not sorted_frame["Day_of_week"].eq(ts.day_name()).all():
        raise ValueError("Day_of_week disagrees with date")
    expected_week_status = np.where(ts.dayofweek >= 5, "Weekend", "Weekday")
    if not sorted_frame["WeekStatus"].eq(expected_week_status).all():
        raise ValueError("WeekStatus disagrees with date")
    if not sorted_frame["Load_Type"].isin(("Light_Load", "Medium_Load", "Maximum_Load")).all():
        raise ValueError("Unexpected Load_Type category")
    return sorted_frame


def load_official_raw(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read the exact downloaded UCI CSV and return sorted data plus audit."""
    path = Path(path)
    checksum = sha256_file(path)
    if checksum != EXPECTED_SHA256:
        raise ValueError(f"Raw Steel checksum mismatch: {checksum}")
    original = pd.read_csv(path)
    original_ts = pd.to_datetime(original["date"], format=DATE_FORMAT, errors="raise")
    sorted_frame = validate_and_sort_raw(original)
    if len(sorted_frame) != 35040:
        raise ValueError("Unexpected number of rows in official Steel CSV")
    if sorted_frame["observation_time"].iloc[0] != SPLIT_STARTS["train"]:
        raise ValueError("Official Steel CSV does not start on 2018-01-01")
    if sorted_frame["observation_time"].iloc[-1] != SPLIT_STARTS["end"] - FREQUENCY:
        raise ValueError("Official Steel CSV does not end on 2018-12-31 23:45")
    audit = {
        "source_url": SOURCE_URL,
        "source_file": "data/steel/raw/Steel_industry_data.csv",
        "source_sha256": checksum,
        "source_bytes": path.stat().st_size,
        "raw_rows": len(original),
        "raw_columns": len(original.columns),
        "raw_order_monotonic": bool(original_ts.is_monotonic_increasing),
        "raw_order_backward_jumps": int(original_ts.diff().dropna().lt(pd.Timedelta(0)).sum()),
        "sorted_start": str(sorted_frame["observation_time"].iloc[0]),
        "sorted_end": str(sorted_frame["observation_time"].iloc[-1]),
        "frequency_minutes": 15,
        "missing_cells": int(original.isna().sum().sum()),
        "duplicate_timestamps": int(original_ts.duplicated().sum()),
        "unexpected_intervals_after_sort": 0,
        "zero_usage_rows": int(sorted_frame["Usage_kWh"].eq(0).sum()),
        "negative_usage_rows": int(sorted_frame["Usage_kWh"].lt(0).sum()),
        "load_type_counts": sorted_frame["Load_Type"].value_counts().sort_index().to_dict(),
    }
    return sorted_frame, audit


def build_supervised_rows(sorted_raw: pd.DataFrame) -> pd.DataFrame:
    """Construct point-in-time tabular features and the next-hour energy target."""
    if not sorted_raw["observation_time"].is_monotonic_increasing:
        raise ValueError("Sort and validate raw timestamps before feature engineering")
    if not sorted_raw["observation_time"].diff().dropna().eq(FREQUENCY).all():
        raise ValueError("Feature engineering requires a complete 15-minute grid")
    usage = sorted_raw["Usage_kWh"].astype(float)
    if not np.isfinite(usage.to_numpy()).all() or (usage < 0).any():
        raise ValueError("Feature engineering requires finite, nonnegative Usage_kWh")
    issue_time = sorted_raw["observation_time"]
    forecast_start = issue_time + FREQUENCY
    rows = pd.DataFrame({
        "observation_time": issue_time,
        "forecast_start": forecast_start,
        "forecast_end": issue_time + HORIZON_STEPS * FREQUENCY,
    })
    rows[TARGET] = sum(usage.shift(-step) for step in range(1, HORIZON_STEPS + 1))

    for lag in (0, 1, 4, 96, LONGEST_LAG_STEPS):
        rows[f"usage_lag_{lag}"] = usage.shift(lag)
    rows["usage_sum_last_1h"] = usage.rolling(4, min_periods=4).sum()
    rows["usage_mean_last_4h"] = usage.rolling(16, min_periods=16).mean()
    rows["usage_std_last_4h"] = usage.rolling(16, min_periods=16).std(ddof=0)
    rows["usage_mean_last_24h"] = usage.rolling(96, min_periods=96).mean()
    rows["usage_max_last_24h"] = usage.rolling(96, min_periods=96).max()

    hour = forecast_start.dt.hour + forecast_start.dt.minute / 60
    weekday = forecast_start.dt.dayofweek
    rows["forecast_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    rows["forecast_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    rows["forecast_weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    rows["forecast_weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    rows["forecast_is_weekend"] = (weekday >= 5).astype(int)

    for source, output in zip(SENSOR_SOURCE_COLUMNS, OPTIONAL_SENSOR_FEATURES, strict=True):
        rows[output] = sorted_raw[source].astype(float)
    return rows


def build_row_manifest(rows: pd.DataFrame) -> pd.DataFrame:
    """Explain the fate of every forecast origin without inspecting target values."""
    required = set(BASE_FEATURES) | set(OPTIONAL_SENSOR_FEATURES) | {
        "observation_time", "forecast_end", TARGET,
    }
    if not required.issubset(rows.columns):
        raise ValueError("Supervised table is missing required feature or target columns")
    times = rows["observation_time"]
    if not times.is_monotonic_increasing or not times.is_unique:
        raise ValueError("Supervised rows must have unique, sorted observation_time")
    if not times.diff().dropna().eq(FREQUENCY).all():
        raise ValueError("Supervised rows must have a complete 15-minute grid")

    period = np.full(len(rows), "out_of_range", dtype=object)
    safe_label = np.zeros(len(rows), dtype=bool)
    for name, next_name in (
        ("train", "validation"),
        ("validation", "calibration"),
        ("calibration", "test"),
        ("test", "end"),
    ):
        start = SPLIT_STARTS[name]
        end = SPLIT_STARTS[next_name]
        mask = times.ge(start) & times.lt(end)
        period[mask.to_numpy()] = name
        safe_label[mask.to_numpy()] = rows.loc[mask, "forecast_end"].lt(end).to_numpy()
    if (period == "out_of_range").any():
        raise ValueError("Supervised rows fall outside the declared study year")

    has_history = rows[list(BASE_FEATURES)].notna().all(axis=1).to_numpy()
    has_label = rows[TARGET].notna().to_numpy()
    reason = np.full(len(rows), "included", dtype=object)
    reason[~has_history] = "insufficient_history"
    reason[has_history & ~has_label] = "future_label_unavailable"
    reason[has_history & has_label & ~safe_label] = "label_crosses_split_boundary"
    return pd.DataFrame({
        "observation_time": times.to_numpy(),
        "forecast_end": rows["forecast_end"].to_numpy(),
        "period": period,
        "row_status": np.where(reason == "included", "included", "excluded"),
        "reason": reason,
    })


def profile_training_columns(sorted_raw: pd.DataFrame) -> pd.DataFrame:
    """Profile source columns on January-August only for preparation decisions."""
    if not set(EXPECTED_COLUMNS).issubset(sorted_raw.columns) or "observation_time" not in sorted_raw:
        raise ValueError("Training profile requires validated Steel source columns")
    train = sorted_raw.loc[
        sorted_raw["observation_time"].ge(SPLIT_STARTS["train"])
        & sorted_raw["observation_time"].lt(SPLIT_STARTS["validation"])
    ]
    if train.empty:
        raise ValueError("No training observations available for column profiling")
    records = []
    for name in EXPECTED_COLUMNS:
        series = train[name]
        record = {
            "column": name,
            "dtype": str(series.dtype),
            "rows": len(series),
            "missing_rows": int(series.isna().sum()),
            "unique_values": int(series.nunique(dropna=True)),
            "zero_rows": None,
            "min": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
        if pd.api.types.is_numeric_dtype(series):
            record.update({
                "zero_rows": int(series.eq(0).sum()),
                "min": float(series.min()),
                "median": float(series.median()),
                "p95": float(series.quantile(0.95)),
                "p99": float(series.quantile(0.99)),
                "max": float(series.max()),
            })
        records.append(record)
    profile = pd.DataFrame.from_records(records)
    profile["zero_rows"] = profile["zero_rows"].astype("Int64")
    return profile


def split_supervised_rows(rows: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    """Apply fixed calendar splits and purge labels crossing any boundary."""
    if "forecast_start" not in rows:
        raise ValueError("Supervised table is missing forecast_start")
    manifest = build_row_manifest(rows)
    splits: dict[str, pd.DataFrame] = {}
    for name, next_name in (
        ("train", "validation"),
        ("validation", "calibration"),
        ("calibration", "test"),
        ("test", "end"),
    ):
        end = SPLIT_STARTS[next_name]
        selected = manifest["period"].eq(name) & manifest["row_status"].eq("included")
        part = rows.loc[selected.to_numpy()].copy()
        if part[list(BASE_FEATURES) + list(OPTIONAL_SENSOR_FEATURES) + [TARGET]].isna().any().any():
            raise ValueError(f"Missing model values remain in {name}")
        if not part["forecast_end"].lt(end).all():
            raise ValueError(f"Future labels cross the {name} boundary")
        splits[name] = part.reset_index(drop=True)

    actual_counts = {name: len(part) for name, part in splits.items()}
    if actual_counts != EXPECTED_SPLIT_ROWS:
        raise ValueError(f"Unexpected split sizes: {actual_counts}")
    split_audit = {
        "horizon_steps": HORIZON_STEPS,
        "horizon_minutes": 60,
        "target_column": TARGET,
        "longest_past_lag_steps": LONGEST_LAG_STEPS,
        "base_feature_columns": list(BASE_FEATURES),
        "optional_current_sensor_columns": list(OPTIONAL_SENSOR_FEATURES),
        "excluded_future_or_ambiguous_columns": ["CO2(tCO2)", "Load_Type"],
        "split_rows": actual_counts,
        "row_disposition_counts": manifest["reason"].value_counts().sort_index().to_dict(),
        "split_boundaries": {key: str(value) for key, value in SPLIT_STARTS.items()},
        "warmup_rows_without_weekly_lag": LONGEST_LAG_STEPS,
        "purged_rows_at_three_boundaries": HORIZON_STEPS * 3,
        "rows_without_full_future_label_at_end": HORIZON_STEPS,
        "train_target_p95_kWh": float(splits["train"][TARGET].quantile(0.95)),
        "train_usage_quantiles_kWh": {
            str(q): float(value)
            for q, value in rows.loc[
                rows["observation_time"].lt(SPLIT_STARTS["validation"]), "usage_lag_0"
            ].quantile([0, 0.5, 0.9, 0.95, 0.99, 1]).items()
        },
    }
    return splits, split_audit


def write_processed_data(
    splits: dict[str, pd.DataFrame],
    summary: dict,
    output_dir: Path,
    *,
    manifest: pd.DataFrame | None = None,
    train_profile: pd.DataFrame | None = None,
) -> None:
    """Persist model-ready splits and a small machine-readable provenance record."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(output_dir / f"steel_next_60m_{name}.csv", index=False)
    if manifest is not None:
        manifest.to_csv(output_dir / "steel_row_manifest.csv", index=False)
    if train_profile is not None:
        train_profile.to_csv(output_dir / "steel_train_column_profile.csv", index=False)
    (output_dir / "steel_quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
