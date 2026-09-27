"""Regression checks for the UCI Steel forecasting data contract."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.prepare_steel_data import write_train_eda
from src.steel.preprocessing import (
    BASE_FEATURES,
    EXPECTED_SHA256,
    EXPECTED_SPLIT_ROWS,
    FREQUENCY,
    OPTIONAL_SENSOR_FEATURES,
    SPLIT_STARTS,
    TARGET,
    build_row_manifest,
    build_supervised_rows,
    load_official_raw,
    profile_training_columns,
    sha256_file,
    split_supervised_rows,
    validate_and_sort_raw,
    write_processed_data,
)
from src.steel.sequences import SteelSequenceView


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data/steel/raw/Steel_industry_data.csv"


class SteelPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw, cls.raw_audit = load_official_raw(RAW_PATH)
        cls.rows = build_supervised_rows(cls.raw)
        cls.splits, cls.split_audit = split_supervised_rows(cls.rows)

    def test_official_source_is_unchanged_and_sorted_grid_is_complete(self) -> None:
        self.assertEqual(sha256_file(RAW_PATH), EXPECTED_SHA256)
        self.assertEqual(len(self.raw), 35040)
        self.assertFalse(self.raw_audit["raw_order_monotonic"])
        self.assertEqual(self.raw_audit["missing_cells"], 0)
        self.assertEqual(self.raw_audit["duplicate_timestamps"], 0)
        self.assertTrue(self.raw["observation_time"].diff().dropna().eq(FREQUENCY).all())
        self.assertEqual(self.raw["observation_time"].iloc[0], pd.Timestamp("2018-01-01 00:00"))
        self.assertEqual(self.raw["observation_time"].iloc[-1], pd.Timestamp("2018-12-31 23:45"))

    def test_target_is_exactly_four_future_intervals(self) -> None:
        for index in (672, 5000, 23323, 23328, 29183, 35035):
            with self.subTest(index=index):
                expected = self.raw["Usage_kWh"].iloc[index + 1:index + 5].sum()
                self.assertAlmostEqual(self.rows.loc[index, TARGET], expected, places=8)
                self.assertEqual(
                    self.rows.loc[index, "forecast_end"],
                    self.raw.loc[index, "observation_time"] + pd.Timedelta(hours=1),
                )
        self.assertTrue(self.rows[TARGET].tail(4).isna().all())

    def test_mutating_future_usage_changes_label_but_not_current_features(self) -> None:
        index = 5000
        changed_raw = self.raw.copy()
        changed_raw.loc[index + 1, "Usage_kWh"] += 500
        changed_rows = build_supervised_rows(changed_raw)
        cols = list(BASE_FEATURES) + list(OPTIONAL_SENSOR_FEATURES)
        np.testing.assert_allclose(
            self.rows.loc[index, cols].to_numpy(dtype=float),
            changed_rows.loc[index, cols].to_numpy(dtype=float),
            rtol=0,
            atol=0,
        )
        self.assertAlmostEqual(changed_rows.loc[index, TARGET] - self.rows.loc[index, TARGET], 500)

    def test_split_sizes_and_label_boundaries(self) -> None:
        self.assertEqual({name: len(frame) for name, frame in self.splits.items()}, EXPECTED_SPLIT_ROWS)
        for name, next_name in (
            ("train", "validation"),
            ("validation", "calibration"),
            ("calibration", "test"),
            ("test", "end"),
        ):
            with self.subTest(split=name):
                part = self.splits[name]
                self.assertTrue(part["forecast_end"].lt(SPLIT_STARTS[next_name]).all())
                self.assertTrue(part["observation_time"].ge(SPLIT_STARTS[name]).all())
                self.assertFalse(part[list(BASE_FEATURES) + [TARGET]].isna().any().any())
        self.assertEqual(self.splits["train"]["forecast_end"].iloc[-1], pd.Timestamp("2018-08-31 23:45"))
        self.assertEqual(self.splits["validation"]["observation_time"].iloc[0], pd.Timestamp("2018-09-01 00:00"))
        self.assertEqual(self.split_audit["warmup_rows_without_weekly_lag"], 672)
        self.assertEqual(self.split_audit["purged_rows_at_three_boundaries"], 12)

    def test_manifest_accounts_for_every_raw_row_and_every_exclusion(self) -> None:
        manifest = build_row_manifest(self.rows)
        self.assertEqual(len(manifest), len(self.raw))
        self.assertEqual(manifest["reason"].value_counts().to_dict(), {
            "included": 34352,
            "insufficient_history": 672,
            "label_crosses_split_boundary": 12,
            "future_label_unavailable": 4,
        })
        self.assertEqual(manifest["reason"].iloc[0], "insufficient_history")
        self.assertEqual(manifest["reason"].iloc[-1], "future_label_unavailable")
        for name, part in self.splits.items():
            selected = manifest.loc[
                manifest["period"].eq(name) & manifest["row_status"].eq("included"),
                "observation_time",
            ].reset_index(drop=True)
            pd.testing.assert_series_equal(selected, part["observation_time"])

    def test_train_profile_does_not_change_when_future_months_change(self) -> None:
        profile = profile_training_columns(self.raw)
        self.assertEqual(profile["column"].tolist(), list(self.raw.columns.drop("observation_time")))
        self.assertTrue(profile["missing_rows"].eq(0).all())
        usage = profile.set_index("column").loc["Usage_kWh"]
        self.assertEqual(int(usage["rows"]), 23328)
        self.assertAlmostEqual(float(usage["p95"]), 102.02, places=2)
        changed_raw = self.raw.copy()
        changed_raw.loc[
            changed_raw["observation_time"].ge(SPLIT_STARTS["validation"]), "Usage_kWh"
        ] += 1000
        pd.testing.assert_frame_equal(profile, profile_training_columns(changed_raw))

    def test_train_eda_does_not_use_later_months(self) -> None:
        changed_raw = self.raw.copy()
        changed_raw.loc[
            changed_raw["observation_time"].ge(SPLIT_STARTS["validation"]), "Usage_kWh"
        ] += 1000
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            original_stats = write_train_eda(self.raw, self.splits["train"], Path(first))
            changed_stats = write_train_eda(changed_raw, self.splits["train"], Path(second))
            self.assertEqual(original_stats, changed_stats)
            self.assertEqual(len(list(Path(first).glob("*.png"))), 5)

    def test_sequence_view_ends_at_issue_time_and_uses_same_target(self) -> None:
        part = self.splits["validation"].head(2)
        view = SteelSequenceView(self.raw, part, lookback=96)
        self.assertEqual(len(view), 2)
        window, target = view[0]
        self.assertEqual(window.shape, (96, 1))
        issue_time = part["observation_time"].iloc[0]
        raw_index = self.raw.index[self.raw["observation_time"].eq(issue_time)][0]
        np.testing.assert_allclose(window[:, 0], self.raw["Usage_kWh"].iloc[raw_index - 95:raw_index + 1])
        self.assertAlmostEqual(float(target), float(part[TARGET].iloc[0]), places=4)
        self.assertEqual(view.observation_times[0], issue_time)

    def test_quality_gate_rejects_invalid_measurements_and_timestamps(self) -> None:
        source = pd.read_csv(RAW_PATH)
        duplicate = source.copy()
        duplicate.loc[0, "date"] = duplicate.loc[1, "date"]
        missing_interval = source.drop(index=0)
        negative_usage = source.copy()
        negative_usage.loc[0, "Usage_kWh"] = -1
        invalid_nsm = source.copy()
        invalid_nsm.loc[0, "NSM"] = -1
        missing_usage = source.copy()
        missing_usage.loc[0, "Usage_kWh"] = np.nan
        infinite_usage = source.copy()
        infinite_usage.loc[0, "Usage_kWh"] = np.inf
        for name, broken in {
            "duplicate": duplicate,
            "missing_interval": missing_interval,
            "negative_usage": negative_usage,
            "invalid_nsm": invalid_nsm,
            "missing_usage": missing_usage,
            "infinite_usage": infinite_usage,
        }.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                validate_and_sort_raw(broken)

        with self.assertRaisesRegex(ValueError, "complete 15-minute grid"):
            build_supervised_rows(self.raw.drop(index=1000))

    def test_written_artifacts_keep_schema_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = build_row_manifest(self.rows)
            profile = profile_training_columns(self.raw)
            write_processed_data(
                self.splits, self.split_audit, Path(temp_dir),
                manifest=manifest, train_profile=profile,
            )
            summary = json.loads((Path(temp_dir) / "steel_quality_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["split_rows"], EXPECTED_SPLIT_ROWS)
            self.assertEqual(summary["row_disposition_counts"]["included"], 34352)
            self.assertEqual(len(pd.read_csv(Path(temp_dir) / "steel_row_manifest.csv")), 35040)
            self.assertEqual(len(pd.read_csv(Path(temp_dir) / "steel_train_column_profile.csv")), 11)
            for name, expected in EXPECTED_SPLIT_ROWS.items():
                saved = pd.read_csv(Path(temp_dir) / f"steel_next_60m_{name}.csv")
                self.assertEqual(len(saved), expected)
                self.assertEqual(saved.columns.tolist(), self.splits[name].columns.tolist())


if __name__ == "__main__":
    unittest.main()
