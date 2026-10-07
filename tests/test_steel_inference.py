"""Inference tests with original measurements and unmodified subsets of them."""
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.steel.inference import REQUIRED_HISTORY_ROWS, SteelForecaster, features_at_issue
from src.steel.preprocessing import BASE_FEATURES, load_official_raw

ROOT = Path(__file__).resolve().parents[1]


class SteelInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw, _ = load_official_raw(ROOT / "data/steel/raw/Steel_industry_data.csv")
        cls.raw = raw.loc[raw.observation_time.lt("2018-10-01"), ["observation_time", "Usage_kWh"]].reset_index(drop=True)
        cls.validation = pd.read_csv(ROOT / "data/steel/processed/steel_next_60m_validation.csv", parse_dates=["observation_time"])

    def window(self, t):
        pos = self.raw.index[self.raw.observation_time.eq(t)][0]
        return self.raw.iloc[pos-REQUIRED_HISTORY_ROWS+1:pos+1]

    def test_live_features_equal_batch_at_real_day_and_week_boundaries(self):
        for index in (0, 95, 96, 671, 672, 2875):
            row = self.validation.iloc[index]
            x = features_at_issue(self.window(row.observation_time), row.observation_time)
            self.assertEqual(x.columns.tolist(), list(BASE_FEATURES))
            np.testing.assert_allclose(x.iloc[0].to_numpy(), row[list(BASE_FEATURES)].to_numpy(dtype=float), atol=1e-7, rtol=1e-10)

    def test_more_measured_history_does_not_change_current_features(self):
        t = self.validation.observation_time.iloc[100]
        full_prefix = self.raw.loc[self.raw.observation_time.le(t)]
        pd.testing.assert_frame_equal(features_at_issue(full_prefix, t), features_at_issue(self.window(t), t))

    def test_incomplete_real_history_is_rejected(self):
        t = self.validation.observation_time.iloc[0]
        with self.assertRaisesRegex(ValueError, "673"):
            features_at_issue(self.window(t).iloc[1:], t)

    def test_future_and_stale_real_windows_are_rejected(self):
        t = self.validation.observation_time.iloc[0]
        for offset in (-pd.Timedelta(minutes=15), pd.Timedelta(minutes=15)):
            with self.assertRaisesRegex(ValueError, "exactly at issue_time"):
                features_at_issue(self.window(t), t+offset)

    def test_gaps_between_real_observations_are_rejected(self):
        t = self.validation.observation_time.iloc[0]
        prefix = self.raw.loc[self.raw.observation_time.le(t)].tail(674)
        measured_subset = pd.concat([prefix.iloc[:10], prefix.iloc[11:]])
        with self.assertRaisesRegex(ValueError, "complete aligned"):
            features_at_issue(measured_subset, t)

    def test_loaded_model_matches_saved_predictions_and_returns_json(self):
        run_dir = ROOT / "reports/steel/duy_2026-10-02"
        forecaster = SteelForecaster(run_dir)
        batch = pd.read_csv(run_dir / "validation_predictions.csv", float_precision="round_trip")
        for i in (0, 95, 672, 2875):
            t = self.validation.observation_time.iloc[i]
            result = forecaster.predict(self.window(t), t)
            self.assertAlmostEqual(result["predicted_next_60m_kWh"], batch.loc[i, f"pred_{forecaster.configuration}_kWh"], places=9)
            self.assertEqual(pd.Timestamp(result["forecast_end"]), t+pd.Timedelta(hours=1))
            self.assertEqual(pd.Timestamp(result["forecast_start"]), t+pd.Timedelta(minutes=15))
            self.assertEqual(result["status"], "forecast_only_no_alert_policy")
            json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
