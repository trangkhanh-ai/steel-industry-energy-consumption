"""Diagnostic invariants checked against the real train/validation records."""
import unittest

import numpy as np
import pandas as pd

from scripts.analyze_and_tune_steel import ROOT, diagnostic_rows, grouped_errors
from scripts.train_steel_models import TIMESTAMPS
from src.steel.preprocessing import TARGET


class SteelDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = pd.read_csv(ROOT / "data/steel/processed/steel_next_60m_train.csv", parse_dates=TIMESTAMPS)
        cls.val = pd.read_csv(ROOT / "data/steel/processed/steel_next_60m_validation.csv", parse_dates=TIMESTAMPS)
        # Last-hour persistence calculated from measured energy, without fake values.
        cls.pred = {"last_hour": cls.val.usage_sum_last_1h.to_numpy()}

    def test_diagnostic_thresholds_use_train_and_do_not_change_with_validation_subset(self):
        rows, thresholds = diagnostic_rows(self.val, self.pred, self.train)
        _, prefix_thresholds = diagnostic_rows(self.val.iloc[:100], {"last_hour": self.pred["last_hour"][:100]}, self.train)
        self.assertEqual(thresholds, prefix_thresholds)
        self.assertAlmostEqual(thresholds["high_target_kWh"], np.quantile(self.train[TARGET], .95))
        change = self.train[TARGET] - self.train.usage_sum_last_1h
        self.assertAlmostEqual(thresholds["large_rise_kWh"], np.quantile(change[change > 0], .95))
        self.assertEqual(int(rows.high_load.sum()), 93)

    def test_grouped_errors_reconcile_to_overall_measured_error(self):
        rows, _ = diagnostic_rows(self.val, self.pred, self.train)
        overall = np.abs(self.pred["last_hour"] - self.val[TARGET]).mean()
        for group in ("hour", "weekday", "high_load", "change_group"):
            result = grouped_errors(rows, self.pred, group)
            self.assertEqual(result.rows.sum(), len(self.val))
            self.assertAlmostEqual(np.average(result.mae_kWh, weights=result.rows), overall)

    def test_retrospective_change_and_calendar_match_observed_rows(self):
        rows, thresholds = diagnostic_rows(self.val, self.pred, self.train)
        np.testing.assert_allclose(rows.change_next_minus_last_kWh, self.val[TARGET] - self.val.usage_sum_last_1h)
        np.testing.assert_array_equal(rows.hour, self.val.forecast_start.dt.hour)
        np.testing.assert_array_equal(rows.change_group.eq("large_rise"), rows.change_next_minus_last_kWh > thresholds["large_rise_kWh"])
        np.testing.assert_array_equal(rows[TARGET], self.val[TARGET])


if __name__ == "__main__":
    unittest.main()
