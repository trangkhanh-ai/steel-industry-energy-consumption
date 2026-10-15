"""Kiểm phân tích lỗi bằng nhãn thật; bản sao sửa giá trị chỉ là fault fixture."""
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from scripts.analyze_forecast_errors import classify, monthly_metrics, validate_predictions
from src.steel.preprocessing import (load_official_raw, build_supervised_rows,
                                     split_supervised_rows)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'reports/steel/release_2026-10-14'


class ForecastErrorAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, _ = load_official_raw(ROOT/'data/steel/raw/Steel_industry_data.csv')
        cls.frame = pd.read_csv(RUN/'test_predictions.csv', parse_dates=['issue_time'])
        cls.origins = split_supervised_rows(build_supervised_rows(cls.raw))[0]['test'].observation_time
        cls.threshold = 379.7710000000002
        cls.margin = .31192691877947

    def test_all_labels_match_the_actual_four_records_and_fixed_origins(self):
        actual = validate_predictions(self.raw, self.frame, self.origins)
        self.assertEqual(actual.shape, (5852, 4))

    def test_changed_label_is_detected_without_modifying_source_csv(self):
        damaged = self.frame.copy()
        damaged.loc[0, 'actual_15m_kWh'] += 1  # Invalid software-test copy only.
        with self.assertRaises(AssertionError):
            validate_predictions(self.raw, damaged, self.origins)

    def test_missing_or_reordered_origins_cannot_silently_align(self):
        for damaged in (self.frame.iloc[1:], self.frame.iloc[::-1]):
            with self.subTest(rows=len(damaged)), self.assertRaises(ValueError):
                validate_predictions(self.raw, damaged, self.origins)

    def test_khanh_old_miss_is_current_true_positive_not_a_new_false_negative(self):
        row = self.frame.loc[self.frame.issue_time.eq('2018-11-22 08:15')].iloc[0]
        usage = self.raw.loc[self.raw.observation_time.eq(row.issue_time), 'Usage_kWh'].item()
        self.assertAlmostEqual(usage, 41.51)
        self.assertAlmostEqual(row.hour_total_actual_kWh, 486.25)
        result = classify([row.hour_total_actual_kWh],
                           [row.hour_total_calibrated_upper_kWh > self.threshold], self.threshold)
        self.assertEqual(result.item(), 'TP')

    def test_monthly_confusion_counts_account_for_every_test_origin(self):
        monthly = monthly_metrics(self.frame, self.threshold, self.margin)
        self.assertEqual(monthly.origins.sum(), 5852)
        self.assertEqual(monthly.tp.sum(), 155)
        self.assertEqual(monthly.fp.sum(), 186)
        self.assertEqual(monthly.fn.sum(), 24)
        self.assertTrue(np.array_equal(monthly[['tp', 'fp', 'fn', 'tn']].sum(axis=1), monthly.origins))


if __name__ == '__main__':
    unittest.main()
