"""Kiểm rủi ro thật của policy: dữ liệu calibration, biên thời gian và đơn vị."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_final_evaluation import make_policy, predictions_for_baselines, run
from src.steel.evaluation import calibrate_upper_buffer, episodes, evaluate_alerts, inspection_cost
from src.steel.preprocessing import BASE_FEATURES, TARGET


def handoff():
    return {"selected_model": "hgb_31_leaves", "model_sha256": "model-hash",
            "feature_implementation": "duy_numpy_windows_v1", "features": list(BASE_FEATURES),
            "source_sha256": "source-hash", "validation_mae_kWh": 15.882850491592112}


def calibration_rows():
    times = pd.date_range("2018-10-01", periods=4, freq="15min")
    return pd.DataFrame({"observation_time": times,
                         "forecast_end": times + pd.Timedelta(hours=1),
                         TARGET: [10., 20., 30., 40.]})


class FinalEvaluationTests(unittest.TestCase):
    def test_policy_fits_clipped_residuals_only_on_october(self):
        part = calibration_rows()
        prediction = np.array([30., 15., 20., 25.])
        policy = make_policy(handoff(), 100, part, prediction, np.arange(100.))
        self.assertAlmostEqual(policy["upper_buffer_kWh"], np.quantile([0, 5, 10, 15], .9))
        self.assertEqual(policy["model_sha256"], "model-hash")
        self.assertEqual(policy["target_unit"], "kWh")
        self.assertEqual(policy["valid_from"], "2018-11-01T00:00:00")
        self.assertFalse(policy["conditional_P90_estimated"])
        self.assertFalse(policy["coverage_guaranteed"])
        self.assertAlmostEqual(sum(policy["reference_usage_probabilities"]), 1.)
        self.assertEqual(len(policy["reference_usage_probabilities"]), len(policy["reference_usage_bins"]) + 1)
        # Dữ liệu test nằm ngoài hàm: thay đổi test không thể đổi buffer này.
        self.assertNotIn("test", policy["buffer_rule"].lower())

    def test_policy_rejects_labels_from_validation_test_or_crossing_boundary(self):
        for start in ("2018-09-01", "2018-11-01", "2018-10-31 23:00"):
            part = calibration_rows()
            times = pd.date_range(start, periods=4, freq="15min")
            part["observation_time"] = times
            part["forecast_end"] = times + pd.Timedelta(hours=1)
            with self.subTest(start=start), self.assertRaises(ValueError):
                make_policy(handoff(), 100, part, [10, 20, 30, 40])

    def test_policy_rejects_empty_duplicate_and_reversed_origins(self):
        part = calibration_rows()
        for broken in (part.head(0), part.iloc[::-1], pd.concat([part.iloc[[0]], part.iloc[[0]]])):
            with self.assertRaises(ValueError):
                make_policy(handoff(), 100, broken, np.zeros(len(broken)))

    def test_policy_rejects_missing_interval_off_grid_or_wrong_horizon(self):
        part = calibration_rows()
        gap = part.drop(index=1)
        off_grid = part.copy()
        off_grid["observation_time"] += pd.Timedelta(seconds=1)
        wrong_horizon = part.copy()
        wrong_horizon["forecast_end"] += pd.Timedelta(minutes=15)
        for broken in (gap, off_grid, wrong_horizon):
            with self.assertRaises(ValueError):
                make_policy(handoff(), 100, broken, np.zeros(len(broken)))

    def test_buffer_clips_before_linear_quantile(self):
        # Khi quantile nội suy qua residual âm/dương, clip từng residual trước
        # cho kết quả khác max(0,Q90(residual)); phải ghi đúng convention.
        actual, prediction = [0, 10], [100, 0]
        self.assertAlmostEqual(calibrate_upper_buffer(actual, prediction, .9), 9.)
        self.assertNotEqual(calibrate_upper_buffer(actual, prediction, .9), max(0, np.quantile([-100, 10], .9)))

    def test_confusion_and_upper_coverage_use_strict_exceedance(self):
        times = pd.date_range("2018-11-01", periods=4, freq="15min")
        metric = evaluate_alerts([5, 20, 30, 10], [15, 10, 25, 10], times, threshold=10, buffer=0)
        self.assertEqual((metric["tp"], metric["fp"], metric["fn"], metric["tn"]), (1, 1, 1, 1))
        self.assertEqual(metric["precision"], .5)
        self.assertEqual(metric["recall"], .5)
        self.assertEqual(metric["false_positive_rate"], .5)
        self.assertEqual(metric["empirical_upper_coverage"], .5)
        self.assertEqual(metric["peak_rows"], 2)
        self.assertFalse(metric["lead_time_measured"])
        self.assertFalse(metric["savings_measured"])
        self.assertIsNotNone(metric["roc_auc_ranking"])

    def test_episodes_never_bridge_missing_interval(self):
        times = pd.to_datetime(["2018-11-01 00:00", "2018-11-01 00:15", "2018-11-01 01:00"])
        self.assertEqual(episodes([True, True, True], times), [{0, 1}, {2}])
        metric = evaluate_alerts([20, 20, 20], [20, 20, 20], times, threshold=10, buffer=0)
        self.assertEqual(metric["true_origin_episodes"], 2)
        self.assertEqual(metric["detected_true_episodes"], 2)

    def test_auc_is_none_if_no_positive_class(self):
        times = pd.date_range("2018-11-01", periods=2, freq="15min")
        metric = evaluate_alerts([1, 2], [1, 2], times, 100, 0)
        self.assertIsNone(metric["roc_auc_ranking"])
        self.assertIsNone(metric["peak_mae_kWh"])
        self.assertEqual(metric["recall"], 0.)

    def test_cost_scenario_is_explicit_points_not_money(self):
        self.assertEqual(inspection_cost({"fp": 3, "fn": 2}, 1, 5), 13.)
        for invalid in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                inspection_cost({"fp": 3, "fn": 2}, invalid, 5)

    def test_train_fitted_baseline_does_not_read_test_labels(self):
        times = pd.date_range("2018-01-01", periods=1000, freq="15min")
        raw = pd.DataFrame({"observation_time": times, "Usage_kWh": np.arange(1000, dtype=float)})
        part = pd.DataFrame({"observation_time": times[700:704],
                             "forecast_start": times[700:704] + pd.Timedelta(minutes=15),
                             "usage_sum_last_1h": [1., 2., 3., 4.], TARGET: [5, 6, 7, 8]})
        state = {"global_mean_kWh": 123., "slots": []}
        before = predictions_for_baselines(raw, part, state)
        part[TARGET] = 100000
        after = predictions_for_baselines(raw, part, state)
        for name in before:
            np.testing.assert_array_equal(before[name], after[name])
        self.assertEqual(before["previous_day"][0], np.arange(605, 609).sum())
        self.assertEqual(before["previous_week"][0], np.arange(29, 33).sum())

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            sentinel = path / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run(path)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
