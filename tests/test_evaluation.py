"""Các ca lỗi ảnh hưởng kết luận của lớp cảnh báo."""

import unittest

import numpy as np
import pandas as pd

from src.steel.evaluation import calibrate_upper_buffer, episodes, evaluate_alerts


class EvaluationTests(unittest.TestCase):
    def test_upper_buffer_uses_underforecast_errors(self):
        self.assertAlmostEqual(calibrate_upper_buffer([10, 10, 10], [12, 9, 5], .5), 1)

    def test_episode_does_not_bridge_a_missing_timestamp(self):
        times = pd.to_datetime(["2018-01-01 00:00", "2018-01-01 00:15", "2018-01-01 00:45"])
        self.assertEqual(episodes([True, True, True], times), [{0, 1}, {2}])

    def test_false_alarm_denominators_are_explicit(self):
        times = pd.date_range("2018-01-01", periods=4, freq="15min")
        report = evaluate_alerts([20, 1, 1, 20], [20, 20, 1, 1], times, 10, 0)
        self.assertEqual([report[key] for key in ("tp", "fp", "fn", "tn")], [1, 1, 1, 1])
        self.assertEqual(report["false_positive_rate"], .5)
        self.assertEqual(report["false_discovery_rate"], .5)
        self.assertFalse(report["lead_time_measured"])

    def test_empty_nonfinite_or_invalid_policy_is_rejected(self):
        for actual, predicted in (([], []), ([1], [np.nan]), ([1], [-1])):
            with self.assertRaises(ValueError):
                calibrate_upper_buffer(actual, predicted)
        with self.assertRaises(ValueError):
            evaluate_alerts([1], [1], ["2018-01-01"], -1, 0)


if __name__ == "__main__":
    unittest.main()
