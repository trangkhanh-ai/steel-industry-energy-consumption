"""Các proxy phải dùng đúng số đo tại t, độc lập target và cột tương lai."""
import unittest

import numpy as np
import pandas as pd

from src.steel.feature_extension import add_extended_features, NEW_FEATURES


class FeatureExtensionTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"usage_lag_0": [0., 5., 10.], "usage_lag_1": [2., 5., 7.],
                                   "known_lagging_reactive_kVarh": [1., 2., 3.],
                                   "known_lagging_power_factor": [95., 90., 89.]})

    def test_proxy_values_and_small_denominator_mask(self):
        result = add_extended_features(self.frame)
        np.testing.assert_array_equal(result.steel_ramp_rate, [-2, 0, 3])
        np.testing.assert_allclose(result.apparent_energy_proxy, np.hypot([0, 5, 10], [1, 2, 3]))
        np.testing.assert_allclose(result.reactive_ratio, [0, 0, .3])
        np.testing.assert_array_equal(result.low_lagging_pf_flag, [0, 0, 1])
        self.assertNotIn("steel_ramp_rate", self.frame)

    def test_target_and_future_columns_do_not_change_features(self):
        original = add_extended_features(self.frame)[list(NEW_FEATURES)]
        changed = self.frame.assign(target_next_60m_kWh=1e9, future_reactive_energy=1e9)
        pd.testing.assert_frame_equal(original, add_extended_features(changed)[list(NEW_FEATURES)])

    def test_invalid_sensors_are_rejected(self):
        for column, value in (("known_lagging_power_factor", 101),
                              ("known_lagging_reactive_kVarh", -1), ("usage_lag_0", np.nan)):
            changed = self.frame.copy()
            changed.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                add_extended_features(changed)

    def test_missing_current_sensor_is_explicit(self):
        with self.assertRaises(KeyError):
            add_extended_features(self.frame.drop(columns="known_lagging_power_factor"))


if __name__ == "__main__":
    unittest.main()
