"""Modeling checks using only original Steel observations (no synthetic rows)."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.steel.modeling import historical_baselines, make_candidates, regression_metrics
from src.steel.preprocessing import BASE_FEATURES, TARGET, load_official_raw

ROOT = Path(__file__).resolve().parents[1]


class SteelModelingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw, _ = load_official_raw(ROOT / "data/steel/raw/Steel_industry_data.csv")
        cls.history = raw.set_index("observation_time").Usage_kWh.loc[:"2018-09-30 23:45"]

    def test_baselines_match_actual_historical_hour_windows(self):
        origins = pd.DatetimeIndex(["2018-09-01 00:00", "2018-09-15 08:15", "2018-09-30 22:45"])
        predicted = historical_baselines(self.history, origins)
        for t in origins:
            self.assertAlmostEqual(predicted.loc[t, "last_hour"], self.history.loc[t-pd.Timedelta(minutes=45):t].sum())
            for name, days in (("previous_day", 1), ("previous_week", 7)):
                start = t - pd.Timedelta(days=days) + pd.Timedelta(minutes=15)
                end = t - pd.Timedelta(days=days) + pd.Timedelta(hours=1)
                self.assertEqual(len(self.history.loc[start:end]), 4)
                self.assertAlmostEqual(predicted.loc[t, name], self.history.loc[start:end].sum())

    def test_forecasts_match_with_only_real_history_available_at_issue_time(self):
        # Truncation uses existing measured rows; no observations are changed.
        origins = pd.DatetimeIndex(["2018-09-01 00:00", "2018-09-15 08:15", "2018-09-30 22:45"])
        whole = historical_baselines(self.history, origins)
        for t in origins:
            prefix = historical_baselines(self.history.loc[:t], pd.DatetimeIndex([t]))
            np.testing.assert_allclose(prefix.iloc[0].to_numpy(), whole.loc[t].to_numpy())

    def test_ridge_scaler_fits_real_train_only(self):
        train = pd.read_csv(ROOT / "data/steel/processed/steel_next_60m_train.csv")
        val = pd.read_csv(ROOT / "data/steel/processed/steel_next_60m_validation.csv")
        model = make_candidates()["ridge"]
        model.fit(train[list(BASE_FEATURES)], train[TARGET])
        scaler = model.named_steps["scaler"]
        np.testing.assert_allclose(scaler.mean_, train[list(BASE_FEATURES)].mean().to_numpy())
        self.assertEqual(int(scaler.n_samples_seen_), len(train))
        before = scaler.mean_.copy()
        pred = model.predict(val[list(BASE_FEATURES)])
        np.testing.assert_array_equal(before, scaler.mean_)
        self.assertTrue(np.isfinite(pred).all())
        measured = regression_metrics(val[TARGET].to_numpy(), pred)
        self.assertAlmostEqual(measured["mae_kWh"], float(np.mean(np.abs(pred-val[TARGET]))))

    def test_hgb_has_no_random_internal_validation_split(self):
        self.assertIs(make_candidates()["hist_gradient_boosting"].early_stopping, False)


if __name__ == "__main__":
    unittest.main()
