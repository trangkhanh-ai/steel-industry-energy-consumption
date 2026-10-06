"""Persistent/delayed scoring checks using only original measured rows."""
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.steel.monitoring import ForecastMonitor
from src.steel.preprocessing import load_official_raw

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "reports/steel/duy_2026-10-02"


@unittest.skipUnless((MODEL / "model_handoff.json").exists(), "Publish the Duy model first")
class SteelMonitoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw, _ = load_official_raw(ROOT / "data/steel/raw/Steel_industry_data.csv")
        cls.t = pd.Timestamp("2018-09-01")
        cls.real = raw.loc[raw.observation_time.between(cls.t-pd.Timedelta(days=7), cls.t+pd.Timedelta(hours=2)), ["observation_time", "Usage_kWh"]].reset_index(drop=True)
        cls.initial = cls.real.loc[cls.real.observation_time.le(cls.t)]

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "monitor.sqlite3"
        self.monitor = ForecastMonitor(MODEL, self.path)

    def tearDown(self):
        self.monitor.close()
        self.temp.cleanup()

    def until(self, t):
        return self.real.loc[self.real.observation_time.le(t)]

    def test_actual_is_pending_until_full_hour_then_scored_once(self):
        response = self.monitor.forecast(self.initial, self.t)
        self.assertEqual(response["status"], "forecast_ready")
        self.assertIsNone(self.monitor.db.execute("SELECT actual_kWh FROM forecasts").fetchone()[0])
        before = self.t+pd.Timedelta(minutes=45)
        self.assertEqual(self.monitor.observe(self.until(before), before)["newly_evaluated"], 0)
        end = self.t+pd.Timedelta(hours=1)
        self.assertEqual(self.monitor.observe(self.until(end), end)["newly_evaluated"], 1)
        value = self.monitor.db.execute("SELECT actual_kWh FROM forecasts").fetchone()[0]
        expected = self.real.loc[self.real.observation_time.gt(self.t) & self.real.observation_time.le(end), "Usage_kWh"].sum()
        self.assertAlmostEqual(value, expected)
        self.assertEqual(self.monitor.observe(self.until(end), end)["newly_evaluated"], 0)
        self.assertEqual(self.monitor.snapshot(before)["models"][0]["all_matured"]["rows"], 0)
        self.assertEqual(self.monitor.snapshot(end)["models"][0]["all_matured"]["rows"], 1)

    def test_future_actuals_are_rejected_and_logged(self):
        self.monitor.forecast(self.initial, self.t)
        with self.assertRaisesRegex(ValueError, "before as_of"):
            self.monitor.observe(self.real, self.t)
        self.assertIsNone(self.monitor.db.execute("SELECT actual_kWh FROM forecasts").fetchone()[0])
        self.assertEqual(self.monitor.db.execute("SELECT status FROM requests ORDER BY id DESC LIMIT 1").fetchone()[0], "invalid_actual_history")

    def test_missing_measured_interval_keeps_label_pending(self):
        self.monitor.forecast(self.initial, self.t)
        end = self.t+pd.Timedelta(hours=1)
        measured_subset = self.until(end).loc[lambda f: f.observation_time.ne(self.t+pd.Timedelta(minutes=30))]
        result = self.monitor.observe(measured_subset, end)
        self.assertEqual(result["newly_evaluated"], 0)
        self.assertEqual(result["matured_waiting_for_observations"], 1)
        self.assertEqual(self.monitor.observe(self.until(end), end)["newly_evaluated"], 1)

    def test_invalid_forecast_input_returns_explicit_status_without_zero_forecast(self):
        response = self.monitor.forecast(self.initial.iloc[1:], self.t)
        self.assertEqual(response["status"], "invalid_history")
        self.assertIsNone(response["prediction"])
        self.assertEqual(self.monitor.db.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0], 0)

    def test_restart_and_duplicate_request_preserve_forecast_and_log_both_requests(self):
        first = self.monitor.forecast(self.initial, self.t)
        self.monitor.close()
        self.monitor = ForecastMonitor(MODEL, self.path)
        second = self.monitor.forecast(self.initial, self.t)
        self.assertEqual(first["forecast_id"], second["forecast_id"])
        self.assertEqual(self.monitor.db.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0], 1)
        self.assertEqual(self.monitor.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0], 2)

    def test_absent_model_is_logged_and_never_falls_back_silently(self):
        with ForecastMonitor(Path(self.temp.name) / "absent_model", Path(self.temp.name) / "missing.sqlite3") as monitor:
            result = monitor.forecast(self.initial, self.t)
            self.assertEqual(result["status"], "model_unavailable")
            self.assertIsNone(result["prediction"])
            self.assertEqual(monitor.db.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0], 0)
            with self.assertRaises(ValueError):
                monitor.record_decision(result["request_id"], self.t, "QA", "acknowledge", "Model absent", "ack")
            monitor.record_decision(result["request_id"], self.t, "QA", "inspect", "Model absent", "inspect")
            self.assertIsNone(monitor.decisions(self.t)[0]["context"]["predicted_kWh"])
            self.assertEqual(monitor.decisions(self.t)[0]["context"]["status"], "model_unavailable")
            monitor.forecast(self.until(self.t + pd.Timedelta(minutes=15)), self.t + pd.Timedelta(minutes=15))
            # No future request or model-loading diagnostic in the denominator.
            self.assertEqual(monitor.snapshot(self.t)["input_quality"]["forecast_attempts"], 1)

    def test_input_error_rate_counts_retries_but_scores_forecasts_once(self):
        self.monitor.forecast(self.initial, self.t)
        self.monitor.forecast(self.initial, self.t)
        self.monitor.forecast(self.initial.iloc[1:], self.t)
        snapshot = self.monitor.snapshot(self.t)
        self.assertEqual(snapshot["forecasts"], 1)
        self.assertEqual(snapshot["input_quality"]["forecast_attempts"], 3)
        self.assertAlmostEqual(snapshot["input_quality"]["invalid_history_rate"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
