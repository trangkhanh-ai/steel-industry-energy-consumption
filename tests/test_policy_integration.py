"""Integration checks for policy identity, delayed labels, and human logs.

Policy numbers below are isolated test fixtures, not calibration results.
Observed histories and actual labels come from the unmodified UCI CSV.
"""
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.steel.monitoring import ForecastMonitor
from src.steel.policy import performance_watch
from src.steel.preprocessing import load_official_raw

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "reports/steel/duy_2026-10-02"


class PolicyIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A missing release fails this integration check; it is never skipped.
        cls.metadata = json.loads((MODEL / "model_handoff.json").read_text(encoding="utf-8"))
        cls.raw, _ = load_official_raw(ROOT / "data/steel/raw/Steel_industry_data.csv")
        cls.origin = pd.Timestamp("2018-11-01 08:00")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.policy_path = self.folder / "fixture_policy.json"
        self.policy_data = {key: self.metadata[key] for key in
                            ("model_sha256", "source_sha256", "feature_implementation", "target", "features")}
        self.policy_data.update(target_unit="kWh", threshold_kWh=379.771,
                                upper_buffer_kWh=80.0, reference_validation_mae_kWh=15.882850491592112,
                                valid_from="2018-11-01T00:00:00", policy_version="isolated_unit_test_fixture")
        self.write_policy()

    def tearDown(self):
        self.temporary.cleanup()

    def write_policy(self):
        self.policy_path.write_text(json.dumps(self.policy_data, ensure_ascii=False, indent=2), encoding="utf-8")

    def history_until(self, t):
        return self.raw.loc[self.raw.observation_time.le(t),
                            ["observation_time", "Usage_kWh"]].tail(673).reset_index(drop=True)

    def open_monitor(self, name="monitor", with_policy=True, model=MODEL):
        return ForecastMonitor(model, self.folder / (name + ".sqlite3"),
                               policy_path=self.policy_path if with_policy else None)

    def record(self, monitor, response, action="inspect", token="unit-test"):
        return monitor.record_decision(response["request_id"], self.origin,
                                       "QA integration", action, "Fixture; không điều khiển máy.", token)

    def test_active_policy_rejects_duplicate_json_keys(self):
        content = json.dumps(self.policy_data)
        self.policy_path.write_text(content[:-1] + ', "threshold_kWh": 10}', encoding="utf-8")
        with self.open_monitor() as monitor:
            self.assertIsNone(monitor.policy)
            self.assertIn("JSON", monitor.policy_error)

    def test_active_policy_rejects_nonfinite_and_boolean_parameters(self):
        for value in (float("nan"), True):
            self.policy_data["threshold_kWh"] = value
            self.write_policy()
            with self.subTest(value=value), self.open_monitor(name=str(value)) as monitor:
                self.assertIsNone(monitor.policy)
                response = monitor.forecast(self.history_until(self.origin), self.origin)
                self.assertIsNone(response["decision"]["alert"])

    def test_valid_policy_uses_matching_model_and_buffer_without_equipment_action(self):
        with self.open_monitor() as monitor:
            response = monitor.forecast(self.history_until(self.origin), self.origin)
            self.assertEqual(response["status"], "forecast_ready")
            predicted = response["prediction"]["predicted_next_60m_kWh"]
            decision = response["decision"]
            self.assertAlmostEqual(decision["upper_proxy_kWh"], predicted + 80.0)
            self.assertEqual(decision["alert"], predicted + 80.0 > 379.771)
            self.assertFalse(decision["automatic_equipment_action"])
            self.assertEqual(decision["policy_sha256"], monitor.policy.sha256)
            self.assertIsNone(monitor.snapshot(self.origin).get("alert_metrics"))

    def test_policy_model_hash_mismatch_preserves_prediction_but_never_alert(self):
        self.policy_data["model_sha256"] = "0" * 64
        self.write_policy()
        with self.open_monitor() as monitor:
            response = monitor.forecast(self.history_until(self.origin), self.origin)
            self.assertEqual(response["status"], "forecast_ready")
            self.assertIsNotNone(response["prediction"])
            self.assertEqual(response["decision"]["status"], "policy_unavailable")
            self.assertIsNone(response["decision"]["alert"])
            self.assertIn("model_sha256", response["decision"]["detail"])

    def test_policy_cannot_make_alert_before_calibration_period_finishes(self):
        before = pd.Timestamp("2018-09-01 08:00")
        with self.open_monitor() as monitor:
            response = monitor.forecast(self.history_until(before), before)
            self.assertEqual(response["status"], "forecast_ready")
            self.assertEqual(response["decision"]["status"], "policy_unavailable")
            self.assertIsNone(response["decision"]["alert"])
            self.assertNotIn("upper_proxy_kWh", response["decision"])

    def test_invalid_reference_distribution_never_displays_valid_alert(self):
        self.policy_data.update(reference_usage_bins=[10.0, 20.0],
                                reference_usage_probabilities=[1.0])
        self.write_policy()
        with self.open_monitor() as monitor:
            response = monitor.forecast(self.history_until(self.origin), self.origin)
            self.assertEqual(response["status"], "forecast_ready")
            self.assertEqual(response["decision"]["status"], "policy_unavailable")
            self.assertIsNone(response["decision"]["alert"])

    def test_saved_decision_has_immutable_policy_and_no_future_actual(self):
        with self.open_monitor() as monitor:
            response = monitor.forecast(self.history_until(self.origin), self.origin)
            self.record(monitor, response)
            initial = monitor.decisions(self.origin)
            self.assertEqual(initial[0]["context"]["decision"], response["decision"])
            self.assertNotIn("actual_kWh", initial[0]["context"])
            # Changing a policy file does not rewrite an already saved context.
            self.policy_data["upper_buffer_kWh"] = 180.0
            self.write_policy()
            end = self.origin + pd.Timedelta(hours=1)
            monitor.observe(self.history_until(end), end)
            self.assertEqual(monitor.decisions(end), initial)
        with self.open_monitor() as reopened:
            self.assertEqual(reopened.decisions(end), initial)
            conflict = reopened.forecast(self.history_until(self.origin), self.origin)
            self.assertEqual(conflict["status"], "forecast_conflict")
            self.assertIsNone(conflict["prediction"])
            self.assertIsNone(conflict["decision"]["alert"])

    def test_actual_and_alert_score_remain_pending_until_four_measured_intervals(self):
        with self.open_monitor() as monitor:
            monitor.forecast(self.history_until(self.origin), self.origin)
            before = self.origin + pd.Timedelta(minutes=45)
            monitor.observe(self.history_until(before), before)
            self.assertIsNone(monitor.snapshot(before)["alert_metrics"])
            end = self.origin + pd.Timedelta(hours=1)
            self.assertEqual(monitor.observe(self.history_until(end), end)["newly_evaluated"], 1)
            actual = self.raw.loc[self.raw.observation_time.gt(self.origin) &
                                  self.raw.observation_time.le(end), "Usage_kWh"].sum()
            self.assertAlmostEqual(monitor.db.execute("SELECT actual_kWh FROM forecasts").fetchone()[0], actual)
            self.assertIsNotNone(monitor.snapshot(end)["alert_metrics"])
            self.assertIsNone(monitor.snapshot(before)["alert_metrics"])

    def test_missing_model_canonical_time_allows_inspection_but_blocks_acknowledge(self):
        with self.open_monitor(model=self.folder / "absent_model") as monitor:
            response = monitor.forecast(self.history_until(self.origin), self.origin)
            self.assertEqual(response["status"], "model_unavailable")
            self.assertIsNone(response["prediction"])
            self.assertIsNone(response["decision"]["alert"])
            with self.assertRaises(ValueError):
                self.record(monitor, response, action="acknowledge")
            self.record(monitor, response)
            context = monitor.decisions(self.origin)[0]["context"]
            self.assertEqual(context["issue_time"], self.origin.isoformat())
            self.assertIsNone(context["predicted_kWh"])

    def test_missing_observation_with_policy_never_alerts_or_creates_prediction(self):
        incomplete = self.history_until(self.origin).iloc[1:]
        with self.open_monitor() as monitor:
            response = monitor.forecast(incomplete, self.origin)
            self.assertEqual(response["status"], "invalid_history")
            self.assertIsNone(response["prediction"])
            self.assertIsNone(response["decision"]["alert"])
            self.assertEqual(monitor.db.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0], 0)
            self.record(monitor, response)

    def test_without_policy_still_forecasts_without_an_alert(self):
        with self.open_monitor(with_policy=False) as monitor:
            response = monitor.forecast(self.history_until(self.origin), self.origin)
            self.assertEqual(response["status"], "forecast_ready")
            self.assertEqual(response["decision"]["status"], "not_configured")
            self.assertIsNone(response["decision"]["alert"])

    def test_performance_review_requires_labels_contiguous_windows_and_human_review(self):
        observations = self.history_until(self.origin).tail(8)
        scored = pd.DataFrame({"issue_time": observations.observation_time,
                               "actual_kWh": observations.Usage_kWh,
                               "predicted_kWh": observations.Usage_kWh + 25.0})
        insufficient = performance_watch(scored.iloc[:7], reference_mae=10.0, week_rows=4)
        self.assertEqual(insufficient["status"], "insufficient_labels")
        reviewed = performance_watch(scored, reference_mae=10.0, week_rows=4)
        self.assertEqual(reviewed["status"], "needs_review")
        self.assertEqual(reviewed["weekly_mae_kWh"], [25.0, 25.0])
        self.assertFalse(reviewed["automatic_retraining"])
        gaps = scored.copy()
        gaps.iloc[4, gaps.columns.get_loc("issue_time")] += pd.Timedelta(minutes=15)
        self.assertEqual(performance_watch(gaps, reference_mae=10.0, week_rows=4)["status"],
                         "non_contiguous_labels")


if __name__ == "__main__":
    unittest.main()
