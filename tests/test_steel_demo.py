"""Exercise replay navigation and payloads using the real local model/data."""
from pathlib import Path
import json
import tempfile
import unittest

import pandas as pd

from src.steel.demo import ReplayDemo

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless((ROOT / "reports/steel/duy_2026-10-02/model_handoff.json").exists(), "Publish the Duy model first")
class SteelDemoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.demo = ReplayDemo(ROOT, sessions_dir=Path(self.temp.name))

    def tearDown(self):
        self.demo.close()
        self.temp.cleanup()

    def assert_no_future(self, state):
        now = pd.Timestamp(state["clock"])
        self.assertTrue(all(pd.Timestamp(row["time"]) <= now for row in state["history"]))
        for row in state["forecasts"]:
            if pd.Timestamp(row["forecast_end"]) > now:
                self.assertIsNone(row["actual"])
                self.assertIsNone(row["error"])

    def test_actual_appears_after_four_steps_only_and_reset_is_new_session(self):
        state = self.demo.dispatch("reset", "2018-09-01T08:00")
        first_session = state["session"]
        self.assertIsNone(state["forecasts"][0]["actual"])
        for i in range(4):
            state = self.demo.dispatch("step")
            self.assert_no_future(state)
            self.assertEqual(state["metrics"]["models"][0]["all_matured"]["rows"], int(i == 3))
        state = self.demo.dispatch("reset", "2018-09-01T08:00")
        self.assertNotEqual(first_session, state["session"])
        self.assertEqual(state["metrics"]["forecasts"], 1)
        self.assertIsNone(state["forecasts"][0]["actual"])

    def test_end_drains_pending_labels_without_new_out_of_range_forecasts(self):
        state = self.demo.dispatch("reset", "2018-09-30T22:45")
        for _ in range(4):
            state = self.demo.dispatch("step")
            self.assert_no_future(state)
        self.assertTrue(state["finished"])
        self.assertEqual(state["metrics"]["forecasts"], 1)
        self.assertEqual(state["metrics"]["models"][0]["pending"], 0)
        self.assertIsNone(state["current_forecast"])
        again = self.demo.dispatch("step")
        self.assertEqual(again["clock"], state["clock"])

    def test_invalid_date_leaves_existing_session_intact(self):
        state = self.demo.dispatch("reset", "2018-09-01T00:00")
        with self.assertRaises(ValueError):
            self.demo.dispatch("reset", "2018-10-01T00:00")
        after = self.demo.dispatch("state")
        self.assertEqual(state["session"], after["session"])
        self.assertEqual(state["clock"], after["clock"])

    def test_missing_model_payload_has_no_fake_prediction_or_actual(self):
        with tempfile.TemporaryDirectory() as path:
            demo = ReplayDemo(ROOT, model_dir=Path(path) / "missing", sessions_dir=Path(path) / "sessions")
            try:
                state = demo.dispatch("reset", "2018-09-01T00:00")
                self.assertEqual(state["current_forecast"]["status"], "model_unavailable")
                self.assertIsNone(state["current_forecast"]["prediction"])
                self.assertEqual(state["forecasts"], [])
                self.assertTrue(state["errors"])
                self.assertIsNone(state["model_info"])
            finally:
                demo.close()

    def test_display_metadata_identifies_the_loaded_model_and_its_validation(self):
        self.assertIsNone(self.demo.state()["model_info"])
        state = self.demo.dispatch("reset", "2018-09-01T08:00")
        manifest = json.loads((self.demo.model_dir / "model_handoff.json").read_text())
        info = state["model_info"]
        self.assertEqual(info["name"], manifest["selected_model"])
        self.assertEqual(info["run"], self.demo.model_dir.name)
        self.assertEqual(info["model_sha256"], state["current_forecast"]["prediction"]["model_sha256"])
        self.assertEqual(info["validation_mae_kWh"], manifest["validation_mae_kWh"])
        self.assertEqual(info["validation_rmse_kWh"], manifest["validation_rmse_kWh"])
        self.assertEqual(info["feature_implementation"], manifest["feature_implementation"])

    def test_matured_forecast_with_missing_real_observation_is_not_scored(self):
        self.demo.dispatch("reset", "2018-09-01T08:00")
        # Remove a real row only in this session; do not change measurements or CSV.
        self.demo.history = self.demo.history.loc[
            self.demo.history.observation_time.ne(pd.Timestamp("2018-09-01 08:30"))]
        for _ in range(4):
            state = self.demo.dispatch("step")
        original = next(r for r in state["forecasts"] if r["issue_time"] == "2018-09-01T08:00:00")
        self.assertEqual(original["label_status"], "waiting_for_observations")
        self.assertIsNone(original["actual"])
        self.assertIsNone(original["error"])
        self.assertEqual(state["metrics"]["models"][0]["all_matured"]["rows"], 0)

    def test_midnight_keeps_same_session_and_waits_for_real_actuals(self):
        initial = self.demo.dispatch("reset", "2018-09-01T23:45")
        for i in range(4):
            state = self.demo.dispatch("step")
            self.assert_no_future(state)
            self.assertEqual(state["session"], initial["session"])
            self.assertEqual(state["metrics"]["models"][0]["all_matured"]["rows"], int(i == 3))
        self.assertEqual(state["clock"], "2018-09-02T00:45:00")

    def decision(self, state, **changes):
        return {"session": state["session"], "clock": state["clock"],
                "request_id": state["current_forecast"]["request_id"],
                "operator": "QA demo", "action": "inspect", "note": "Kiểm tra dự báo đang hiển thị.",
                "token": "qa-decision-1", **changes}

    def test_decision_retry_is_idempotent_and_context_never_gains_future_actual(self):
        state = self.demo.dispatch("reset", "2018-09-01T08:00")
        command = self.decision(state)
        saved = self.demo.dispatch("decision", decision=command)
        again = self.demo.dispatch("decision", decision=command)
        self.assertEqual(saved["decisions"], again["decisions"])
        self.assertEqual(len(saved["decisions"]), 1)
        context = saved["decisions"][0]["context"]
        self.assertNotIn("actual", context)
        self.assertEqual(context["alert_policy"], "not_configured")
        self.assertEqual(context["model_sha256"], state["current_forecast"]["prediction"]["model_sha256"])
        with self.assertRaises(ValueError):
            self.demo.dispatch("decision", decision={**command, "note": "Khác nội dung cùng mã"})
        for _ in range(4):
            matured = self.demo.dispatch("step")
        self.assertEqual(matured["decisions"], saved["decisions"])
        self.assert_no_future(matured)

    def test_stale_clock_or_session_cannot_write_to_new_observation(self):
        state = self.demo.dispatch("reset", "2018-09-01T08:00")
        command = self.decision(state)
        self.demo.dispatch("step")
        with self.assertRaises(ValueError):
            self.demo.dispatch("decision", decision=command)
        self.demo.dispatch("reset", "2018-09-01T08:00")
        with self.assertRaises(ValueError):
            self.demo.dispatch("decision", decision=command)
        self.assertEqual(self.demo.dispatch("state")["decisions"], [])

    def test_bad_inputs_do_not_write_and_corrections_append(self):
        state = self.demo.dispatch("reset", "2018-09-01T08:00")
        for changes in ({"operator": " "}, {"note": ""}, {"note": "x"*501},
                        {"action": "switch_off"}, {"token": None}, {"request_id": 99999}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.demo.dispatch("decision", decision=self.decision(state, **changes))
        self.assertEqual(self.demo.state()["decisions"], [])
        self.demo.dispatch("decision", decision=self.decision(state, action="acknowledge"))
        self.demo.dispatch("decision", decision=self.decision(state, token="correction", note="Bổ sung kiểm tra."))
        self.assertEqual(len(self.demo.state()["decisions"]), 2)

    def test_missing_real_history_rejects_forecast_but_allows_inspection_log(self):
        # Only take a subset of real rows; never invent or alter a measured value.
        start = "2018-09-01T08:00"
        self.demo.history = self.demo.history.loc[self.demo.history.observation_time.ne(pd.Timestamp(start))]
        state = self.demo.dispatch("reset", start)
        self.assertEqual(state["current_forecast"]["status"], "invalid_history")
        self.assertIsNone(state["current_forecast"]["prediction"])
        self.assertEqual(state["metrics"]["input_quality"]["invalid_history_rate"], 1)
        with self.assertRaises(ValueError):
            self.demo.dispatch("decision", decision=self.decision(state, action="acknowledge"))
        saved = self.demo.dispatch("decision", decision=self.decision(state))
        self.assertIsNone(saved["decisions"][0]["context"]["predicted_kWh"])
        self.assertEqual(saved["decisions"][0]["context"]["status"], "invalid_history")

    def test_decisions_survive_database_reopen_but_do_not_leak_into_reset(self):
        from src.steel.monitoring import ForecastMonitor
        state = self.demo.dispatch("reset", "2018-09-01T08:00")
        saved = self.demo.dispatch("decision", decision=self.decision(state))
        path = Path(self.temp.name) / state["session"] / "monitor.sqlite3"
        self.demo.close()
        with ForecastMonitor(self.demo.model_dir, path) as monitor:
            self.assertEqual(monitor.decisions(state["clock"]), saved["decisions"])
        reset = self.demo.dispatch("reset", "2018-09-01T08:00")
        self.assertEqual(reset["decisions"], [])


if __name__ == "__main__":
    unittest.main()
