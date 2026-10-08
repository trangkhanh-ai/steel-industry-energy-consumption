"""Handoff checks using existing artifacts; altered copies are config tests only."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.steel.inference import SteelForecaster, DEFAULT_MODEL_SUBDIR
from src.steel.policy import DEFAULT_POLICY, ROOT, inspect_policy
from src.steel.monitoring import ForecastMonitor
from src.steel.demo import ReplayDemo


class SteelPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = SteelForecaster(ROOT / DEFAULT_MODEL_SUBDIR)
        cls.original = json.loads(DEFAULT_POLICY.read_text())

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "notebooks/artifacts/policy.json"
        self.path.parent.mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def inspect_copy(self, data):
        self.path.write_text(json.dumps(data), encoding="utf-8")
        return inspect_policy(self.path, self.model, ROOT / "notebooks")

    def test_actual_handoff_is_incompatible_and_never_activates(self):
        result = inspect_policy(model=self.model)
        self.assertEqual(result["status"], "incompatible")
        self.assertEqual({x["code"] for x in result["issues"]}, {"model_mismatch", "artifact_hash_mismatch"})
        self.assertFalse(result["alerts_enabled"])
        self.assertEqual(result["policy_sha256"], hashlib.sha256(DEFAULT_POLICY.read_bytes()).hexdigest())
        self.assertNotIn("threshold_T_value_kWh", result)

    def test_missing_and_unreadable_policy_do_not_break_replay_contract(self):
        self.assertEqual(inspect_policy(self.path, self.model)["status"], "missing")
        self.assertEqual(inspect_policy(self.path.parent, self.model)["status"], "invalid")

    def test_bad_json_duplicate_keys_and_nonfinite_config_rejected(self):
        for content in ('{', '[]', '{"policy_metadata":{},"policy_metadata":{}}', '{"bad":NaN}'):
            with self.subTest(content=content):
                self.path.write_text(content)
                result = inspect_policy(self.path, self.model)
                self.assertEqual(result["status"], "invalid")
                self.assertFalse(result["alerts_enabled"])

    def test_invalid_version_numbers_rule_and_path_rejected(self):
        changes = [("policy_metadata", "policy_version", "unknown"),
                   ("policy_metadata", "model_artifact_path", "../../outside.joblib"),
                   ("policy_metadata", "model_sha256", "not-a-hash"),
                   ("calibration_parameters", "buffer_b_value_kWh", -1),
                   ("calibration_parameters", "threshold_T_value_kWh", True),
                   ("decision_logic", "trigger_condition", "untrusted expression")]
        for part, key, value in changes:
            data = copy.deepcopy(self.original)
            data[part][key] = value
            with self.subTest(key=key):
                self.assertEqual(self.inspect_copy(data)["status"], "invalid")

    def test_changed_feature_order_is_not_compatible(self):
        data = copy.deepcopy(self.original)
        data["policy_metadata"]["feature_schema"].reverse()
        result = self.inspect_copy(data)
        self.assertIn("schema_mismatch", [x["code"] for x in result["issues"]])
        self.assertFalse(result["alerts_enabled"])

    def test_model_unavailable_is_explicit_without_using_policy_as_forecast(self):
        result = inspect_policy()
        self.assertIn("model_unavailable", [x["code"] for x in result["issues"]])
        self.assertFalse(result["alerts_enabled"])

    def test_snapshot_is_frozen_across_file_change_and_database_reopen(self):
        self.path.write_bytes(DEFAULT_POLICY.read_bytes())
        database = Path(self.temp.name) / "monitor.sqlite3"
        with ForecastMonitor(ROOT / DEFAULT_MODEL_SUBDIR, database, self.path) as monitor:
            snapshot = monitor.policy_info
        self.path.write_text("{")
        with ForecastMonitor(ROOT / DEFAULT_MODEL_SUBDIR, database, self.path) as monitor:
            self.assertEqual(monitor.policy_info, snapshot)
        with ForecastMonitor(ROOT / DEFAULT_MODEL_SUBDIR, Path(self.temp.name) / "new.sqlite3", self.path) as monitor:
            self.assertEqual(monitor.policy_info["status"], "invalid")

    def test_real_replay_logs_policy_without_future_labels_or_alerts(self):
        demo = ReplayDemo(ROOT, sessions_dir=Path(self.temp.name) / "sessions")
        try:
            state = demo.dispatch("reset", "2018-09-01T08:00")
            command = {"session": state["session"], "clock": state["clock"],
                       "request_id": state["current_forecast"]["request_id"],
                       "operator": "QA policy", "action": "inspect", "note": "Kiểm bản bàn giao.", "token": "policy-log"}
            saved = demo.dispatch("decision", decision=command)
            context = saved["decisions"][0]["context"]
            self.assertEqual(context["policy_inspection"], state["policy_info"])
            self.assertFalse(context["policy_inspection"]["alerts_enabled"])
            self.assertNotIn("actual", context)
            self.assertEqual(context["alert_policy"], "not_configured")
            for _ in range(4):
                matured = demo.dispatch("step")
            self.assertEqual(matured["decisions"], saved["decisions"])
            self.assertEqual(matured["metrics"]["models"][0]["all_matured"]["rows"], 1)
        finally:
            demo.close()


if __name__ == "__main__":
    unittest.main()
