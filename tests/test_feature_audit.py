"""QA phải phát hiện dữ liệu sai thật, thiếu file hoặc pipeline nhìn tương lai."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.audit_features_and_leakage import audit, check_causality, rebuild_rows
from src.steel.preprocessing import load_official_raw

ROOT = Path(__file__).resolve().parents[1]


class FeatureAuditTests(unittest.TestCase):
    def test_missing_required_files_block_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(audit(Path(directory))["passed"])

    def test_wrong_manifest_and_csv_are_not_reported_as_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "processed"
            shutil.copytree(ROOT / "data/steel/processed", path)
            manifest = pd.read_csv(path / "steel_row_manifest.csv")
            manifest.loc[0, "reason"] = "included"
            manifest.to_csv(path / "steel_row_manifest.csv", index=False)
            train = pd.read_csv(path / "steel_next_60m_train.csv")
            train.loc[0, "usage_sum_last_1h"] += 100
            train.to_csv(path / "steel_next_60m_train.csv", index=False)
            report = audit(path)
            self.assertFalse(report["passed"])
            status = {row["id"]: row["status"] for row in report["checks"]}
            self.assertEqual(status["FE-05"], "BLOCKER")
            self.assertEqual(status["FE-06"], "BLOCKER")

    def test_extra_input_in_metadata_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "processed"
            shutil.copytree(ROOT / "data/steel/processed", path)
            summary_path = path / "steel_quality_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["base_feature_columns"].append("CO2(tCO2)")
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            report = audit(path)
            status = {row["id"]: row["status"] for row in report["checks"]}
            self.assertEqual(status["FE-01"], "BLOCKER")
            self.assertEqual(status["FE-03"], "BLOCKER")
            self.assertEqual(status["FE-04"], "BLOCKER")

    def test_causality_checks_the_real_builder(self):
        raw, _ = load_official_raw(ROOT / "data/steel/raw/Steel_industry_data.csv")

        def leaky_builder(data, implementation):
            rows = rebuild_rows(data, implementation)
            rows["usage_lag_0"] = data["Usage_kWh"].shift(-1)
            return rows

        with patch("scripts.audit_features_and_leakage.rebuild_rows", side_effect=leaky_builder):
            with self.assertRaises(AssertionError):
                check_causality(raw, "legacy_pandas_rolling")


if __name__ == "__main__":
    unittest.main()
