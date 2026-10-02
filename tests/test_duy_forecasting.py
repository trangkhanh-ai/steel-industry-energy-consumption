"""Kiểm rủi ro thật: lookahead, split, scaler, batch/replay và dữ liệu lỗi."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import sklearn

from scripts.run_duy_pipeline import git_snapshot, run, select_model, write_json
from src.steel.duy_forecasting import (
    FEATURE_IMPLEMENTATION, baseline_predictions, features_at_issue, fit_diurnal_state,
    fixed_models, forecast_features, predict_energy,
)
from src.steel.duy_inference import DuyForecaster
from src.steel.preprocessing import (
    BASE_FEATURES, SPLIT_STARTS, TARGET, build_supervised_rows,
    load_official_raw, sha256_file, split_supervised_rows,
)

ROOT = Path(__file__).resolve().parents[1]


class DuyForecastingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, _ = load_official_raw(ROOT / "data/steel/raw/Steel_industry_data.csv")
        cls.features = forecast_features(cls.raw)
        cls.rows = build_supervised_rows(cls.raw)
        cls.rows.loc[:, list(BASE_FEATURES)] = cls.features.to_numpy()
        cls.splits, _ = split_supervised_rows(cls.rows)

    def test_future_changes_target_but_never_present_features(self):
        index = 5000
        changed = self.raw.copy()
        changed.loc[index + 1:, "Usage_kWh"] += 123
        np.testing.assert_array_equal(
            self.features.iloc[index].to_numpy(), forecast_features(changed).iloc[index].to_numpy()
        )
        target = build_supervised_rows(changed).loc[index, TARGET]
        self.assertAlmostEqual(target - self.rows.loc[index, TARGET], 4 * 123)

    def test_full_batch_and_short_history_are_exactly_equal(self):
        for index in (672, 5000, 23328, 24008, 25000, 30000):
            with self.subTest(index=index):
                history = self.raw.iloc[index - 672:index + 1][["observation_time", "Usage_kWh"]]
                online = features_at_issue(history, history["observation_time"].iloc[-1])
                np.testing.assert_array_equal(self.features.iloc[index].to_numpy(), online.iloc[0].to_numpy())

    def test_labels_never_cross_split_boundary(self):
        names = list(SPLIT_STARTS)
        for name, part in self.splits.items():
            end = SPLIT_STARTS[names[names.index(name) + 1]]
            self.assertTrue(part["forecast_end"].lt(end).all())

    def test_previous_day_baseline_uses_matching_future_hour(self):
        train, validation = self.splits["train"], self.splits["validation"].head(2)
        values = baseline_predictions(self.raw, train, validation)
        index = self.raw.index[self.raw["observation_time"].eq(validation.iloc[0]["observation_time"])][0]
        self.assertAlmostEqual(values["previous_day"][0], self.raw["Usage_kWh"].iloc[index - 95:index - 91].sum())
        self.assertAlmostEqual(values["previous_week"][0], self.raw["Usage_kWh"].iloc[index - 671:index - 667].sum())

    def test_diurnal_baseline_is_fitted_only_on_train_and_uses_actual_hour(self):
        train = self.splits["train"]
        validation = self.splits["validation"].head(12).copy()
        original = baseline_predictions(self.raw, train, validation)["diurnal_train_mean"]
        validation[TARGET] += 10000
        np.testing.assert_array_equal(original, baseline_predictions(self.raw, train, validation)["diurnal_train_mean"])
        position = 2  # forecast_start 00:45; must stay hour 0, not round to 1.
        start = validation.iloc[position]["forecast_start"]
        matching = train.loc[(train["forecast_start"].dt.hour == start.hour)
                             & (train["forecast_start"].dt.dayofweek == start.dayofweek)]
        self.assertAlmostEqual(original[position], matching[TARGET].mean())

    def test_ridge_scaler_is_fitted_on_train_not_validation(self):
        train = self.splits["train"].head(1000)
        validation = self.splits["validation"].head(10)
        model = fixed_models()["ridge_alpha_1"]
        model.fit(train[list(BASE_FEATURES)], train[TARGET])
        np.testing.assert_allclose(model[0].mean_, train[list(BASE_FEATURES)].mean().to_numpy())
        mean_before = model[0].mean_.copy()
        predict_energy(model, validation)
        np.testing.assert_array_equal(model[0].mean_, mean_before)

    def test_invalid_history_is_rejected_instead_of_filled(self):
        history = self.raw.iloc[1000:1673][["observation_time", "Usage_kWh"]].copy()
        issue = history["observation_time"].iloc[-1]
        for broken in (history.iloc[:-1], history.drop(history.index[20]), history.iloc[::-1]):
            with self.assertRaises(ValueError):
                features_at_issue(broken, issue)
        for value in (np.nan, np.inf, -1):
            broken = history.copy()
            broken.iloc[-1, broken.columns.get_loc("Usage_kWh")] = value
            with self.assertRaises(ValueError):
                features_at_issue(broken, issue)
        with self.assertRaises(ValueError):
            features_at_issue(history.head(672), history["observation_time"].iloc[671])

    def test_selection_compares_all_baselines_and_ai(self):
        leaderboard = pd.DataFrame([
            {"model": "last_hour", "kind": "baseline", "mae_kWh": 40},
            {"model": "previous_day", "kind": "baseline", "mae_kWh": 10},
            {"model": "hgb", "kind": "AI", "mae_kWh": 20},
        ])
        self.assertEqual(select_model(leaderboard), "previous_day")
        leaderboard.loc[2, "mae_kWh"] = 10
        self.assertEqual(select_model(leaderboard), "previous_day")

    def test_long_history_matches_short_history_and_rejects_future(self):
        history = self.raw.iloc[:5001][["observation_time", "Usage_kWh"]]
        issue = history["observation_time"].iloc[-1]
        pd.testing.assert_frame_equal(
            features_at_issue(history, issue), features_at_issue(history.tail(673), issue)
        )
        with self.assertRaises(ValueError):
            features_at_issue(self.raw.iloc[:5002], issue)

    def test_input_schema_and_off_grid_timestamp_are_rejected(self):
        history = self.raw.iloc[1000:1673][["observation_time", "Usage_kWh"]].copy()
        issue = history["observation_time"].iloc[-1]
        for broken in (
            history.drop(columns="Usage_kWh"),
            pd.concat([history, history[["Usage_kWh"]]], axis=1),
            history.assign(observation_time=history["observation_time"] + pd.Timedelta(seconds=1)),
            history.assign(observation_time=history["observation_time"].dt.tz_localize("Asia/Seoul")),
        ):
            with self.assertRaises(ValueError):
                features_at_issue(broken, issue)

    def test_zip_without_git_still_has_honest_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(git_snapshot(Path(directory)), {
                "local_git_commit": None, "git_worktree_dirty": None,
            })

    def test_hashed_json_is_stable_under_git_lf_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            write_json(path, {"slots": [1, 2], "note": "điện năng"})
            original = path.read_bytes()
            self.assertEqual(original, original.replace(b"\r\n", b"\n"))
            self.assertEqual(json.loads(original), {"slots": [1, 2], "note": "điện năng"})

    def test_empty_and_invalid_selection_fail_clearly(self):
        for board in (
            pd.DataFrame(),
            pd.DataFrame([{"model": "x", "kind": "AI", "mae_kWh": np.nan}]),
            pd.DataFrame([{"model": "x", "kind": "unknown", "mae_kWh": 1}]),
        ):
            with self.assertRaises(ValueError):
                select_model(board)

    def test_all_baseline_winners_can_be_deployed(self):
        train = self.splits["train"]
        part = self.splits["validation"].iloc[[2]]  # forecast_start 00:45
        issue = part["observation_time"].iloc[0]
        history = self.raw.loc[self.raw["observation_time"] <= issue].tail(673)
        expected = baseline_predictions(self.raw, train, part)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            state_path = path / "baseline_state.json"
            state_path.write_text(json.dumps(fit_diurnal_state(train)), encoding="utf-8")
            for name, values in expected.items():
                manifest = {"kind": "baseline", "selected_model": name,
                            "features": list(BASE_FEATURES), "target": TARGET,
                            "feature_implementation": FEATURE_IMPLEMENTATION,
                            "sklearn": sklearn.__version__, "model_sha256": None,
                            "baseline_state_path": "baseline_state.json",
                            "baseline_state_sha256": sha256_file(state_path)}
                (path / "model_handoff.json").write_text(json.dumps(manifest), encoding="utf-8")
                prediction = DuyForecaster(path).predict(history, issue)
                self.assertAlmostEqual(prediction["predicted_next_60m_kWh"], values[0])
            state_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                DuyForecaster(path)

    def test_never_overwrites_an_existing_run(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run(Path(directory))
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_inference_loads_correct_artifact_and_never_returns_future_actual(self):
        model = fixed_models()["ridge_alpha_1"]
        train = self.splits["train"].head(1000)
        model.fit(train[list(BASE_FEATURES)], train[TARGET])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "models").mkdir()
            joblib.dump(model, path / "models/model.joblib")
            manifest = {"kind": "AI", "selected_model": "ridge_alpha_1",
                        "features": list(BASE_FEATURES), "target": TARGET,
                        "feature_implementation": FEATURE_IMPLEMENTATION,
                        "sklearn": sklearn.__version__, "model_path": "models\\model.joblib",
                        "model_sha256": sha256_file(path / "models/model.joblib")}
            (path / "model_handoff.json").write_text(json.dumps(manifest), encoding="utf-8")
            forecaster = DuyForecaster(path)
            history = self.raw.iloc[1000:1673][["observation_time", "Usage_kWh"]]
            output = forecaster.predict(history, history["observation_time"].iloc[-1])
            self.assertEqual(output["status"], "forecast_only_no_alert_policy")
            self.assertGreaterEqual(output["predicted_next_60m_kWh"], 0)
            self.assertNotIn(TARGET, output)
            self.assertNotIn("actual", output)
            # Model hỏng phải bị chặn TRƯỚC khi joblib.load.
            (path / "models/model.joblib").write_bytes(b"broken")
            with self.assertRaisesRegex(ValueError, "checksum"):
                DuyForecaster(path)


if __name__ == "__main__":
    unittest.main()
