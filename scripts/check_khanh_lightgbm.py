"""Kiểm lại artifact LightGBM 15 feature đã lưu, không huấn luyện/chọn bằng test.

Đây là đối chiếu hồi cứu; test đã được xem trước đó. Policy cũ không được bật.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.steel.evaluation import calibrate_upper_buffer, evaluate_alerts
from src.steel.preprocessing import BASE_FEATURES, TARGET, sha256_file

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODEL_SHA256 = "8d671810b3ba241b95c6caac2982466e8223856aa446a4b8051ae71b02fc3c93"


def run(output):
    import lightgbm as lgb
    artifact = ROOT / "notebooks/artifacts/lightgbm_model.txt"
    if sha256_file(artifact) != EXPECTED_MODEL_SHA256:
        raise ValueError("Artifact lịch sử thay đổi; không dùng kết quả cũ")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model = lgb.Booster(model_file=str(artifact))
    if model.num_feature() != len(BASE_FEATURES):
        raise ValueError("Model phải có đúng 15 feature theo notebook lịch sử")
    data = ROOT / "data/steel/processed"
    def read(part):
        return pd.read_csv(data / f"steel_next_60m_{part}.csv", parse_dates=["observation_time"])
    train, validation, calibration = read("train"), read("validation"), read("calibration")
    threshold = float(np.quantile(train[TARGET], .95))
    cal_pred = np.maximum(model.predict(calibration[list(BASE_FEATURES)].to_numpy(), num_threads=1), 0)
    buffer = calibrate_upper_buffer(calibration[TARGET], cal_pred)
    # Save exact calibration identity before reading the already-observed test.
    calibration_card = {"model_sha256": EXPECTED_MODEL_SHA256, "features": list(BASE_FEATURES),
                        "source_sha256": sha256_file(ROOT / "data/steel/raw/Steel_industry_data.csv"),
                        "feature_implementation": "legacy_pandas_windows_v1",
                        "threshold_kWh": threshold, "upper_buffer_kWh": buffer,
                        "buffer_rule": "Q0.90(max(actual - prediction, 0)), October only",
                        "calibration_file_sha256": sha256_file(data / "steel_next_60m_calibration.csv"),
                        "valid_from": "2018-11-01", "alerts_enabled_in_HGB_demo": False,
                        "historical_policy_replaced_for_activation": False}
    (output / "calibration_card.json").write_text(json.dumps(calibration_card, indent=2), encoding="utf-8")
    test = read("test")
    prediction = np.maximum(model.predict(test[list(BASE_FEATURES)].to_numpy(), num_threads=1), 0)
    metric = evaluate_alerts(test[TARGET], prediction, test.observation_time, threshold, buffer)
    recorded = json.loads((ROOT / "notebooks/artifacts/test_results.json").read_text(encoding="utf-8"))
    pairs = {"test_mae_kWh": "mae_kWh", "test_rmse_kWh": "rmse_kWh",
             "precision": "precision", "recall": "recall", "q_buffer_kWh": None}
    differences = {old: float((buffer if new is None else metric[new]) - recorded[old]) for old, new in pairs.items()}
    if any(abs(value) > 1e-10 for value in differences.values()):
        raise AssertionError(f"Số ghi cũ không khớp: {differences}")
    val_pred = np.maximum(model.predict(validation[list(BASE_FEATURES)].to_numpy(), num_threads=1), 0)
    days = int(test.observation_time.dt.normalize().nunique())
    result = {"passed": True, "model_sha256": EXPECTED_MODEL_SHA256,
              "feature_count": model.num_feature(), "tree_count": model.num_trees(),
              "validation": {"mae_kWh": float(mean_absolute_error(validation[TARGET], val_pred)),
                             "rmse_kWh": float(root_mean_squared_error(validation[TARGET], val_pred))},
              "test": metric, "buffer_kWh": buffer, "recorded_metric_differences": differences,
              "calendar_days": days, "alert_origins_per_day": metric["alert_rows"] / days,
              "alert_origin_episodes_per_day": metric["alert_origin_episodes"] / days,
              "test_already_observed": True, "model_refitted": False,
              "runtime_latency_rechecked": False, "scope": "Historical 15-feature LightGBM; not extended 19-feature model"}
    (output / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args().output_dir)
