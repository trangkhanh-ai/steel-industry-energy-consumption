"""Replay validation with real history only, verify feature/prediction parity, time local inference."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn

from src.steel.inference import REQUIRED_HISTORY_ROWS, SteelForecaster, features_at_issue
from src.steel.modeling import regression_metrics
from src.steel.preprocessing import BASE_FEATURES, SPLIT_STARTS, TARGET, load_official_raw, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def run(output: Path, model_dir: Path):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new output directory: {output}")
    raw, source_audit = load_official_raw(ROOT / "data/steel/raw/Steel_industry_data.csv")
    history = raw.loc[raw.observation_time.lt(SPLIT_STARTS["calibration"]), ["observation_time", "Usage_kWh"]].reset_index(drop=True)
    validation_path = ROOT / "data/steel/processed/steel_next_60m_validation.csv"
    val = pd.read_csv(validation_path, parse_dates=["observation_time"])
    model_manifest = json.loads((model_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if sha256_file(validation_path) != model_manifest["hashes"]["data/steel/processed/steel_next_60m_validation.csv"]:
        raise ValueError("Validation data differs from the selected model's run")
    saved = pd.read_csv(model_dir / "search_predictions.csv", parse_dates=["observation_time"], float_precision="round_trip")
    batch_features = pd.read_csv(model_dir / "validation_features.csv", parse_dates=["observation_time"], float_precision="round_trip")
    pd.testing.assert_series_equal(batch_features.observation_time, val.observation_time)
    pd.testing.assert_series_equal(saved.observation_time, val.observation_time)
    np.testing.assert_allclose(saved[TARGET], val[TARGET], atol=1e-10, rtol=1e-10)
    start = perf_counter()
    forecaster = SteelForecaster(model_dir)
    model_load_ms = 1000*(perf_counter()-start)
    lookup = pd.Series(history.index, index=history.observation_time)
    first_pos = int(lookup.loc[val.observation_time.iloc[0]])
    first_window = history.iloc[first_pos-REQUIRED_HISTORY_ROWS+1:first_pos+1]
    # Warmup measured separately; it also provides a real CLI input/output example.
    start = perf_counter()
    example = forecaster.predict(first_window, val.observation_time.iloc[0])
    warmup_ms = 1000*(perf_counter()-start)
    records, features = [], []
    for i, t in enumerate(val.observation_time):
        pos = int(lookup.loc[t])
        window = history.iloc[pos-REQUIRED_HISTORY_ROWS+1:pos+1]
        assert len(window) == REQUIRED_HISTORY_ROWS and window.observation_time.max() == t
        start = perf_counter()
        result = forecaster.predict(window, t)
        elapsed_ms = 1000*(perf_counter()-start)
        # Audit is outside the timed call; inference itself creates its features.
        x = features_at_issue(window, t)
        expected = batch_features.loc[i, list(BASE_FEATURES)].to_numpy(dtype=float)
        np.testing.assert_array_equal(x.iloc[0].to_numpy(), expected)
        batch_prediction = float(saved.loc[i, forecaster.configuration])
        np.testing.assert_allclose(result["predicted_next_60m_kWh"], batch_prediction, atol=1e-9, rtol=1e-12)
        features.append({"observation_time": t, **x.iloc[0].to_dict()})
        records.append({**result, "actual_next_60m_kWh": float(val.loc[i, TARGET]),
                        "batch_prediction_kWh": batch_prediction,
                        "prediction_difference_kWh": result["predicted_next_60m_kWh"]-batch_prediction,
                        "max_feature_absolute_difference": float(np.abs(x.iloc[0].to_numpy()-expected).max()),
                        "history_start": window.observation_time.iloc[0], "history_end": t,
                        "history_rows": len(window), "local_pipeline_ms": elapsed_ms})
    results = pd.DataFrame(records)
    summary = {
        "finished_utc": datetime.now(timezone.utc).isoformat(), "rows": len(results),
        "model_configuration": forecaster.configuration, "model_sha256": forecaster.model_sha256,
        "source_sha256": source_audit["source_sha256"],
        "max_prediction_absolute_difference_kWh": float(results.prediction_difference_kWh.abs().max()),
        "max_feature_absolute_difference": float(results.max_feature_absolute_difference.max()),
        "feature_check_atol": 0, "feature_check_rtol": 0,
        "model_load_ms": model_load_ms, "first_call_ms": warmup_ms,
        "local_pipeline_ms": {"p50": float(results.local_pipeline_ms.median()), "p95": float(results.local_pipeline_ms.quantile(.95)), "max": float(results.local_pipeline_ms.max())},
        "timing_scope": "Warm sequential calls: validate 673 in-memory records, build features, apply thread limit, predict and format response. Excludes model loading, source-file reading, history-window retrieval, sensor/network/UI and audit comparisons.",
        "metrics": regression_metrics(results.actual_next_60m_kWh.to_numpy(), results.predicted_next_60m_kWh.to_numpy()),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "code_sha256": {p: sha256_file(ROOT / p) for p in ("src/steel/inference.py", "src/steel/preprocessing.py", "scripts/replay_steel.py", "scripts/predict_steel.py")},
        "run_artifact_sha256": {p: sha256_file(model_dir / p) for p in ("run_manifest.json", "frozen_config.json", "search_predictions.csv", "validation_features.csv")},
        "limits": "Validation replay only; no refit, calibration/test scoring, alert policy, fault injection or industrial validation. Floating-point feature equality uses a stated tolerance; predictions are checked against saved batch results."
    }
    # Publish outputs only once the full replay has passed.
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "replay_predictions.csv", index=False)
    pd.DataFrame(features).to_csv(output / "replay_features.csv", index=False)
    first_window.to_csv(output / "example_history.csv", index=False)
    (output / "example_prediction.json").write_text(json.dumps(example, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "replay_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    latency = summary["local_pipeline_ms"]
    report = f"""# Kiểm tra phát lại dữ liệu thật và chức năng dự báo

Đã chạy {len(results):,} thời điểm validation tháng 9 theo thứ tự. Mỗi lần chỉ đưa vào 673 bản đo thật kết thúc đúng thời điểm dự báo; không truyền target, Power Factor, Load_Type hoặc các bản đo tương lai cho mô hình. Đây là phát lại offline theo thứ tự, không phải chạy chờ đủ 15 phút hoặc kết nối nhà máy.

Tất cả 15 feature khớp chính xác bảng validation của bản đóng gói inference_v1. Chênh lệch feature tuyệt đối lớn nhất: {summary['max_feature_absolute_difference']:.12g}. Chênh lệch dự báo lớn nhất so với kết quả chạy theo bảng: {summary['max_prediction_absolute_difference_kWh']:.12g} kWh.

MAE validation được tái tạo: {summary['metrics']['mae_kWh']:.4f} kWh. Đây là đối chiếu chức năng dùng mô hình, không phải kết quả kiểm định mới trên dữ liệu chưa từng xem.

| Phép đo trên máy hiện tại | Thời gian ms |
|---|---:|
| Nạp và kiểm tra mô hình một lần | {model_load_ms:.3f} |
| Lần gọi đầu tiên | {warmup_ms:.3f} |
| p50 của {len(results)} lần gọi tiếp theo | {latency['p50']:.3f} |
| p95 | {latency['p95']:.3f} |
| Lớn nhất | {latency['max']:.3f} |

Mỗi lần đo bao gồm kiểm tra đầu vào trong bộ nhớ, tạo feature, giới hạn luồng, predict và đóng gói kết quả. Không gồm đọc file/cảm biến, lấy cửa sổ lịch sử, mạng hoặc giao diện. Mô hình được nạp một lần và gọi tuần tự; số đo này chưa đại diện cho nhiều người dùng đồng thời hay độ trễ nhà máy. Các phép đối chiếu với nhãn thật chạy ngoài đoạn đo.

`replay_predictions.csv` lưu từng thời điểm, phạm vi lịch sử, kết quả, target thật dùng để đối chiếu và latency. `replay_features.csv` lưu feature. `example_history.csv` là trích nguyên 673 bản đo thật cho lần dự báo đầu tiên, không phải dữ liệu mô phỏng. Hash và môi trường nằm trong `replay_summary.json`.

Cấu hình HGB giữ nguyên từ tuning_v1, đã fit lại đúng một lần trên train với phép tính feature dùng chung cho batch và cửa sổ. Lý do: sai số tích lũy của rolling ở cách cũ có thể khiến cây đi nhánh khác; tại 2018-09-08 02:00, chênh lệch dự báo của lần thử dùng cửa sổ với mô hình cũ là khoảng 0,345 kWh dù sai khác feature rất nhỏ. Không chấp nhận sai khác đó bằng cách nới tolerance. Bản cũ được giữ nguyên để đối chiếu, không mở thêm tuning.

MAE trước đóng gói: {model_manifest['previous_validation_mae_kWh']:.4f} kWh; sau đóng gói: {model_manifest['metrics']['mae_kWh']:.4f} kWh. Sau đóng gói vẫn dự báo thấp tại {model_manifest['metrics']['high_underpredictions']}/93 mẫu tải cao. Đây là sửa tính nhất quán khi triển khai, không phải cải tiến mô hình đã được kiểm định độc lập. Chưa có cảnh báo, tự điều phối tải, fallback tự động hay tuyên bố tiết kiệm điện. Cần B kiểm tra và C tích hợp theo hướng dẫn bàn giao.
"""
    (output / "replay_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/steel/modeling/replay_v1")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "reports/steel/modeling/inference_v1")
    args = parser.parse_args()
    run(args.output_dir.resolve(), args.model_dir.resolve())
