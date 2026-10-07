"""Run the team's first baseline/Ridge/HGB comparison on real validation rows.

No calibration/test evaluation, synthetic data, alert calibration or tuning.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from src.steel.modeling import historical_baselines, make_candidates, regression_metrics
from src.steel.preprocessing import (
    BASE_FEATURES, EXPECTED_SPLIT_ROWS, SPLIT_STARTS, TARGET,
    build_supervised_rows, load_official_raw, sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
TIMESTAMPS = ["observation_time", "forecast_start", "forecast_end"]


def read_verified_split(root: Path, name: str, reconstructed: pd.DataFrame) -> pd.DataFrame:
    path = root / f"data/steel/processed/steel_next_60m_{name}.csv"
    saved = pd.read_csv(path, parse_dates=TIMESTAMPS)
    boundary = SPLIT_STARTS["validation" if name == "train" else "calibration"]
    expected = reconstructed.loc[
        reconstructed.observation_time.ge(SPLIT_STARTS[name])
        & reconstructed.observation_time.lt(boundary)
        & reconstructed.forecast_end.lt(boundary)
    ].dropna().reset_index(drop=True)
    if len(saved) != EXPECTED_SPLIT_ROWS[name]:
        raise ValueError(f"Unexpected {name} row count")
    pd.testing.assert_frame_equal(
        saved, expected, check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10,
    )
    return saved


def run(root: Path, output: Path) -> pd.DataFrame:
    # Existing outputs are not silently overwritten: keep each measured run.
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    raw, raw_audit = load_official_raw(root / "data/steel/raw/Steel_industry_data.csv")
    # Later measured values are not used to construct training/validation inputs.
    history = raw.loc[raw.observation_time.lt(SPLIT_STARTS["calibration"])].copy()
    reconstructed = build_supervised_rows(history)
    train = read_verified_split(root, "train", reconstructed)
    validation = read_verified_split(root, "validation", reconstructed)
    x_train, y_train = train[list(BASE_FEATURES)], train[TARGET].to_numpy()
    x_val, y_val = validation[list(BASE_FEATURES)], validation[TARGET].to_numpy()
    origins = pd.DatetimeIndex(validation.observation_time)
    high_threshold = float(np.quantile(y_train, .95))
    high_mask = y_val > high_threshold

    candidates = make_candidates()
    config = {
        "scope": "initial fixed-configuration comparison on validation only",
        "target": TARGET,
        "features": list(BASE_FEATURES),
        "seed": 0,
        "tuning": "none; default Ridge/HGB settings except HGB random_state=0 and early_stopping=False",
        "selection": "Compare validation MAE; if ML MAE differs by less than 1% of the best ML MAE, prefer Ridge. Final model/policy approval remains with the group.",
        "high_load_definition": "actual target strictly greater than train target p95; diagnostic, not an operational limit",
        "prediction_postprocessing": "none; count negative predictions rather than silently clip",
        "thread_limit": 1,
        "excluded_evaluations": ["calibration", "test"],
        "parameters": {name: model.get_params(deep=True) for name, model in candidates.items()},
    }
    output.mkdir(parents=True, exist_ok=True)
    # Write the configuration before fitting or reading validation metrics.
    (output / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    predictions = validation[TIMESTAMPS + [TARGET]].copy()
    measured = []

    def record(name, pred, fit_seconds, predict_seconds):
        stats = {"method": name, **regression_metrics(y_val, pred)}
        high = regression_metrics(y_val[high_mask], pred[high_mask]) if high_mask.any() else {}
        stats.update({"high_" + key: value for key, value in high.items()})
        stats.update({
            "fit_seconds": fit_seconds, "validation_batch_predict_seconds": predict_seconds,
            "negative_predictions": int((pred < 0).sum()),
        })
        predictions[name] = pred
        measured.append(stats)

    start = perf_counter()
    baseline = historical_baselines(history.set_index("observation_time").Usage_kWh, origins)
    baseline_seconds = perf_counter() - start
    for name in baseline:
        # The three baselines are computed together; do not attribute joint time to one method.
        record(name, baseline[name].to_numpy(), 0.0, None)
    with threadpool_limits(limits=1):
        for name, model in candidates.items():
            start = perf_counter()
            model.fit(x_train, y_train)
            fit_seconds = perf_counter() - start
            start = perf_counter()
            pred = model.predict(x_val)
            predict_seconds = perf_counter() - start
            record(name, pred, fit_seconds, predict_seconds)
            joblib.dump(model, output / f"{name}.joblib")
    metrics = pd.DataFrame(measured).sort_values("mae_kWh").reset_index(drop=True)
    metrics.to_csv(output / "validation_metrics.csv", index=False)
    predictions.to_csv(output / "validation_predictions.csv", index=False)
    ml = metrics.loc[metrics.method.isin(candidates)].set_index("method")
    best_ml = str(ml.mae_kWh.idxmin())
    if ml.loc["ridge", "mae_kWh"] < ml.mae_kWh.min() * 1.01:
        best_ml = "ridge"
    baseline_metrics = metrics.loc[~metrics.method.isin(candidates)]
    best_baseline = baseline_metrics.iloc[0]
    improvement = float(100 * (1 - ml.loc[best_ml, "mae_kWh"] / best_baseline.mae_kWh))
    manifest = {
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "code_sha256": {p: sha256_file(root / p) for p in (
            "src/steel/modeling.py", "scripts/train_steel_models.py", "src/steel/preprocessing.py",
        )},
        "source_sha256": raw_audit["source_sha256"],
        "processed_sha256": {name: sha256_file(root / f"data/steel/processed/steel_next_60m_{name}.csv") for name in ("train", "validation")},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "train_rows": len(train), "validation_rows": len(validation),
        "train_target_p95_kWh": high_threshold,
        "baseline_joint_seconds": baseline_seconds,
        "provisional_ml_candidate": best_ml,
        "best_baseline": str(best_baseline.method),
        "validation_mae_improvement_over_best_baseline_percent": improvement,
        "not_done": ["hyperparameter search", "calibration", "test model scoring", "alert policy", "industrial benefit measurement"],
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Kết quả cổng 7 — lần so sánh đầu tiên", "",
        "Nguồn công việc: vai trò A, cổng 7 trong bản phân công ngày 29/09/2026.",
        "Dùng dữ liệu UCI thật; các cột dự báo là đầu ra tính toán, không phải bản đo giả.",
        f"Train: {len(train):,} dòng; validation tháng 9: {len(validation):,} dòng. Target là tổng kWh 60 phút tới.", "",
        "| Phương pháp | MAE kWh | RMSE kWh | Bias dự báo trừ thực tế kWh | Dự báo âm |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics.itertuples():
        lines.append(f"| {row.method} | {row.mae_kWh:.4f} | {row.rmse_kWh:.4f} | {row.bias_pred_minus_actual_kWh:.4f} | {row.negative_predictions} |")
    lines.extend([
        "", f"Ứng viên ML theo quy tắc MAE trong kế hoạch: **{best_ml}**. Đây là lựa chọn tạm cho cấu hình cố định, chưa phải phê duyệt mô hình/policy cuối.",
        f"MAE thay đổi so với baseline tốt nhất ({best_baseline.method}): cải thiện {improvement:.2f}% trên validation.",
        "", "Ridge dùng StandardScaler fit trên train. HGB tắt early stopping tự động để không tạo holdout ngẫu nhiên bên trong train. Các tham số còn lại giữ mặc định; chưa tìm kiếm hyperparameter.",
        "Ngưỡng train p95 chỉ dùng mô tả lỗi vùng tải cao, không phải ngưỡng vận hành. Không chặn dự báo âm; cột đếm cho biết giới hạn cần nhóm xem xét.",
        "Thời gian dự báo là thời gian của cả batch validation, không phải latency end-to-end của dashboard. Thời gian phụ thuộc máy chạy.",
        "", "Chưa chấm mô hình trên calibration/test, chưa hiệu chỉnh cảnh báo, chưa có dashboard hoặc kết luận tiết kiệm điện/tiền. Kết quả cần B kiểm tra chéo theo phân công.",
        "Các CSV calibration/test trước đó chỉ được đối chiếu tính toàn vẹn; không dùng để chọn mô hình trong lần chạy này.",
        "", "Tái chạy: `python -m scripts.train_steel_models --output-dir <thu_muc_moi>`.",
        "File cấu hình được ghi trước khi fit; manifest ghi hash nguồn, code, dữ liệu và phiên bản thư viện. File .joblib chỉ nạp từ nguồn tin cậy.",
    ])
    (output / "model_card.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/steel/modeling/initial_validation")
    args = parser.parse_args()
    print(run(ROOT, args.output_dir.resolve()).to_string(index=False))
