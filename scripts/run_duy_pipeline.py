"""Chạy phần Duy: audit -> EDA -> preparation -> baseline -> Ridge/HGB.

python -m scripts.run_duy_pipeline --output-dir reports/steel/duy_2026-10-01
Thư mục đích phải mới. Không huấn luyện trên calibration/test hoặc sửa policy cũ.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from threadpoolctl import threadpool_limits

from scripts.prepare_steel_data import write_train_eda
from src.steel.duy_forecasting import (
    FEATURE_IMPLEMENTATION, baseline_predictions, features_at_issue,
    fit_diurnal_state, fixed_models, forecast_features, predict_baseline, predict_energy,
)
from src.steel.preprocessing import (
    BASE_FEATURES, SPLIT_STARTS, TARGET, build_row_manifest, build_supervised_rows,
    load_official_raw, profile_training_columns, sha256_file,
    split_supervised_rows, write_processed_data,
)


def write_json(path: Path, value: dict) -> None:
    # Hash JSON phải giữ nguyên khi Git checkout trên Windows/Linux.
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def metrics(actual, predicted, threshold: float) -> dict:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    peak = actual > threshold
    return {
        "mae_kWh": float(mean_absolute_error(actual, predicted)),
        "rmse_kWh": float(root_mean_squared_error(actual, predicted)),
        "bias_kWh": float(np.mean(predicted - actual)),
        "peak_rows": int(peak.sum()),
        "peak_mae_kWh": float(mean_absolute_error(actual[peak], predicted[peak])) if peak.any() else None,
        "peak_mean_underprediction_kWh": float(np.maximum(actual[peak] - predicted[peak], 0).mean()) if peak.any() else None,
    }


def select_model(leaderboard: pd.DataFrame) -> str:
    """MAE validation thấp nhất; bằng nhau ưu tiên baseline rồi tên model."""
    if leaderboard.empty or not {"model", "kind", "mae_kWh"}.issubset(leaderboard.columns):
        raise ValueError("Bảng chọn model rỗng hoặc thiếu cột")
    if not leaderboard["kind"].isin(["AI", "baseline"]).all():
        raise ValueError("Loại model phải là AI hoặc baseline")
    if not np.isfinite(leaderboard["mae_kWh"]).all() or (leaderboard["mae_kWh"] < 0).any():
        raise ValueError("Không chọn model khi MAE không hữu hạn")
    ordered = leaderboard.assign(ai_order=leaderboard["kind"].eq("AI").astype(int))
    return ordered.sort_values(["mae_kWh", "ai_order", "model"]).iloc[0]["model"]


def git_snapshot(root: Path) -> dict:
    """Bản tải ZIP không có .git vẫn chạy được; không bịa commit nguồn."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip())
        return {"local_git_commit": commit, "git_worktree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"local_git_commit": None, "git_worktree_dirty": None}


def additional_data_figures(raw, train, directory: Path) -> None:
    train_raw = raw.loc[raw["observation_time"] < SPLIT_STARTS["validation"]]
    groups = [part["Usage_kWh"].to_numpy() for _, part in
              train_raw.groupby(train_raw["observation_time"].dt.hour)]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.boxplot(groups, tick_labels=range(24), showfliers=False)
    ax.set(title="Distribution by hour — train only (fliers hidden, data retained)",
           xlabel="Hour", ylabel="kWh per 15-minute record")
    fig.tight_layout()
    fig.savefig(directory / "train_hour_boxplots.png", dpi=160)
    plt.close(fig)

    corr = train[list(BASE_FEATURES) + [TARGET]].corr()
    corr.to_csv(directory.parent / "train_feature_correlations.csv")
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr)), labels=corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), labels=corr.columns, fontsize=7)
    ax.set_title("Pearson correlations — training rows only; not causality")
    fig.colorbar(im, ax=ax, label="Correlation")
    fig.tight_layout()
    fig.savefig(directory / "train_feature_correlations.png", dpi=160)
    plt.close(fig)


def validation_figures(validation, predictions, selected, leaderboard, directory: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(leaderboard["model"], leaderboard["mae_kWh"], color="#176B87")
    ax.set(title="Model comparison — September validation only", ylabel="MAE (kWh)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(directory / "validation_model_comparison.png", dpi=160)
    plt.close(fig)

    actual = validation[TARGET].to_numpy()
    prediction = predictions[selected]
    sample = validation["observation_time"] < pd.Timestamp("2018-09-04")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(validation.loc[sample, "observation_time"], actual[sample], label="Actual", linewidth=1)
    ax.plot(validation.loc[sample, "observation_time"], prediction[sample], label=selected, linewidth=1)
    ax.set(title="Next-hour energy — first 3 validation days", xlabel="Forecast issue time",
           ylabel="kWh in next 60 minutes")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(directory / "validation_forecast.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(actual, prediction, s=5, alpha=0.25)
    axes[0].plot([0, actual.max()], [0, actual.max()], color="black", linestyle="--")
    axes[0].set(xlabel="Actual (kWh)", ylabel="Predicted (kWh)", title="All validation forecasts")
    axes[1].hist(prediction - actual, bins=60, color="#176B87")
    axes[1].axvline(0, color="black", linestyle="--")
    axes[1].set(xlabel="Prediction minus actual (kWh)", ylabel="Rows", title="Negative = underforecast")
    fig.tight_layout()
    fig.savefig(directory / "validation_errors.png", dpi=160)
    plt.close(fig)


def run(output: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    if output.exists():
        raise FileExistsError("Chọn thư mục kết quả mới; không ghi đè lần chạy cũ")
    raw_path = root / "data/steel/raw/Steel_industry_data.csv"
    raw, raw_audit = load_official_raw(raw_path)
    rows = build_supervised_rows(raw)
    rows.loc[:, list(BASE_FEATURES)] = forecast_features(raw).to_numpy()
    splits, split_audit = split_supervised_rows(rows)
    train, validation = splits["train"], splits["validation"]
    threshold = split_audit["train_target_p95_kWh"]
    output.mkdir(parents=True)
    figures, model_dir = output / "figures", output / "models"
    figures.mkdir()
    model_dir.mkdir()
    models = fixed_models()
    baseline_state = fit_diurnal_state(train)
    write_json(output / "baseline_state.json", baseline_state)

    # Lưu protocol TRƯỚC khi fit; không đổi cấu hình theo kết quả test cũ.
    write_json(output / "protocol.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": raw_audit["source_sha256"],
        "fit_period": "Jan-Aug 2018", "selection_period": "Sep 2018",
        "calibration_period_owner": "Oct 2018 - Khanh", "test_period_owner": "Nov-Dec 2018 - Khanh",
        "selection_rule": "Lowest validation MAE across all baselines and AI; ties prefer baseline",
        "feature_implementation": FEATURE_IMPLEMENTATION,
        "features": list(BASE_FEATURES),
        "models": {name: model.get_params(deep=False) if name.startswith("hgb") else
                   {"scaler": "StandardScaler fitted on train", "alpha": model[-1].alpha}
                   for name, model in models.items()},
        "old_test_already_observed": True,
        "limitations": "Reproducible development run, not a new blind test; no savings measured",
        "sensor_feature_decision": "Retain 15 Usage/calendar features; proposed electrical formulas need unit/availability review",
    })
    summary = {**raw_audit, **split_audit, "feature_implementation": FEATURE_IMPLEMENTATION,
               "timestamp_assumption": "Declared timestamp is available after its 15-min measurement completes; provider has not confirmed start/end convention"}
    summary["train_eda"] = write_train_eda(raw, train, figures)
    train_raw = raw.loc[raw["observation_time"] < SPLIT_STARTS["validation"]]
    summary["train_co2_usage_correlation"] = float(train_raw["Usage_kWh"].corr(train_raw["CO2(tCO2)"]))
    write_processed_data(splits, summary, output / "processed", manifest=build_row_manifest(rows),
                         train_profile=profile_training_columns(raw))
    additional_data_figures(raw, train, figures)

    # Ba nhãn tính tay dùng TRAIN, đủ để giải thích phép shift(-1)..shift(-4).
    examples = []
    for index in (672, 5000, 20000):
        readings = raw["Usage_kWh"].iloc[index + 1:index + 5].to_list()
        expected = sum(readings)
        np.testing.assert_allclose(rows.loc[index, TARGET], expected, rtol=0, atol=1e-10)
        examples.append({"observation_time": str(raw.loc[index, "observation_time"]),
                         "next_four_kWh": readings, "hand_sum_kWh": expected})
    write_json(output / "target_examples.json", {"examples": examples})

    x_train, x_val = train[list(BASE_FEATURES)], validation[list(BASE_FEATURES)]
    predictions = baseline_predictions(raw, train, validation)
    train_predictions = baseline_predictions(raw, train, train)
    leaderboard = []
    fit_seconds, model_bytes = {}, {}
    with threadpool_limits(limits=1):
        for name, model in models.items():
            started = time.perf_counter()
            model.fit(x_train, train[TARGET])
            fit_seconds[name] = time.perf_counter() - started
            predictions[name] = predict_energy(model, x_val)
            train_predictions[name] = predict_energy(model, x_train)
            path = model_dir / f"{name}.joblib"
            joblib.dump(model, path, compress=3)
            model_bytes[name] = path.stat().st_size
        for name, prediction in predictions.items():
            record = {"model": name, "kind": "AI" if name in models else "baseline",
                      "train_mae_kWh": float(mean_absolute_error(train[TARGET], train_predictions[name])),
                      "fit_seconds": fit_seconds.get(name, 0), "model_bytes": model_bytes.get(name, 0),
                      **metrics(validation[TARGET], prediction, threshold)}
            if name in models:
                one = x_val.iloc[[0]]
                for _ in range(5):
                    predict_energy(models[name], one)
                timings = []
                for _ in range(50):
                    start = time.perf_counter()
                    predict_energy(models[name], one)
                    timings.append((time.perf_counter() - start) * 1000)
                record["model_predict_p95_ms"] = float(np.quantile(timings, .95))
            leaderboard.append(record)
        leaderboard = pd.DataFrame(leaderboard).sort_values(["mae_kWh", "model"]).reset_index(drop=True)
        selected = select_model(leaderboard)

        # Kiểm feature batch == feature replay, có ca Huy từng báo lệch.
        origins = pd.concat([validation["observation_time"].iloc[::96],
                             pd.Series([pd.Timestamp("2018-09-08 02:00")])]).drop_duplicates()
        parity, end_to_end = [], []
        for issue in origins:
            history = raw.loc[raw["observation_time"] <= issue, ["observation_time", "Usage_kWh"]].tail(673)
            start = time.perf_counter()
            online = features_at_issue(history, issue)
            offline = validation.loc[validation["observation_time"] == issue, list(BASE_FEATURES)].reset_index(drop=True)
            np.testing.assert_array_equal(online.to_numpy(), offline.to_numpy())
            difference = 0.0
            if selected in models:
                online_prediction = predict_energy(models[selected], online)[0]
                offline_prediction = predict_energy(models[selected], offline)[0]
                difference = float(abs(online_prediction - offline_prediction))
                np.testing.assert_allclose(online_prediction, offline_prediction, rtol=0, atol=1e-10)
            else:
                online_prediction = predict_baseline(selected, history, issue, baseline_state)
                index = validation.index[validation["observation_time"].eq(issue)][0]
                offline_prediction = predictions[selected][index]
                difference = float(abs(online_prediction - offline_prediction))
                np.testing.assert_allclose(online_prediction, offline_prediction, rtol=0, atol=1e-10)
            end_to_end.append((time.perf_counter() - start) * 1000)
            parity.append({"issue_time": str(issue), "max_feature_difference": 0.0,
                           "prediction_difference_kWh": difference})

    leaderboard.to_csv(output / "validation_leaderboard.csv", index=False)
    prediction_file = validation[["observation_time", "forecast_start", "forecast_end", TARGET]].copy()
    for name, values in predictions.items():
        prediction_file[f"pred_{name}_kWh"] = values
    prediction_file.to_csv(output / "validation_predictions.csv", index=False)
    pd.DataFrame(parity).to_csv(output / "batch_replay_parity.csv", index=False)

    selected_prediction = predictions[selected]
    details = validation[["observation_time", TARGET]].copy()
    details["prediction_kWh"] = selected_prediction
    details["error_kWh"] = selected_prediction - validation[TARGET]
    details["absolute_error_kWh"] = details["error_kWh"].abs()
    details.nlargest(20, "absolute_error_kWh").to_csv(output / "largest_validation_errors.csv", index=False)
    # Theo tuần: chẩn đoán độ ổn định; KHÔNG dùng để search thêm cấu hình.
    weekly = []
    for week, part in details.groupby(details["observation_time"].dt.isocalendar().week):
        weekly.append({"week": int(week), "rows": len(part),
                       **metrics(part[TARGET], part["prediction_kWh"], threshold)})
    pd.DataFrame(weekly).to_csv(output / "validation_by_week.csv", index=False)
    validation_figures(validation, predictions, selected, leaderboard, figures)

    best_baseline = leaderboard.loc[leaderboard["kind"] == "baseline"].iloc[0]
    best = leaderboard.loc[leaderboard["model"] == selected].iloc[0]
    model_path = model_dir / f"{selected}.joblib"
    selected_file = model_path.relative_to(output).as_posix() if selected in models else None
    manifest = {
        "selected_model": selected, "kind": best["kind"], "model_path": selected_file,
        "model_sha256": sha256_file(model_path) if selected in models else None,
        "baseline_state_path": "baseline_state.json",
        "baseline_state_sha256": sha256_file(output / "baseline_state.json"),
        "source_sha256": raw_audit["source_sha256"], "feature_implementation": FEATURE_IMPLEMENTATION,
        "features": list(BASE_FEATURES), "target": TARGET, "history_rows_required": 673,
        "prediction_postprocessing": "max(prediction, 0)",
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "sklearn": sklearn.__version__, "processor": platform.processor(), "math_threads": 1,
        **git_snapshot(root),
        "validation_mae_kWh": float(best["mae_kWh"]), "validation_rmse_kWh": float(best["rmse_kWh"]),
        "peak_mae_kWh": float(best["peak_mae_kWh"]),
        "best_baseline": best_baseline["model"],
        "mae_improvement_vs_best_baseline_pct": float(100 * (1 - best["mae_kWh"] / best_baseline["mae_kWh"])),
        "threshold_train_p95_kWh": threshold,
        "calibration_completed": False, "test_scored_in_this_run": False,
        "old_test_already_observed": True, "alert_policy": None,
        "parity_checked_origins": len(parity),
        "parity_max_feature_difference": 0.0, "parity_max_prediction_difference_kWh": max(row["prediction_difference_kWh"] for row in parity),
        "parity_check_total_p95_ms": float(np.quantile(end_to_end, .95)),
        "latency_note": "Local CPU; model-only includes feature selection; parity total includes feature construction, comparisons and two predictions; not production latency",
        "code_sha256": {path.relative_to(root).as_posix(): sha256_file(path) for path in (
            root / "src/steel/duy_forecasting.py", root / "scripts/run_duy_pipeline.py",
            root / "src/steel/duy_inference.py",
            root / "src/steel/preprocessing.py", root / "scripts/prepare_steel_data.py")},
    }
    write_json(output / "model_handoff.json", manifest)
    report = ["# Kết quả phần Duy — lần chạy riêng", "", "Chỉ train Jan–Aug và chọn trên validation Sep.",
              "Test Nov–Dec đã được xem trong lịch sử; lần này không tính test hoặc calibration.", "",
              f"- Source: {raw_audit['raw_rows']:,} hàng; split: {split_audit['split_rows']}",
              f"- Model chọn: **{selected}**; MAE {best['mae_kWh']:.4f} kWh; RMSE {best['rmse_kWh']:.4f} kWh.",
              f"- MAE cải thiện {manifest['mae_improvement_vs_best_baseline_pct']:.2f}% so với {best_baseline['model']}.",
              f"- MAE vùng cao: {best['peak_mae_kWh']:.4f} kWh, {int(best['peak_rows'])} mẫu; phải bàn giao Khánh kiểm cảnh báo.",
              f"- Batch/replay: {len(parity)} mốc đã kiểm, feature và dự báo khớp.",
              "- Không có policy cảnh báo hoặc số tiết kiệm tiền điện trong lần chạy này.", "",
              "## Bảng validation", "", "| Model | MAE kWh | RMSE kWh | Peak MAE kWh |", "|---|---:|---:|---:|"]
    for record in leaderboard.to_dict("records"):
        report.append(f"| {record['model']} | {record['mae_kWh']:.4f} | {record['rmse_kWh']:.4f} | {record['peak_mae_kWh']:.4f} |")
    (output / "RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(leaderboard[["model", "mae_kWh", "rmse_kWh", "peak_mae_kWh"]].to_string(index=False), flush=True)
    print(f"Selected: {selected}; handoff: {output / 'model_handoff.json'}", flush=True)
    return manifest


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_dir.resolve())


if __name__ == "__main__":
    main()
