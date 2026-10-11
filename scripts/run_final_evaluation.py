"""Khóa policy trên tháng 10 và đánh giá model đã chọn trên tháng 11–12.

Không fit lại AI, không chọn cấu hình trên test. Test đã được xem trước đây:
đây là kiểm chứng tái lập, không phải blind test mới hay đo tiết kiệm điện.
Chạy: python -m scripts.run_final_evaluation --output-dir reports/steel/final_2026-10-07
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from src.steel.duy_forecasting import FEATURE_IMPLEMENTATION, forecast_features, predict_energy
from src.steel.duy_inference import DuyForecaster
from src.steel.evaluation import calibrate_upper_buffer, evaluate_alerts, inspection_cost
from src.steel.preprocessing import (
    BASE_FEATURES, FREQUENCY, TARGET, build_supervised_rows, load_official_raw,
    sha256_file, split_supervised_rows,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "reports/steel/duy_2026-10-02"
TARGET_DEFINITION = "Sum Usage_kWh at t+15, t+30, t+45 and t+60 minutes; issued after measurement at t is available"
CALIBRATION_PERIOD = "2018-10-01 00:00 through 2018-10-31 22:45; all forecast_end before 2018-11-01"


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def predictions_for_baselines(raw: pd.DataFrame, part: pd.DataFrame, state: dict) -> dict:
    """Bốn baseline với thống kê diurnal đã fit trên train trong release cũ."""
    origins = pd.DatetimeIndex(part["observation_time"])
    meter = raw.set_index("observation_time")["Usage_kWh"]
    result = {"last_hour": part["usage_sum_last_1h"].to_numpy()}
    for name, lag in (("previous_day", 96), ("previous_week", 672)):
        result[name] = sum(meter.reindex(origins - (lag - step) * FREQUENCY).to_numpy()
                           for step in range(1, 5))
    lookup = {(slot["weekday"], slot["hour"]): slot["mean_kWh"] for slot in state["slots"]}
    result["diurnal_train_mean"] = np.asarray([
        lookup.get((start.dayofweek, start.hour), state["global_mean_kWh"])
        for start in part["forecast_start"]
    ])
    return result


def make_policy(handoff: dict, threshold: float, calibration: pd.DataFrame, prediction, train_usage=None) -> dict:
    """Chỉ nhận nhãn nằm trong tháng 10. Không tham số nào được chọn từ test."""
    start = pd.Timestamp("2018-10-01")
    end = pd.Timestamp("2018-11-01")
    times = pd.DatetimeIndex(calibration["observation_time"])
    if (len(calibration) == 0 or not times.is_unique or not times.is_monotonic_increasing
            or times.min() < start or times.max() >= end
            or not calibration["forecast_end"].lt(end).all()):
        raise ValueError("Policy chỉ được hiệu chỉnh từ calibration tháng 10, không nhãn vượt ranh giới")
    if (times.tz is not None or not times.equals(times.floor("15min"))
            or (len(times) > 1 and not (times[1:] - times[:-1] == FREQUENCY).all())
            or not np.array_equal(pd.DatetimeIndex(calibration["forecast_end"]), times + 4 * FREQUENCY)):
        raise ValueError("Calibration cần mốc 15 phút liên tục và forecast_end = origin + 60 phút")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Ngưỡng train phải dương và hữu hạn")
    buffer = calibrate_upper_buffer(calibration[TARGET], prediction, quantile=0.90)
    policy = {
        "policy_version": "steel_alert_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": handoff["selected_model"],
        "model_sha256": handoff["model_sha256"],
        "feature_implementation": handoff["feature_implementation"],
        "features": handoff["features"],
        "source_sha256": handoff["source_sha256"],
        "target": TARGET,
        "target_unit": "kWh",
        "target_definition": TARGET_DEFINITION,
        "reference_validation_mae_kWh": handoff["validation_mae_kWh"],
        "threshold_kWh": float(threshold),
        "threshold_rule": "p95 of TRAIN target only; research threshold, not contract power",
        "upper_buffer_kWh": buffer,
        "buffer_rule": "Q0.90(max(actual - prediction, 0)) on October calibration only",
        "quantile_method": "numpy linear interpolation; clip EACH residual before quantile",
        "calibration_period": {"start": str(times.min()), "end": str(calibration["forecast_end"].max())},
        "valid_from": "2018-11-01T00:00:00",
        "calibration_rows": len(calibration),
        "calibration_first_origin": str(times.min()),
        "calibration_last_origin": str(times.max()),
        "alert_rule": "prediction + upper_buffer_kWh > threshold_kWh",
        "decision": "Operator checks meter and operating plan, records approve/reject; no automatic machine control",
        "old_test_already_observed": True,
        "conditional_P90_estimated": False,
        "coverage_guaranteed": False,
        "savings_measured": False,
        "limitations": "One factory/year; overlapping next-hour targets; no tariff, contract kW or controllable-equipment data",
    }
    if train_usage is not None:
        reference = np.asarray(train_usage, dtype=float)
        if not len(reference) or not np.isfinite(reference).all() or (reference < 0).any():
            raise ValueError("Train reference Usage phải hữu hạn và không âm")
        # Bin học chỉ từ train; smoothing tránh log(0), không sửa dữ liệu nguồn.
        bins = np.unique(np.quantile(reference, np.arange(.1, 1, .1)))
        counts = np.histogram(reference, bins=np.r_[-np.inf, bins, np.inf])[0]
        probabilities = (counts + .5) / (len(reference) + .5 * len(counts))
        policy.update({
            "reference_usage_bins": bins.tolist(),
            "reference_usage_probabilities": probabilities.tolist(),
            "reference_usage_rows": len(reference),
            "reference_usage_period": "Jan-Aug 2018 raw meter records only",
            "monitor_window_rows": 672,
            "psi_review_threshold": .20,
            "performance_review_factor": 1.20,
            "performance_review_weeks": 2,
            "monitoring_threshold_note": "Design heuristics for human review, not validated drift hypothesis tests; 0.5 histogram smoothing",
        })
    return policy


def time_calls(function, warmup=5, repeats=50) -> dict:
    for _ in range(warmup):
        function()
    elapsed = []
    cpu_started = time.process_time()
    wall_started = time.perf_counter()
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        elapsed.append((time.perf_counter() - started) * 1000)
    cpu_seconds = time.process_time() - cpu_started
    wall_seconds = time.perf_counter() - wall_started
    return {
        "warmup_calls": warmup, "measured_calls": repeats,
        "p50_ms": float(np.quantile(elapsed, .50)),
        "p95_ms": float(np.quantile(elapsed, .95)),
        "min_ms": float(min(elapsed)), "max_ms": float(max(elapsed)),
        "process_cpu_seconds": cpu_seconds, "wall_seconds": wall_seconds,
        "mean_process_cpu_ms_per_call": cpu_seconds / repeats * 1000,
    }


def system_benchmarks(forecaster, raw, test, model_path) -> dict:
    first = test.iloc[0]
    issue = first["observation_time"]
    history = raw.loc[raw["observation_time"] <= issue, ["observation_time", "Usage_kWh"]].tail(673)
    one = test.iloc[[0]][list(BASE_FEATURES)]
    with threadpool_limits(limits=1):
        model_only = time_calls(lambda: predict_energy(forecaster.model, one))
        complete = time_calls(lambda: forecaster.predict(history, issue))
        # Đo allocation riêng: không để tracemalloc làm chậm benchmark latency.
        tracemalloc.start()
        for _ in range(20):
            forecaster.predict(history, issue)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return {
        "model_only": model_only,
        "complete_forecast_path": complete,
        "model_bytes": model_path.stat().st_size,
        "python_allocation_current_bytes": current,
        "python_allocation_peak_bytes_20_predictions": peak,
        "memory_scope": "Python-tracked allocation during predictions; excludes full native library/RSS memory and model load",
        "latency_scope": "Local warm CPU, one forecast, 673 input records; complete path includes validation/features/model/output, excludes HTTP/browser/network/model load",
        "math_threads": 1, "logical_cpu_count": os.cpu_count(),
        "processor": platform.processor(), "platform": platform.platform(),
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "sklearn": sklearn.__version__,
        "production_SLA_validated": False,
    }


def input_fault_checks(forecaster, raw, test) -> list:
    issue = test.iloc[0]["observation_time"]
    history = raw.loc[raw["observation_time"] <= issue, ["observation_time", "Usage_kWh"]].tail(673).reset_index(drop=True)
    cases = {
        "insufficient_history": history.head(672),
        "missing_interval": history.drop(index=100),
        "reversed_time": history.iloc[::-1],
        "missing_column": history.drop(columns="Usage_kWh"),
        "duplicate_column": pd.concat([history, history[["Usage_kWh"]]], axis=1),
        "off_grid": history.assign(observation_time=history["observation_time"] + pd.Timedelta(seconds=1)),
        "timezone_changed": history.assign(observation_time=history["observation_time"].dt.tz_localize("Asia/Seoul")),
    }
    for name, value in (("negative_usage", -1), ("missing_usage", np.nan), ("infinite_usage", np.inf)):
        broken = history.copy()
        broken.loc[672, "Usage_kWh"] = value
        cases[name] = broken
    duplicate = history.copy()
    duplicate.loc[100, "observation_time"] = duplicate.loc[99, "observation_time"]
    cases["duplicate_time"] = duplicate
    future = pd.concat([history, pd.DataFrame({"observation_time": [issue + FREQUENCY], "Usage_kWh": [10.]})], ignore_index=True)
    cases["future_after_issue"] = future
    results = []
    for name, broken in cases.items():
        try:
            forecaster.predict(broken, issue)
            results.append({"case": name, "rejected": False, "error": None})
        except (ValueError, TypeError) as exc:
            results.append({"case": name, "rejected": True, "error": str(exc)})
    if not all(case["rejected"] for case in results):
        raise AssertionError("Có đầu vào hỏng chưa được chặn")
    return results


def shift_stress(forecaster, raw, test, policy) -> dict:
    """Kịch bản tổng hợp +20% điện năng, không phải sự kiện nhà máy đã quan sát."""
    shifted = raw.copy()
    shifted["Usage_kWh"] *= 1.20
    features = forecast_features(shifted)
    lookup = pd.Series(np.arange(len(raw)), index=raw["observation_time"])
    indices = lookup.loc[test["observation_time"]].to_numpy()
    with threadpool_limits(limits=1):
        prediction = predict_energy(forecaster.model, features.iloc[indices])
    result = evaluate_alerts(test[TARGET].to_numpy() * 1.20, prediction,
                             test["observation_time"], policy["threshold_kWh"], policy["upper_buffer_kWh"])
    return {"scenario": "All historical Usage and corresponding future target multiplied by 1.20",
            "generated_data": True, "real_operating_drift_measured": False,
            "policy_recalibrated": False, "metrics": result,
            "interpretation": "Sensitivity to a stipulated level shift; not a statistical drift detector or guarantee for unseen factories"}


def draw_figures(test, prediction, policy, comparison, output):
    figures = output / "figures"
    figures.mkdir()
    actual = test[TARGET].to_numpy()
    upper = prediction + policy["upper_buffer_kWh"]
    sample = test["observation_time"] < pd.Timestamp("2018-11-04")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(test.loc[sample, "observation_time"], actual[sample], label="Actual", linewidth=1)
    ax.plot(test.loc[sample, "observation_time"], prediction[sample], label="HGB point forecast", linewidth=1)
    ax.plot(test.loc[sample, "observation_time"], upper[sample], label="Forecast + calibrated buffer", linewidth=1, linestyle="--")
    ax.axhline(policy["threshold_kWh"], color="red", linestyle=":", label="Train p95 research threshold")
    ax.set(title="Next-hour energy — first 3 test days (previously observed test)", ylabel="kWh in next 60 minutes", xlabel="Forecast origin")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figures / "test_forecast_and_buffer.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(comparison["model"], comparison["mae_kWh"], color="#176B87")
    axes[0].set(title="Test MAE — no tuning on test", ylabel="MAE (kWh)")
    axes[0].tick_params(axis="x", rotation=25)
    x = np.arange(len(comparison))
    axes[1].bar(x - .18, comparison["precision"], .36, label="Precision")
    axes[1].bar(x + .18, comparison["recall"], .36, label="Recall")
    axes[1].set_xticks(x, comparison["model"], rotation=25)
    axes[1].set(title="Alert quality — each model calibrated separately", ylabel="Fraction", ylim=(0, 1))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figures / "test_model_and_alert_comparison.png", dpi=160)
    plt.close(fig)

    true_flags, alerts = actual > policy["threshold_kWh"], upper > policy["threshold_kWh"]
    matrix = np.array([[np.sum(~true_flags & ~alerts), np.sum(~true_flags & alerts)],
                       [np.sum(true_flags & ~alerts), np.sum(true_flags & alerts)]])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    ax.set_xticks([0, 1], ["No alert", "Alert"])
    ax.set_yticks([0, 1], ["Actual not high", "Actual high"])
    ax.set(title="Test confusion matrix — forecast origins", xlabel="Policy output", ylabel="Ground truth")
    fig.tight_layout()
    fig.savefig(figures / "test_alert_confusion.png", dpi=160)
    plt.close(fig)


def run(output: Path, model_run: Path = DEFAULT_RUN) -> dict:
    output, model_run = Path(output).resolve(), Path(model_run).resolve()
    if output.exists():
        raise FileExistsError("Thư mục kết quả phải mới; không ghi đè evaluation/policy cũ")
    forecaster = DuyForecaster(model_run)
    handoff = forecaster.manifest
    if handoff["kind"] != "AI" or handoff["feature_implementation"] != FEATURE_IMPLEMENTATION:
        raise ValueError("Lần nghiệm thu này cần model AI với feature NumPy đã khóa")
    raw_path = ROOT / "data/steel/raw/Steel_industry_data.csv"
    if sha256_file(raw_path) != handoff["source_sha256"]:
        raise ValueError("Nguồn CSV không khớp model handoff")
    state_path = model_run / handoff["baseline_state_path"]
    if sha256_file(state_path) != handoff["baseline_state_sha256"]:
        raise ValueError("Trạng thái baseline không khớp checksum")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    # Protocol được lưu trước khi đọc số đo, tính calibration hoặc chấm test.
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": handoff["selected_model"], "model_sha256": handoff["model_sha256"],
        "source_sha256": handoff["source_sha256"], "feature_implementation": FEATURE_IMPLEMENTATION,
        "features": list(BASE_FEATURES), "target": TARGET, "target_definition": TARGET_DEFINITION,
        "training_period": "Jan-Aug 2018 — existing fitted artifact",
        "model_selection_period": "Sep 2018 — existing validation selection, not repeated",
        "calibration_period": CALIBRATION_PERIOD,
        "test_period": "2018-11-01 through 2018-12-31; forecast_end before 2019-01-01",
        "threshold_rule": "p95 of train target; verify agreement with frozen handoff",
        "buffer_rule": "Q0.90(max(actual-prediction,0)); numpy linear quantile; October only",
        "alert_rule": "prediction + own October buffer > shared train p95 threshold",
        "models": [handoff["selected_model"], "last_hour", "previous_day", "previous_week", "diurnal_train_mean"],
        "AI_refitted": False, "test_used_for_tuning": False, "old_test_already_observed": True,
        "stress_design": "Synthetic +20% Usage history and target; frozen AI/policy",
        "cost_scenarios": [{"name": "equal_cost", "false_alarm_points": 1, "missed_high_points": 1},
                           {"name": "miss_cost_5x", "false_alarm_points": 1, "missed_high_points": 5},
                           {"name": "miss_cost_10x", "false_alarm_points": 1, "missed_high_points": 10}],
        "savings_measured": False,
        "code_sha256": {name: sha256_file(ROOT / name) for name in (
            "scripts/run_final_evaluation.py", "src/steel/evaluation.py", "src/steel/duy_forecasting.py")},
    }
    write_json(output / "protocol.json", protocol)
    raw, raw_audit = load_official_raw(raw_path)
    rows = build_supervised_rows(raw)
    rows.loc[:, list(BASE_FEATURES)] = forecast_features(raw).to_numpy()
    splits, split_audit = split_supervised_rows(rows)
    calibration, test = splits["calibration"], splits["test"]
    threshold = split_audit["train_target_p95_kWh"]
    if not np.isclose(threshold, handoff["threshold_train_p95_kWh"], rtol=0, atol=1e-9):
        raise ValueError("Ngưỡng train không khớp frozen handoff")
    with threadpool_limits(limits=1):
        calibration_predictions = predictions_for_baselines(raw, calibration, state)
        calibration_predictions[handoff["selected_model"]] = predict_energy(forecaster.model, calibration[list(BASE_FEATURES)])
    train_usage = raw.loc[raw["observation_time"] < pd.Timestamp("2018-09-01"), "Usage_kWh"]
    policy = make_policy(handoff, threshold, calibration, calibration_predictions[handoff["selected_model"]], train_usage)
    write_json(output / "locked_policy.json", policy)
    # Freeze tất cả buffer trước khi tính bất kỳ dự báo/metric test nào.
    buffers = {name: calibrate_upper_buffer(calibration[TARGET], values)
               for name, values in calibration_predictions.items()}
    write_json(output / "baseline_and_ai_buffers.json", {
        "calibration_period": CALIBRATION_PERIOD, "threshold_kWh": threshold,
        "own_buffer_by_model_kWh": buffers, "rule": protocol["buffer_rule"],
    })
    with threadpool_limits(limits=1):
        test_predictions = predictions_for_baselines(raw, test, state)
        test_predictions[handoff["selected_model"]] = predict_energy(forecaster.model, test[list(BASE_FEATURES)])

    summaries = {}
    for split, part, predictions in (("calibration", calibration, calibration_predictions),
                                     ("test", test, test_predictions)):
        summaries[split] = {}
        exported = part[["observation_time", "forecast_start", "forecast_end", TARGET]].copy()
        for name, values in predictions.items():
            metric = evaluate_alerts(part[TARGET], values, part["observation_time"], threshold, buffers[name])
            metric["upper_buffer_kWh"] = buffers[name]
            metric["threshold_kWh"] = threshold
            summaries[split][name] = metric
            exported[f"pred_{name}_kWh"] = values
            exported[f"upper_{name}_kWh"] = values + buffers[name]
            exported[f"alert_{name}"] = values + buffers[name] > threshold
        exported.to_csv(output / f"{split}_predictions.csv", index=False)
    comparison = pd.DataFrame([{"model": name, **metric} for name, metric in summaries["test"].items()])
    comparison.to_csv(output / "test_comparison.csv", index=False)
    monthly = []
    for month, part in test.groupby(test["observation_time"].dt.month):
        indices = part.index.to_numpy()
        monthly.append({"month": int(month), **evaluate_alerts(part[TARGET], test_predictions[handoff["selected_model"]][indices],
            part["observation_time"], threshold, policy["upper_buffer_kWh"])})
    pd.DataFrame(monthly).to_csv(output / "test_by_month.csv", index=False)
    model_path = model_run / handoff["model_path"]
    system = system_benchmarks(forecaster, raw, test, model_path)
    faults = input_fault_checks(forecaster, raw, test)
    stress = shift_stress(forecaster, raw, test, policy)
    ai = summaries["test"][handoff["selected_model"]]
    baseline = summaries["test"]["last_hour"]
    costs = []
    for scenario in protocol["cost_scenarios"]:
        for name, metric in summaries["test"].items():
            costs.append({"scenario": scenario["name"], "model": name,
                          "false_alarm_cost_points": scenario["false_alarm_points"],
                          "missed_high_cost_points": scenario["missed_high_points"],
                          "total_cost_points": inspection_cost(metric, scenario["false_alarm_points"], scenario["missed_high_points"])})
    pd.DataFrame(costs).to_csv(output / "simulated_inspection_costs.csv", index=False)
    industrial = {
        "unit": "forecast-origin inspection opportunities, not unique equipment incidents",
        "AI_alert_origins": ai["alert_rows"], "baseline_alert_origins": baseline["alert_rows"],
        "AI_alert_origin_episodes": ai["alert_origin_episodes"],
        "baseline_alert_origin_episodes": baseline["alert_origin_episodes"],
        "AI_false_alarm_origins": ai["fp"], "baseline_false_alarm_origins": baseline["fp"],
        "AI_missed_high_origins": ai["fn"], "baseline_missed_high_origins": baseline["fn"],
        "simulated_costs": costs, "cost_scenarios_chosen_before_test": True,
        "energy_saved_kWh": None, "money_saved": None, "savings_measured": False,
        "physical_event_lead_time_measured": False,
        "limitations": "No observed operator interventions, equipment constraints, tariff or counterfactual; 60-minute overlapping labels cannot prove avoided kW peak/demand charges",
    }
    result = {
        "protocol_sha256": sha256_file(output / "protocol.json"),
        "locked_policy_sha256": sha256_file(output / "locked_policy.json"),
        "model_name": handoff["selected_model"], "model_sha256": handoff["model_sha256"],
        "data_audit": raw_audit, "split_audit": split_audit,
        "calibration": summaries["calibration"], "test": summaries["test"],
        "system_level": system, "input_fault_checks": faults,
        "synthetic_level_shift_stress": stress, "industrial_level": industrial,
        "old_test_already_observed": True, "AI_refitted": False, "test_used_for_tuning": False,
        "reference_validation_mae_kWh": handoff["validation_mae_kWh"],
        "test_mae_improvement_vs_last_hour_pct": float(100 * (1 - ai["mae_kWh"] / baseline["mae_kWh"])),
        "final_model_sha256": sha256_file(model_path),
    }
    if result["final_model_sha256"] != handoff["model_sha256"]:
        raise AssertionError("Frozen model đã thay đổi trong evaluation")
    write_json(output / "metrics.json", result)
    draw_figures(test, test_predictions[handoff["selected_model"]], policy, comparison, output)
    report = ["# Đánh giá cuối — model và policy đã khóa", "",
              "Test này đã được xem trong lịch sử, không phải blind test mới. Không fit lại AI hoặc chọn tham số trên test.", "",
              f"- Model: {handoff['selected_model']}; SHA256: {handoff['model_sha256']}.",
              f"- Train threshold: {threshold:.4f} kWh; October buffer: {policy['upper_buffer_kWh']:.4f} kWh.",
              "- Buffer là phân vị lỗi thiếu thực nghiệm, không phải P90 có điều kiện hoặc bảo đảm độ phủ.",
              f"- Test AI MAE: {ai['mae_kWh']:.4f} kWh; RMSE: {ai['rmse_kWh']:.4f} kWh; peak MAE: {ai['peak_mae_kWh']:.4f} kWh.",
              f"- Test baseline last_hour MAE: {baseline['mae_kWh']:.4f} kWh; cải thiện MAE: {result['test_mae_improvement_vs_last_hour_pct']:.2f}%.",
              f"- AI alert precision: {ai['precision']:.4f}; recall: {ai['recall']:.4f}; FP: {ai['fp']}; FN: {ai['fn']}.",
              f"- Empirical upper coverage test: {ai['empirical_upper_coverage']:.4f} (cần đo, không mặc định 90%).",
              f"- Warm local p95 model-only: {system['model_only']['p95_ms']:.3f} ms; complete forecast: {system['complete_forecast_path']['p95_ms']:.3f} ms.",
              f"- Input faults rejected: {sum(case['rejected'] for case in faults)}/{len(faults)}.",
              "- Có kịch bản tổng hợp +20% để đo nhạy cảm; không phải drift thực tế đã quan sát.",
              "- Chi phí cảnh báo chỉ là điểm mô phỏng theo hàng; chưa đo tiết kiệm điện/tiền hoặc lead time sự kiện thiết bị.", "",
              "## Bảng test", "", "| Model | MAE kWh | RMSE kWh | Buffer kWh | Precision | Recall | FP | FN |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in summaries["test"].items():
        report.append(f"| {name} | {m['mae_kWh']:.4f} | {m['rmse_kWh']:.4f} | {m['upper_buffer_kWh']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['fp']} | {m['fn']} |")
    (output / "RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8", newline="\n")
    print(comparison[["model", "mae_kWh", "rmse_kWh", "precision", "recall", "fp", "fn"]].to_string(index=False), flush=True)
    print(f"Policy: {output / 'locked_policy.json'}", flush=True)
    return result


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run(args.output_dir, args.model_run)


if __name__ == "__main__":
    main()
