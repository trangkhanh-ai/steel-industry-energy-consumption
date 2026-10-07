"""Bounded Gate 7 search and observed-error analysis; never score calibration/test."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from scripts.train_steel_models import ROOT, TIMESTAMPS, read_verified_split
from src.steel.modeling import historical_baselines, make_candidates, regression_metrics
from src.steel.preprocessing import BASE_FEATURES, SPLIT_STARTS, TARGET, build_supervised_rows, load_official_raw, sha256_file


def diagnostic_rows(validation, predictions, train):
    """Retrospective labels use measured outcomes and must never enter X."""
    rows = validation[TIMESTAMPS + [TARGET, "usage_sum_last_1h"]].copy()
    rows["hour"] = rows.forecast_start.dt.hour
    rows["weekday"] = rows.forecast_start.dt.dayofweek  # Monday=0
    rows["change_next_minus_last_kWh"] = rows[TARGET] - rows.usage_sum_last_1h
    train_change = train[TARGET] - train.usage_sum_last_1h
    thresholds = {
        "high_target_kWh": float(train[TARGET].quantile(.95)),
        "large_rise_kWh": float(train_change[train_change > 0].quantile(.95)),
    }
    rows["high_load"] = rows[TARGET] > thresholds["high_target_kWh"]
    rows["change_group"] = np.select(
        [rows.change_next_minus_last_kWh > thresholds["large_rise_kWh"], rows.change_next_minus_last_kWh > 0, rows.change_next_minus_last_kWh < 0],
        ["large_rise", "other_rise", "fall"], default="unchanged",
    )
    for name, pred in predictions.items():
        rows[name] = np.asarray(pred)
    return rows, thresholds


def grouped_errors(rows, names, group):
    records = []
    for key, part in rows.groupby(group, observed=True):
        for name in names:
            pred = part[name].to_numpy()
            actual = part[TARGET].to_numpy()
            records.append({"group": str(key), "method": name,
                            **regression_metrics(actual, pred),
                            "underpredictions": int((pred < actual).sum()),
                            "negative_predictions": int((pred < 0).sum())})
    return pd.DataFrame(records)


def run(output: Path):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new output directory: {output}")
    config_path = ROOT / "configs/steel_tuning_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    candidates = []
    for alpha in config["ridge_alpha"]:
        candidates.append((f"ridge_a{alpha:g}", "ridge", make_candidates()["ridge"].set_params(model__alpha=alpha)))
    for rate, leaves, iterations in product(config["hgb_learning_rate"], config["hgb_max_leaf_nodes"], config["hgb_max_iter"]):
        model = make_candidates()["hist_gradient_boosting"].set_params(
            learning_rate=rate, max_leaf_nodes=leaves, max_iter=iterations,
            random_state=config["random_state"], early_stopping=config["early_stopping"],
        )
        candidates.append((f"hgb_lr{rate:g}_leaves{leaves}_iter{iterations}", "hgb", model))
    assert len(candidates) == config["budget_fits"]
    output.mkdir(parents=True, exist_ok=True)
    # Freeze the entire budget before fitting or seeing new candidate metrics.
    config["frozen_utc"] = datetime.now(timezone.utc).isoformat()
    config["parameters"] = {name: model.get_params(deep=True) for name, _, model in candidates}
    (output / "frozen_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    raw, audit = load_official_raw(ROOT / "data/steel/raw/Steel_industry_data.csv")
    history = raw.loc[raw.observation_time.lt(SPLIT_STARTS["calibration"])].copy()
    reconstructed = build_supervised_rows(history)
    train = read_verified_split(ROOT, "train", reconstructed)
    val = read_verified_split(ROOT, "validation", reconstructed)
    x_train, x_val = train[list(BASE_FEATURES)], val[list(BASE_FEATURES)]
    y = val[TARGET].to_numpy()
    high = y > train[TARGET].quantile(.95)
    predictions, records, models = {}, [], {}
    with threadpool_limits(limits=config["thread_limit"]):
        for name, family, model in candidates:
            start = perf_counter()
            model.fit(x_train, train[TARGET])
            fit_seconds = perf_counter() - start
            start = perf_counter()
            pred = model.predict(x_val)
            elapsed = perf_counter() - start
            predictions[name], models[name] = pred, model
            records.append({"method": name, "family": family, **regression_metrics(y, pred),
                            **{"high_" + k: v for k, v in regression_metrics(y[high], pred[high]).items()},
                            "negative_predictions": int((pred < 0).sum()), "fit_seconds": fit_seconds,
                            "batch_predict_seconds": elapsed})
    metrics = pd.DataFrame(records)
    ordered = metrics.sort_values("mae_kWh", kind="stable")
    best_ridge = ordered.loc[ordered.family.eq("ridge")].iloc[0]
    selected = ordered.iloc[0]
    if best_ridge.mae_kWh - selected.mae_kWh < .01 * selected.mae_kWh:
        selected = best_ridge
    metrics.to_csv(output / "search_metrics.csv", index=False)
    all_pred = val[TIMESTAMPS + [TARGET]].assign(**predictions)
    all_pred.to_csv(output / "search_predictions.csv", index=False)
    joblib.dump(models[selected.method], output / "selected_model.joblib")
    joblib.dump(models[best_ridge.method], output / "best_ridge.joblib")
    baseline = historical_baselines(history.set_index("observation_time").Usage_kWh, pd.DatetimeIndex(val.observation_time))
    # Initial configurations are part of the frozen grid; reproduce them without another fit.
    compared = {name: baseline[name].to_numpy() for name in baseline}
    compared.update(initial_ridge=predictions["ridge_a1"], initial_hgb=predictions["hgb_lr0.1_leaves31_iter100"],
                    tuned_ridge=predictions[best_ridge.method], selected=predictions[selected.method])
    rows, thresholds = diagnostic_rows(val, compared, train)
    rows.to_csv(output / "diagnostic_rows.csv", index=False)
    for group in ("hour", "weekday", "high_load", "change_group"):
        grouped_errors(rows, compared, group).to_csv(output / f"errors_by_{group}.csv", index=False)
    rows.loc[rows.high_load].to_csv(output / "high_load_cases.csv", index=False)
    rows.loc[rows.initial_ridge < 0].to_csv(output / "initial_ridge_negative_cases.csv", index=False)
    rows.assign(abs_error=(rows.selected - rows[TARGET]).abs()).nlargest(20, "abs_error").to_csv(output / "largest_errors.csv", index=False)

    ridge = models["ridge_a1"]
    contributions = ridge.named_steps["scaler"].transform(x_val) * ridge.named_steps["model"].coef_
    np.testing.assert_allclose(contributions.sum(axis=1) + ridge.named_steps["model"].intercept_, rows.initial_ridge, atol=1e-10)
    negative = rows.initial_ridge < 0
    pd.DataFrame({"feature": BASE_FEATURES,
                  "mean_contribution_negative_kWh": contributions[negative].mean(axis=0),
                  "mean_contribution_nonnegative_kWh": contributions[~negative].mean(axis=0),
                  "coefficient_standardized": ridge.named_steps["model"].coef_}).to_csv(output / "initial_ridge_contributions.csv", index=False)

    latency = []
    with threadpool_limits(limits=config["thread_limit"]):
        for label, name in (("tuned_ridge", best_ridge.method), ("selected", selected.method)):
            model = models[name]
            model.predict(x_val.iloc[:1])
            for i in range(min(256, len(x_val))):
                sample = x_val.iloc[i:i+1]
                start = perf_counter()
                model.predict(sample)
                latency.append({"method": label, "observation_time": val.observation_time.iloc[i], "milliseconds": 1000 * (perf_counter() - start)})
    latency = pd.DataFrame(latency)
    latency.to_csv(output / "model_only_latency.csv", index=False)
    latency_summary = latency.groupby("method").milliseconds.agg(["median", lambda x: x.quantile(.95), "max"])
    latency_summary.columns = ["p50_ms", "p95_ms", "max_ms"]
    latency_summary.to_csv(output / "model_only_latency_summary.csv")
    summary = pd.DataFrame({"method": name, **regression_metrics(y, p),
                            **{"high_" + k: v for k, v in regression_metrics(y[high], p[high]).items()},
                            "negative_predictions": int((p < 0).sum())} for name, p in compared.items())
    summary.to_csv(output / "comparison.csv", index=False)
    make_figure(rows, output)
    manifest = {"finished_utc": datetime.now(timezone.utc).isoformat(), "selected_configuration": selected.method,
                "best_ridge_configuration": best_ridge.method, "thresholds": thresholds,
                "ridge_intercept_kWh": float(ridge.named_steps["model"].intercept_),
                "train_rows": len(train), "validation_rows": len(val), "fits": len(candidates),
                "source_sha256": audit["source_sha256"],
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "hashes": {p: sha256_file(ROOT / p) for p in ("scripts/analyze_and_tune_steel.py", "scripts/train_steel_models.py", "src/steel/modeling.py", "src/steel/preprocessing.py", "configs/steel_tuning_v1.json", "data/steel/processed/steel_next_60m_train.csv", "data/steel/processed/steel_next_60m_validation.csv")},
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
                "scope": "Repeated validation model selection; no independent final-test estimate, no calibration or alert policy"}
    (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output, rows, summary, manifest, latency_summary)
    print(summary.to_string(index=False))
    print("Selected:", selected.method)


def make_figure(rows, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for name, label in (("initial_hgb", "Initial HGB"), ("selected", "Selected")):
        axes[0, 0].scatter(rows[TARGET], rows[name], s=6, alpha=.3, label=label)
    lim = max(rows[TARGET].max(), rows.selected.max())
    axes[0, 0].plot([0, lim], [0, lim], color="black", linewidth=1)
    axes[0, 0].set(xlabel="Measured next-hour energy (kWh)", ylabel="Prediction (kWh)", title="Validation predictions")
    axes[0, 0].legend()
    hourly = rows.assign(error=(rows.selected-rows[TARGET]).abs()).groupby("hour").error.mean()
    axes[0, 1].bar(hourly.index, hourly)
    axes[0, 1].set(xlabel="Forecast start hour", ylabel="MAE (kWh)", title="Selected model by hour")
    high = rows.loc[rows.high_load]
    axes[1, 0].scatter(high.observation_time, high.initial_hgb-high[TARGET], s=12, label="Initial HGB")
    axes[1, 0].scatter(high.observation_time, high.selected-high[TARGET], s=12, label="Selected")
    axes[1, 0].axhline(0, color="black", linewidth=1)
    axes[1, 0].set(ylabel="Predicted minus actual (kWh)", title="High-load cases (train p95)")
    axes[1, 0].tick_params(axis="x", labelrotation=25)
    axes[1, 0].legend()
    neg = rows.loc[rows.initial_ridge < 0]
    axes[1, 1].scatter(neg[TARGET], neg.initial_ridge, s=9, alpha=.5)
    axes[1, 1].set(xlabel="Measured next-hour energy (kWh)", ylabel="Initial Ridge prediction (kWh)", title=f"Initial Ridge negative predictions (n={len(neg)})")
    fig.savefig(output / "validation_diagnostics.png", dpi=160)
    plt.close(fig)


def write_report(output, rows, summary, manifest, latency):
    high = rows.loc[rows.high_load]
    neg = rows.loc[rows.initial_ridge < 0]
    hourly = rows.assign(ae=(rows.selected-rows[TARGET]).abs()).groupby("hour").ae.mean()
    rising = rows.loc[rows.change_group.eq("large_rise")]
    lines = ["# Phần kỹ thuật của Huy — phân tích lỗi và tinh chỉnh cổng 7", "",
             "## Phương pháp có thể đưa vào báo cáo", "",
             "Mục tiêu là dự báo tổng Usage_kWh của bốn khoảng 15 phút tiếp theo, tại thời điểm bản đo hiện tại đã có. Đây là tổng điện năng 60 phút tới, không phải công suất đỉnh tức thời hoặc bốn dự báo riêng.",
             "Nguồn: CSV UCI Steel có 35.040 bản ghi. Giữ nguyên dữ liệu đo, không thêm quan sát giả, không loại bỏ tải cao. Pipeline sắp xếp timestamp, kiểm tra lưới 15 phút và loại nhãn vượt ranh giới tập.",
             f"Train tháng 1–8: {manifest['train_rows']:,} dòng; validation tháng 9: {manifest['validation_rows']:,} dòng. Calibration tháng 10 và test tháng 11–12 chưa được chấm mô hình trong công việc này.",
             "Hai mô hình dùng cùng 15 feature: lag Usage ở bước 0/1/4/96/672; tổng giờ vừa qua; trung bình/độ lệch chuẩn 4 giờ; trung bình/cực đại 24 giờ; sin/cos giờ và thứ; cờ cuối tuần. Chỉ dùng thông tin có sẵn tại thời điểm dự báo. Các cột mô tả thay đổi tương lai trong phân tích lỗi không phải feature.",
             "Ba baseline: tổng giờ vừa qua, tổng đúng khung giờ dự báo của ngày trước và tuần trước. Ridge dùng StandardScaler chỉ fit train. HGB tắt early stopping để không tạo validation ngẫu nhiên trong train, seed 0, giới hạn một luồng.",
             "Ngân sách cố định trước lượt chạy: 5 Ridge (alpha 0,1/1/10/100/1000), 8 HGB (learning_rate 0,05/0,1 × max_leaf_nodes 15/31 × max_iter 100/200). Đây là lựa chọn triển khai của A theo quyền định dải tham số trong kế hoạch, không phải tham số do dataset cung cấp. Không mở rộng lưới sau khi xem kết quả.",
             "Chọn MAE thấp nhất trong từng họ; nếu Ridge chênh dưới 1% tương đối so với họ tốt nhất thì ưu tiên Ridge. Không chặn dự báo âm hoặc đổi hàm mục tiêu sau khi xem điểm.", "", "## Kết quả validation", "",
             "| Phương pháp | MAE kWh | RMSE kWh | MAE tải cao kWh | Dự báo âm |", "|---|---:|---:|---:|---:|"]
    for r in summary.itertuples():
        lines.append(f"| {r.method} | {r.mae_kWh:.4f} | {r.rmse_kWh:.4f} | {r.high_mae_kWh:.4f} | {r.negative_predictions} |")
    lines += ["", f"Ứng viên được chọn: **{manifest['selected_configuration']}**; Ridge tốt nhất: **{manifest['best_ridge_configuration']}**.", "",
              "## Phân tích lỗi trên các bản đo thật", "",
              f"Tải cao được xác định hồi cứu bằng target thực tế > p95 train = {manifest['thresholds']['high_target_kWh']:.3f} kWh. Có {len(high)} dòng; đây là các cửa sổ chồng lấp, không phải {len(high)} sự cố độc lập và không phải giới hạn vận hành.",
              f"HGB ban đầu dự báo thấp ở {(high.initial_hgb < high[TARGET]).sum()}/{len(high)} dòng tải cao; mô hình được chọn dự báo thấp ở {(high.selected < high[TARGET]).sum()}/{len(high)} dòng. Xem đầy đủ timestamp và dự báo trong high_load_cases.csv.",
              f"Theo giờ bắt đầu dự báo, MAE của mô hình được chọn lớn nhất ở giờ {hourly.idxmax():02d}: {hourly.max():.3f} kWh. errors_by_hour.csv và errors_by_weekday.csv ghi cả số mẫu để tránh suy luận chỉ từ trung bình.",
              f"Định nghĩa tăng mạnh: tổng giờ tới trừ tổng giờ vừa qua > p95 phần thay đổi dương trên train ({manifest['thresholds']['large_rise_kWh']:.3f} kWh). Validation có {len(rising)} dòng như vậy; MAE mô hình được chọn là {(rising.selected-rising[TARGET]).abs().mean():.3f} kWh. Nhãn này cần dữ liệu tương lai, chỉ dùng giải thích lỗi sau sự kiện.",
              f"Ridge ban đầu có {len(neg)} dự báo âm, thấp nhất {neg.initial_ridge.min():.3f} kWh. Target thật trong nhóm này nằm trong [{neg[TARGET].min():.3f}, {neg[TARGET].max():.3f}] kWh. Ridge là hàm tuyến tính không ràng buộc đầu ra không âm; các đóng góp feature cộng intercept ({manifest['ridge_intercept_kWh']:.3f} kWh) tái tạo chính xác dự báo.",
              "initial_ridge_contributions.csv đối chiếu đóng góp tuyến tính trung bình giữa nhóm dự báo âm/không âm; đây là phân rã toán học của mô hình, không chứng minh nguyên nhân vật lý. Các lag/rolling có tương quan nên không diễn giải hệ số như quan hệ nhân quả.",
              "Dataset không có lịch máy, sản lượng hay nhật ký sự cố để xác nhận nguyên nhân sai số. Không gán các mốc lỗi thành hỏng thiết bị, lãng phí hay sự cố đã xác nhận.", "",
              "![Phân tích validation](validation_diagnostics.png)", "", "## Độ trễ và khả năng tái lập", "",
              "Đo từng dòng trong 256 dòng validation đầu tiên sau một lần warmup, một luồng. Đây chỉ là thời gian gọi model.predict trên feature đã tạo, không gồm đọc cảm biến, tạo feature, truyền dữ liệu hoặc dashboard."]
    for name, r in latency.iterrows():
        lines.append(f"- {name}: p50 {r.p50_ms:.3f} ms; p95 {r.p95_ms:.3f} ms; max {r.max_ms:.3f} ms.")
    lines += ["", "Chạy lại: `python -m scripts.analyze_and_tune_steel --output-dir <thu_muc_moi>`. frozen_config.json được ghi trước các lần fit; run_manifest.json chứa cấu hình được chọn, hash code/dữ liệu và phiên bản. File CSV kèm theo tái tạo bảng và biểu đồ. Mô hình .joblib lưu trên máy, được Git bỏ qua.", "",
              "## Giới hạn và bàn giao", "",
              "Đây là kết quả chọn mô hình trên validation đã được xem ở lần chạy trước, không phải ước lượng độc lập về hiệu quả tương lai. Không coi cải thiện sau tuning là bằng chứng tổng quát hóa; không tiếp tục thử vô hạn trên tháng 9. Các cửa sổ 60 phút cách nhau 15 phút chồng lấp nên số dòng không tương đương số quan sát độc lập.",
              "Mẫu tải cao được chọn theo target thực tế nên phải xem thêm toàn bộ sai số; thiên lệch trong nhóm này không tự chứng minh mô hình bị lệch ở mọi trạng thái. Ngưỡng p95 chỉ để chẩn đoán, chưa phải ngưỡng cảnh báo hoặc công suất hợp đồng.",
              "Chưa có policy cảnh báo, confusion matrix cảnh báo, fault-injection, dashboard, kiểm định trên test hay đo tiết kiệm điện/tiền. Không tạo dữ liệu giả theo yêu cầu Huy. Vì vậy chưa tuyên bố hoàn tất toàn bộ cổng 7 hoặc các chương triển khai hệ thống.",
              "B cần kiểm tra chéo target/feature/split, baseline và kết quả. Sau khi nhóm chốt mô hình mới chuyển sang hiệu chỉnh cảnh báo bằng calibration và đánh giá cuối theo kế hoạch. Phần phương pháp/kết quả/giới hạn ở đây có thể dùng cho báo cáo kỹ thuật; các mục hệ thống chưa triển khai phải ghi rõ là chưa thực hiện."]
    (output / "technical_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/steel/modeling/tuning_v1")
    run(parser.parse_args().output_dir.resolve())
