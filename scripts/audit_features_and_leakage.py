"""QA trên pipeline và CSV thật: không dùng chuỗi giả để báo PASS.

python -m scripts.audit_features_and_leakage --processed-dir data/steel/processed
Chỉ kiểm dữ liệu/nhãn; không huấn luyện hoặc chấm điểm model trên test.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.steel.duy_forecasting import FEATURE_IMPLEMENTATION, forecast_features
from src.steel.preprocessing import (
    BASE_FEATURES, SPLIT_STARTS, TARGET, build_row_manifest,
    build_supervised_rows, load_official_raw, split_supervised_rows,
)

ROOT = Path(__file__).resolve().parents[1]
TIME_COLUMNS = ["observation_time", "forecast_start", "forecast_end"]


def rebuild_rows(raw, implementation):
    rows = build_supervised_rows(raw)
    if implementation == FEATURE_IMPLEMENTATION:
        rows.loc[:, list(BASE_FEATURES)] = forecast_features(raw).to_numpy()
    elif implementation != "legacy_pandas_rolling":
        raise ValueError(f"Không biết phiên bản feature: {implementation}")
    return rows


def check_causality(raw, implementation):
    original = rebuild_rows(raw, implementation)
    # Sửa bản đo tương lai thật. Feature tại t phải giữ nguyên, target phải đổi.
    for index in (672, 5000, 23328):
        changed = raw.copy()
        changed.loc[index + 1, "Usage_kWh"] += 500
        perturbed = rebuild_rows(changed, implementation)
        np.testing.assert_array_equal(
            original.loc[index, list(BASE_FEATURES)].to_numpy(),
            perturbed.loc[index, list(BASE_FEATURES)].to_numpy(),
        )
        np.testing.assert_allclose(
            perturbed.loc[index, TARGET] - original.loc[index, TARGET], 500,
            rtol=0, atol=1e-10,
        )


def audit(processed_dir: Path) -> dict:
    results = []

    def check(check_id, title, action):
        try:
            detail = action()
            results.append({"id": check_id, "title": title, "status": "PASS", "detail": detail})
        except (AssertionError, ValueError, KeyError, OSError, TypeError) as error:
            results.append({"id": check_id, "title": title, "status": "BLOCKER", "detail": str(error)})

    try:
        summary = json.loads((processed_dir / "steel_quality_summary.json").read_text(encoding="utf-8"))
        raw, raw_audit = load_official_raw(ROOT / "data/steel/raw/Steel_industry_data.csv")
        if summary["source_sha256"] != raw_audit["source_sha256"]:
            raise ValueError("Nguồn processed không khớp CSV UCI")
        implementation = summary.get("feature_implementation", "legacy_pandas_rolling")
        rows = rebuild_rows(raw, implementation)
        expected, split_audit = split_supervised_rows(rows)
        actual = {name: pd.read_csv(
            processed_dir / f"steel_next_60m_{name}.csv",
            parse_dates=TIME_COLUMNS, float_precision="round_trip",
        ) for name in expected}
    except (ValueError, KeyError, OSError) as error:
        return {"passed": False, "checks": [{"id": "INPUT", "status": "BLOCKER", "detail": str(error)}]}

    def schema():
        if summary["base_feature_columns"] != list(BASE_FEATURES):
            raise ValueError("Metadata phải đúng 15 feature và đúng thứ tự")
        for name, frame in actual.items():
            if list(frame.columns) != list(expected[name].columns):
                raise ValueError(f"Schema CSV {name} khác pipeline")
        return "Schema metadata và bốn CSV khớp pipeline; sensor optional không là model input"

    def causal():
        check_causality(raw, implementation)
        return "Sửa Usage(t+1) tại ba mốc thật: feature(t) giữ nguyên, target(t) tăng đúng 500"

    def availability():
        features = summary["base_feature_columns"]
        allowed = set(BASE_FEATURES)
        if any(column not in allowed for column in features):
            raise ValueError("Feature ngoài hợp đồng Usage/calendar; cần duyệt thời điểm có sẵn")
        return "Input chỉ là Usage tới t và lịch biết trước; lag_0 hợp lệ khi bản đo t đã hoàn tất"

    def context():
        if any(column in summary["base_feature_columns"] for column in ("CO2(tCO2)", "Load_Type")):
            raise ValueError("CO2/Load_Type không được dùng làm input mặc định")
        train_raw = raw.loc[raw["observation_time"] < SPLIT_STARTS["validation"]]
        corr = train_raw["Usage_kWh"].corr(train_raw["CO2(tCO2)"])
        return f"CO2/Load_Type loại khỏi input; corr TRAIN={corr:.6f}; tương quan cao không tự chứng minh leakage"

    def values():
        for name in expected:
            pd.testing.assert_frame_equal(
                actual[name], expected[name], check_dtype=False,
                check_exact=False, rtol=0, atol=1e-10,
            )
        return "Toàn bộ giá trị/thời gian/nhãn bốn CSV khớp tái tạo; không scale hoặc chèn mẫu trong CSV. Scaler train-only được kiểm riêng trong unittest"

    def accounting():
        manifest = pd.read_csv(processed_dir / "steel_row_manifest.csv",
                               parse_dates=["observation_time", "forecast_end"])
        pd.testing.assert_frame_equal(manifest, build_row_manifest(rows), check_dtype=False)
        if summary["split_rows"] != split_audit["split_rows"]:
            raise ValueError("Số hàng split trong metadata sai")
        if summary["row_disposition_counts"] != split_audit["row_disposition_counts"]:
            raise ValueError("Thống kê loại hàng trong metadata sai")
        if not np.isclose(summary["train_target_p95_kWh"], split_audit["train_target_p95_kWh"], rtol=0, atol=1e-10):
            raise ValueError("Ngưỡng không khớp p95 TRAIN")
        return f"Đối chiếu từng hàng manifest: {split_audit['row_disposition_counts']}"

    for check_id, title, action in (
        ("FE-01", "Schema", schema), ("FE-02", "Causality", causal),
        ("FE-03", "Availability", availability), ("FE-04", "Background", context),
        ("FE-05", "Reconstructed values", values), ("FE-06", "Manifest and boundaries", accounting),
    ):
        check(check_id, title, action)
    return {"passed": all(row["status"] == "PASS" for row in results),
            "feature_implementation": implementation, "checks": results}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/steel/processed")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    report = audit(args.processed_dir)
    for row in report["checks"]:
        print(f"[{row['status']}] {row['id']}: {row['detail']}")
    if args.output_json:
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
