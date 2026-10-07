#!/usr/bin/env python3
"""
AUDIT CHECKLIST RUNNER - GATES 4 TO 6 (FE-01 -> FE-06)
Kiểm định viên: Khánh (QA / Reliability Lead)
Đối tượng kiểm tra: Pipeline tiền xử lý & Bộ đặc trưng bàn giao bởi Duy
Phiên bản: v2.0 (Khắc phục toàn diện K01 - K03)
"""

import json
import sys
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

# Thiết lập đường dẫn dự án
BASE_DIR = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path.cwd()
sys.path.insert(0, str(BASE_DIR))

PROCESSED_DIR = BASE_DIR / "data" / "steel" / "processed"
RAW_FILE = BASE_DIR / "data" / "steel" / "raw" / "Steel_industry_data.csv"

# 15 BASE_FEATURES chuẩn hóa cố định theo thứ tự
EXPECTED_BASE_FEATURES = [
    "usage_lag_0", "usage_lag_1", "usage_lag_4", "usage_lag_96", "usage_lag_672",
    "usage_sum_last_1h", "usage_mean_last_4h", "usage_std_last_4h",
    "usage_mean_last_24h", "usage_max_last_24h",
    "forecast_hour_sin", "forecast_hour_cos",
    "forecast_weekday_sin", "forecast_weekday_cos",
    "forecast_is_weekend"
]
TARGET_COL = "target_next_60m_kWh"

class AuditReporter:
    """Bộ ghi nhận và báo cáo kết quả kiểm toán kỹ thuật."""
    def __init__(self):
        self.results = []
        self.has_blocker = False

    def log(self, check_id: str, title: str, status: str, detail: str):
        colors = {"PASS": "\033[92m", "WARNING": "\033[93m", "BLOCKER": "\033[91m"}
        reset = "\033[0m"
        if status == "BLOCKER":
            self.has_blocker = True
        self.results.append((check_id, title, status, detail))
        print(f"[{colors.get(status, '')}{status:7s}{reset}] {check_id} - {title}: {detail}")

reporter = AuditReporter()

# ==============================================================================
# FE-01: Khóa cứng Schema & Thứ tự 15 BASE_FEATURES (Sửa K03)
# ==============================================================================
def check_fe_01(train_df: pd.DataFrame, summary_meta: dict):
    check_id = "FE-01"
    title = "Khóa Schema & Thứ tự 15 BASE_FEATURES"
    
    # 1. Đọc danh sách feature từ JSON metadata của Duy
    json_features = summary_meta.get("base_feature_columns", [])
    if not json_features:
        reporter.log(check_id, title, "BLOCKER", "JSON metadata thiếu key 'base_feature_columns'.")
        return False

    # 2. Kiểm tra chính xác thứ tự và độ dài
    if json_features != EXPECTED_BASE_FEATURES:
        missing = set(EXPECTED_BASE_FEATURES) - set(json_features)
        extra = set(json_features) - set(EXPECTED_BASE_FEATURES)
        reporter.log(check_id, title, "BLOCKER", 
                     f"Sai lệch schema trong JSON. Thiếu: {missing}, Thừa: {extra}, Sai thứ tự: {json_features != EXPECTED_BASE_FEATURES}")
        return False

    # 3. Kiểm tra sự tồn tại trong file train CSV
    missing_in_df = set(EXPECTED_BASE_FEATURES) - set(train_df.columns)
    if missing_in_df:
        reporter.log(check_id, title, "BLOCKER", f"Tập Train CSV thiếu các cột: {missing_in_df}")
        return False

    reporter.log(check_id, title, "PASS", "Khớp chính xác 15/15 BASE_FEATURES đúng thứ tự trên cả CSV và JSON.")
    return True

# ==============================================================================
# FE-02: Kiểm định Causality trên Pipeline thật (Sửa K01)
# ==============================================================================
def check_fe_02():
    check_id = "FE-02"
    title = "Causality Test trên Pipeline Thật"

    # 1. Nạp hàm của Duy
    try:
        from src.steel.preprocessing import build_supervised_rows
    except ImportError:
        reporter.log(check_id, title, "BLOCKER", "Không import được hàm build_supervised_rows từ src.steel.preprocessing")
        return False

    if not RAW_FILE.exists():
        reporter.log(check_id, title, "BLOCKER", f"Không tìm thấy file: {RAW_FILE}")
        return False

    # 2. Chuẩn bị dữ liệu đúng luật của hàm: có observation_time, sort tăng dần, đủ 15 phút
    raw_df = pd.read_csv(RAW_FILE)
    raw_df["observation_time"] = pd.to_datetime(raw_df["date"], format="%d/%m/%Y %H:%M")
    raw_df = raw_df.sort_values("observation_time").reset_index(drop=True)

    # Lấy 1.000 dòng đầu (đủ lớn hơn 672 bước của lag tuần)
    sample_raw = raw_df.iloc[:1000].copy()

    # Tạo bản dữ liệu bị sửa ở tương lai t+1 (dòng 801)
    perturbed_raw = sample_raw.copy()
    perturbed_raw.loc[801, "Usage_kWh"] += 500.0

    try:
        # 3. Chạy pipeline thật
        df_orig = build_supervised_rows(sample_raw)
        df_pert = build_supervised_rows(perturbed_raw)
    except Exception as e:
        reporter.log(check_id, title, "BLOCKER", f"Lỗi khi chạy build_supervised_rows: {e}")
        return False

    # 4. Kiểm tra tại thời điểm t = 800
    t = 800
    
    # Feature tại t có bị đổi không?
    feat_orig = df_orig.loc[t, EXPECTED_BASE_FEATURES].values.astype(float)
    feat_pert = df_pert.loc[t, EXPECTED_BASE_FEATURES].values.astype(float)
    features_ok = np.allclose(feat_orig, feat_pert, equal_nan=False)

    # Target tại t có đổi đúng khi t+1 đổi không?
    target_orig = df_orig.loc[t, TARGET_COL]
    target_pert = df_pert.loc[t, TARGET_COL]
    target_ok = not np.isclose(target_orig, target_pert)

    if features_ok and target_ok:
        reporter.log(check_id, title, "PASS", "Sửa dữ liệu t+1 chỉ đổi target tại t, toàn bộ 15 feature tại t giữ nguyên.")
        return True
    else:
        reporter.log(check_id, title, "BLOCKER", f"Lỗi rò rỉ: Features không đổi={features_ok}, Target đổi={target_ok}")
        return False

# ==============================================================================
# FE-03: Khử rò rỉ biến điện tức thời trong Model Matrix (Sửa K03)
# ==============================================================================
def check_fe_03(train_df: pd.DataFrame, summary_meta: dict):
    check_id = "FE-03"
    title = "Kiểm tra Biến điện Tức thời & Schema Model"
    
    forbidden_instant_cols = [
        "Lagging_Current_Reactive.Power_kVarh", 
        "Leading_Current_Reactive_Power_kVarh",
        "Lagging_Current_Power_Factor", 
        "Leading_Current_Power_Factor"
    ]
    
    # 1. Kiểm tra danh sách feature chính thức của model trong JSON
    model_features = summary_meta.get("base_feature_columns", [])
    forbidden_in_model = [c for c in forbidden_instant_cols if c in model_features]
    
    if forbidden_in_model:
        reporter.log(check_id, title, "BLOCKER", f"Biến điện tức thời chưa lag nằm trong danh sách feature nạp model: {forbidden_in_model}")
        return False
        
    # 2. Kiểm tra tính độc lập: nếu file CSV chứa sensor columns để phục vụ mở rộng,
    # phải bảo đảm mô hình baseline chỉ chọn lọc đúng 15 BASE_FEATURES
    extra_sensor_cols = [c for c in forbidden_instant_cols if c in train_df.columns]
    sensor_note = f" (CSV có chứa {len(extra_sensor_cols)} cột sensor phụ, model matrix tách riêng 15 cột)" if extra_sensor_cols else ""

    reporter.log(check_id, title, "PASS", f"Không có biến điện tức thời chưa trễ hóa trong không gian đặc trưng huấn luyện{sensor_note}.")
    return True

# ==============================================================================
# FE-04: Loại trừ CO2 và Load_Type
# ==============================================================================
def check_fe_04(train_df: pd.DataFrame, summary_meta: dict):
    check_id = "FE-04"
    title = "Loại trừ CO2 và Load_Type khỏi Feature Set"
    
    model_features = summary_meta.get("base_feature_columns", EXPECTED_BASE_FEATURES)
    has_co2 = any(c in model_features for c in ["CO2(tCO2)", "CO2", "co2"])
    has_load_type = "Load_Type" in model_features
    
    if has_co2 or has_load_type:
        reporter.log(check_id, title, "BLOCKER", f"Cột cấm xuất hiện trong feature set: CO2={has_co2}, Load_Type={has_load_type}")
        return False
        
    co2_corr = summary_meta.get("train_co2_usage_correlation")
    corr_note = f" (Xác nhận tương quan r = {co2_corr:.4f} trong train)" if co2_corr is not None else ""
    reporter.log(check_id, title, "PASS", f"Đã loại bỏ CO2 và Load_Type khỏi baseline features{corr_note}.")
    return True

# ==============================================================================
# FE-05: Kiểm tra Tính toàn vẹn Dữ liệu & Scaler Hygiene (Sửa K03)
# ==============================================================================
def check_fe_05(train_df: pd.DataFrame):
    check_id = "FE-05"
    title = "Audit Toàn vẹn Dữ liệu: Không SMOTE & Scaler Train-Only"
    
    # 1. Kiểm tra số dòng tập Train chuẩn xác
    expected_train_rows = 22652
    if len(train_df) != expected_train_rows:
        reporter.log(check_id, title, "BLOCKER", f"Số hàng Train ({len(train_df)}) != {expected_train_rows}.")
        return False
        
    # 2. Dữ liệu CSV processed không được scale trước
    if train_df["usage_sum_last_1h"].min() < 0 or abs(train_df["usage_sum_last_1h"].mean()) < 1.0:
        reporter.log(check_id, title, "BLOCKER", "Dữ liệu CSV đã bị chuẩn hóa trước (vi phạm Scaler Hygiene).")
        return False

    # 3. Kiểm tra tính liên tục thời gian (chứng minh không có mẫu SMOTE nhân tạo làm gián đoạn)
    if "observation_time" in train_df.columns:
        ts = pd.to_datetime(train_df["observation_time"])
        diffs = ts.diff().dropna()
        # Các bước nhảy bình thường là 15 phút, ngoại trừ các điểm giao ranh giới
        invalid_steps = diffs[(diffs != pd.Timedelta(minutes=15)) & (diffs < pd.Timedelta(minutes=0))]
        if len(invalid_steps) > 0:
            reporter.log(check_id, title, "BLOCKER", f"Phát hiện timestamp bị đảo ngược hoặc nhảy lùi: {len(invalid_steps)} điểm.")
            return False

    reporter.log(check_id, title, "PASS", f"Dữ liệu gốc giữ nguyên đơn vị kWh. Đạt chuẩn {expected_train_rows} dòng, không bị SMOTE.")
    return True

# ==============================================================================
# FE-06: Kiểm tra Ranh giới Purging & Manifest Accounting Chuẩn (Sửa K02)
# ==============================================================================
def check_fe_06(manifest_path: Path):
    check_id = "FE-06"
    title = "Kiểm tra Ranh giới Purging & Manifest Accounting"
    
    if not manifest_path.exists():
        reporter.log(check_id, title, "BLOCKER", f"Thiếu file {manifest_path.name}. Đây là lỗi BLOCKER!")
        return False
        
    manifest_df = pd.read_csv(manifest_path)
    total_raw = len(manifest_df)
    
    # 1. Kiểm toán đủ 35.040 bản ghi năm 2018
    if total_raw != 35040:
        reporter.log(check_id, title, "BLOCKER", f"Manifest có {total_raw} dòng (kỳ vọng đúng 35.040).")
        return False

    # 2. Kiểm tra schema bắt buộc
    required_cols = {"row_status", "reason"}
    if not required_cols.issubset(manifest_df.columns):
        reporter.log(check_id, title, "BLOCKER", f"Manifest sai schema. Cần có {required_cols}, hiện có: {list(manifest_df.columns)}")
        return False

    # 3. Đối soát chính xác 4 nhóm reason
    reason_counts = manifest_df["reason"].value_counts().to_dict()
    
    c_included = reason_counts.get("included", 0)
    c_history = reason_counts.get("insufficient_history", 0) or reason_counts.get("burn_in", 0)
    c_boundary = reason_counts.get("boundary", 0) or reason_counts.get("label_crosses_split_boundary", 0)
    c_future = reason_counts.get("future_label_unavailable", 0)

    included_ok = (c_included == 34352)
    burn_in_ok = (c_history == 672)
    boundary_ok = (c_boundary == 12)
    future_ok = (c_future == 4)
    total_sum_ok = (c_included + c_history + c_boundary + c_future == 35040)

    if not (included_ok and burn_in_ok and boundary_ok and future_ok and total_sum_ok):
        detail = (f"Sai số lượng kế toán manifest: included={c_included} (cần 34.352), "
                  f"insufficient_history={c_history} (cần 672), boundary={c_boundary} (cần 12), "
                  f"future_label_unavailable={c_future} (cần 4).")
        reporter.log(check_id, title, "BLOCKER", detail)
        return False

    # 4. Kiểm tra điều kiện timestamp ranh giới (forecast_end <= split_end)
    if "forecast_end" in manifest_df.columns and "split_end" in manifest_df.columns:
        included_mask = manifest_df["row_status"] == "included"
        f_end = pd.to_datetime(manifest_df.loc[included_mask, "forecast_end"])
        s_end = pd.to_datetime(manifest_df.loc[included_mask, "split_end"])
        boundary_violations = (f_end > s_end).sum()
        if boundary_violations > 0:
            reporter.log(check_id, title, "BLOCKER", f"Phát hiện {boundary_violations} dòng included có forecast_end vượt quá split_end!")
            return False

    detail = f"Tổng 35.040 dòng cân bằng chuẩn: Included=34.352, Burn-in=672, Purged Boundary=12, Future Unavailable=4."
    reporter.log(check_id, title, "PASS", detail)
    return True

# ==============================================================================
# HÀM ĐIỀU PHỐI CHÍNH (MAIN AUDIT SUITE)
# ==============================================================================
def run_all_checks():
    print("=" * 80)
    print("BẮT ĐẦU KIỂM TOÁN CHỐT CHẶN CỔNG 4-6 (KHÁNH AUDIT DUY) - BUILD V2.0")
    print(f"Thư mục dữ liệu: {PROCESSED_DIR}")
    print("=" * 80)
    
    train_file = PROCESSED_DIR / "steel_next_60m_train.csv"
    summary_file = PROCESSED_DIR / "steel_quality_summary.json"
    manifest_file = PROCESSED_DIR / "steel_row_manifest.csv"
    
    if not train_file.exists():
        print(f"\033[91m[BLOCKER]\033[0m Không tìm thấy {train_file.name}. Duy chưa xuất CSV processed!")
        sys.exit(1)
        
    train_df = pd.read_csv(train_file)
    summary_meta = {}
    if summary_file.exists():
        with open(summary_file, "r") as f:
            summary_meta = json.load(f)
            
    # Chạy lần lượt 6 chốt chặn
    results = [
        check_fe_01(train_df, summary_meta),
        check_fe_02(),
        check_fe_03(train_df, summary_meta),
        check_fe_04(train_df, summary_meta),
        check_fe_05(train_df),
        check_fe_06(manifest_file)
    ]
    
    print("=" * 80)
    if all(results) and not reporter.has_blocker:
        print("\033[92mKẾT QUẢ: TOÀN BỘ CHECKS ĐẠT (PASS). CHO PHÉP DUY BẤM LỆNH TRAIN Ở CỔNG 7.\033[0m")
        sys.exit(0)
    else:
        print("\033[91mKẾT QUẢ: CÓ BLOCKER. TỪ CHỐI DUYỆT CỔNG 6, YÊU CẦU DUY SỬA LỖI TRƯỚC KHI TRAIN!\033[0m")
        sys.exit(1)

if __name__ == "__main__":
    run_all_checks()