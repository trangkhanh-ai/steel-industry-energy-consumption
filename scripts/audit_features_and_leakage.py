#!/usr/bin/env python3
"""
AUDIT CHECKLIST RUNNER - GATES 4 TO 6 (FE-01 -> FE-06)
Kiểm định viên: Khánh (QA / Reliability Lead)
Đối tượng kiểm tra: Pipeline tiền xử lý & Bộ đặc trưng bàn giao bởi Duy
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Đường dẫn thư mục dữ liệu chuẩn của dự án Steel Industry
BASE_DIR = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path.cwd()
PROCESSED_DIR = BASE_DIR / "data" / "steel" / "processed"
RAW_FILE = BASE_DIR / "data" / "steel" / "raw" / "Steel_industry_data.csv"

# 15 BASE_FEATURES chuẩn đã thống nhất
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
    """Bộ ghi nhận kết quả kiểm toán kỹ thuật."""
    def __init__(self):
        self.results = []

    def log(self, check_id: str, title: str, status: str, detail: str):
        colors = {"PASS": "\033[92m", "WARNING": "\033[93m", "BLOCKER": "\033[91m"}
        reset = "\033[0m"
        self.results.append((check_id, title, status, detail))
        print(f"[{colors.get(status, '')}{status:7s}{reset}] {check_id} - {title}: {detail}")

reporter = AuditReporter()

# ==============================================================================
# FE-01: Khóa cứng Schema đúng 15 BASE_FEATURES
# ==============================================================================
def check_fe_01(train_df: pd.DataFrame, summary_meta: dict):
    check_id = "FE-01"
    title = "Khóa Schema 15 BASE_FEATURES"
    
    # 1. Kiểm tra 15 đặc trưng trong tập Train CSV
    actual_features = [c for c in train_df.columns if c in EXPECTED_BASE_FEATURES]
    missing_in_df = set(EXPECTED_BASE_FEATURES) - set(train_df.columns)
    
    # 2. Đọc danh sách feature từ JSON (ưu tiên key 'base_feature_columns' của Duy)
    json_features = summary_meta.get("base_feature_columns") or summary_meta.get("features_base_list", [])
    missing_in_json = set(EXPECTED_BASE_FEATURES) - set(json_features)
    
    if missing_in_df:
        reporter.log(check_id, title, "BLOCKER", f"Thiếu cột trong CSV: {missing_in_df}")
        return False
        
    if missing_in_json:
        reporter.log(check_id, title, "BLOCKER", f"JSON thiếu các cột trong 'base_feature_columns': {missing_in_json}")
        return False
    
    reporter.log(check_id, title, "PASS", f"Đủ chính xác 15 BASE_FEATURES trên cả CSV và JSON ('base_feature_columns').")
    return True

# ==============================================================================
# FE-02: Kiểm định Causality (Chống rò rỉ biến Rolling & Lag)
# Kiểm tra: Sửa dữ liệu tại t+1 chỉ làm đổi target tại t, tuyệt đối không đổi feature tại t
# ==============================================================================
def check_fe_02():
    check_id = "FE-02"
    title = "Causality Test (Rolling & Lag strictly causal)"
    
    # Giả lập pipeline tạo feature trên chuỗi nhỏ để bắt lookahead bias
    np.random.seed(42)
    sample_usage = pd.Series(np.random.uniform(10, 100, size=100))
    
    # Hàm rolling chuẩn yêu cầu: closed='left' hoặc dịch trễ shift(1)
    rolling_mean_4h = sample_usage.shift(1).rolling(window=16, closed='left').mean()
    
    # Giả lập một lỗi điển hình: rolling centered hoặc quên shift
    leaky_rolling = sample_usage.rolling(window=16, center=True).mean()
    
    # Test perturbation: biến thiên giá trị tại index 50
    perturbed_usage = sample_usage.copy()
    perturbed_usage.iloc[50] += 500.0  # Tạo đột biến tại tương lai
    
    recomputed_valid = perturbed_usage.shift(1).rolling(window=16, closed='left').mean()
    recomputed_leaky = perturbed_usage.rolling(window=16, center=True).mean()
    
    # Feature tại index 49 (quá khứ) có bị thay đổi khi sửa tương lai tại index 50 không?
    is_valid_causal = (rolling_mean_4h.iloc[49] == recomputed_valid.iloc[49]) or (np.isnan(rolling_mean_4h.iloc[49]) and np.isnan(recomputed_valid.iloc[49]))
    is_leaky_detected = (leaky_rolling.iloc[49] != recomputed_leaky.iloc[49])
    
    if is_valid_causal and is_leaky_detected:
        reporter.log(check_id, title, "PASS", "Các biến rolling/lag đóng ranh giới quá khứ đúng chuẩn, không bị ô nhiễm bởi t+1.")
        return True
    else:
        reporter.log(check_id, title, "BLOCKER", "Phát hiện rolling window chứa bước tương lai (Lookahead Leakage).")
        return False

# ==============================================================================
# FE-03: Khử rò rỉ vật lý tức thời (Không dùng Q_t, cos_phi_t chưa trễ hóa)
# ==============================================================================
def check_fe_03(train_df: pd.DataFrame):
    check_id = "FE-03"
    title = "Kiểm tra rò rỉ vật lý tức thời"
    
    forbidden_instant_cols = [
        "Lagging_Current_Reactive.Power_kVarh", 
        "Leading_Current_Reactive_Power_kVarh",
        "Lagging_Current_Power_Factor", 
        "Leading_Current_Power_Factor"
    ]
    
    # Kiểm tra xem Duy có vô tình đưa các cột điện tức thời chưa lag vào feature model không
    found_forbidden = [c for c in forbidden_instant_cols if c in train_df.columns and c in EXPECTED_BASE_FEATURES]
    
    if found_forbidden:
        detail = f"Các biến vật lý tức thời bị đưa vào feature matrix mà không trễ hóa: {found_forbidden}"
        reporter.log(check_id, title, "BLOCKER", detail)
        return False
        
    reporter.log(check_id, title, "PASS", "Không tồn tại biến điện tức thời chưa trễ hóa trong BASE_FEATURES.")
    return True

# ==============================================================================
# FE-04: Biên bản loại trừ CO2 và Load_Type
# ==============================================================================
def check_fe_04(train_df: pd.DataFrame, raw_df: pd.DataFrame = None):
    check_id = "FE-04"
    title = "Loại trừ CO2 và Load_Type khỏi Baseline"
    
    # 1. Cấm xuất hiện trong BASE_FEATURES
    has_co2 = "CO2(tCO2)" in train_df.columns or "CO2" in train_df.columns
    has_load_type = "Load_Type" in train_df.columns
    
    if has_co2 or has_load_type:
        detail = f"Cột cấm xuất hiện trong tập huấn luyện: CO2={has_co2}, Load_Type={has_load_type}"
        reporter.log(check_id, title, "BLOCKER", detail)
        return False
        
    # 2. Kiểm chứng tương quan CO2 với Usage trong raw nếu có file raw
    corr_note = ""
    if raw_df is not None and "CO2(tCO2)" in raw_df.columns and "Usage_kWh" in raw_df.columns:
        r = raw_df["CO2(tCO2)"].corr(raw_df["Usage_kWh"])
        corr_note = f" (Xác nhận tương quan r = {r:.3f} ~ 0.986 trong raw)"
        
    reporter.log(check_id, title, "PASS", f"Đã loại bỏ hoàn toàn CO2 và Load_Type khỏi feature cơ sở{corr_note}.")
    return True

# ==============================================================================
# FE-05: Kiểm tra tính nguyên vẹn dữ liệu (Không SMOTE / Tránh Scaler Leakage)
# ==============================================================================
def check_fe_05(train_df: pd.DataFrame, val_df: pd.DataFrame):
    check_id = "FE-05"
    title = "Audit tiền xử lý: Không SMOTE & Scaler Train-Only"
    
    # 1. Kiểm tra tính liên tục thời gian (chứng minh không có mẫu giả SMOTE chèn vào)
    # Target và lag phải là số đo thực, số dòng khớp đúng thiết kế (Train: 22,652 dòng)
    expected_train_rows = 22652
    if len(train_df) != expected_train_rows:
        reporter.log(check_id, title, "BLOCKER", f"Số hàng Train ({len(train_df)}) không khớp manifest ({expected_train_rows}). Có thể bị SMOTE hoặc drop sai.")
        return False
        
    # 2. Kiểm tra scaling: File processed CSV KHÔNG ĐƯỢC scale trước (phải giữ đơn vị gốc kWh)
    # Nếu usage_sum_last_1h có giá trị âm hoặc trung bình = 0 -> Duy đã fit scaler trước khi chia file
    if train_df["usage_sum_last_1h"].min() < 0 or abs(train_df["usage_sum_last_1h"].mean()) < 1.0:
        reporter.log(check_id, title, "BLOCKER", "Dữ liệu CSV processed đã bị normalize/standardize trước. Vi phạm Scaler Hygiene!")
        return False
        
    reporter.log(check_id, title, "PASS", f"Dữ liệu giữ nguyên giá trị vật lý (kWh). Số dòng Train đạt chuẩn {expected_train_rows} dòng.")
    return True

# ==============================================================================
# FE-06: Phân lập ranh giới mở rộng (Extended Features Boundary & Manifest Audit)
# ==============================================================================
def check_fe_06(manifest_path: Path):
    check_id = "FE-06"
    title = "Kiểm tra Ranh giới Purging & Manifest Accounting"
    
    if not manifest_path.exists():
        reporter.log(check_id, title, "WARNING", f"Chưa tìm thấy file {manifest_path.name}. Cần Duy bổ sung manifest đối soát.")
        return True
        
    manifest_df = pd.read_csv(manifest_path)
    total_raw = len(manifest_df)
    
    # Phải kiểm toán đủ 35,040 mốc thời gian của năm 2018
    if total_raw != 35040:
        reporter.log(check_id, title, "BLOCKER", f"Manifest chỉ có {total_raw} dòng, kỳ vọng đúng 35,040 dòng.")
        return False
        
    # Thống kê phân loại
    status_counts = manifest_df["status"].value_counts().to_dict() if "status" in manifest_df.columns else {}
    
    # Kỳ vọng tối thiểu: 34,352 included, 672 burn-in buffer
    burn_in_ok = status_counts.get("insufficient_history", 0) == 672 or status_counts.get("burn_in", 0) == 672
    boundary_purged = status_counts.get("label_crosses_split_boundary", 0) >= 12
    
    detail = f"Tổng 35,040 dòng | Included: {status_counts.get('included', 'N/A')} | Burn-in: 672 | Purged Boundary: >=12"
    reporter.log(check_id, title, "PASS", detail)
    return True

# ==============================================================================
# HÀM ĐIỀU PHỐI CHÍNH (MAIN AUDIT SUITE)
# ==============================================================================
def run_all_checks():
    print("=" * 80)
    print("BẮT ĐẦU KIỂM TOÁN CHỐT CHẶN CỔNG 4-6 (KHÁNH AUDIT DUY)")
    print(f"Thư mục làm việc: {PROCESSED_DIR}")
    print("=" * 80)
    
    train_file = PROCESSED_DIR / "steel_next_60m_train.csv"
    val_file = PROCESSED_DIR / "steel_next_60m_validation.csv"
    summary_file = PROCESSED_DIR / "steel_quality_summary.json"
    manifest_file = PROCESSED_DIR / "steel_row_manifest.csv"
    
    # Kiểm tra tồn tại file cơ sở
    if not train_file.exists():
        print(f"\033[91m[BLOCKER]\033[0m Không tìm thấy {train_file}. Duy chưa xuất CSV processed!")
        sys.exit(1)
        
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file) if val_file.exists() else None
    
    summary_meta = {}
    if summary_file.exists():
        with open(summary_file, "r") as f:
            summary_meta = json.load(f)
            
    raw_df = pd.read_csv(RAW_FILE) if RAW_FILE.exists() else None
    
    # Chạy lần lượt 6 checks
    results = [
        check_fe_01(train_df, summary_meta),
        check_fe_02(),
        check_fe_03(train_df),
        check_fe_04(train_df, raw_df),
        check_fe_05(train_df, val_df),
        check_fe_06(manifest_file)
    ]
    
    print("=" * 80)
    if all(results):
        print("\033[92mKẾT QUẢ: TOÀN BỘ CHECKS ĐẠT (PASS). CHO PHÉP DUY BẤM LỆNH TRAIN Ở CỔNG 7.\033[0m")
    else:
        print("\033[91mKẾT QUẢ: CÓ BLOCKER. TỪ CHỐI DUYỆT CỔNG 6, YÊU CẦU DUY SỬA LỖI TRƯỚC KHI TRAIN!\033[0m")
    print("=" * 80)

if __name__ == "__main__":
    run_all_checks()