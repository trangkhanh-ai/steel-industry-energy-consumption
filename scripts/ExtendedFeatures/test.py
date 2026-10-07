#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gate 7: Feature Extension Validation (Strictly Causal & Clean Proxies)
So sánh BASE_FEATURES vs EXTENDED_FEATURES trên tập Validation.
Tự động lưu bảng kết quả vào thư mục hiện tại.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb

warnings.filterwarnings("ignore")

# 1. Tự động xác định thư mục hiện tại và đường dẫn dữ liệu
CURRENT_DIR = Path(__file__).resolve().parent

# Tìm thư mục processed linh hoạt
if (CURRENT_DIR / "data" / "steel" / "processed").exists():
    DATA_DIR = CURRENT_DIR / "data" / "steel" / "processed"
elif (CURRENT_DIR.parent / "data" / "steel" / "processed").exists():
    DATA_DIR = CURRENT_DIR.parent / "data" / "steel" / "processed"
elif (CURRENT_DIR.parent.parent / "data" / "steel" / "processed").exists():
    DATA_DIR = CURRENT_DIR.parent.parent / "data" / "steel" / "processed"
else:
    DATA_DIR = CURRENT_DIR.parent.parent.parent / "data" / "steel" / "processed"

# 2. Danh sách 15 đặc trưng cơ sở (BASE_FEATURES)
BASE_FEATURES = [
    "usage_lag_0", "usage_lag_1", "usage_lag_4", "usage_lag_96", "usage_lag_672",
    "usage_sum_last_1h", "usage_mean_last_4h", "usage_std_last_4h",
    "usage_mean_last_24h", "usage_max_last_24h",
    "forecast_hour_sin", "forecast_hour_cos",
    "forecast_weekday_sin", "forecast_weekday_cos",
    "forecast_is_weekend"
]
TARGET = "target_next_60m_kWh"

# 3. Hàm trích xuất đặc trưng mở rộng an toàn (Strictly Causal)
def add_safe_extended_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sinh các đặc trưng mở rộng tại thời điểm t:
    - steel_ramp_rate: Tốc độ thay đổi phụ tải tức thời (kWh / 15 phút)
    - apparent_energy_proxy: Biến đại diện năng lượng biểu kiến kVAh
    - reactive_ratio: Tỷ số phản kháng có mặt nạ lọc khi P <= 5 kWh
    - low_lagging_pf_flag: Cờ cảnh báo chất lượng điện (PF < 90%)
    """
    df = df.copy()
    
    # 3.1. Ramp rate: Usage(t) - Usage(t-1)
    df["steel_ramp_rate"] = df["usage_lag_0"] - df["usage_lag_1"]
    
    # 3.2. Biến đại diện năng lượng biểu kiến: sqrt(P^2 + Q^2)
    p_energy = df["usage_lag_0"]
    q_energy = df["known_lagging_reactive_kVarh"]
    df["apparent_energy_proxy"] = np.sqrt(p_energy**2 + q_energy**2)
    
    # 3.3. Tỷ số Q/P có mặt nạ lọc: Chỉ tính khi máy đang chạy (P > 5 kWh)
    active_mask = p_energy > 5.0
    df["reactive_ratio"] = 0.0
    df.loc[active_mask, "reactive_ratio"] = (
        q_energy[active_mask] / p_energy[active_mask]
    )
    
    # 3.4. Cờ cảnh báo hệ số công suất thấp (ngưỡng nghiên cứu 90%)
    pf_raw = df["known_lagging_power_factor"]
    df["low_lagging_pf_flag"] = (pf_raw < 90.0).astype(int)
    
    return df

# 4. Đọc dữ liệu Train (Jan-Aug) và Validation (Sep)
train_file = DATA_DIR / "steel_next_60m_train.csv"
val_file = DATA_DIR / "steel_next_60m_validation.csv"

if not train_file.exists():
    raise FileNotFoundError(f"Không tìm thấy file: {train_file.resolve()}")

print(f"Đang đọc dữ liệu từ: {DATA_DIR.resolve()}...")
train_raw = pd.read_csv(train_file)
val_raw = pd.read_csv(val_file)

# Áp dụng sinh đặc trưng
train_df = add_safe_extended_features(train_raw)
val_df = add_safe_extended_features(val_raw)

NEW_FEATURES = [
    "steel_ramp_rate", 
    "apparent_energy_proxy", 
    "reactive_ratio", 
    "low_lagging_pf_flag"
]
EXTENDED_FEATURES = BASE_FEATURES + NEW_FEATURES

print(f"- Số biến BASE: {len(BASE_FEATURES)}")
print(f"- Số biến EXTENDED: {len(EXTENDED_FEATURES)}")

# 5. Huấn luyện và kiểm nghiệm đối chuẩn
results = []

def calculate_rmse(y_true, y_pred):
    """Tính RMSE tương thích mọi phiên bản scikit-learn."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def evaluate_models(features: list, feature_set_name: str):
    X_tr = train_df[features].values
    y_tr = train_df[TARGET].values
    X_va = val_df[features].values
    y_va = val_df[TARGET].values

    # --- Thử nghiệm A: Ridge Regression ---
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    
    ridge = Ridge(alpha=1.0, random_state=42).fit(X_tr_scaled, y_tr)
    ridge_pred = ridge.predict(X_va_scaled)
    ridge_mae = mean_absolute_error(y_va, ridge_pred)
    ridge_rmse = calculate_rmse(y_va, ridge_pred)
    
    results.append({
        "Feature Set": feature_set_name,
        "Model": "Ridge",
        "Val MAE (kWh)": round(ridge_mae, 2),
        "Val RMSE (kWh)": round(ridge_rmse, 2)
    })

    # --- Thử nghiệm B: LightGBM (Regression L1 Loss) ---
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=train_data)
    
    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "seed": 42,
        "subsample_freq": 1
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    lgb_pred = model.predict(X_va)
    lgb_mae = mean_absolute_error(y_va, lgb_pred)
    lgb_rmse = calculate_rmse(y_va, lgb_pred)
    
    results.append({
        "Feature Set": feature_set_name,
        "Model": "LightGBM",
        "Val MAE (kWh)": round(lgb_mae, 2),
        "Val RMSE (kWh)": round(lgb_rmse, 2)
    })

# Thực thi kiểm nghiệm cả 2 bộ đặc trưng
evaluate_models(BASE_FEATURES, "BASE (15 biến)")
evaluate_models(EXTENDED_FEATURES, "EXTENDED (19 biến)")

# 6. Hiển thị và in kết quả ra file trong thư mục hiện tại
summary_df = pd.DataFrame(results)

output_csv = CURRENT_DIR / "gate7_validation_results.csv"
output_md = CURRENT_DIR / "gate7_validation_results.md"

summary_df.to_csv(output_csv, index=False, encoding="utf-8")

with open(output_md, "w", encoding="utf-8") as f:
    f.write("### Bảng kết quả đối chuẩn Gate 7 (Tập Validation - Tháng 9)\n\n")
    f.write(summary_df.to_markdown(index=False))
    f.write("\n")

print("\n" + "="*55)
print("BẢNG KẾT QUẢ ĐỐI CHUẨN TRÊN TẬP VALIDATION (THÁNG 9)")
print("="*55)
print(summary_df.to_string(index=False))
print("="*55)
print(f"Đã lưu kết quả CSV tại: {output_csv.resolve()}")
print(f"Đã lưu kết quả MD tại:  {output_md.resolve()}")