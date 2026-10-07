#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script tính toán ma trận tương quan Pearson và xuất trực tiếp biểu đồ nhiệt
vào đúng thư mục có sẵn: reports/steel/figures/extendedFeats/
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. Định vị chuẩn xác thư mục gốc (lùi 2 cấp từ scripts/ExtendedFeatures/)
CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parents[2]

# Dự phòng tự động tìm thư mục gốc nếu chạy từ môi trường khác
if not (ROOT_DIR / "data" / "steel" / "processed").exists():
    for parent in CURRENT_FILE.parents:
        if (parent / "data" / "steel" / "processed").exists():
            ROOT_DIR = parent
            break

DATA_DIR = ROOT_DIR / "data" / "steel" / "processed"
# Trỏ trực tiếp vào đúng thư mục extendedFeats có sẵn trong cây thư mục
OUTPUT_DIR = ROOT_DIR / "reports" / "steel" / "figures" / "extendedFeats"

# 2. Đọc dữ liệu Train
train_file = DATA_DIR / "steel_next_60m_train.csv"
if not train_file.exists():
    raise FileNotFoundError(f"Không tìm thấy file: {train_file.resolve()}")

print(f"Đang đọc dữ liệu Train từ: {train_file.resolve()}")
df_train = pd.read_csv(train_file)

# 3. Hàm tạo 4 đặc trưng mở rộng (an toàn, chống nổ số)
def add_safe_extended_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 3.1. Tốc độ thay đổi phụ tải tức thời
    df["steel_ramp_rate"] = df["usage_lag_0"] - df["usage_lag_1"]
    
    # Nhận diện linh hoạt tên cột phản kháng và PF (hỗ trợ cả tên thô lẫn tên đã tiền xử lý)
    q_col = "known_lagging_reactive_kVarh" if "known_lagging_reactive_kVarh" in df.columns else "Lagging_Current_Reactive.Power_kVarh"
    pf_col = "known_lagging_power_factor" if "known_lagging_power_factor" in df.columns else "Lagging_Current_Power_Factor"
    
    p_energy = df["usage_lag_0"]
    q_energy = df[q_col]
    
    # 3.2. Biến đại diện năng lượng biểu kiến: sqrt(P^2 + Q^2)
    df["apparent_energy_proxy"] = np.sqrt(p_energy**2 + q_energy**2)
    
    # 3.3. Tỷ số phản kháng có mặt nạ lọc khi P > 5 kWh
    active_mask = p_energy > 5.0
    df["reactive_ratio"] = 0.0
    df.loc[active_mask, "reactive_ratio"] = q_energy[active_mask] / p_energy[active_mask]
    
    # 3.4. Cờ cảnh báo hệ số công suất thấp (< 90%)
    pf_raw = df[pf_col]
    df["low_lagging_pf_flag"] = (pf_raw < 90.0).astype(int)
    
    return df

df_extended = add_safe_extended_features(df_train)

# 4. Danh sách các đặc trưng
BASE_FEATURES = [
    "usage_lag_0", "usage_lag_1", "usage_lag_4", "usage_lag_96", "usage_lag_672",
    "usage_sum_last_1h", "usage_mean_last_4h", "usage_std_last_4h",
    "usage_mean_last_24h", "usage_max_last_24h",
    "forecast_hour_sin", "forecast_hour_cos",
    "forecast_weekday_sin", "forecast_weekday_cos",
    "forecast_is_weekend"
]
NEW_FEATURES = [
    "steel_ramp_rate",
    "apparent_energy_proxy",
    "reactive_ratio",
    "low_lagging_pf_flag"
]
TARGET = "target_next_60m_kWh"

# =====================================================================
# 5. Xuất ảnh 1: Ma trận 15 đặc trưng cơ sở (BASE)
# =====================================================================
print("Đang tính và vẽ Heatmap 15 đặc trưng cơ sở...")
base_cols = BASE_FEATURES + [TARGET]
corr_base = df_extended[base_cols].corr(method="pearson")

plt.figure(figsize=(14, 11), dpi=150)
sns.heatmap(
    corr_base,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1.0,
    vmax=1.0,
    cbar=True,
    linewidths=0.5,
    square=True,
    annot_kws={"size": 8}
)
plt.title("Correlation Matrix: 15 Base Features & Target (Train Set)", fontsize=14, pad=15)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()

base_img_path = OUTPUT_DIR / "steel_base_feature_correlation_heatmap.png"
plt.savefig(base_img_path, dpi=300)
plt.close()
print(f"-> Đã lưu: {base_img_path.resolve()}")

# =====================================================================
# 6. Xuất ảnh 2: Ma trận 19 đặc trưng mở rộng (EXTENDED)
# =====================================================================
print("Đang tính và vẽ Heatmap 19 đặc trưng mở rộng...")
extended_cols = BASE_FEATURES + NEW_FEATURES + [TARGET]
corr_extended = df_extended[extended_cols].corr(method="pearson")

plt.figure(figsize=(16, 13), dpi=150)
sns.heatmap(
    corr_extended,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1.0,
    vmax=1.0,
    cbar=True,
    linewidths=0.5,
    square=True,
    annot_kws={"size": 7.5}
)
plt.title("Correlation Matrix: 19 Extended Features & Target (Train Set)", fontsize=14, pad=15)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()

extended_img_path = OUTPUT_DIR / "steel_extended_feature_correlation_heatmap.png"
plt.savefig(extended_img_path, dpi=300)
plt.close()
print(f"-> Đã lưu: {extended_img_path.resolve()}")

print(f"\nHoàn tất! Cả 2 ảnh đã được ghi thẳng vào: {OUTPUT_DIR.resolve()}")