"""Bốn feature nghiên cứu của Khánh, dùng số đo đã có tại t.

Chỉ dành cho nghiên cứu LightGBM/Ridge; không thay schema HGB đang chạy demo.
P > 5 là mặt nạ chống chia cho số nhỏ, không xác nhận máy đang vận hành.
"""
import numpy as np

from .preprocessing import BASE_FEATURES

NEW_FEATURES = (
    "steel_ramp_rate", "apparent_energy_proxy", "reactive_ratio", "low_lagging_pf_flag",
)
EXTENDED_FEATURES = tuple(BASE_FEATURES) + NEW_FEATURES
SENSOR_COLUMNS = ("known_lagging_reactive_kVarh", "known_lagging_power_factor")


def add_extended_features(frame):
    """Các cột known_* phải là số đo tại t, đã hoàn tất khi phát dự báo."""
    result = frame.copy()
    required = ["usage_lag_0", "usage_lag_1", *SENSOR_COLUMNS]
    values = result[required].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Feature đầu vào phải hữu hạn và không âm")
    if (result[SENSOR_COLUMNS[1]] > 100).any():
        raise ValueError("Power factor phải có đơn vị %, trong khoảng 0–100")
    active = result["usage_lag_0"]
    reactive = result[SENSOR_COLUMNS[0]]
    result["steel_ramp_rate"] = active - result["usage_lag_1"]  # ΔkWh / bước 15 phút
    # Proxy từ điện năng tích phân; không phải công suất biểu kiến tức thời.
    result["apparent_energy_proxy"] = np.hypot(active, reactive)
    result["reactive_ratio"] = np.divide(
        reactive, active, out=np.zeros(len(result)), where=active > 5.0,
    )
    result["low_lagging_pf_flag"] = (result[SENSOR_COLUMNS[1]] < 90.0).astype(int)
    return result
