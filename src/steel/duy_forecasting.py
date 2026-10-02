"""Phần Duy: feature tại t, baseline và model dự báo tổng kWh giờ tiếp theo.

Batch training và dự báo từng hàng dùng CÙNG hàm tính rolling. Module này
không thay cách tính của các artifact cũ trong src.steel.preprocessing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .preprocessing import BASE_FEATURES, FREQUENCY, LONGEST_LAG_STEPS, TARGET


FEATURE_IMPLEMENTATION = "duy_numpy_windows_v1"


def validate_history(history: pd.DataFrame) -> pd.DataFrame:
    """Kiểm dữ liệu đầu vào, không tự điền số hoặc sắp xếp lại lịch sử lỗi."""
    required = ["observation_time", "Usage_kWh"]
    if not history.columns.is_unique or not set(required).issubset(history.columns):
        raise ValueError("Cần cột observation_time và Usage_kWh, không trùng tên cột")
    times = pd.to_datetime(history["observation_time"], errors="raise")
    usage = pd.to_numeric(history["Usage_kWh"], errors="raise").to_numpy(dtype=float)
    if len(times) == 0 or times.isna().any() or not times.is_unique:
        raise ValueError("Cần timestamp hợp lệ, không trùng")
    if not times.diff().dropna().eq(FREQUENCY).all():
        raise ValueError("Cần bản đo liên tục, tăng dần, cách nhau 15 phút")
    if not times.eq(times.dt.floor("15min")).all():
        raise ValueError("Timestamp phải đúng mốc 00/15/30/45 phút")
    if times.dt.tz is not None:
        raise ValueError("Dữ liệu nguồn dùng timestamp không timezone")
    if not np.isfinite(usage).all() or (usage < 0).any():
        raise ValueError("Usage_kWh phải hữu hạn và không âm")
    return pd.DataFrame({"observation_time": times.to_numpy(), "Usage_kWh": usage})


def forecast_features(history: pd.DataFrame) -> pd.DataFrame:
    """Chỉ đọc bản đo đã có tới t; hàng thiếu lịch sử giữ NaN cho manifest."""
    clean = validate_history(history)
    times = clean["observation_time"]
    usage = clean["Usage_kWh"].to_numpy()

    frame = pd.DataFrame(index=range(len(history)))
    series = pd.Series(usage)
    for lag in (0, 1, 4, 96, LONGEST_LAG_STEPS):
        frame[f"usage_lag_{lag}"] = series.shift(lag)

    # Mỗi hàng được tính trực tiếp từ cửa sổ của nó. Cùng dữ liệu của cửa sổ
    # cho cùng kết quả số học khi chạy toàn chuỗi hoặc chỉ 673 bản đo gần nhất.
    specifications = (
        ("usage_sum_last_1h", 4, "sum"),
        ("usage_mean_last_4h", 16, "mean"),
        ("usage_std_last_4h", 16, "std"),
        ("usage_mean_last_24h", 96, "mean"),
        ("usage_max_last_24h", 96, "max"),
    )
    for name, window, operation in specifications:
        values = np.full(len(usage), np.nan)
        if len(usage) >= window:
            windows = sliding_window_view(usage, window)
            values[window - 1:] = getattr(windows, operation)(axis=1)
        frame[name] = values

    start = times.reset_index(drop=True) + FREQUENCY
    hour = start.dt.hour + start.dt.minute / 60
    weekday = start.dt.dayofweek
    frame["forecast_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["forecast_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    frame["forecast_weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    frame["forecast_weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    frame["forecast_is_weekend"] = (weekday >= 5).astype(int)
    return frame.loc[:, list(BASE_FEATURES)]


def features_at_issue(history: pd.DataFrame, issue_time) -> pd.DataFrame:
    """Hợp đồng giao Huy: chỉ nhận lịch sử đã tới t, cần ít nhất 673 hàng."""
    issue_time = pd.Timestamp(issue_time)
    # Kiểm cả đầu vào nhưng chỉ tính rolling trên 673 bản đo cần thiết.
    clean = validate_history(history)
    if len(clean) < LONGEST_LAG_STEPS + 1 or clean["observation_time"].iloc[-1] != issue_time:
        raise ValueError("Cần >=673 bản đo, bản cuối phải đúng issue_time")
    features = forecast_features(clean.tail(LONGEST_LAG_STEPS + 1))
    return features.tail(1).reset_index(drop=True)


def fixed_models() -> dict:
    """Bốn cấu hình nhỏ chốt trước khi chạy; không search trên test."""
    return {
        "ridge_alpha_1": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "ridge_alpha_100": make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
        "hgb_15_leaves": HistGradientBoostingRegressor(
            max_iter=180, max_leaf_nodes=15, learning_rate=0.08,
            l2_regularization=1.0, early_stopping=False, random_state=42,
        ),
        "hgb_31_leaves": HistGradientBoostingRegressor(
            max_iter=180, max_leaf_nodes=31, learning_rate=0.08,
            l2_regularization=1.0, early_stopping=False, random_state=42,
        ),
    }


def baseline_predictions(raw: pd.DataFrame, train: pd.DataFrame, part: pd.DataFrame) -> dict:
    """Persistence, ngày/tuần trước, trung bình lịch chỉ học từ TRAIN.

    Ngày trước đối chiếu đúng cửa sổ TƯƠNG LAI: t-95..t-92, không phải
    t-99..t-96. Diurnal dùng giờ nguyên từ timestamp, không làm tròn sin/cos.
    """
    energy = raw.set_index("observation_time")["Usage_kWh"]
    origins = pd.DatetimeIndex(part["observation_time"])
    result = {"last_hour": part["usage_sum_last_1h"].to_numpy()}
    for name, lag in (("previous_day", 96), ("previous_week", 672)):
        measurements = [energy.reindex(origins - (lag - step) * FREQUENCY).to_numpy()
                        for step in range(1, 5)]
        result[name] = np.sum(measurements, axis=0)
    state = fit_diurnal_state(train)
    lookup = {(row["weekday"], row["hour"]): row["mean_kWh"] for row in state["slots"]}
    slots = list(zip(part["forecast_start"].dt.dayofweek, part["forecast_start"].dt.hour))
    result["diurnal_train_mean"] = np.array([lookup.get(slot, state["global_mean_kWh"]) for slot in slots])
    if not all(np.isfinite(values).all() for values in result.values()):
        raise ValueError("Baseline thiếu bản đo lịch sử hoặc giá trị không hợp lệ")
    return result


def fit_diurnal_state(train: pd.DataFrame) -> dict:
    """Lưu trung bình theo 168 ô lịch, chỉ học từ TRAIN."""
    start = train["forecast_start"].dt
    lookup = train.groupby([start.dayofweek, start.hour])[TARGET].mean()
    return {"global_mean_kWh": float(train[TARGET].mean()), "slots": [
        {"weekday": int(day), "hour": int(hour), "mean_kWh": float(value)}
        for (day, hour), value in lookup.items()
    ]}


def predict_baseline(name: str, history: pd.DataFrame, issue_time, state: dict) -> float:
    """Một dự báo baseline với cùng hợp đồng lịch sử như model AI."""
    features = features_at_issue(history, issue_time)
    if name == "last_hour":
        return float(features["usage_sum_last_1h"].iloc[0])
    if name in ("previous_day", "previous_week"):
        lag = 96 if name == "previous_day" else 672
        usage = validate_history(history)["Usage_kWh"].to_numpy()
        return float(np.sum(usage[len(usage) - lag:len(usage) - lag + 4]))
    if name == "diurnal_train_mean":
        start = pd.Timestamp(issue_time) + FREQUENCY
        lookup = {(row["weekday"], row["hour"]): row["mean_kWh"] for row in state["slots"]}
        return float(lookup.get((start.dayofweek, start.hour), state["global_mean_kWh"]))
    raise ValueError(f"Baseline chưa được hỗ trợ: {name}")


def predict_energy(model, features: pd.DataFrame) -> np.ndarray:
    """Một hậu xử lý duy nhất cho train, validation và replay: chặn kWh âm."""
    if not features.columns.is_unique or not set(BASE_FEATURES).issubset(features.columns):
        raise ValueError("Model cần 15 feature, không trùng tên cột")
    matrix = features.loc[:, list(BASE_FEATURES)]
    if matrix.empty or not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError("Model cần 15 feature hữu hạn theo đúng thứ tự")
    values = np.asarray(model.predict(matrix), dtype=float)
    if values.shape != (len(matrix),) or not np.isfinite(values).all():
        raise ValueError("Model phải trả một dự báo hữu hạn cho mỗi hàng")
    return np.maximum(values, 0)
