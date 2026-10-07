"""Công cụ Khánh dùng sau khi model đã khóa: buffer và metric cảnh báo.

Ngưỡng/buffer có đơn vị kWh giờ tới. Không phải công suất hợp đồng,
phân vị có bảo đảm hay bằng chứng tiết kiệm điện.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, mean_absolute_error, precision_recall_fscore_support,
    root_mean_squared_error,
)


def checked_arrays(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or actual.shape != predicted.shape or not len(actual):
        raise ValueError("Cần hai mảng một chiều, cùng độ dài và không rỗng")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual/prediction phải hữu hạn")
    if (actual < 0).any() or (predicted < 0).any():
        raise ValueError("Điện năng phải không âm")
    return actual, predicted


def calibrate_upper_buffer(actual, predicted, quantile=0.90):
    """Phân vị thực nghiệm của lỗi dự báo thiếu trên CALIBRATION riêng."""
    actual, predicted = checked_arrays(actual, predicted)
    if not 0 < quantile < 1:
        raise ValueError("Quantile phải nằm giữa 0 và 1")
    return float(np.quantile(np.maximum(actual - predicted, 0), quantile))


def episodes(flags, times):
    """Các cụm mốc phát dự báo dương; không nối cụm qua khoảng thời gian thiếu."""
    flags = np.asarray(flags, dtype=bool)
    times = pd.DatetimeIndex(times)
    if len(flags) != len(times) or times.hasnans or not times.is_unique:
        raise ValueError("Flag và timestamp phải khớp, không trùng/thiếu")
    if not times.is_monotonic_increasing:
        raise ValueError("Timestamp phải tăng dần")
    groups, current = [], set()
    for index, positive in enumerate(flags):
        gap = index > 0 and times[index] - times[index - 1] != pd.Timedelta(minutes=15)
        if current and (not positive or gap):
            groups.append(current)
            current = set()
        if positive:
            current.add(index)
    if current:
        groups.append(current)
    return groups


def evaluate_alerts(actual, predicted, times, threshold, buffer):
    """Báo metric hồi quy, cảnh báo theo hàng và cụm chồng lấp tại forecast origin."""
    actual, predicted = checked_arrays(actual, predicted)
    if not np.isfinite(threshold) or threshold <= 0 or not np.isfinite(buffer) or buffer < 0:
        raise ValueError("Ngưỡng phải dương; buffer hữu hạn, không âm")
    upper = predicted + buffer
    true_flags, alerts = actual > threshold, upper > threshold
    tn, fp, fn, tp = confusion_matrix(true_flags, alerts, labels=[False, True]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_flags, alerts, average="binary", zero_division=0,
    )
    true_episodes, alert_episodes = episodes(true_flags, times), episodes(alerts, times)
    detected = sum(any(true & alert for alert in alert_episodes) for true in true_episodes)
    valid = sum(any(alert & true for true in true_episodes) for alert in alert_episodes)
    return {
        "mae_kWh": float(mean_absolute_error(actual, predicted)),
        "rmse_kWh": float(root_mean_squared_error(actual, predicted)),
        "rows": len(actual), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "false_discovery_rate": float(fp / (fp + tp)) if fp + tp else None,
        "empirical_upper_coverage": float(np.mean(actual <= upper)),
        "true_origin_episodes": len(true_episodes), "alert_origin_episodes": len(alert_episodes),
        "detected_true_episodes": detected, "valid_alert_episodes": valid,
        "overlap_episode_recall": detected / len(true_episodes) if true_episodes else None,
        "overlap_episode_precision": valid / len(alert_episodes) if alert_episodes else None,
        "lead_time_measured": False, "savings_measured": False,
        "episode_note": "Many-to-many overlap at forecast origins; not physical load events or 60-minute lead-time proof",
    }
