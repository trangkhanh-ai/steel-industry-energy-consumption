"""Cảnh báo nghiên cứu: model đã khóa + buffer từ calibration, không điều khiển máy."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocessing import sha256_file
# Backward compatible, read-only inspection of Khánh's historical handoff.
from .policy_handoff import DEFAULT_POLICY, ROOT, inspect_policy


class AlertPolicy:
    def __init__(self, path, model_metadata):
        path = Path(path)
        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Policy có khóa JSON trùng")
                result[key] = value
            return result

        def reject_constant(value):
            raise ValueError("Policy có số không hữu hạn")

        self.data = json.loads(path.read_text(encoding="utf-8-sig"),
                               object_pairs_hook=unique_keys, parse_constant=reject_constant)
        if not isinstance(self.data, dict):
            raise ValueError("Policy phải là JSON object")
        self.sha256 = sha256_file(path)
        for key in ("model_sha256", "source_sha256", "feature_implementation", "target", "features"):
            if self.data.get(key) != model_metadata.get(key):
                raise ValueError(f"Policy không khớp model: {key}")
        if self.data.get("target_unit") != "kWh":
            raise ValueError("Policy phải dùng tổng kWh, không phải ngưỡng kW hợp đồng")
        for key in ("threshold_kWh", "upper_buffer_kWh", "reference_validation_mae_kWh"):
            if type(self.data.get(key)) not in (int, float):
                raise ValueError(f"Tham số policy phải là số: {key}")
        self.threshold = float(self.data["threshold_kWh"])
        self.buffer = float(self.data["upper_buffer_kWh"])
        self.reference_mae = float(self.data["reference_validation_mae_kWh"])
        if not np.isfinite([self.threshold, self.buffer, self.reference_mae]).all():
            raise ValueError("Tham số policy phải hữu hạn")
        if self.threshold <= 0 or self.buffer < 0 or self.reference_mae <= 0:
            raise ValueError("Ngưỡng/MAE phải dương, buffer không âm")
        self.valid_from = pd.Timestamp(self.data["valid_from"])
        if pd.isna(self.valid_from) or self.valid_from.tzinfo is not None:
            raise ValueError("valid_from phải là timestamp nguồn không kèm múi giờ")
        period = self.data.get("calibration_period")
        if isinstance(period, dict):
            end = pd.Timestamp(period["end"])
            if pd.isna(end) or end.tzinfo is not None or self.valid_from <= end:
                raise ValueError("Policy chỉ có hiệu lực sau khi calibration kết thúc")

    def information(self):
        return {"status": "configured", "policy_sha256": self.sha256,
                "policy_version": self.data.get("policy_version", "steel_alert_v1"),
                "threshold_kWh": self.threshold, "upper_buffer_kWh": self.buffer,
                "valid_from": self.valid_from.isoformat(),
                "note": "Ngưỡng nghiên cứu; buffer thực nghiệm không phải P90 có bảo đảm."}

    def apply(self, predicted, issue_time):
        t = pd.Timestamp(issue_time)
        if pd.isna(t) or t.tzinfo is not None or t < self.valid_from:
            raise ValueError("Chỉ dùng policy sau giai đoạn calibration")
        if not np.isfinite(predicted) or predicted < 0:
            raise ValueError("Không ra quyết định từ dự báo lỗi")
        upper = float(predicted) + self.buffer
        alert = bool(upper > self.threshold)
        return {**self.information(), "status": "inspection_recommended" if alert else "below_research_threshold",
                "alert": alert, "upper_proxy_kWh": upper,
                "recommendation": "Kiểm tra đồng hồ và lịch vận hành; người phụ trách xác nhận."
                if alert else "Tiếp tục theo dõi; dưới ngưỡng nghiên cứu không xác nhận an toàn.",
                "automatic_equipment_action": False}

    def usage_watch(self, usage):
        """PSI train vs 7 ngày quan sát: tín hiệu rà soát heuristic, không chẩn đoán drift."""
        rows = int(self.data.get("monitor_window_rows", 672))
        if len(usage) < rows or "reference_usage_bins" not in self.data:
            return {"status": "insufficient_reference_or_history", "psi": None}
        values = np.asarray(usage, dtype=float)[-rows:]
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Không tính PSI từ số đo lỗi")
        edges = np.asarray(self.data["reference_usage_bins"], dtype=float)
        reference = np.asarray(self.data["reference_usage_probabilities"], dtype=float)
        if (len(reference) != len(edges) + 1 or not np.isfinite(edges).all()
                or not (np.diff(edges) > 0).all() or not np.isfinite(reference).all()
                or (reference <= 0).any() or not np.isclose(reference.sum(), 1)):
            raise ValueError("Phân bố tham chiếu PSI không hợp lệ")
        counts, _ = np.histogram(values, bins=np.r_[-np.inf, edges, np.inf])
        current = (counts + 0.5) / (len(values) + 0.5 * len(counts))
        psi = float(np.sum((current - reference) * np.log(current / reference)))
        threshold = float(self.data.get("psi_review_threshold", 0.2))
        return {"status": "needs_review" if psi > threshold else "no_review_trigger",
                "psi": psi, "rows": rows, "threshold": threshold,
                "note": "PSI/0,2 là heuristic; thay đổi mùa/ca cũng có thể gây tín hiệu."}


def performance_watch(scored, reference_mae, week_rows=672, factor=1.2):
    """Hai cửa sổ tuần liên tục đều xấu hơn reference: yêu cầu người kiểm tra."""
    required = 2 * week_rows
    result = {"status": "insufficient_labels", "required_labels": required,
              "available_labels": len(scored), "automatic_retraining": False,
              "note": "Heuristic rà soát, không phải kiểm định thống kê drift."}
    if len(scored) < required:
        return result
    recent = scored.sort_values("issue_time").tail(required)
    times = pd.DatetimeIndex(pd.to_datetime(recent.issue_time))
    if not times.is_unique or not (times[1:] - times[:-1] == pd.Timedelta(minutes=15)).all():
        return {**result, "status": "non_contiguous_labels"}
    errors = (recent.predicted_kWh - recent.actual_kWh).abs().to_numpy()
    weekly = [float(errors[:week_rows].mean()), float(errors[week_rows:].mean())]
    limit = float(reference_mae) * factor
    return {**result, "status": "needs_review" if all(x > limit for x in weekly) else "no_review_trigger",
            "weekly_mae_kWh": weekly, "limit_mae_kWh": limit,
            "recommendation": "Kiểm dữ liệu/ca sản xuất, phân tích residual; đánh giá model mới trước khi người phụ trách duyệt."}
