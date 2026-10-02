"""Bàn giao model AI của Duy cho demo Huy; chưa gắn chính sách cảnh báo."""

import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from .duy_forecasting import (
    FEATURE_IMPLEMENTATION, features_at_issue, predict_baseline, predict_energy,
)
from .preprocessing import BASE_FEATURES, FREQUENCY, TARGET, sha256_file


class DuyForecaster:
    def __init__(self, run_dir: Path):
        run_dir = Path(run_dir).resolve()
        self.manifest = json.loads((run_dir / "model_handoff.json").read_text(encoding="utf-8"))
        metadata = self.manifest
        if metadata["features"] != list(BASE_FEATURES) or metadata["target"] != TARGET:
            raise ValueError("Schema hoặc target không đúng hợp đồng")
        if metadata["feature_implementation"] != FEATURE_IMPLEMENTATION:
            raise ValueError("Model và cách tính feature không cùng phiên bản")
        if metadata["sklearn"] != sklearn.__version__:
            raise ValueError("Cần cùng phiên bản scikit-learn với lúc fit")
        def checked_path(path_key, hash_key):
            # Hỗ trợ cả manifest Windows cũ và đường dẫn POSIX mới.
            path = (run_dir / metadata[path_key].replace("\\", "/")).resolve()
            if not path.is_relative_to(run_dir):
                raise ValueError("Đường dẫn artifact phải nằm trong thư mục lần chạy")
            if sha256_file(path) != metadata[hash_key]:
                raise ValueError("Artifact checksum không khớp manifest")
            return path

        self.baseline_state = {}
        if metadata["kind"] == "baseline":
            if metadata["selected_model"] not in (
                "last_hour", "previous_day", "previous_week", "diurnal_train_mean"
            ):
                raise ValueError("Baseline chưa được hỗ trợ")
            state_path = checked_path("baseline_state_path", "baseline_state_sha256")
            self.baseline_state = json.loads(state_path.read_text(encoding="utf-8"))
            return
        if metadata["kind"] != "AI":
            raise ValueError("Loại model phải là AI hoặc baseline")
        model_path = checked_path("model_path", "model_sha256")
        # Chỉ nạp artifact do nhóm tạo và tin cậy. Hash xác nhận tính toàn vẹn,
        # không xác nhận file từ người lạ là an toàn.
        self.model = joblib.load(model_path)
        if list(self.model.feature_names_in_) != list(BASE_FEATURES):
            raise ValueError("Model không nhận đúng 15 feature")

    def predict(self, history: pd.DataFrame, issue_time) -> dict:
        issue_time = pd.Timestamp(issue_time)
        with threadpool_limits(limits=1):
            if self.manifest["kind"] == "baseline":
                prediction = predict_baseline(
                    self.manifest["selected_model"], history, issue_time, self.baseline_state
                )
            else:
                features = features_at_issue(history, issue_time)
                prediction = float(predict_energy(self.model, features)[0])
        return {
            "issue_time": str(issue_time),
            "forecast_start": str(issue_time + FREQUENCY),
            "forecast_end": str(issue_time + 4 * FREQUENCY),
            "predicted_next_60m_kWh": prediction,
            "model_name": self.manifest["selected_model"],
            "model_sha256": self.manifest["model_sha256"],
            "feature_implementation": FEATURE_IMPLEMENTATION,
            "status": "forecast_only_no_alert_policy",
        }
