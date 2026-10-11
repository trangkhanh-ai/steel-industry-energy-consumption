"""Inspect a policy handoff without executing rules, changing data or enabling alerts."""
import hashlib
import json
import math
from pathlib import Path
import re

from .preprocessing import BASE_FEATURES

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = ROOT / "notebooks/artifacts/policy.json"


def inspect_policy(path=DEFAULT_POLICY, model=None, artifact_root=None):
    """Validate v1 identity and structure; never approve calibration or activate it.

    This handoff's artifact paths are relative to notebooks/, not artifacts/.
    """
    path = Path(path)
    root = Path(artifact_root) if artifact_root else path.parent.parent
    result = {"status": "missing", "alerts_enabled": False, "policy_sha256": None,
              "version": None, "target_model": None, "expected_model_sha256": None,
              "artifact_sha256": None, "loaded_model_sha256": getattr(model, "model_sha256", None),
              "issues": [], "activation": "disabled_pending_team_protocol"}

    def problem(code, detail):
        result["issues"].append({"code": code, "detail": detail})

    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        problem("policy_missing", "Chưa tìm thấy file policy bàn giao.")
        return result
    except OSError:
        result["status"] = "invalid"
        problem("policy_unreadable", "Không đọc được file policy bàn giao.")
        return result
    result["policy_sha256"] = hashlib.sha256(raw).hexdigest()
    try:
        def unique(pairs):
            obj = {}
            for key, value in pairs:
                if key in obj:
                    raise ValueError("Duplicate JSON key")
                obj[key] = value
            return obj
        def reject_constant(value):
            raise ValueError("Non-finite JSON number")
        data = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=unique,
                          parse_constant=reject_constant)
        meta, cal, logic = (data[key] for key in ("policy_metadata", "calibration_parameters", "decision_logic"))
        if not all(isinstance(part, dict) for part in (meta, cal, logic)):
            raise ValueError("Expected objects")
        if meta["policy_version"] != "1.0.0":
            raise ValueError("Unsupported version")
        for key in ("target_model", "model_artifact_path"):
            if not isinstance(meta[key], str) or not meta[key].strip():
                raise ValueError("Missing identity/path")
        if not isinstance(meta["model_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", meta["model_sha256"]):
            raise ValueError("Invalid hash")
        schema = meta["feature_schema"]
        if (not isinstance(schema, list) or not schema or not all(isinstance(x, str) for x in schema)
                or len(schema) != len(set(schema))):
            raise ValueError("Invalid feature schema")
        for key in ("threshold_T_value_kWh", "buffer_b_value_kWh"):
            v = cal[key]
            if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
                raise ValueError("Invalid threshold/buffer")
        if cal["threshold_T_value_kWh"] == 0:
            raise ValueError("Threshold must be positive")
        if cal["threshold_T_source"] != "train_target_p95_kWh" or cal["calibration_split"] != "October_2018":
            raise ValueError("Unsupported calibration provenance")
        if cal["buffer_b_formula"] != "max(0, Q90(y_true - y_pred))":
            raise ValueError("Unsupported buffer formula")
        if logic["trigger_condition"] != "(y_pred + buffer_b) > threshold_T":
            raise ValueError("Unsupported rule")
        episode = logic["episode_grouping"]
        if type(episode["max_gap_minutes"]) is not int or episode["max_gap_minutes"] != 15:
            raise ValueError("Unsupported episode gap")
        if type(episode["min_consecutive_steps"]) is not int or episode["min_consecutive_steps"] < 1:
            raise ValueError("Invalid episode length")
        if not isinstance(logic["fallback_policy"], dict):
            raise ValueError("Missing fallback specification")
        artifact = (root / meta["model_artifact_path"].replace("\\", "/")).resolve()
        if not artifact.is_relative_to(root.resolve()):
            raise ValueError("Artifact outside handoff root")
    except (UnicodeError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
        result["status"] = "invalid"
        problem("invalid_contract", "Policy sai cấu trúc, thiếu trường hoặc dùng quy tắc/phiên bản chưa hỗ trợ.")
        return result

    result.update(version=meta["policy_version"], target_model=meta["target_model"],
                  expected_model_sha256=meta["model_sha256"])
    try:
        result["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if result["artifact_sha256"] != meta["model_sha256"]:
            problem("artifact_hash_mismatch", "File model bàn giao không khớp SHA-256 trong policy.")
    except OSError:
        problem("artifact_unavailable", "Không đọc được file model mà policy tham chiếu.")
    if model is None:
        problem("model_unavailable", "Chưa nạp được model của phiên để đối chiếu policy.")
    else:
        if model.model_sha256 != meta["model_sha256"]:
            problem("model_mismatch", "Policy dành cho model khác với model đang chạy.")
        if schema != list(BASE_FEATURES):
            problem("schema_mismatch", "Thứ tự hoặc tên đặc trưng khác hợp đồng inference hiện tại.")
    codes = {x["code"] for x in result["issues"]}
    if codes & {"model_mismatch", "schema_mismatch"}:
        result["status"] = "incompatible"
    elif codes & {"artifact_hash_mismatch", "artifact_unavailable"}:
        result["status"] = "invalid"
    elif model is None:
        result["status"] = "model_unavailable"
    else:
        result["status"] = "verified_inactive"
    # Textual fallback/severity rules are never evaluated as code or activated.
    return result
