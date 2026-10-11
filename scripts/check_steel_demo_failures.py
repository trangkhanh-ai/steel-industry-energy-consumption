"""Record explicit failure behavior on copies of real replay observations.

This check never edits the source CSV, model, or policy. Failure fixtures are
clearly named, isolated in temporary sessions, and are not model scores.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from uuid import uuid4

import pandas as pd

from src.steel.demo import ReplayDemo
from src.steel.inference import DEFAULT_MODEL_SUBDIR

ROOT = Path(__file__).resolve().parents[1]


def decision_for(state, action):
    return {"session": state["session"], "clock": state["clock"],
            "request_id": state["current_forecast"]["request_id"],
            "operator": "QA local; dữ liệu kiểm thử",
            "action": action, "note": "Kiểm tra đường lỗi; không điều khiển thiết bị.",
            "token": uuid4().hex}


def check(model_dir, policy_path, output):
    if output.exists():
        raise FileExistsError("Use a new failure-check output file")
    if not (model_dir / "model_handoff.json").exists():
        raise FileNotFoundError("The real model handoff is required for this check")
    scenarios = []
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        cases = ("missing_model", "missing_interval", "negative_measurement", "invalid_policy")
        for name in cases:
            selected_model = temporary / "missing_model" if name == "missing_model" else model_dir
            selected_policy = policy_path
            if name == "invalid_policy":
                selected_policy = temporary / "invalid_policy.json"
                selected_policy.write_text('{"schema": "invalid-test-fixture"}', encoding="utf-8")
            demo = ReplayDemo(ROOT, model_dir=selected_model,
                              sessions_dir=temporary / name, policy_path=selected_policy)
            try:
                origin = demo.START + pd.Timedelta(hours=8)
                if name == "missing_interval":
                    demo.history = demo.history.loc[
                        demo.history.observation_time.ne(origin-pd.Timedelta(minutes=15))].copy()
                if name == "negative_measurement":
                    demo.history = demo.history.copy()
                    demo.history.loc[demo.history.observation_time.eq(origin), "Usage_kWh"] = -1.0
                state = demo.dispatch("reset", origin.isoformat())
                response = state["current_forecast"]
                assert all(pd.Timestamp(row["time"]) <= origin for row in state["history"])
                if name == "invalid_policy":
                    assert response["status"] == "forecast_ready"
                    assert response["prediction"] is not None
                    decision = response["decision"]
                    assert decision["alert"] is None
                    assert decision["status"] == "policy_unavailable"
                else:
                    assert response["status"] in ("model_unavailable", "invalid_history")
                    assert response["prediction"] is None
                    assert not state["forecasts"]
                    try:
                        demo.dispatch("decision", decision=decision_for(state, "acknowledge"))
                    except ValueError:
                        acknowledge_blocked = True
                    else:
                        raise AssertionError("Invalid forecasts must not be acknowledged")
                saved = demo.dispatch("decision", decision=decision_for(state, "inspect"))
                assert len(saved["decisions"]) == 1
                context = saved["decisions"][0]["context"]
                assert context["issue_time"] == state["clock"]
                assert "actual" not in context and "actual_kWh" not in context
                scenarios.append({"scenario": name, "status": response["status"],
                                  "prediction_displayed": response["prediction"] is not None,
                                  "decision": response.get("decision"),
                                  "acknowledge_blocked": acknowledge_blocked if name != "invalid_policy" else None,
                                  "inspection_log_saved": True,
                                  "no_future_payload": True})
            finally:
                demo.close()
    summary = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
               "model_dir": str(model_dir), "policy_path": str(policy_path) if policy_path else None,
               "scenarios": scenarios,
               "scope": "Local controller failures on isolated test fixtures; not sensor accuracy or factory safety certification."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    # ASCII console output also works with Windows' legacy terminal encoding.
    print(json.dumps(summary, ensure_ascii=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / DEFAULT_MODEL_SUBDIR)
    parser.add_argument("--policy-path", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    check(args.model_dir.resolve(), args.policy_path.resolve() if args.policy_path else None,
          args.output.resolve())
