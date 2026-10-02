"""Kiểm bản clone/ZIP: nguồn code, model và một dự báo validation đã lưu."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.steel.duy_forecasting import features_at_issue
from src.steel.duy_inference import DuyForecaster
from src.steel.preprocessing import BASE_FEATURES, load_official_raw, sha256_file


def check_handoff(run_dir: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    forecaster = DuyForecaster(run_dir)
    manifest = forecaster.manifest
    state_path = run_dir / manifest["baseline_state_path"]
    if sha256_file(state_path) != manifest["baseline_state_sha256"]:
        raise ValueError("Baseline state checksum không khớp sau checkout")
    for relative, expected in manifest["code_sha256"].items():
        path = (root / relative.replace("\\", "/")).resolve()
        if not path.is_relative_to(root) or sha256_file(path) != expected:
            raise ValueError(f"Nguồn code khác lần train: {relative}")
    raw, audit = load_official_raw(root / "data/steel/raw/Steel_industry_data.csv")
    if audit["source_sha256"] != manifest["source_sha256"]:
        raise ValueError("Nguồn UCI không khớp model")
    issue = pd.Timestamp("2018-09-08 02:00")
    history = raw.loc[raw["observation_time"] <= issue, ["observation_time", "Usage_kWh"]].tail(673)
    validation = pd.read_csv(
        run_dir / "processed/steel_next_60m_validation.csv",
        parse_dates=["observation_time"], float_precision="round_trip",
    )
    offline = validation.loc[validation["observation_time"].eq(issue), list(BASE_FEATURES)]
    online = features_at_issue(history, issue)
    np.testing.assert_array_equal(offline.to_numpy(), online.to_numpy())
    prediction = forecaster.predict(history, issue)["predicted_next_60m_kWh"]
    saved = pd.read_csv(run_dir / "validation_predictions.csv", parse_dates=["observation_time"],
                        float_precision="round_trip")
    expected = saved.loc[saved["observation_time"].eq(issue),
                         f"pred_{manifest['selected_model']}_kWh"].iloc[0]
    np.testing.assert_allclose(prediction, expected, rtol=0, atol=1e-10)
    return {"passed": True, "model": manifest["selected_model"],
            "code_files_verified": len(manifest["code_sha256"]),
            "issue_time": str(issue), "feature_difference": 0.0,
            "prediction_difference_kWh": float(abs(prediction - expected))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(check_handoff(args.run_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
