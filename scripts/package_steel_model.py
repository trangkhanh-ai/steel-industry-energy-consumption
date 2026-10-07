"""Refit the already chosen HGB configuration with shared batch/online features.

No parameter search. Preserves the earlier runs and original data artifacts.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from scripts.train_steel_models import ROOT, read_verified_split, TIMESTAMPS
from src.steel.inference import FEATURE_CONTRACT, features_from_valid_history
from src.steel.modeling import make_candidates, regression_metrics
from src.steel.preprocessing import BASE_FEATURES, SPLIT_STARTS, TARGET, load_official_raw, build_supervised_rows, sha256_file


def run(output, original_run):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new model directory: {output}")
    previous = json.loads((original_run / "run_manifest.json").read_text(encoding="utf-8"))
    frozen = json.loads((original_run / "frozen_config.json").read_text(encoding="utf-8"))
    selected = previous["selected_configuration"]
    model = make_candidates()["hist_gradient_boosting"].set_params(**frozen["parameters"][selected])
    raw, audit = load_official_raw(ROOT / "data/steel/raw/Steel_industry_data.csv")
    history = raw.loc[raw.observation_time.lt(SPLIT_STARTS["calibration"])].reset_index(drop=True)
    old_rows = build_supervised_rows(history)
    train = read_verified_split(ROOT, "train", old_rows)
    val = read_verified_split(ROOT, "validation", old_rows)
    features = features_from_valid_history(history.observation_time, history.Usage_kWh.to_numpy())
    features.index = pd.DatetimeIndex(history.observation_time)
    x_train = features.loc[train.observation_time].reset_index(drop=True)
    x_val = features.loc[val.observation_time].reset_index(drop=True)
    assert x_train.notna().all().all() and x_val.notna().all().all()
    # Small numerical changes only: same definitions, target, periods and columns.
    np.testing.assert_allclose(x_train, train[list(BASE_FEATURES)], atol=1e-7, rtol=1e-10)
    np.testing.assert_allclose(x_val, val[list(BASE_FEATURES)], atol=1e-7, rtol=1e-10)
    with threadpool_limits(limits=1):
        model.fit(x_train, train[TARGET])
        pred = model.predict(x_val)
    old = pd.read_csv(original_run / "search_predictions.csv")
    np.testing.assert_allclose(old[TARGET], val[TARGET])
    high = val[TARGET] > train[TARGET].quantile(.95)
    metrics = regression_metrics(val[TARGET].to_numpy(), pred)
    metrics.update({"high_"+k: v for k, v in regression_metrics(val.loc[high, TARGET].to_numpy(), pred[high]).items()})
    metrics["high_underpredictions"] = int((pred[high] < val.loc[high, TARGET]).sum())
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "selected_model.joblib")
    (output / "frozen_config.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    val[TIMESTAMPS + [TARGET]].assign(**{selected: pred, "previous_selected_prediction": old[selected].to_numpy()}).to_csv(output / "search_predictions.csv", index=False)
    pd.concat([val[TIMESTAMPS], x_val], axis=1).to_csv(output / "validation_features.csv", index=False)
    manifest = {"finished_utc": datetime.now(timezone.utc).isoformat(), "selected_configuration": selected,
                "feature_contract": FEATURE_CONTRACT, "fits": 1, "tuning": "none; refit previously selected parameters",
                "metrics": metrics, "model_sha256": sha256_file(output / "selected_model.joblib"),
                "source_sha256": audit["source_sha256"], "train_rows": len(train), "validation_rows": len(val),
                "max_prediction_change_from_original_kWh": float(np.abs(pred-old[selected]).max()),
                "previous_validation_mae_kWh": float(np.abs(old[selected]-val[TARGET]).mean()),
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
                "hashes": {p: sha256_file(ROOT / p) for p in ("data/steel/processed/steel_next_60m_train.csv", "data/steel/processed/steel_next_60m_validation.csv", "src/steel/inference.py", "scripts/package_steel_model.py")},
                "prior_run_manifest_sha256": sha256_file(original_run / "run_manifest.json"),
                "scope": "Same model parameters, real data and splits; shared numerical feature implementation; validation only, not independent final evaluation"}
    (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/steel/modeling/inference_v1")
    parser.add_argument("--original-run", type=Path, default=ROOT / "reports/steel/modeling/tuning_v1")
    args = parser.parse_args()
    run(args.output_dir.resolve(), args.original_run.resolve())
