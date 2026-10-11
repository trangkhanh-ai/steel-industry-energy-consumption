"""Nghiên cứu Gate 7 của Khánh: so sánh 15/19 feature trên validation, không mở test.

python -m scripts.run_feature_extension --output-dir reports/steel/local_gate7_01
Requires requirements-khanh.txt. Không ghi đè các lần chạy trước.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from threadpoolctl import threadpool_limits

from src.steel.feature_extension import add_extended_features, EXTENDED_FEATURES
from src.steel.preprocessing import BASE_FEATURES, TARGET, sha256_file

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = dict(objective="regression_l1", metric="mae", learning_rate=.03,
                  num_leaves=31, subsample=.8, colsample_bytree=.8,
                  verbosity=-1, seed=42, subsample_freq=1, num_threads=1)


def metrics(actual, predicted):
    return {"mae_kWh": float(mean_absolute_error(actual, predicted)),
            "rmse_kWh": float(root_mean_squared_error(actual, predicted))}


def draw_correlations(train, features, path):
    columns = [*features, TARGET]
    correlation = train[columns].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(13, 11))
    heatmap = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)), columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(columns)), columns, fontsize=8)
    for row in range(len(columns)):
        for col in range(len(columns)):
            ax.text(col, row, f"{correlation[row, col]:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title(f"TRAIN only — {len(features)} features; correlation does not prove causality")
    fig.colorbar(heatmap, ax=ax, shrink=.75)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(output):
    import lightgbm as lgb  # Optional dependency; core demo does not import it.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = ROOT / "data/steel/processed"
    files = {part: data / f"steel_next_60m_{part}.csv" for part in ("train", "validation")}
    # Read only train/validation; old pandas-window implementation stays separate from HGB.
    frames = {name: add_extended_features(pd.read_csv(file)) for name, file in files.items()}
    protocol = {"contribution": "Khánh: Gate 7, 07/10/2026; refactored for reproducibility",
                "input_sha256": {name: sha256_file(file) for name, file in files.items()},
                "feature_implementation": "legacy_pandas_windows_plus_khanh_4_v1",
                "lightgbm_version": lgb.__version__, "lightgbm_parameters": PARAMETERS,
                "max_rounds": 1000, "early_stopping_validation_rounds": 50,
                "ridge_alpha": 1.0, "test_opened": False, "calibration_opened": False,
                "primary_prediction_rule": "max(prediction, 0) for both methods",
                "sensor_availability_assumption": "Meter Usage, reactive energy and PF at t are available together",
                "selection": "Validation-only group comparison; not automatic replacement of deployed HGB"}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    train, validation = frames["train"], frames["validation"]
    results = []
    with threadpool_limits(limits=1):
        for name, features in (("base_15", BASE_FEATURES), ("extended_19", EXTENDED_FEATURES)):
            x_train, x_val = train[list(features)].to_numpy(), validation[list(features)].to_numpy()
            scaler = StandardScaler().fit(x_train)  # Fit on TRAIN only.
            ridge = Ridge(alpha=1.0).fit(scaler.transform(x_train), train[TARGET])
            ridge_prediction = ridge.predict(scaler.transform(x_val))
            booster = lgb.train(PARAMETERS, lgb.Dataset(x_train, label=train[TARGET]),
                                num_boost_round=1000,
                                valid_sets=[lgb.Dataset(x_val, label=validation[TARGET])],
                                callbacks=[lgb.early_stopping(50, verbose=False)])
            for model, prediction in (("ridge", ridge_prediction), ("lightgbm", booster.predict(x_val))):
                results.append({"feature_set": name, "model": model,
                                **metrics(validation[TARGET], np.maximum(prediction, 0)),
                                "unclipped": metrics(validation[TARGET], prediction),
                                "best_iteration": booster.best_iteration if model == "lightgbm" else None})
            draw_correlations(train, features, output / f"train_correlations_{name}.png")
    table = pd.DataFrame([{k: v for k, v in row.items() if k != "unclipped"} for row in results])
    table.to_csv(output / "validation_comparison.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([r["model"] + " / " + r["feature_set"] for r in results], [r["mae_kWh"] for r in results])
    ax.set_ylabel("Validation MAE (kWh)")
    ax.set_title("September validation; nonnegative predictions; test unopened")
    fig.tight_layout()
    fig.savefig(output / "validation_comparison.png", dpi=160)
    plt.close(fig)
    print(table.to_string(index=False))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args().output_dir)
