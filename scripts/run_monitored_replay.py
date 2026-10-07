"""Exercise local logs and delayed actuals on real September observations."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.steel.monitoring import ForecastMonitor
from src.steel.inference import DEFAULT_MODEL_SUBDIR
from src.steel.preprocessing import load_official_raw, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def run(output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory")
    raw, audit = load_official_raw(ROOT / "data/steel/raw/Steel_industry_data.csv")
    model_dir = ROOT / DEFAULT_MODEL_SUBDIR
    validation = pd.read_csv(model_dir / "processed/steel_next_60m_validation.csv", parse_dates=["observation_time"], float_precision="round_trip")
    source = raw.loc[raw.observation_time.between(validation.observation_time.iloc[0]-pd.Timedelta(days=7), "2018-09-30 23:45"), ["observation_time", "Usage_kWh"]].reset_index(drop=True)
    origins = set(validation.observation_time)
    output.mkdir(parents=True, exist_ok=True)
    snapshots = []
    with ForecastMonitor(model_dir, output / "monitor.sqlite3") as monitor:
        if monitor.model is None:
            raise RuntimeError(monitor.model_error)
        for pos in range(672, len(source)):
            t = source.observation_time.iloc[pos]
            # Only measurements already seen by the replay clock are exposed.
            observed = source.iloc[max(0,pos-672):pos+1]
            monitor.observe(observed, t)
            if t in origins:
                result = monitor.forecast(observed, t)
                if result["status"] != "forecast_ready":
                    raise RuntimeError(result)
            if t.hour == 23 and t.minute == 45:
                snapshots.append(monitor.snapshot(t))
        last_time = source.observation_time.iloc[-1]
        final = monitor.snapshot(last_time)
        stored = pd.read_sql_query("SELECT * FROM forecasts ORDER BY issue_time", monitor.db)
        assert len(stored) == len(validation) and stored.actual_kWh.notna().all()
        assert (pd.to_datetime(stored.evaluated_as_of) >= pd.to_datetime(stored.forecast_end)).all()
        np.testing.assert_allclose(stored.actual_kWh, validation.target_next_60m_kWh, atol=1e-10)
        reference = pd.read_csv(model_dir / "validation_predictions.csv", float_precision="round_trip")
        column = f"pred_{monitor.model.configuration}_kWh"
        np.testing.assert_allclose(stored.predicted_kWh, reference[column].to_numpy(), rtol=0, atol=1e-10)
        stored.to_csv(output / "scored_forecasts.csv", index=False)
    (output / "daily_monitor_snapshots.json").write_text(json.dumps(snapshots, indent=2), encoding="utf-8")
    final.update(source_sha256=audit["source_sha256"], evaluation="chronological validation replay; actual recorded only at forecast_end or later",
                 code_sha256={p:sha256_file(ROOT/p) for p in ("src/steel/monitoring.py", "scripts/run_monitored_replay.py")})
    (output / "run_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/steel/modeling/monitoring_v1")
    run(parser.parse_args().output_dir.resolve())
