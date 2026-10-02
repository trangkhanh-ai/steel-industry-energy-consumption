"""Print a forecast as JSON from a real, chronological two-column history CSV."""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.steel.inference import DEFAULT_MODEL_SUBDIR, SteelForecaster

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-csv", required=True, type=Path)
    parser.add_argument("--issue-time", required=True, help="Dataset timestamp, e.g. 2018-09-01T00:00:00")
    parser.add_argument("--model-dir", type=Path, default=ROOT / DEFAULT_MODEL_SUBDIR)
    args = parser.parse_args()
    result = SteelForecaster(args.model_dir).predict(
        pd.read_csv(args.history_csv, float_precision="round_trip"), args.issue_time
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
