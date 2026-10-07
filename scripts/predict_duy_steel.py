"""Demo một dự báo thật từ artifact Duy; không hiển thị actual tương lai."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.steel.duy_inference import DuyForecaster
from src.steel.preprocessing import load_official_raw


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--issue-time", required=True)
    parser.add_argument("--history-csv", type=Path, help="CSV chỉ có observation_time, Usage_kWh")
    args = parser.parse_args()
    issue = pd.Timestamp(args.issue_time)
    if args.history_csv:
        history = pd.read_csv(args.history_csv, float_precision="round_trip")
    else:
        root = Path(__file__).resolve().parents[1]
        raw, _ = load_official_raw(root / "data/steel/raw/Steel_industry_data.csv")
        history = raw.loc[raw["observation_time"] <= issue, ["observation_time", "Usage_kWh"]].tail(673)
    forecaster = DuyForecaster(args.run_dir)
    print(json.dumps(forecaster.predict(history, issue), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
