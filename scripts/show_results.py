"""Xem đáp án cố định và kiểm checksum, không train/notebook."""
import json
from pathlib import Path
from src.steel.frozen_results import read_results

if __name__ == '__main__':
    summary = read_results(Path(__file__).resolve().parents[1])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
