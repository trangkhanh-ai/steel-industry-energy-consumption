"""Read-only startup checks for the local demo; never installs or trains anything."""
from importlib.metadata import version
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    try:
        for line in (ROOT / "requirements.txt").read_text().splitlines():
            if "==" in line and not line.lstrip().startswith("#"):
                package, expected = line.strip().split("==")
                actual = version(package)
                if actual != expected:
                    raise ValueError(f"{package}: expected {expected}, found {actual}")
        from src.steel.inference import SteelForecaster
        from src.steel.preprocessing import load_official_raw
        raw, _ = load_official_raw(ROOT / "data/steel/raw/Steel_industry_data.csv")
        model = SteelForecaster(ROOT / "reports/steel/modeling/inference_v1")
        t = raw.observation_time.searchsorted("2018-09-01")
        model.predict(raw.iloc[t-672:t+1][["observation_time", "Usage_kWh"]], raw.observation_time.iloc[t])
        for file in ("index.html", "app.js", "style.css"):
            if not (ROOT / "web/steel_demo" / file).is_file():
                raise FileNotFoundError(file)
    except Exception as error:
        print(f"NOT READY: {type(error).__name__}: {error}")
        print("Install requirements.txt in the active environment; see README to recreate the local model if absent.")
        return 1
    print(f"READY: Python {sys.version.split()[0]}, pinned libraries, original data, model and UI files verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
