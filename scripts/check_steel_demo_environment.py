"""Read-only demo readiness; requirement ranges/includes are checked explicitly."""
import argparse
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def check_requirements(path, version_lookup=version, stack=()):
    from packaging.requirements import Requirement
    path = Path(path).resolve()
    if path in stack:
        raise ValueError(f"Recursive requirements include: {path.name}")
    errors = []
    for original in path.read_text(encoding="utf-8-sig").splitlines():
        line = original.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith(("-r ", "--requirement ")):
            errors.extend(check_requirements(path.parent / line.split(maxsplit=1)[1], version_lookup, (*stack, path)))
            continue
        requirement = Requirement(line)
        if requirement.url or requirement.extras:
            raise ValueError(f"Unsupported requirement: {line}")
        if requirement.marker and not requirement.marker.evaluate():
            continue
        try:
            actual = version_lookup(requirement.name)
        except PackageNotFoundError:
            errors.append(f"{requirement.name}: not installed")
        else:
            if actual not in requirement.specifier:
                errors.append(f"{requirement.name}: expected {requirement.specifier}, found {actual}")
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, default=ROOT / "requirements-demo.txt")
    args = parser.parse_args(argv)
    try:
        errors = check_requirements(args.requirements)
        if errors:
            raise ValueError("; ".join(errors))
        from src.steel.inference import DEFAULT_MODEL_SUBDIR, SteelForecaster
        from src.steel.preprocessing import load_official_raw
        from src.steel.policy import inspect_policy
        raw, _ = load_official_raw(ROOT / "data/steel/raw/Steel_industry_data.csv")
        model = SteelForecaster(ROOT / DEFAULT_MODEL_SUBDIR)
        t = raw.observation_time.searchsorted("2018-09-01")
        model.predict(raw.iloc[t-672:t+1][["observation_time", "Usage_kWh"]], raw.observation_time.iloc[t])
        for file in ("index.html", "app.js", "style.css"):
            if not (ROOT / "web/steel_demo" / file).is_file():
                raise FileNotFoundError(file)
        policy = inspect_policy(model=model)
    except Exception as error:
        print(f"NOT READY: {type(error).__name__}: {error}")
        print(f"Check dependencies in {args.requirements.name}; use a separate environment for each workflow.")
        return 1
    print(f"READY FOR REPLAY: Python {sys.version.split()[0]}, {args.requirements.name}, original data/model/UI verified.")
    print(f"POLICY: {policy['status']}; alerts disabled, calibration not approved by this check.")
    for issue in policy["issues"]:
        print(f"  {issue['code']}: {issue['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
