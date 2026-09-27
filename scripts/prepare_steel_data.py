"""Create validated Steel forecasting splits and train-only EDA figures.

Run from the repository root with: python -m scripts.prepare_steel_data
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.steel.preprocessing import (
    SPLIT_STARTS,
    TARGET,
    build_row_manifest,
    build_supervised_rows,
    load_official_raw,
    profile_training_columns,
    split_supervised_rows,
    write_processed_data,
)


def write_train_eda(raw, train, figure_dir: Path) -> dict:
    """Make descriptive figures and statistics from training dates only."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    train_raw = raw.loc[raw["observation_time"].lt(SPLIT_STARTS["validation"])].copy()

    first_week = train_raw.loc[
        train_raw["observation_time"].lt(SPLIT_STARTS["train"] + np.timedelta64(7, "D"))
    ]
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(first_week["observation_time"], first_week["Usage_kWh"], linewidth=0.75)
    ax.set(title="Steel electricity usage — first training week", ylabel="kWh per record", xlabel="Time")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figure_dir / "steel_train_first_week.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].hist(train_raw["Usage_kWh"], bins=60, color="#176B87")
    axes[0].set(title="15-minute electricity use", xlabel="kWh per record", ylabel="Records")
    axes[1].hist(train[TARGET], bins=60, color="#C65D07")
    axes[1].set(title="Next-hour target", xlabel="kWh in next 60 min", ylabel="Forecast origins")
    fig.tight_layout()
    fig.savefig(figure_dir / "steel_train_distributions.png", dpi=160)
    plt.close(fig)

    hour = train_raw["observation_time"].dt.hour
    weekday = train_raw["observation_time"].dt.dayofweek
    grid = train_raw.assign(hour=hour, weekday=weekday).pivot_table(
        index="weekday", columns="hour", values="Usage_kWh", aggfunc="mean"
    ).reindex(index=range(7), columns=range(24))
    fig, ax = plt.subplots(figsize=(12, 3.8))
    im = ax.imshow(grid.to_numpy(), aspect="auto", origin="upper", cmap="YlOrRd")
    ax.set(title="Mean 15-minute usage by weekday and hour (train only)", xlabel="Hour", ylabel="Weekday")
    ax.set_yticks(range(7), labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(0, 24, 2))
    fig.colorbar(im, ax=ax, label="kWh per record")
    fig.tight_layout()
    fig.savefig(figure_dir / "steel_train_hour_weekday.png", dpi=160)
    plt.close(fig)

    # Daily totals reveal sustained changes that a short time-series plot misses.
    daily_kwh = train_raw.set_index("observation_time")["Usage_kWh"].resample("D").sum()
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(daily_kwh.index, daily_kwh, color="#A9C8D5", linewidth=0.9, label="Daily total")
    ax.plot(
        daily_kwh.index, daily_kwh.rolling(7, min_periods=7).mean(),
        color="#176B87", linewidth=1.7, label="Past 7-day average",
    )
    ax.set(title="Daily electricity use (train only)", xlabel="Date", ylabel="kWh per day")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figure_dir / "steel_train_daily_totals.png", dpi=160)
    plt.close(fig)

    # Correlation is descriptive evidence for past-value features, not a model score.
    usage = train_raw["Usage_kWh"]
    lag_names = {1: "15 min", 4: "1 hour", 96: "1 day", 672: "1 week"}
    lag_correlations = {name: float(usage.corr(usage.shift(lag))) for lag, name in lag_names.items()}
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(list(lag_correlations), list(lag_correlations.values()), color="#176B87")
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set(title="Usage correlation with past readings (train only)",
           xlabel="Past reading", ylabel="Pearson correlation", ylim=(-0.1, 1.05))
    fig.tight_layout()
    fig.savefig(figure_dir / "steel_train_lag_correlations.png", dpi=160)
    plt.close(fig)

    return {
        "daily_kWh_median": float(daily_kwh.median()),
        "daily_kWh_p95": float(daily_kwh.quantile(0.95)),
        "daily_kWh_max": float(daily_kwh.max()),
        "lag_correlations": lag_correlations,
        "usage_above_train_p99_rows": int(usage.gt(usage.quantile(0.99)).sum()),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw_path = root / "data/steel/raw/Steel_industry_data.csv"
    output_dir = root / "data/steel/processed"
    figure_dir = root / "reports/steel/figures"
    raw, raw_audit = load_official_raw(raw_path)
    rows = build_supervised_rows(raw)
    splits, split_audit = split_supervised_rows(rows)
    manifest = build_row_manifest(rows)
    train_profile = profile_training_columns(raw)
    train_raw = raw.loc[raw["observation_time"].lt(SPLIT_STARTS["validation"])]
    train_eda = write_train_eda(raw, splits["train"], figure_dir)
    summary = {
        **raw_audit,
        **split_audit,
        "train_eda": train_eda,
        "train_co2_usage_correlation": float(train_raw[["Usage_kWh", "CO2(tCO2)"]].corr().iloc[0, 1]),
        "preprocessing_policy": {
            "timestamps": "parse DD/MM/YYYY HH:MM, then sort; no timestamp correction inferred",
            "missing_values": "reject incomplete official source; no imputation applied",
            "outliers": "retain observed high-consumption values; no clipping applied",
            "normalization": "not applied; fit model-specific scaler on train only",
            "feature_availability": "meter values at or before observation_time; calendar known in advance",
            "target": "sum of Usage_kWh at the next four 15-minute timestamps",
        },
    }
    write_processed_data(
        splits, summary, output_dir, manifest=manifest, train_profile=train_profile
    )
    print(f"Source SHA-256: {raw_audit['source_sha256']}")
    print(f"Sorted {len(raw)} raw rows; original file order monotonic: {raw_audit['raw_order_monotonic']}")
    print(f"Model-ready rows: {split_audit['split_rows']}")
    print(f"Row disposition: {split_audit['row_disposition_counts']}")
    print(f"Train p95 next-hour target: {split_audit['train_target_p95_kWh']:.2f} kWh")
    print(f"Outputs: {output_dir.relative_to(root)} and {figure_dir.relative_to(root)}")


if __name__ == "__main__":
    main()
