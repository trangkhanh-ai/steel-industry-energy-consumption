"""Vẽ ảnh README từ CSV TRAIN và prediction đã lưu; không tạo số liệu vận hành."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

from src.steel.preprocessing import load_official_raw
from src.steel.probabilistic import DEFAULT_PROBABILISTIC_SUBDIR

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'assets'
    output.mkdir(exist_ok=True)
    raw, _ = load_official_raw(ROOT / 'data/steel/raw/Steel_industry_data.csv')
    train = raw.loc[raw.observation_time < '2018-09-01']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.7))
    week = train.iloc[:672]
    axes[0].plot(week.observation_time, week.Usage_kWh, color='#0f766e', linewidth=1)
    axes[0].set(title='1. Understand the measured data — first TRAIN week', ylabel='Usage per 15-minute record (kWh)')
    axes[1].hist(train.Usage_kWh, bins=55, color='#2563eb', alpha=.8)
    axes[1].set(title='TRAIN distribution — keep real high-load records', xlabel='Usage (kWh)', ylabel='Records')
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(output / 'data_preview.png', dpi=160); plt.close(fig)

    # Architecture labels describe actual implemented components, no factory claims.
    fig, ax = plt.subplots(figsize=(13, 2.8))
    ax.set(xlim=(0, 13), ylim=(0, 2.8)); ax.axis('off')
    labels = [('UCI CSV', 'Measured history'), ('Quality + features', '15 causal inputs'),
              ('Ridge / HGB', '4 steps + quantiles'), ('Research alert', 'TRAIN p95 + margin'),
              ('Human review', 'Inspect / record')]
    for i, (title, detail) in enumerate(labels):
        x = .15 + i*2.6
        ax.add_patch(FancyBboxPatch((x, 1.05), 2.25, 1.1, boxstyle='round,pad=.1',
                                    facecolor='#eff6ff', edgecolor='#2563eb', linewidth=1.4))
        ax.text(x+1.125, 1.75, title, ha='center', fontsize=11, weight='bold', color='#0f172a')
        ax.text(x+1.125, 1.38, detail, ha='center', fontsize=9, color='#475569')
        if i < 4:
            ax.annotate('', xy=(x+2.5, 1.6), xytext=(x+2.3, 1.6),
                        arrowprops={'arrowstyle': '->', 'color': '#2563eb', 'lw': 1.7})
    ax.text(6.5, .45, 'SQLite: predictions + delayed actuals + decisions   |   Local replay; no equipment control',
            ha='center', fontsize=11, color='#0f766e')
    fig.tight_layout(); fig.savefig(output / 'pipeline.png', dpi=160); plt.close(fig)

    table = pd.read_csv(ROOT / DEFAULT_PROBABILISTIC_SUBDIR / 'point_model_comparison.csv')
    fig, ax = plt.subplots(figsize=(9, 4))
    test = table.loc[table.split.eq('test')]
    bars = ax.bar(test.model, test.mae_kWh, color=['#94a3b8', '#60a5fa', '#0f766e'])
    ax.bar_label(bars, fmt='%.4f', padding=4)
    ax.set(title='2. Compare methods — retrospective TEST, Nov–Dec 2018',
           ylabel='Mean MAE over four forecast steps (kWh)', ylim=(0, test.mae_kWh.max()*1.18))
    fig.tight_layout(); fig.savefig(output / 'model_comparison.png', dpi=160); plt.close(fig)
    print('Created assets/data_preview.png, assets/pipeline.png, assets/model_comparison.png')


if __name__ == '__main__':
    main()
