"""Thực nghiệm hồi cứu có giới hạn; giữ nguyên model đã công bố ngày 14/10.

Mỗi fold fit quá khứ, purge nhãn một giờ, dự báo tháng giữ lại.
Không dùng TEST chọn model, feature, margin hay ngưỡng công nghiệp.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.run_probabilistic_forecast import PARAMETERS, errors, save_json
from src.steel.duy_forecasting import forecast_features
from src.steel.evaluation import evaluate_alerts
from src.steel.preprocessing import BASE_FEATURES, TARGET, build_supervised_rows, load_official_raw, sha256_file, split_supervised_rows
from src.steel.probabilistic import future_measurements

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT/'reports/steel/release_2026-10-14'
METHODS = ['last_value', 'seasonal_day', 'seasonal_week', 'ridge', 'hgb', 'hgb_absolute', 'hgb_small']


def temporal_fold(rows, month):
    """Chỉ origin có đủ feature và nhãn nằm hoàn toàn trong cùng phần dữ liệu."""
    start = pd.Timestamp(2018, month, 1)
    end = start + pd.offsets.MonthBegin(1)
    ready = rows[list(BASE_FEATURES)+[TARGET]].notna().all(axis=1)
    train = rows.loc[ready & rows.forecast_end.lt(start)]
    valid = rows.loc[ready & rows.observation_time.ge(start) & rows.forecast_end.lt(end)]
    if train.empty or valid.empty or train.forecast_end.max() >= valid.observation_time.min():
        raise ValueError('Fold rỗng hoặc nhãn train lấn sang validation')
    return train, valid


def seasonal_prediction(raw, times, days):
    meter = raw.set_index('observation_time').Usage_kWh
    origins = pd.DatetimeIndex(times)
    source = [origins+pd.Timedelta(minutes=15*k)-pd.Timedelta(days=days) for k in range(1, 5)]
    if not all((t <= origins).all() for t in source):
        raise ValueError('Baseline không được đọc tương lai')
    predicted = np.column_stack([meter.reindex(t).to_numpy() for t in source])
    if not np.isfinite(predicted).all():
        raise ValueError('Baseline mùa thiếu lịch sử')
    return predicted


def feature_matrix(part, variant='base'):
    x = part[list(BASE_FEATURES)].to_numpy()
    if variant == 'no_calendar':
        return x[:, :10]
    if variant == 'no_rolling':
        return x[:, list(range(5))+list(range(10, 15))]
    if variant == 'two_deltas':
        return np.column_stack([x, x[:, 0]-x[:, 1], x[:, 0]-x[:, 2]])
    return x


def fit_predict(method, raw, train, valid, variant='base'):
    x, z = feature_matrix(train, variant), feature_matrix(valid, variant)
    y = future_measurements(raw, train.observation_time)
    if method == 'last_value':
        return np.repeat(z[:, 0, None], 4, axis=1)
    if method.startswith('seasonal_'):
        return seasonal_prediction(raw, valid.observation_time, 1 if method == 'seasonal_day' else 7)
    if method == 'ridge':
        return np.maximum(make_pipeline(StandardScaler(), Ridge(alpha=1)).fit(x, y).predict(z), 0)
    parameters = dict(PARAMETERS)
    if method == 'hgb_absolute':
        parameters['loss'] = 'absolute_error'
    if method == 'hgb_small':
        parameters.update(max_leaf_nodes=15, l2_regularization=3.)
    if method == 'lightgbm':
        from lightgbm import LGBMRegressor
        factory = lambda: LGBMRegressor(objective='regression_l1', n_estimators=180,
            learning_rate=.08, num_leaves=31, reg_lambda=1., random_state=42,
            n_jobs=1, verbosity=-1, deterministic=True, force_col_wise=True)
    else:
        factory = lambda: HistGradientBoostingRegressor(**parameters)
    if method == 'lightgbm':
        x, z = pd.DataFrame(x), pd.DataFrame(z)
    return np.maximum(np.column_stack([factory().fit(x, y[:, k]).predict(z) for k in range(4)]), 0)


def upper_margin(actual, upper, scale=None, coverage=.9):
    """Hiệu chỉnh một phía bằng nhãn CAL; không nhận bảo đảm conformal cho TS."""
    y, u = np.asarray(actual, float), np.asarray(upper, float)
    s = np.ones(len(y)) if scale is None else np.asarray(scale, float)
    if not len(y) or s.shape != y.shape or u.shape != y.shape or not np.isfinite([y, u, s]).all() or (s <= 0).any():
        raise ValueError('CAL/scale phải đầy đủ, hữu hạn và scale dương')
    scores = np.maximum(y-u, 0)/s
    level = min(1., np.ceil((len(y)+1)*coverage)/len(y))
    return float(np.quantile(scores, level, method='higher'))


def calibration_study(output, threshold):
    cal = pd.read_csv(PUBLISHED/'calibration_predictions.csv')
    test = pd.read_csv(PUBLISHED/'test_predictions.csv', parse_dates=['issue_time'])
    # Fit cả hai margin trước khi scoring TEST. Không tìm tham số trên TEST.
    floor = max(float(np.median(cal.hour_total_p90_kWh-cal.hour_total_p10_kWh)), 1e-12)
    scale = np.maximum(cal.hour_total_p90_kWh-cal.hour_total_p10_kWh, floor)
    constant = upper_margin(cal.hour_total_actual_kWh, cal.hour_total_p90_kWh)
    normalized = upper_margin(cal.hour_total_actual_kWh, cal.hour_total_p90_kWh, scale)
    config = {'calibration_period': 'October 2018', 'nominal_one_sided_coverage': .9,
              'constant_margin_kWh': constant, 'normalized_margin': normalized, 'width_floor_kWh': floor,
              'coverage_guaranteed': False, 'deployed': False,
              'decision': 'Research comparison only; no operational false-alarm cost or new blind holdout'}
    save_json(output/'calibration_candidates.json', config)
    upper = test.hour_total_p90_kWh.to_numpy()
    candidates = {'raw_p90': upper,
                  'published_central80_upper': test.hour_total_calibrated_upper_kWh.to_numpy(),
                  'one_sided90_constant': upper+constant,
                  'one_sided90_normalized': upper+normalized*np.maximum(test.hour_total_p90_kWh-test.hour_total_p10_kWh, floor)}
    rows = []
    actual = test.hour_total_actual_kWh.to_numpy()
    high = actual > threshold
    predictions = test[['issue_time', 'hour_total_actual_kWh']].copy()
    for name, bound in candidates.items():
        score = evaluate_alerts(actual, bound, test.issue_time, threshold, 0)
        rows.append({'method': name, 'rows': len(actual), 'upper_coverage': float(np.mean(actual <= bound)),
            'high_actual_upper_coverage': float(np.mean(actual[high] <= bound[high])),
            'mean_upper_addition_kWh': float(np.mean(bound-upper)),
            **{k: score[k] for k in ('tp', 'fp', 'fn', 'tn', 'precision', 'recall', 'f1', 'false_positive_rate', 'false_discovery_rate')}})
        predictions[name+'_upper_kWh'] = bound
    pd.DataFrame(rows).to_csv(output/'calibration_comparison.csv', index=False)
    predictions.to_csv(output/'calibration_test_predictions.csv', index=False)


def diagnostics(output, splits, published=PUBLISHED):
    records = []
    p50, p95 = splits['train'][TARGET].quantile([.5, .95])
    test = pd.read_csv(published/'test_predictions.csv', parse_dates=['issue_time'])
    y = test.hour_total_actual_kWh
    pred = test[[f'predicted_{15*k}m_kWh' for k in range(1, 5)]].sum(axis=1)
    residual = y-pred  # Dương = dự báo thiếu. Khác bias prediction-actual trên dashboard.
    covered = y.between(test.hour_total_calibrated_lower_kWh, test.hour_total_calibrated_upper_kWh)
    groups = {'actual_energy': pd.cut(y, [-np.inf, p50, p95, np.inf], labels=['low', 'middle', 'high']),
              'hour': test.issue_time.dt.hour, 'weekday': test.issue_time.dt.dayofweek,
              'month': test.issue_time.dt.month}
    for kind, labels in groups.items():
        for label in sorted(labels.unique()):
            mask = labels == label
            records.append({'group_by': kind, 'group': str(label), 'rows': int(mask.sum()),
                'mae_hour_kWh': float(residual[mask].abs().mean()), 'mean_actual_minus_prediction_kWh': float(residual[mask].mean()),
                'interval_coverage': float(covered[mask].mean()),
                'upper_coverage': float((y[mask] <= test.loc[mask, 'hour_total_calibrated_upper_kWh']).mean())})
    pd.DataFrame(records).to_csv(output/'conditional_errors.csv', index=False)
    scores = []
    for k in range(1, 5):
        e = test[f'actual_{15*k}m_kWh']-test[f'predicted_{15*k}m_kWh']
        scores.append({'minutes': 15*k, 'rows': len(e), 'mae_kWh': float(e.abs().mean()),
                       'rmse_kWh': float(np.sqrt(np.mean(e**2))), 'mean_actual_minus_prediction_kWh': float(e.mean())})
    pd.DataFrame(scores).to_csv(output/'horizon_errors.csv', index=False)
    save_json(output/'residual_dependence.json', {'overlapping_15m_lag1_acf': float(residual.autocorr()),
        'nonoverlap_hourly_lag1_acf': float(residual.iloc[::4].autocorr()), 'train_actual_p50_kWh': p50,
        'train_actual_p95_kWh': p95, 'conditional_groups_known_at_inference': False,
        'note': 'Overlapping labels and temporal dependence; no physical failure cause inferred'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].hist(residual, bins=70); axes[0, 0].set_title('Actual minus point sum (kWh)')
    axes[0, 1].scatter(pred, residual, s=3, alpha=.25); axes[0, 1].set_xlabel('Predicted hour total (kWh)')
    axes[0, 1].set_ylabel('Actual - predicted (kWh)')
    axes[1, 0].plot(test.issue_time, residual, lw=.4); axes[1, 0].set_title('Residual over measured TEST')
    table = pd.DataFrame(records).query("group_by == 'actual_energy'").set_index('group').loc[['low', 'middle', 'high']]
    axes[1, 1].bar(table.index, 100*table.interval_coverage); axes[1, 1].axhline(80, color='r', ls='--')
    axes[1, 1].set_title('Central interval coverage by actual group (%)'); axes[1, 1].set_ylim(0, 100)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(output/'residual_analysis.png', dpi=150); plt.close(fig)


def run(output, include_lightgbm=False):
    output.mkdir(parents=True, exist_ok=False)
    raw, audit = load_official_raw(ROOT/'data/steel/raw/Steel_industry_data.csv')
    rows = build_supervised_rows(raw)
    rows.loc[:, list(BASE_FEATURES)] = forecast_features(raw).to_numpy()
    splits, split_info = split_supervised_rows(rows)
    before = {str(p.relative_to(ROOT)): sha256_file(p) for p in PUBLISHED.rglob('*') if p.is_file()}
    methods = METHODS + (['lightgbm'] if include_lightgbm else [])
    save_json(output/'protocol.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': audit['source_sha256'], 'methods': methods, 'hgb_parameters': PARAMETERS,
        'fold_months': [6, 7, 8], 'validation_month': 9, 'purge': 'forecast_end strictly before next month',
        'variants': ['base', 'two_deltas', 'no_calendar', 'no_rolling'],
        'selection_data': 'Only held-out past months and September; TEST never selects changes',
        'model_deployment': 'Retain published HGB; challengers exploratory, not automatically deployed',
        'test_already_observed': True, 'retrospective_study': True, 'split': split_info,
        'synthetic_operating_data_used': False, 'savings_measured': False})
    results, ablations = [], []
    with threadpool_limits(limits=1):
        for month in (6, 7, 8, 9):
            train, valid = temporal_fold(rows, month) if month < 9 else (splits['train'], splits['validation'])
            actual = future_measurements(raw, valid.observation_time)
            predictions = pd.DataFrame({'issue_time': valid.observation_time.to_numpy()})
            for k in range(4): predictions[f'actual_{15*(k+1)}m_kWh'] = actual[:, k]
            for method in methods:
                start = perf_counter()
                predicted = fit_predict(method, raw, train, valid)
                score = errors(actual, predicted)
                results.append({'month': month, 'model': method, 'train_rows': len(train), 'valid_rows': len(valid),
                    'mae_per_step_kWh': score['mae_kWh'], 'rmse_per_step_kWh': score['rmse_kWh'],
                    'mae_hour_sum_kWh': score['sum_hour_mae_kWh'], 'fit_predict_seconds': perf_counter()-start})
                for k in range(4): predictions[f'{method}_{15*(k+1)}m_kWh'] = predicted[:, k]
                if method == 'hgb':
                    ablations.append({'month': month, 'variant': 'base', 'valid_rows': len(valid), 'mae_per_step_kWh': score['mae_kWh']})
                print(month, method, round(score['mae_kWh'], 6), flush=True)
            for variant in ('two_deltas', 'no_calendar', 'no_rolling'):
                predicted = fit_predict('hgb', raw, train, valid, variant)
                ablations.append({'month': month, 'variant': variant, 'valid_rows': len(valid),
                                  'mae_per_step_kWh': errors(actual, predicted)['mae_kWh']})
            predictions.to_csv(output/f'predictions_month_{month:02}.csv', index=False)
    results = pd.DataFrame(results)
    results.to_csv(output/'walk_forward_metrics.csv', index=False)
    summary = []
    for method, part in results.loc[results.month < 9].groupby('model', sort=False):
        summary.append({'model': method, 'origins': int(part.valid_rows.sum()),
            'weighted_mae_per_step_kWh': float(np.average(part.mae_per_step_kWh, weights=part.valid_rows)),
            'weighted_mae_hour_sum_kWh': float(np.average(part.mae_hour_sum_kWh, weights=part.valid_rows)),
            'worst_fold_mae_per_step_kWh': float(part.mae_per_step_kWh.max())})
    pd.DataFrame(summary).to_csv(output/'walk_forward_summary.csv', index=False)
    pd.DataFrame(ablations).to_csv(output/'feature_ablation.csv', index=False)
    diagnostics(output, splits)
    manifest = json.loads((PUBLISHED/'manifest.json').read_text())
    calibration_study(output, manifest['train_research_thresholds_kWh']['hour_total'])
    cal = pd.read_csv(PUBLISHED/'calibration_predictions.csv')
    point = cal[[f'predicted_{15*k}m_kWh' for k in range(1, 5)]].sum(axis=1)
    save_json(output/'monitoring_reference.json', {'model_sha256': manifest['model_sha256'],
        'source_sha256': manifest['source_sha256'], 'reference_period': '2018-10 calibration; no TEST reference',
        'rows': len(cal), 'point_hour_mae_kWh': float((point-cal.hour_total_actual_kWh).abs().mean()),
        'required_labels': 1344, 'window_labels': 672, 'mae_factor': 1.2, 'coverage_drop': .10,
        'central_coverage': float(cal.hour_total_actual_kWh.between(cal.hour_total_calibrated_lower_kWh, cal.hour_total_calibrated_upper_kWh).mean()),
        'heuristic_only': True, 'automatic_retraining': False})
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for method, part in results.query('month < 9').groupby('model', sort=False):
        ax.plot(part.month, part.mae_per_step_kWh, marker='o', label=method)
    ax.set(xticks=[6, 7, 8], xlabel='Held-out month (2018)', ylabel='MAE per 15-min output (kWh)', title='Retrospective expanding-window comparison')
    ax.legend(ncol=3, fontsize=8); fig.tight_layout(); fig.savefig(output/'walk_forward.png', dpi=150); plt.close(fig)
    if before != {name: sha256_file(ROOT/name) for name in before}:
        raise AssertionError('Published run changed')
    save_json(output/'verification.json', {'published_artifacts_unchanged': True, 'measured_source_sha256': audit['source_sha256'],
        'folds': 3, 'validation': 1, 'test_used_for_selection': False, 'predictions_saved': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--include-lightgbm', action='store_true', help='Cần requirements-khanh.txt; study cùng 15 feature/4 nhãn')
    args = parser.parse_args()
    run(args.output_dir.resolve(), args.include_lightgbm)
