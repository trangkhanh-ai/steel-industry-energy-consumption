"""Huấn luyện/đánh giá từ CSV thật: 4 bước 15 phút + phân vị tổng/đỉnh kWh.

Không giả định P_limit, giá điện, lịch tải, BESS hoặc tiền tiết kiệm.
Train Jan-Aug; chọn point trên Sep; calibration Oct; test hồi cứu Nov-Dec.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.steel.duy_forecasting import forecast_features
from src.steel.preprocessing import BASE_FEATURES, TARGET, build_supervised_rows, load_official_raw, sha256_file, split_supervised_rows
from src.steel.probabilistic import CONTRACT, QUANTILES, calibration_margin, future_measurements, interval_metrics, point_prediction, quantile_prediction, research_alert_metrics

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = dict(max_iter=180, max_leaf_nodes=31, learning_rate=.08,
                  l2_regularization=1., early_stopping=False, random_state=42)


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def errors(y, predicted):
    return {'mae_kWh': float(mean_absolute_error(y, predicted)),
            'rmse_kWh': float(root_mean_squared_error(y, predicted)),
            'per_step': [{'minutes': 15*(i+1),
                          'mae_kWh': float(mean_absolute_error(y[:, i], predicted[:, i])),
                          'rmse_kWh': float(root_mean_squared_error(y[:, i], predicted[:, i]))}
                         for i in range(4)],
            'sum_hour_mae_kWh': float(mean_absolute_error(y.sum(axis=1), predicted.sum(axis=1))),
            'sum_hour_rmse_kWh': float(root_mean_squared_error(y.sum(axis=1), predicted.sum(axis=1)))}


def run(output, point_loss='squared_error'):
    if output.exists():
        raise FileExistsError('Chọn thư mục run mới; không ghi đè kết quả cũ')
    raw, audit = load_official_raw(ROOT/'data/steel/raw/Steel_industry_data.csv')
    rows = build_supervised_rows(raw)
    rows.loc[:, list(BASE_FEATURES)] = forecast_features(raw).to_numpy()
    splits, split_audit = split_supervised_rows(rows)
    x = {name: frame[list(BASE_FEATURES)].to_numpy() for name, frame in splits.items()}
    y = {name: future_measurements(raw, frame.observation_time) for name, frame in splits.items()}
    for name, part in splits.items():
        np.testing.assert_allclose(y[name].sum(axis=1), part[TARGET], rtol=0, atol=1e-10)
    output.mkdir(parents=True)
    (output/'figures').mkdir()
    protocol = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': audit['source_sha256'],
        'targets': 'Four future measured Usage_kWh; hour_total=sum; max_interval=max, never instantaneous kW',
        'split_audit': split_audit, 'features': list(BASE_FEATURES),
        'point_candidates': ['last_value', 'ridge', 'hgb'],
        'point_selection': 'lowest mean MAE over four outputs on September; ties prefer last_value',
        'ridge': {'alpha': 1, 'scaler_fit': 'train only'}, 'hgb': PARAMETERS,
        'point_loss': point_loss, 'quantile_loss': 'quantile',
        'quantiles': list(QUANTILES), 'quantile_targets': ['hour_total', 'max_interval'],
        'quantile_postprocessing': 'clip nonnegative and sort three estimates; report crossings',
        'calibration': 'October only; nonnegative CQR-style margin; nominal 80%, no TS coverage guarantee',
        'test_already_observed': True, 'quantile_configuration_search': False,
        'contract_power_limit_available': False, 'tariff_available': False,
        'savings_measured': False, 'synthetic_operating_data_used': False,
        'sources': ['https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption',
                   'https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html',
                   'https://arxiv.org/abs/1905.03222'],
    }
    save_json(output/'protocol.json', protocol)  # Ghi trước fit và trước test.
    point_bundles = {'last_value': {'point_method': 'last_value'}}
    with threadpool_limits(limits=1):
        begin = perf_counter()
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1))
        ridge.fit(x['train'], y['train'])
        point_bundles['ridge'] = {'point_method': 'ridge', 'point_model': ridge}
        hgb = [HistGradientBoostingRegressor(loss=point_loss, **PARAMETERS).fit(x['train'], y['train'][:, i]) for i in range(4)]
        point_bundles['hgb'] = {'point_method': 'hgb', 'point_model': hgb}
        validation = {name: errors(y['validation'], point_prediction(bundle, x['validation']))
                      for name, bundle in point_bundles.items()}
        selected = min(point_bundles, key=lambda name: (validation[name]['mae_kWh'], name!='last_value', name))
        print('Selected on validation:', selected, flush=True)
        bundle = dict(point_bundles[selected], quantile_models={})
        margins = {}
        for target in ('hour_total', 'max_interval'):
            actual = {name: values.sum(axis=1) if target=='hour_total' else values.max(axis=1)
                      for name, values in y.items()}
            models = []
            for quantile in QUANTILES:
                model = HistGradientBoostingRegressor(loss='quantile', quantile=quantile, **PARAMETERS)
                model.fit(x['train'], actual['train'])
                models.append(model)
                print(f'Fitted {target} P{int(quantile*100)} from TRAIN', flush=True)
            bundle['quantile_models'][target] = models
            cal, _ = quantile_prediction(models, x['calibration'])
            margins[target] = calibration_margin(actual['calibration'], cal)
        fit_seconds = perf_counter()-begin
        model_path = output/'forecast_bundle.joblib'
        joblib.dump(bundle, model_path)
        thresholds = {'hour_total': float(np.quantile(y['train'].sum(axis=1), .95)),
                      'max_interval': float(np.quantile(y['train'].max(axis=1), .95))}
        manifest = {'contract': CONTRACT, 'features': list(BASE_FEATURES), 'source_sha256': audit['source_sha256'],
                    'sklearn': sklearn.__version__, 'model_file': model_path.name,
                    'model_sha256': sha256_file(model_path), 'point_method': selected,
                    'valid_from': '2018-11-01T00:00:00', 'calibration_margins_kWh': margins,
                    'train_research_thresholds_kWh': thresholds,
                    'nominal_central_coverage': .8, 'coverage_guaranteed': False,
                    'point_loss': point_loss,
                    'fit_seconds': fit_seconds, 'protocol_sha256': sha256_file(output/'protocol.json')}
        save_json(output/'manifest.json', manifest)
        # Chỉ scoring test sau khi model, margin, thresholds và manifest đã khóa.
        metrics = {'validation_point_candidates': validation, 'selected_point_method': selected,
                   'splits': {}, 'synthetic_operating_data_used': False, 'savings_measured': False}
        for name in ('validation', 'calibration', 'test'):
            frame = splits[name]
            pred = point_prediction(bundle, x[name])
            metrics['splits'][name] = {'point': errors(y[name], pred), 'quantiles': {}}
            csv = pd.DataFrame({'issue_time': frame.observation_time})
            for i in range(4):
                csv[f'actual_{15*(i+1)}m_kWh'] = y[name][:, i]
                csv[f'predicted_{15*(i+1)}m_kWh'] = pred[:, i]
            for target, models in bundle['quantile_models'].items():
                actual = y[name].sum(axis=1) if target=='hour_total' else y[name].max(axis=1)
                q, crossing = quantile_prediction(models, x[name])
                margin = 0. if name=='validation' else margins[target]
                score = interval_metrics(actual, q, margin)
                score['raw_crossing_rows'] = int(crossing.sum())
                metrics['splits'][name]['quantiles'][target] = score
                if target == 'hour_total':
                    metrics['splits'][name]['research_alert'] = research_alert_metrics(
                        actual, q, frame.observation_time, thresholds[target], margin)
                csv[f'{target}_actual_kWh'] = actual
                for i, quantile in enumerate(QUANTILES):csv[f'{target}_p{int(quantile*100)}_kWh'] = q[:, i]
                csv[f'{target}_calibrated_lower_kWh'] = np.maximum(q[:, 0]-margin, 0)
                csv[f'{target}_calibrated_upper_kWh'] = q[:, 2]+margin
            csv.to_csv(output/f'{name}_predictions.csv', index=False)
            if name=='test':
                # Comparison uses a frozen legacy point artifact, never selects it with new test scores.
                legacy = joblib.load(ROOT/'reports/steel/duy_2026-10-02/models/hgb_31_leaves.joblib')
                old = np.maximum(legacy.predict(frame[list(BASE_FEATURES)]), 0)
                metrics['legacy_hour_test'] = {'mae_kWh': float(mean_absolute_error(y[name].sum(axis=1), old)),
                                              'rmse_kWh': float(root_mean_squared_error(y[name].sum(axis=1), old))}
                # Plot measured first three test days; not a synthetic scenario.
                view = csv.iloc[:288]
                fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
                for ax, target, title in zip(axes, ('hour_total', 'max_interval'),
                        ('Next-hour total energy (kWh)', 'Largest future 15-minute Usage (kWh; NOT peak kW)')):
                    times = pd.to_datetime(view.issue_time)
                    ax.plot(times, view[f'{target}_actual_kWh'], label='Measured actual', linewidth=1)
                    ax.plot(times, view[f'{target}_p50_kWh'], label='P50 estimate', linewidth=1)
                    ax.fill_between(times, view[f'{target}_p10_kWh'], view[f'{target}_p90_kWh'], alpha=.2, label='Raw P10-P90 estimates')
                    ax.set_ylabel('kWh');ax.set_title(title);ax.legend(fontsize=8)
                fig.autofmt_xdate();fig.tight_layout();fig.savefig(output/'figures/test_quantiles.png', dpi=160);plt.close(fig)
        # Giám sát chỉ lấy reference CAL; không dùng TEST làm chuẩn vận hành.
        cal_csv = pd.read_csv(output/'calibration_predictions.csv')
        cal_point = cal_csv[[f'predicted_{15*k}m_kWh' for k in range(1, 5)]].sum(axis=1)
        reference = {'model_sha256': manifest['model_sha256'], 'source_sha256': audit['source_sha256'],
            'reference_period': '2018-10 calibration; no TEST reference', 'rows': len(cal_csv),
            'point_hour_mae_kWh': float(np.abs(cal_point-cal_csv.hour_total_actual_kWh).mean()),
            'central_coverage': float(cal_csv.hour_total_actual_kWh.between(cal_csv.hour_total_calibrated_lower_kWh, cal_csv.hour_total_calibrated_upper_kWh).mean()),
            'window_labels': 672, 'mae_factor': 1.2, 'coverage_drop': .10,
            'automatic_retraining': False, 'heuristic_only': True}
        save_json(output/'monitoring_reference.json', reference)
        manifest.update(monitoring_reference_file='monitoring_reference.json',
                        monitoring_reference_sha256=sha256_file(output/'monitoring_reference.json'))
        save_json(output/'manifest.json', manifest)
        save_json(output/'metrics.json', metrics)
        rows = []
        for name, candidate in point_bundles.items():
            for split in ('validation', 'test'):
                score = errors(y[split], point_prediction(candidate, x[split]))
                rows.append({'model': name, 'split': split, **{k:v for k,v in score.items() if k!='per_step'}})
        pd.DataFrame(rows).to_csv(output/'point_model_comparison.csv', index=False)
    print(json.dumps({'output': str(output), 'point_method': selected, 'fit_seconds': fit_seconds,
                      'test': metrics['splits']['test']}, ensure_ascii=False, indent=2), flush=True)
    return metrics


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--point-loss', choices=['squared_error', 'absolute_error'], default='squared_error')
    args = parser.parse_args()
    run(args.output_dir.resolve(), args.point_loss)
