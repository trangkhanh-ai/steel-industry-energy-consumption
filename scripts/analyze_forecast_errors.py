"""Đánh giá và phân tích lỗi từ CSV UCI thật; không fit hay chỉnh model theo TEST.

python -m scripts.analyze_forecast_errors --output-dir reports/steel/local_error_analysis_01
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.check_probabilistic_run import run as verify_run
from scripts.run_probabilistic_forecast import errors
from src.steel.duy_forecasting import forecast_features
from src.steel.evaluation import calibrate_upper_buffer, evaluate_alerts
from src.steel.preprocessing import (BASE_FEATURES, TARGET, build_supervised_rows,
                                     load_official_raw, sha256_file, split_supervised_rows)
from src.steel.probabilistic import (DEFAULT_PROBABILISTIC_SUBDIR, ProbabilisticForecaster,
                                     future_measurements, interval_metrics, research_alert_metrics)

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / 'reports/steel/release_2026-10-11'


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                    encoding='utf-8', newline='\n')


def validate_predictions(raw, frame, origins):
    """Kiểm đúng tập origin và từng số đo nhãn; không tự sort/sửa CSV sai."""
    times = pd.DatetimeIndex(pd.to_datetime(frame.issue_time, errors='raise'))
    expected = pd.DatetimeIndex(origins)
    if not times.equals(expected):
        raise ValueError('Prediction phải có đúng origin và thứ tự của split đã khóa')
    actual = future_measurements(raw, times)
    columns = [f'actual_{15*k}m_kWh' for k in range(1, 5)]
    np.testing.assert_allclose(frame[columns], actual, rtol=0, atol=1e-10)
    np.testing.assert_allclose(frame.hour_total_actual_kWh, actual.sum(axis=1), rtol=0, atol=1e-10)
    return actual


def classify(actual, alert, threshold):
    """TP/FP/FN/TN theo tổng giờ thật và ngưỡng nghiên cứu, không theo lỗi máy."""
    high = np.asarray(actual) > threshold
    alert = np.asarray(alert, dtype=bool)
    return np.where(high, np.where(alert, 'TP', 'FN'), np.where(alert, 'FP', 'TN'))


def monthly_metrics(frame, threshold, margin):
    records = []
    for month, part in frame.groupby(pd.to_datetime(frame.issue_time).dt.month):
        y = part[[f'actual_{15*k}m_kWh' for k in range(1, 5)]].to_numpy()
        pred = part[[f'predicted_{15*k}m_kWh' for k in range(1, 5)]].to_numpy()
        q = part[[f'hour_total_p{p}_kWh' for p in (10, 50, 90)]].to_numpy()
        point = errors(y, pred)
        band = interval_metrics(y.sum(axis=1), q, margin)
        alert = research_alert_metrics(y.sum(axis=1), q, pd.to_datetime(part.issue_time), threshold, margin)
        records.append({'month': int(month), 'origins': len(part),
            'mae_per_step_kWh': point['mae_kWh'], 'mae_point_sum_kWh': point['sum_hour_mae_kWh'],
            'coverage_after_calibration': band['calibrated_coverage'],
            **{key: alert[key] for key in ('tp', 'fp', 'fn', 'tn', 'precision', 'recall', 'f1')}})
    return pd.DataFrame(records)


def compare_reproduction(published, reproduced):
    result = {}
    for split in ('validation', 'calibration', 'test'):
        first = pd.read_csv(published / f'{split}_predictions.csv')
        second = pd.read_csv(reproduced / f'{split}_predictions.csv')
        pd.testing.assert_series_equal(pd.to_datetime(first.issue_time), pd.to_datetime(second.issue_time))
        np.testing.assert_allclose(first.iloc[:, 1:], second.iloc[:, 1:], rtol=0, atol=1e-10)
        result[split] = {'rows': len(first), 'maximum_absolute_difference':
                         float(np.max(np.abs(first.iloc[:, 1:].to_numpy()-second.iloc[:, 1:].to_numpy())))}
    for filename in ('point_model_comparison.csv',):
        first, second = pd.read_csv(published/filename), pd.read_csv(reproduced/filename)
        pd.testing.assert_frame_equal(first, second, check_exact=False, atol=1e-10, rtol=0)
    return {'passed': True, 'absolute_tolerance': 1e-10, 'splits': result,
            'point_candidates_reproduced': True, 'published_model_changed': False}


def draw_cases(frame, threshold, folder):
    fig, axes = plt.subplots(2, 1, figsize=(7, 5))
    for ax, center, title in zip(axes, ('2018-11-22 08:15', '2018-11-16 19:30'),
            ('22 Nov: legacy FN becomes TP', '16 Nov: current policy still has an FN')):
        time = pd.Timestamp(center)
        view = frame.loc[frame.issue_time.between(time-pd.Timedelta(hours=1), time+pd.Timedelta(hours=1))]
        ax.plot(view.issue_time, view.hour_total_actual_kWh, color='black', marker='o', markersize=3,
                label='True future hour total (offline label)')
        ax.plot(view.issue_time, view.legacy_upper_kWh, color='#64748b', label='Legacy prediction + buffer')
        ax.plot(view.issue_time, view.hour_total_calibrated_upper_kWh, color='#0f766e', label='New calibrated upper')
        ax.axhline(threshold, color='#b91c1c', linestyle='--', label='TRAIN p95 research threshold')
        ax.axvline(time, color='#94a3b8', linestyle=':')
        ax.set(title=title, ylabel='Next-hour energy (kWh)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.tick_params(labelsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc='lower center', ncol=2)
    fig.tight_layout(rect=(0, .11, 1, 1))
    fig.savefig(folder/'error_cases.png', dpi=160); plt.close(fig)


def run(output, run_dir=ROOT / DEFAULT_PROBABILISTIC_SUBDIR, reproduction_dir=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    verify_run(run_dir, output/'artifact_verification.json')
    raw, audit = load_official_raw(ROOT/'data/steel/raw/Steel_industry_data.csv')
    features = forecast_features(raw)
    rows = build_supervised_rows(raw)
    rows.loc[:, list(BASE_FEATURES)] = features.to_numpy()
    splits, _ = split_supervised_rows(rows)
    current = pd.read_csv(run_dir/'test_predictions.csv', parse_dates=['issue_time'])
    actual = validate_predictions(raw, current, splits['test'].observation_time)
    metadata = json.loads((run_dir/'manifest.json').read_text(encoding='utf-8'))
    threshold = metadata['train_research_thresholds_kWh']['hour_total']
    margin = metadata['calibration_margins_kWh']['hour_total']
    np.testing.assert_allclose(threshold, np.quantile(splits['train'][TARGET], .95), rtol=0, atol=1e-10)

    legacy = pd.read_csv(LEGACY/'test_predictions.csv', parse_dates=['observation_time'])
    pd.testing.assert_series_equal(current.issue_time, legacy.observation_time, check_names=False)
    np.testing.assert_allclose(actual.sum(axis=1), legacy[TARGET], rtol=0, atol=1e-10)
    policy = json.loads((LEGACY/'locked_policy.json').read_text(encoding='utf-8'))
    legacy_path = ROOT/'reports/steel/duy_2026-10-02/models/hgb_31_leaves.joblib'
    if sha256_file(legacy_path) != policy['model_sha256']:
        raise ValueError('Sai artifact đối chứng cũ')
    legacy_model = joblib.load(legacy_path)
    with threadpool_limits(limits=1):
        prediction = np.maximum(legacy_model.predict(splits['test'][list(BASE_FEATURES)]), 0)
        cal_prediction = np.maximum(legacy_model.predict(splits['calibration'][list(BASE_FEATURES)]), 0)
    np.testing.assert_allclose(prediction, legacy.pred_hgb_31_leaves_kWh, rtol=0, atol=1e-10)
    np.testing.assert_allclose(calibrate_upper_buffer(splits['calibration'][TARGET], cal_prediction),
                               policy['upper_buffer_kWh'], rtol=0, atol=1e-10)
    old = evaluate_alerts(actual.sum(axis=1), prediction, current.issue_time, threshold, policy['upper_buffer_kWh'])
    quantiles = current[[f'hour_total_p{p}_kWh' for p in (10, 50, 90)]].to_numpy()
    new = research_alert_metrics(actual.sum(axis=1), quantiles, current.issue_time, threshold, margin)
    compare = pd.DataFrame([{'method': name, **{key: score[key] for key in
        ('rows', 'tp', 'fp', 'fn', 'tn', 'precision', 'recall', 'f1', 'alert_rows', 'alert_origin_episodes')}}
        for name, score in (('legacy_hour_buffer', old), ('four_step_quantile', new))])
    compare.to_csv(output/'alert_comparison.csv', index=False)
    months = monthly_metrics(current, threshold, margin)
    months.to_csv(output/'monthly_metrics.csv', index=False)

    frame = current.merge(raw[['observation_time', 'Usage_kWh']], left_on='issue_time',
                          right_on='observation_time', validate='one_to_one')
    frame['legacy_upper_kWh'] = prediction+policy['upper_buffer_kWh']
    frame['legacy_case'] = classify(frame.hour_total_actual_kWh, frame.legacy_upper_kWh > threshold, threshold)
    frame['current_case'] = classify(frame.hour_total_actual_kWh, frame.hour_total_calibrated_upper_kWh > threshold, threshold)
    frame['point_sum_kWh'] = frame[[f'predicted_{15*k}m_kWh' for k in range(1, 5)]].sum(axis=1)
    frame['upper_shortfall_kWh'] = frame.hour_total_actual_kWh-frame.hour_total_calibrated_upper_kWh
    missed = frame.loc[frame.current_case.eq('FN')].sort_values('upper_shortfall_kWh', ascending=False)
    missed.to_csv(output/'false_negatives.csv', index=False)
    selected = pd.to_datetime(['2018-11-22 08:15', '2018-11-22 09:00', '2018-11-01 08:00'])
    selected = selected.union(pd.DatetimeIndex([missed.issue_time.iloc[0]]))
    cases = frame.loc[frame.issue_time.isin(selected)].sort_values('issue_time')
    case_columns = ['issue_time', 'Usage_kWh', 'hour_total_actual_kWh', 'legacy_upper_kWh',
                    'legacy_case', 'point_sum_kWh', 'hour_total_p50_kWh',
                    'hour_total_calibrated_upper_kWh', 'current_case']
    cases[case_columns].to_csv(output/'case_studies.csv', index=False)
    forecaster = ProbabilisticForecaster(run_dir, audit['source_sha256'])
    for _, row in cases.iterrows():
        history = raw.loc[raw.observation_time <= row.issue_time, ['observation_time', 'Usage_kWh']].tail(673)
        online = forecaster.predict(history, row.issue_time)
        np.testing.assert_allclose(online['hour_total']['calibrated_upper_kWh'], row.hour_total_calibrated_upper_kWh, rtol=0, atol=1e-10)
        if online['hour_total']['actual_kWh'] is not None:
            raise AssertionError('Actual tương lai xuất hiện lúc phát dự báo')
    draw_cases(frame, threshold, output)
    scope = {'decision': 'Prioritize review of high-energy forecast origins by a human',
        'true_high_origins': int(new['tp']+new['fn']), 'alerted_origins': int(new['alert_rows']),
        'false_positive_origins': int(new['fp']), 'false_negative_origins': int(new['fn']),
        'unit': 'overlapping forecast origins, not physical incidents or actual human inspections',
        'energy_saved_kWh': None, 'money_saved': None, 'contract_power_limit_kW': None,
        'savings_measured': False, 'physical_event_lead_time_measured': False,
        'inspection_workload_measured': False, 'automatic_equipment_control': False,
        'operating_cause_identified': False, 'synthetic_operating_data_used': False}
    save_json(output/'industrial_scope.json', scope)
    summary = {'passed': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': audit['source_sha256'], 'model_sha256': metadata['model_sha256'],
        'protocol_sha256': metadata['protocol_sha256'], 'labels_checked_against_raw_csv': True,
        'legacy_predictions_recomputed': True, 'legacy_buffer_recomputed_from_october': True,
        'test_origins': len(current), 'threshold_kWh': threshold,
        'case_online_parity_checked': len(cases), 'all_future_actuals_absent_at_issue': True,
        'case_selection': 'Two fixed Khanh examples, fixed initial FP, largest actual-minus-upper current FN',
        'case_selection_purpose': 'Retrospective explanation, not representative sampling or model tuning',
        'test_already_observed': True, 'model_tuned': False, 'synthetic_operating_data_used': False,
        'input_sha256': {str(path.relative_to(ROOT)): sha256_file(path) for path in
             (run_dir/'test_predictions.csv', run_dir/'metrics.json', LEGACY/'test_predictions.csv')},
        'reproduction': compare_reproduction(run_dir, reproduction_dir) if reproduction_dir else None}
    save_json(output/'verification.json', summary)
    print(compare.to_string(index=False)); print(months.to_string(index=False))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, default=ROOT / DEFAULT_PROBABILISTIC_SUBDIR)
    parser.add_argument('--reproduction-dir', type=Path)
    args = parser.parse_args()
    run(args.output_dir.resolve(), args.run_dir.resolve(),
        args.reproduction_dir.resolve() if args.reproduction_dir else None)
