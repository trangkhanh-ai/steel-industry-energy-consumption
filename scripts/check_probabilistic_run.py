"""Tái kiểm nhãn/metric/calibration từ CSV UCI và artifact thật, không fit lại."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.steel.preprocessing import FREQUENCY, load_official_raw, sha256_file
from src.steel.probabilistic import (ProbabilisticForecaster, calibration_margin, future_measurements,
                                     interval_metrics, research_alert_metrics)
from scripts.run_probabilistic_forecast import errors

ROOT = Path(__file__).resolve().parents[1]


def run(folder, output):
    raw, audit = load_official_raw(ROOT/'data/steel/raw/Steel_industry_data.csv')
    model = ProbabilisticForecaster(folder, audit['source_sha256'])
    manifest = model.metadata
    recorded = json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
    results = {}
    for split in ('validation', 'calibration', 'test'):
        frame = pd.read_csv(folder/f'{split}_predictions.csv')
        actual = future_measurements(raw, pd.to_datetime(frame.issue_time))
        saved = frame[[f'actual_{15*k}m_kWh' for k in range(1, 5)]].to_numpy()
        np.testing.assert_array_equal(actual, saved)
        points = frame[[f'predicted_{15*k}m_kWh' for k in range(1, 5)]].to_numpy()
        score = errors(actual, points)
        expected_point = recorded['splits'][split]['point']
        for key in ('mae_kWh', 'rmse_kWh', 'sum_hour_mae_kWh', 'sum_hour_rmse_kWh'):
            np.testing.assert_allclose(score[key], expected_point[key], rtol=0, atol=1e-10)
        for actual_step, saved_step in zip(score['per_step'], expected_point['per_step']):
            assert actual_step['minutes'] == saved_step['minutes']
            for key in ('mae_kWh', 'rmse_kWh'):
                np.testing.assert_allclose(actual_step[key], saved_step[key], rtol=0, atol=1e-10)
        results[split] = {'rows': len(frame), 'point': score, 'quantiles': {}}
        for target in ('hour_total', 'max_interval'):
            y = actual.sum(axis=1) if target=='hour_total' else actual.max(axis=1)
            np.testing.assert_allclose(y, frame[f'{target}_actual_kWh'], rtol=0, atol=1e-10)
            q = frame[[f'{target}_p{p}_kWh' for p in (10, 50, 90)]].to_numpy()
            margin = 0 if split=='validation' else manifest['calibration_margins_kWh'][target]
            if split=='calibration':
                np.testing.assert_allclose(calibration_margin(y, q), margin, rtol=0, atol=1e-10)
            recalculated = interval_metrics(y, q, margin)
            for key in ('p50_mae_kWh', 'p50_rmse_kWh', 'raw_p10_p90_coverage',
                        'calibrated_coverage', 'raw_mean_width_kWh', 'calibrated_mean_width_kWh'):
                np.testing.assert_allclose(recalculated[key], recorded['splits'][split]['quantiles'][target][key], rtol=0, atol=1e-10)
            for key in ('pinball_loss', 'empirical_fraction_y_leq_quantile'):
                for alpha in recalculated[key]:
                    np.testing.assert_allclose(recalculated[key][alpha], recorded['splits'][split]['quantiles'][target][key][alpha], rtol=0, atol=1e-10)
            results[split]['quantiles'][target] = recalculated
            if target=='hour_total':
                results[split]['research_alert'] = research_alert_metrics(y, q, pd.to_datetime(frame.issue_time),
                    manifest['train_research_thresholds_kWh'][target], margin)
                for key in ('tp', 'fp', 'fn', 'tn', 'precision', 'recall', 'f1', 'alert_rows'):
                    np.testing.assert_allclose(results[split]['research_alert'][key], recorded['splits'][split]['research_alert'][key], rtol=0, atol=1e-10)
        if split=='test':
            for i in (0, 73, 500, 2500, len(frame)-1):
                row = frame.iloc[i]; t = pd.Timestamp(row.issue_time)
                history = raw.loc[raw.observation_time.between(t-672*FREQUENCY, t)]
                online = model.predict(history, t)
                np.testing.assert_allclose([s['predicted_kWh'] for s in online['intervals']], points[i], rtol=0, atol=1e-10)
                for target in ('hour_total', 'max_interval'):
                    for p in (10, 50, 90):
                        np.testing.assert_allclose(online[target][f'p{p}_kWh'], row[f'{target}_p{p}_kWh'], rtol=0, atol=1e-10)
    summary = {'checks_passed': True, 'source_sha256': audit['source_sha256'],
               'model_sha256': manifest['model_sha256'], 'protocol_sha256': manifest['protocol_sha256'],
               'labels_match_real_csv': True, 'offline_online_sample_parity': True,
               'calibration_recomputed_from_october': True, 'test_already_observed': True,
               'synthetic_operating_data_used': False, 'savings_measured': False,
               'input_files_sha256': {name: sha256_file(folder/name) for name in ('metrics.json', 'calibration_predictions.csv', 'test_predictions.csv')},
               'results': results}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Chọn output mới để giữ bằng chứng cũ')
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'checks_passed': True, 'output': str(output), 'test_research_alert': results['test']['research_alert']}, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-file', type=Path, required=True)
    args = parser.parse_args()
    run(args.run_dir.resolve(), args.output_file.resolve())
