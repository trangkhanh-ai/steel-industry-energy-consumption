"""Kiểm các rủi ro leakage, label trễ và model mixing của phần bổ sung."""
import unittest
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from scripts.run_improvement_study import temporal_fold, seasonal_prediction, upper_margin
from src.steel.preprocessing import BASE_FEATURES, TARGET
from src.steel.reliability import reliability_watch
from src.steel.frozen_results import read_results
from src.steel.preprocessing import sha256_file


class ImprovementTests(unittest.TestCase):
    def test_fold_purges_labels_crossing_boundary(self):
        times = pd.date_range('2018-05-31 22:00', '2018-06-01 02:00', freq='15min')
        rows = pd.DataFrame({'observation_time': times, 'forecast_end': times+pd.Timedelta(hours=1), TARGET: 1.})
        for name in BASE_FEATURES: rows[name] = 1.
        train, valid = temporal_fold(rows, 6)
        self.assertEqual(train.forecast_end.max(), pd.Timestamp('2018-05-31 23:45'))
        self.assertEqual(valid.observation_time.min(), pd.Timestamp('2018-06-01'))

    def test_seasonal_prediction_reads_past_corresponding_four_intervals(self):
        times = pd.date_range('2018-01-01', periods=200, freq='15min')
        raw = pd.DataFrame({'observation_time': times, 'Usage_kWh': np.arange(200, dtype=float)})
        np.testing.assert_array_equal(seasonal_prediction(raw, times[100:101], 1), [[5., 6., 7., 8.]])
        with self.assertRaises(ValueError): seasonal_prediction(raw, times[100:101], 0)

    def test_one_sided_margin_is_calibration_only_and_rejects_bad_scale(self):
        actual = np.arange(1., 11.)
        self.assertGreaterEqual(upper_margin(actual, actual-2), 2.)
        with self.assertRaises(ValueError): upper_margin(actual, actual, np.zeros(10))

    def test_reliability_requires_labels_and_contiguous_windows(self):
        ref = dict(model_sha256='test', reference_period='calibration', window_labels=4,
                   point_hour_mae_kWh=1., mae_factor=1.2, central_coverage=.8, coverage_drop=.1)
        times = pd.date_range('2018-11-01', periods=8, freq='15min')
        actual = np.ones(8)*10
        pred = actual-2
        args = [times, actual, pred, pred-1, pred+1, ref]
        result = reliability_watch(*args)
        self.assertEqual(result['status'], 'needs_review')
        self.assertEqual(len(result['review_reasons']), 2)
        self.assertFalse(result['automatic_retraining'])
        self.assertEqual(reliability_watch(times[:7], actual[:7], pred[:7], pred[:7]-1, pred[:7]+1, ref)['status'], 'insufficient_labels')
        args[0] = times.delete(3).append(pd.DatetimeIndex([times[-1]+pd.Timedelta(minutes=15)]))
        self.assertEqual(reliability_watch(*args)['status'], 'non_contiguous_labels')

    def test_frozen_reader_detects_changed_results_without_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'summary.json').write_text('{"measured": true}')
            lock = {'summary_file': 'summary.json', 'files_sha256': {'summary.json': sha256_file(root/'summary.json')}}
            (root/'lock.json').write_text(json.dumps(lock))
            self.assertTrue(read_results(root, 'lock.json')['measured'])
            (root/'summary.json').write_text('{"measured": false}')
            with self.assertRaises(ValueError): read_results(root, 'lock.json')

    def test_saved_walk_forward_scores_match_predictions_and_never_overlap_train(self):
        folder = Path(__file__).resolve().parents[1]/'reports/steel/improvements_2026-10-16'
        scores = pd.read_csv(folder/'walk_forward_metrics.csv')
        for month in (6, 7, 8, 9):
            frame = pd.read_csv(folder/f'predictions_month_{month:02}.csv', parse_dates=['issue_time'])
            self.assertTrue(frame.issue_time.dt.month.eq(month).all())
            actual = frame[[f'actual_{15*k}m_kWh' for k in range(1, 5)]].to_numpy()
            for row in scores.loc[scores.month.eq(month)].itertuples():
                predicted = frame[[f'{row.model}_{15*k}m_kWh' for k in range(1, 5)]].to_numpy()
                self.assertAlmostEqual(float(np.abs(actual-predicted).mean()), row.mae_per_step_kWh, places=10)
                self.assertEqual(len(frame), row.valid_rows)


if __name__ == '__main__': unittest.main()
