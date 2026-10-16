"""Kiểm từ CSV thật: đúng nhãn, không lộ tương lai, không dựng dữ liệu vận hành."""
from pathlib import Path
import json
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.steel.demo import ReplayDemo
from src.steel.duy_forecasting import forecast_features
from src.steel.live_features import latest_features
from src.steel.preprocessing import FREQUENCY, load_official_raw
from src.steel.probabilistic import (
    ProbabilisticForecaster, attach_observed_actuals, calibration_margin,
    future_measurements, interval_metrics,
    DEFAULT_PROBABILISTIC_SUBDIR,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / DEFAULT_PROBABILISTIC_SUBDIR
POLICY = ROOT/'reports/steel/release_2026-10-11/locked_policy.json'


class MeasuredFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.audit = load_official_raw(ROOT/'data/steel/raw/Steel_industry_data.csv')

    def test_fast_features_equal_frozen_arithmetic_across_all_months(self):
        batch = forecast_features(self.raw)
        for i in range(672, len(self.raw), 1493):
            history = self.raw.iloc[i-672:i+1]
            fast = latest_features(history, history.observation_time.iloc[-1])
            np.testing.assert_array_equal(fast.to_numpy()[0], batch.iloc[i].to_numpy())

    def test_features_reject_future_or_gap_in_measured_history(self):
        history = self.raw.iloc[1000:1673]
        with self.assertRaises(ValueError):
            latest_features(history, history.observation_time.iloc[-2])
        with self.assertRaises(ValueError):
            latest_features(history.drop(history.index[300]), history.observation_time.iloc[-1])

    def test_labels_are_the_four_future_csv_records(self):
        i = 2000
        labels = future_measurements(self.raw, [self.raw.observation_time.iloc[i]])
        np.testing.assert_array_equal(labels[0], self.raw.Usage_kWh.iloc[i+1:i+5])
        with self.assertRaises(ValueError):
            future_measurements(self.raw, [self.raw.observation_time.iloc[-1]])

    def test_margin_and_coverage_recompute_from_measured_labels(self):
        # Hệ thống kiểm thuật toán, không coi các khoảng này là dự báo của nhà máy.
        actual = self.raw.Usage_kWh.iloc[1000:1100].to_numpy()
        q = np.column_stack((actual*.8, actual*.95, actual*1.05))
        margin = calibration_margin(actual, q)
        score = interval_metrics(actual, q, margin)
        self.assertEqual(score['nominal_central_coverage'], .8)
        self.assertFalse(score['coverage_guaranteed'])
        expected = np.mean((actual >= np.maximum(q[:, 0]-margin, 0)) & (actual <= q[:, 2]+margin))
        self.assertEqual(score['calibrated_coverage'], expected)


@unittest.skipUnless((RUN/'manifest.json').exists(), 'Train scripts.run_probabilistic_forecast to enable real-artifact integration checks')
class ProbabilisticIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.audit = load_official_raw(ROOT/'data/steel/raw/Steel_industry_data.csv')
        cls.model = ProbabilisticForecaster(RUN, cls.audit['source_sha256'])
        cls.csv = pd.read_csv(RUN/'test_predictions.csv')

    def forecast(self, origin):
        time = pd.Timestamp(origin)
        history = self.raw.loc[self.raw.observation_time.between(time-672*FREQUENCY, time)]
        return self.model.predict(history, time)

    def test_online_predictions_match_saved_real_test_rows(self):
        for i in (0, 73, 500, 2500, len(self.csv)-1):
            row = self.csv.iloc[i]
            predicted = self.forecast(row.issue_time)
            for k, step in enumerate(predicted['intervals'], 1):
                self.assertAlmostEqual(step['predicted_kWh'], row[f'predicted_{15*k}m_kWh'], places=10)
                self.assertIsNone(step['actual_kWh'])
            for target in ('hour_total', 'max_interval'):
                for percentile in (10, 50, 90):
                    self.assertAlmostEqual(predicted[target][f'p{percentile}_kWh'], row[f'{target}_p{percentile}_kWh'], places=10)
            self.assertIsNone(predicted['research_alert']['contract_power_limit'])
            self.assertFalse(predicted['research_alert']['automatic_equipment_action'])

    def test_actual_appears_only_as_records_arrive_and_missing_stays_unknown(self):
        origin = pd.Timestamp(self.csv.issue_time.iloc[0])
        saved = self.forecast(origin)
        for k in range(5):
            now = origin+k*FREQUENCY
            observed = self.raw.loc[self.raw.observation_time.between(origin, now)]
            updated = attach_observed_actuals(saved, observed, now)
            self.assertEqual(sum(row['actual_kWh'] is not None for row in updated['intervals']), k)
            if k < 4:
                self.assertIsNone(updated['hour_total']['actual_kWh'])
        actual = future_measurements(self.raw, [origin])[0]
        self.assertEqual(updated['hour_total']['actual_kWh'], float(actual.sum()))
        self.assertEqual(updated['max_interval']['actual_kWh'], float(actual.max()))
        missing = observed.loc[observed.observation_time.ne(origin+FREQUENCY)]
        updated = attach_observed_actuals(updated, missing, now)
        self.assertIsNone(updated['hour_total']['actual_kWh'])
        self.assertIsNone(updated['max_interval']['actual_kWh'])
        self.assertTrue(all(row['actual_kWh'] is None for row in saved['intervals']))
        with self.assertRaises(ValueError):
            attach_observed_actuals(saved, observed, origin)
        malformed = observed.copy()
        malformed.loc[malformed.index[-1], 'observation_time'] += pd.Timedelta(minutes=1)
        with self.assertRaises(ValueError):
            attach_observed_actuals(saved, malformed, now+FREQUENCY)

    def test_manifest_source_margin_protocol_and_artifact_rejected_if_changed(self):
        with self.assertRaises(ValueError):
            ProbabilisticForecaster(RUN, 'wrong-source')
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for name in ('manifest.json', 'protocol.json', 'forecast_bundle.joblib', 'monitoring_reference.json'):
                shutil.copy2(RUN/name, folder/name)
            original = json.loads((folder/'manifest.json').read_text())
            for field, value in (('valid_from', '2018-10-01T00:00:00'), ('coverage_guaranteed', True),
                                 ('protocol_sha256', 'wrong-hash'), ('model_sha256', 'wrong-hash'),
                                 ('calibration_margins_kWh', {'hour_total': -1, 'max_interval': 0}),
                                 ('monitoring_reference_sha256', 'wrong-hash'), ('point_loss', 'squared_error')):
                with self.subTest(field=field):
                    (folder/'manifest.json').write_text(json.dumps({**original, field: value}))
                    with self.assertRaises(ValueError):
                        ProbabilisticForecaster(folder, self.audit['source_sha256'])

    def test_replay_and_human_context_preserve_predictions_before_actuals(self):
        with tempfile.TemporaryDirectory() as temp:
            demo = ReplayDemo(ROOT, sessions_dir=Path(temp), policy_path=POLICY, probabilistic_dir=RUN)
            try:
                initial = demo.dispatch('reset', '2018-11-01T08:00')
                forecast = initial['current_forecast']['probabilistic']
                self.assertIsNotNone(forecast)
                watch = initial['metrics']['probabilistic_metrics']['performance_watch']
                self.assertEqual(watch['status'], 'insufficient_labels')
                self.assertEqual(watch['available_labels'], 0)
                self.assertEqual(watch['model_sha256'], forecast['model_sha256'])
                self.assertEqual(initial['probabilistic_info']['model_sha256'], forecast['model_sha256'])
                demo.dispatch('decision', decision=dict(session=initial['session'], clock=initial['clock'],
                    request_id=initial['current_forecast']['request_id'], operator='Unit test', action='inspect',
                    note='Ghi nhận thử nghiệm phần mềm, không phải hành động công nghiệp', token='test-context'))
                for k in range(4):
                    state = demo.dispatch('step')
                    saved = next(row['probabilistic'] for row in state['forecasts'] if row['issue_time']==initial['clock'])
                    self.assertEqual(sum(row['actual_kWh'] is not None for row in saved['intervals']), k+1)
                context = state['decisions'][0]['context']['probabilistic']
                self.assertEqual(context, forecast)
                self.assertIsNone(context['hour_total']['actual_kWh'])
                full = demo.monitor.snapshot(demo.current)
                self.assertEqual(full['alert_metrics_status'], 'computed')
                self.assertEqual(state['metrics']['alert_metrics_status'], 'available_on_demand')
                self.assertEqual(full['probabilistic_metrics']['rows'], 1)
                self.assertEqual(full['probabilistic_metrics']['performance_watch']['available_labels'], 1)
                self.assertEqual(full['probabilistic_metrics']['performance_watch']['required_labels'], 1344)
                self.assertEqual(full, demo.monitor.snapshot(demo.current))
                earlier = demo.monitor.snapshot(pd.Timestamp(initial['clock']))
                self.assertEqual(earlier['probabilistic_metrics']['rows'], 0)
                self.assertEqual(earlier['probabilistic_metrics']['performance_watch']['available_labels'], 0)
                self.assertEqual(full['probabilistic_metrics']['hour_total']['rows'], 1)
                self.assertEqual(state['metrics']['probabilistic_metrics']['point_hour']['rows'], 1)
                self.assertAlmostEqual(state['metrics']['probabilistic_metrics']['point_hour']['mae_kWh'],
                    abs(forecast['forecast_sum_of_interval_points_kWh']-saved['hour_total']['actual_kWh']))
                self.assertIsNone(state['metrics']['probabilistic_metrics']['research_alert'])
                self.assertEqual(demo.monitor.db.execute('PRAGMA synchronous').fetchone()[0], 2)
                reset = demo.dispatch('reset', '2018-11-01T08:00')
                self.assertNotEqual(reset['session'], initial['session'])
                self.assertIsNone(reset['forecasts'][0]['probabilistic']['hour_total']['actual_kWh'])
            finally:
                demo.close()

    def test_missing_extension_never_generates_dummy_quantiles(self):
        with tempfile.TemporaryDirectory() as temp:
            demo = ReplayDemo(ROOT, sessions_dir=Path(temp)/'sessions', probabilistic_dir=Path(temp)/'missing')
            try:
                state = demo.dispatch('reset', '2018-11-01T08:00')
                self.assertEqual(state['probabilistic_info']['status'], 'unavailable')
                self.assertIsNone(state['current_forecast']['probabilistic'])
                self.assertTrue(state['current_forecast']['probabilistic_error'])
                self.assertIsNotNone(state['current_forecast']['prediction'])
            finally:
                demo.close()


if __name__ == '__main__':
    unittest.main()
