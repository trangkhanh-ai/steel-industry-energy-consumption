"""Display-cache regressions using only released forecasts and measured UCI rows."""
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.steel.demo import ReplayDemo
from src.steel.probabilistic import attach_observed_actuals, DEFAULT_PROBABILISTIC_SUBDIR

ROOT = Path(__file__).resolve().parents[1]


class ReplayDisplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.demo = ReplayDemo(ROOT, sessions_dir=Path(self.temp.name),
            policy_path=ROOT/'reports/steel/release_2026-10-11/locked_policy.json',
            probabilistic_dir=ROOT/DEFAULT_PROBABILISTIC_SUBDIR)
        self.initial = self.demo.dispatch('reset', '2018-11-22T08:15')

    def tearDown(self):
        self.demo.close()
        self.temp.cleanup()

    def assert_reference(self, state):
        observed = self.demo.observed_rows(96)
        saved = {r['issue_time']: json.loads(r['probabilistic_json']) for r in
                 self.demo.monitor.db.execute('SELECT issue_time,probabilistic_json FROM forecasts')}
        for row in state['forecasts']:
            self.assertEqual(row['probabilistic'], attach_observed_actuals(saved[row['issue_time']], observed, self.demo.current))

    def test_cache_matches_reference_with_partial_and_matured_labels(self):
        self.assert_reference(self.initial)
        for _ in range(24):
            self.assert_reference(self.demo.dispatch('step'))
        self.assertLessEqual(len(self.demo._actual_display_cache),20)

    def test_withheld_real_reading_removes_previously_visible_actual(self):
        for _ in range(4): self.demo.dispatch('step')
        self.demo.history = self.demo.history.loc[
            self.demo.history.observation_time.ne(pd.Timestamp('2018-11-22T08:30'))].reset_index(drop=True)
        state = self.demo.dispatch('state')
        self.assert_reference(state)
        first = next(r['probabilistic'] for r in state['forecasts'] if r['issue_time']==self.initial['clock'])
        self.assertIsNone(first['intervals'][0]['actual_kWh'])
        self.assertIsNone(first['hour_total']['actual_kWh'])

    def test_invalid_meter_and_past_clock_never_reuse_future_actuals(self):
        for _ in range(4): self.demo.dispatch('step')
        row = self.demo.monitor.db.execute('SELECT * FROM forecasts ORDER BY id LIMIT 1').fetchone()
        result = self.demo._display_probabilistic(row,None,{})
        self.assertTrue(all(v['actual_kWh'] is None for v in result['intervals']))
        self.demo.current = pd.Timestamp(self.initial['clock'])
        self.assert_reference(self.demo.state())
        result = self.demo.state()['forecasts'][0]['probabilistic']
        self.assertIsNone(result['hour_total']['actual_kWh'])

    def test_returned_objects_and_reset_are_isolated(self):
        state = self.demo.dispatch('step')
        state['forecasts'][0]['probabilistic']['intervals'].clear()
        self.assert_reference(self.demo.dispatch('state'))
        state = self.demo.dispatch('reset','2018-11-22T08:15')
        self.assert_reference(state)
        self.assertIsNone(state['forecasts'][0]['probabilistic']['hour_total']['actual_kWh'])


if __name__=='__main__':
    unittest.main()
