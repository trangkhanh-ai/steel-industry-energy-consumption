"""Verify monitoring from real replay labels and frozen calibration evidence.

No training, fabricated observations, or changes to released artifacts.
Output must be a new directory outside the frozen release folders.
"""
import argparse
import cProfile
import json
from pathlib import Path
import pstats
from time import perf_counter

import numpy as np
import pandas as pd

from src.steel.demo import ReplayDemo
from src.steel.frozen_results import read_results
from src.steel.probabilistic import DEFAULT_PROBABILISTIC_SUBDIR
from src.steel.reliability import reliability_watch

ROOT = Path(__file__).resolve().parents[1]


def run(output, steps=1440):
    output = Path(output).resolve()
    if output.is_relative_to(ROOT/'reports/steel'):
        raise ValueError('Use a separate local evidence directory, not released reports')
    if not 1348 <= steps <= 5000:
        raise ValueError('Need 1348–5000 steps for two weeks and a withheld-label gap check')
    output.mkdir(parents=True, exist_ok=False)
    read_results(ROOT)
    run_dir = ROOT/DEFAULT_PROBABILISTIC_SUBDIR
    reference = json.loads((run_dir/'monitoring_reference.json').read_text(encoding='utf-8'))
    calibration = pd.read_csv(run_dir/'calibration_predictions.csv')
    actual = calibration.hour_total_actual_kWh.to_numpy()
    points = calibration[[f'predicted_{k}m_kWh' for k in (15,30,45,60)]].sum(axis=1).to_numpy()
    coverage = ((actual >= calibration.hour_total_calibrated_lower_kWh) &
                (actual <= calibration.hour_total_calibrated_upper_kWh)).mean()
    assert len(calibration) == reference['rows']
    assert pd.to_datetime(calibration.issue_time).dt.month.eq(10).all()
    assert abs(float(np.abs(points-actual).mean())-reference['point_hour_mae_kWh']) < 1e-10
    assert abs(float(coverage)-reference['central_coverage']) < 1e-10
    observations = {}
    demo = ReplayDemo(ROOT, sessions_dir=output/'sessions',
        policy_path=ROOT/'reports/steel/release_2026-10-11/locked_policy.json', probabilistic_dir=run_dir)
    try:
        state = demo.dispatch('reset','2018-11-01T00:00')
        observations['0'] = state['metrics']['probabilistic_metrics']['performance_watch']
        timings=[]
        for i in range(1, steps+1):
            begin=perf_counter()
            state=demo.dispatch('step')
            timings.append(1000*(perf_counter()-begin))
            metric=state['metrics']['probabilistic_metrics']
            count=metric['rows']
            assert metric['performance_watch']['available_labels']==count
            if count in (1,1343,1344) or i==steps:
                observations[str(count)]=metric['performance_watch']
            if i%192==0:
                print(f'{i}/{steps} steps; {count} matured labels',flush=True)
        assert observations['1343']['status']=='insufficient_labels'
        assert observations['1344']['status'] in ('no_review_trigger','needs_review')
        assert all(w['model_sha256']==reference['model_sha256'] and not w['automatic_retraining'] for w in observations.values())
        rows=demo.monitor.db.execute('SELECT issue_time,actual_kWh,probabilistic_json FROM forecasts WHERE evaluated_as_of<=? ORDER BY issue_time',(state['clock'],)).fetchall()
        predictions=[json.loads(r['probabilistic_json']) for r in rows]
        times=[r['issue_time'] for r in rows]
        actual=np.array([r['actual_kWh'] for r in rows])
        points=np.array([q['forecast_sum_of_interval_points_kWh'] for q in predictions])
        lower=np.array([q['hour_total']['calibrated_lower_kWh'] for q in predictions])
        upper=np.array([q['hour_total']['calibrated_upper_kWh'] for q in predictions])
        expected_mae=[]; expected_coverage=[]
        for begin,end in ((-1344,-672),(-672,None)):
            expected_mae.append(float(np.abs(points[begin:end]-actual[begin:end]).mean()))
            expected_coverage.append(float(((actual[begin:end]>=lower[begin:end]) & (actual[begin:end]<=upper[begin:end])).mean()))
        watch=state['metrics']['probabilistic_metrics']['performance_watch']
        np.testing.assert_allclose(watch['weekly_mae_kWh'],expected_mae,rtol=0,atol=1e-10)
        np.testing.assert_allclose(watch['weekly_coverage'],expected_coverage,rtol=0,atol=1e-10)
        assert abs(watch['mae_limit_kWh']-reference['point_hour_mae_kWh']*1.2)<1e-10
        assert abs(watch['coverage_limit']-(reference['central_coverage']-.1))<1e-10
        # Remove one real matured record, retain >=1344 labels: detect the gap.
        keep=np.arange(len(times))!=len(times)-100
        gap=reliability_watch(np.array(times)[keep],actual[keep],points[keep],lower[keep],upper[keep],reference)
        assert gap['status']=='non_contiguous_labels'
        early=demo.monitor.snapshot(pd.Timestamp('2018-11-01T00:00'))['probabilistic_metrics']['performance_watch']
        assert early['status']=='insufficient_labels' and early['available_labels']==0
        assert demo.monitor.snapshot(pd.Timestamp(state['clock']))['probabilistic_metrics']['performance_watch']==watch
        profiler=cProfile.Profile()
        with profiler:
            for _ in range(10): demo.dispatch('step')
        with (output/'profile.txt').open('w',encoding='utf-8') as f:
            pstats.Stats(profiler,stream=f).sort_stats('cumulative').print_stats(40)
        result={'passed':True,'steps':steps,'matured_labels':len(rows),'reference':reference,
            'checkpoints':observations,'gap_check':gap,'past_snapshot':early,
            'independent_weekly_mae_kWh':expected_mae,'independent_weekly_coverage':expected_coverage,
            'backend_p50_ms':float(np.quantile(timings,.5)),'backend_p95_ms':float(np.quantile(timings,.95)),
            'first_96_p95_ms':float(np.quantile(timings[:96],.95)),'last_96_p95_ms':float(np.quantile(timings[-96:],.95)),
            'timing_scope':'Sequential Python dispatch; no HTTP or browser; profiling excluded',
            'no_synthetic_measurements':True,'automatic_retraining':False}
        (output/'monitoring.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        (output/'timings.json').write_text(json.dumps(timings),encoding='utf-8')
        read_results(ROOT)
        print(json.dumps({k:result[k] for k in ('passed','steps','matured_labels','backend_p50_ms','backend_p95_ms')},indent=2),flush=True)
        return result
    finally:
        demo.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=1440)
    args=parser.parse_args()
    run(args.output_dir,args.steps)
