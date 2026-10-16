"""Đo backend phiên dài từ CSV thật; không dùng số này thay latency trình duyệt."""
import argparse
import cProfile
import json
from pathlib import Path
import pstats
import tempfile
from time import perf_counter

import numpy as np
from src.steel.demo import ReplayDemo
from src.steel.probabilistic import DEFAULT_PROBABILISTIC_SUBDIR

ROOT = Path(__file__).resolve().parents[1]


def run(output, steps):
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='steel-benchmark-') as temp:
        demo = ReplayDemo(ROOT, sessions_dir=Path(temp),
            policy_path=ROOT/'reports/steel/release_2026-10-11/locked_policy.json',
            probabilistic_dir=ROOT/DEFAULT_PROBABILISTIC_SUBDIR)
        try:
            demo.dispatch('reset', '2018-11-01T00:00')
            timings = []
            for _ in range(steps):
                begin = perf_counter()
                state = demo.dispatch('step')
                timings.append(1000*(perf_counter()-begin))
            profiler = cProfile.Profile()
            profiler.enable()
            for _ in range(10): demo.dispatch('step')
            profiler.disable()
            with (output/'profile.txt').open('w', encoding='utf-8') as stream:
                pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(30)
            result = {'steps': steps, 'timing_scope': 'Python backend dispatch; no HTTP/browser',
                'p50_ms': float(np.quantile(timings, .5)), 'p95_ms': float(np.quantile(timings, .95)),
                'first_96_p95_ms': float(np.quantile(timings[:96], .95)),
                'last_96_p95_ms': float(np.quantile(timings[-96:], .95)),
                'matured_rows': state['metrics']['probabilistic_metrics']['rows'],
                'process_rss_bytes': state['performance']['process_rss_bytes'],
                'new_model_watch': state['metrics']['probabilistic_metrics'].get('performance_watch'),
                'timings_ms': timings}
            (output/'timings.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            print(json.dumps({k:v for k,v in result.items() if k!='timings_ms'}, indent=2), flush=True)
        finally:
            demo.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--steps', type=int, default=1440)
    a = p.parse_args()
    if not 96 <= a.steps <= 5000: p.error('steps phải từ 96 đến 5000')
    run(a.output_dir.resolve(), a.steps)
