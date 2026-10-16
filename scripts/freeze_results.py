"""Khóa artifact và số liệu sau kiểm chứng; từ chối ghi đè khóa đã công bố."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import pandas as pd
from scripts.run_probabilistic_forecast import save_json
from src.steel.preprocessing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'reports/steel/release_2026-10-16'
STUDY = ROOT/'reports/steel/improvements_2026-10-16'


def run():
    if (RUN/'results_lock.json').exists():
        raise FileExistsError('Không sửa khóa đã công bố; tạo release mới')
    new = json.loads((RUN/'metrics.json').read_text())
    old = json.loads((ROOT/'reports/steel/release_2026-10-14/metrics.json').read_text())
    summary = {'release': '2026-10-16', 'retrospective_not_new_blind_test': True,
        'dataset': 'UCI Steel Industry Energy Consumption; actual 2018 observations',
        'old_test_point': old['splits']['test']['point'], 'current_test_point': new['splits']['test']['point'],
        'test_quantiles': new['splits']['test']['quantiles'],
        'test_research_alert': new['splits']['test']['research_alert'],
        'walk_forward': pd.read_csv(STUDY/'walk_forward_summary.csv').to_dict('records'),
        'calibration_comparison': pd.read_csv(STUDY/'calibration_comparison.csv').to_dict('records'),
        'selected': {'point': 'HGB absolute_error; better in all 3 past folds and September',
            'features': '15 causal features; delta/calendar/rolling ablations saved',
            'quantile': 'Retain published central calibration; one-sided alternatives did not reduce FN',
            'tradeoff': 'TEST MAE improved; hour RMSE increased. No claim of universal improvement.'},
        'system': json.loads((RUN/'system_verification.json').read_text()),
        'industrial_value': 'Forecast and uncertain high-energy review support; human approval',
        'contract_power_limit_kW': None, 'energy_saved_kWh': None, 'money_saved': None,
        'limitations': ['One facility/year; no production context or tariffs',
            'Central coverage below nominal; high-actual-energy coverage low',
            'Alert origins overlap; not physical events or measured lead time',
            'No equipment control; PSI/MAE/coverage triggers are heuristics']}
    save_json(RUN/'results_summary.json', summary)
    folders = [RUN, STUDY, ROOT/'reports/steel/error_analysis_2026-10-16']
    paths = [p for folder in folders for p in folder.rglob('*') if p.is_file() and p.name!='results_lock.json']
    paths += [ROOT/'data/steel/raw/Steel_industry_data.csv',
              ROOT/'reports/steel/release_2026-10-14/forecast_bundle.joblib',
              ROOT/'reports/steel/release_2026-10-14/test_predictions.csv']
    paths += [ROOT/name for name in ('scripts/run_improvement_study.py', 'scripts/run_probabilistic_forecast.py',
                                   'src/steel/probabilistic.py', 'src/steel/reliability.py', 'src/steel/monitoring.py')]
    save_json(RUN/'results_lock.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_parent_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'summary_file': str((RUN/'results_summary.json').relative_to(ROOT)).replace('\\', '/'),
        'files_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): sha256_file(p) for p in sorted(set(paths))},
        'integrity_only_not_cryptographic_authenticity': True, 'retraining_required_to_read': False})


if __name__ == '__main__': run()
