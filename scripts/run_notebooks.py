"""Execute every public notebook with this Python; save results in a fresh folder.

Requires requirements-notebooks.txt. Does not overwrite source notebooks/models.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_client import AsyncKernelManager

ROOT = Path(__file__).resolve().parents[1]


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    artifacts = [ROOT / 'data/steel/raw/Steel_industry_data.csv',
                 *ROOT.glob('reports/steel/**/models/*.joblib'),
                 *ROOT.glob('reports/steel/release_*/forecast_bundle.joblib'),
                 *ROOT.glob('reports/steel/release_*/locked_policy.json'),
                 *ROOT.glob('notebooks/artifacts/*')]
    before = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in artifacts if p.is_file() and p.suffix != '.md'}
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MPLBACKEND'] = 'Agg'
    results = []
    with tempfile.TemporaryDirectory(prefix='steel-kernel-') as temp:
        kernel = Path(temp) / 'steel-check'
        kernel.mkdir()
        (kernel / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Steel verification', 'language': 'python'}), encoding='utf-8')
        manager = KernelSpecManager(kernel_dirs=[temp])
        class LocalKernelManager(AsyncKernelManager):
            def __init__(self, **kwargs):
                super().__init__(kernel_spec_manager=manager, **kwargs)

        for path in sorted((ROOT / 'notebooks').glob('*.ipynb')):
            book = nbformat.read(path, as_version=4)
            client = NotebookClient(book, timeout=180, kernel_name='steel-check',
                                    kernel_manager_class=LocalKernelManager,
                                    resources={'metadata': {'path': str(ROOT)}})
            client.execute()
            nbformat.write(book, output / path.name)
            results.append({'notebook': path.name, 'passed': True,
                            'code_cells': sum(c.cell_type == 'code' for c in book.cells)})
            print('PASS', path.name, flush=True)
    after = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in before}
    if before != after:
        raise AssertionError('Notebook changed a frozen input artifact')
    summary = {'passed': True, 'python': sys.version.split()[0], 'notebooks': results,
               'frozen_artifacts_unchanged': True}
    (output / 'verification.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run(args.output_dir.resolve())
