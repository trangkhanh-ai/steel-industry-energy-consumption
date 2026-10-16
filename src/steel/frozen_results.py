"""Đọc kết quả đã khóa: không fit model, không cần chạy notebook."""
import json
from pathlib import Path
from .preprocessing import sha256_file

LOCK = Path('reports/steel/release_2026-10-16/results_lock.json')


def read_results(root, lock_path=LOCK):
    root = Path(root).resolve()
    lock = json.loads((root/lock_path).read_text(encoding='utf-8'))
    for name, expected in lock['files_sha256'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or sha256_file(path) != expected:
            raise ValueError('Kết quả không khớp khóa: '+name)
    summary_path = lock['summary_file']
    if summary_path not in lock['files_sha256']:
        raise ValueError('Summary chưa được khóa')
    return json.loads((root/summary_path).read_text(encoding='utf-8'))
