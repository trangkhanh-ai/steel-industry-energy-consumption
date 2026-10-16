"""Giám sát model mới bằng nhãn đã đến, không dùng TEST làm reference.

Hai cửa sổ liên tục 7 ngày; ngưỡng rà soát là heuristic của prototype.
Không biến tín hiệu sai số/coverage thành chẩn đoán máy hoặc lệnh retrain.
"""
import numpy as np
import pandas as pd


def reliability_watch(times, actual, predicted, lower, upper, reference):
    n = len(actual)
    window = reference['window_labels']
    result = {'status': 'insufficient_labels', 'available_labels': n,
        'required_labels': 2*window, 'model_sha256': reference['model_sha256'],
        'reference_period': reference['reference_period'], 'automatic_retraining': False,
        'note': 'Heuristic; hai cửa sổ 7 ngày liên tục, không kiểm định drift hay nguyên nhân vật lý'}
    if n < 2*window:
        return result
    order = np.argsort(pd.DatetimeIndex(times).asi8)[-2*window:]
    t = pd.DatetimeIndex(times)[order]
    if not t.is_unique or not (np.diff(t.asi8) == pd.Timedelta(minutes=15).value).all():
        return {**result, 'status': 'non_contiguous_labels'}
    a, p, lo, hi = [np.asarray(v, dtype=float)[order] for v in (actual, predicted, lower, upper)]
    if not np.isfinite([a, p, lo, hi]).all() or (lo > hi).any():
        raise ValueError('Không giám sát từ nhãn hoặc dải lỗi')
    mae = [float(np.abs(a[i:i+window]-p[i:i+window]).mean()) for i in (0, window)]
    coverage = [float(((a[i:i+window] >= lo[i:i+window]) & (a[i:i+window] <= hi[i:i+window])).mean()) for i in (0, window)]
    mae_limit = reference['point_hour_mae_kWh']*reference['mae_factor']
    coverage_limit = max(0., reference['central_coverage']-reference['coverage_drop'])
    reasons = []
    if all(v > mae_limit for v in mae): reasons.append('point_mae_above_reference_both_windows')
    if all(v < coverage_limit for v in coverage): reasons.append('coverage_below_reference_both_windows')
    return {**result, 'status': 'needs_review' if reasons else 'no_review_trigger',
        'weekly_mae_kWh': mae, 'weekly_coverage': coverage,
        'mae_limit_kWh': mae_limit, 'coverage_limit': coverage_limit, 'review_reasons': reasons,
        'recommendation': 'Người phụ trách kiểm dữ liệu, phân tích lỗi rồi duyệt model mới; chưa có tự điều khiển'}
