"""Dự báo 4 bản đo và phân vị tổng/đỉnh điện năng. Không giá điện/P_limit giả."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, root_mean_squared_error
from threadpoolctl import ThreadpoolController

from .live_features import latest_features
from .evaluation import evaluate_alerts
from .preprocessing import BASE_FEATURES, FREQUENCY, sha256_file

QUANTILES = (0.1, 0.5, 0.9)
TARGETS = tuple(f'usage_next_{15*k}m_kWh' for k in range(1, 5))
CONTRACT = 'steel_measured_energy_4_steps_quantiles_v1'
DEFAULT_PROBABILISTIC_SUBDIR = Path('reports/steel/release_2026-10-16')


def future_measurements(raw, origins):
    """Nhãn OFFLINE từ Usage thật. Không gọi hàm này để phát dự báo online."""
    meter = raw.set_index('observation_time').Usage_kWh
    times = pd.DatetimeIndex(origins)
    labels = np.column_stack([meter.reindex(times+k*FREQUENCY).to_numpy() for k in range(1, 5)])
    if not np.isfinite(labels).all() or (labels < 0).any():
        raise ValueError('Thiếu số đo thực tương lai; không điền/bịa nhãn')
    return labels


def point_prediction(bundle, features):
    x = np.asarray(features, dtype=float)
    if bundle['point_method'] == 'last_value':
        predicted = np.repeat(x[:, 0, None], 4, axis=1)
    elif bundle['point_method'] == 'ridge':
        predicted = bundle['point_model'].predict(x)
    elif bundle['point_method'] == 'hgb':
        predicted = np.column_stack([model.predict(x) for model in bundle['point_model']])
    else:
        raise ValueError('Phương pháp point không được hỗ trợ')
    if predicted.shape != (len(x), 4) or not np.isfinite(predicted).all():
        raise ValueError('Model phải trả 4 giá trị hữu hạn cho mỗi origin')
    return np.maximum(predicted, 0)


def quantile_prediction(models, features):
    raw = np.column_stack([model.predict(np.asarray(features, dtype=float)) for model in models])
    if raw.shape != (len(features), 3) or not np.isfinite(raw).all():
        raise ValueError('Model phân vị trả dữ liệu lỗi')
    crossing = np.any(np.diff(raw, axis=1) < 0, axis=1)
    # Rearrangement công bố trước đánh giá; không sửa theo actual của test.
    ordered = np.sort(np.maximum(raw, 0), axis=1)
    return ordered, crossing


def calibration_margin(actual, quantiles):
    """CQR-style score trên tháng 10; TS phụ thuộc nên không hứa coverage 80%."""
    y, q = np.asarray(actual, dtype=float), np.asarray(quantiles, dtype=float)
    if q.shape != (len(y), 3) or not len(y) or not np.isfinite(q).all() or not np.isfinite(y).all():
        raise ValueError('Nhãn/phân vị calibration không hợp lệ')
    score = np.maximum.reduce([q[:, 0]-y, y-q[:, 2], np.zeros(len(y))])
    level = min(1.0, np.ceil((len(y)+1)*0.8)/len(y))
    return float(np.quantile(score, level, method='higher'))


def interval_metrics(actual, quantiles, margin):
    y, q = np.asarray(actual), np.asarray(quantiles)
    lower, upper = np.maximum(q[:, 0]-margin, 0), q[:, 2]+margin
    return {
        'rows': len(y), 'p50_mae_kWh': float(mean_absolute_error(y, q[:, 1])),
        'p50_rmse_kWh': float(root_mean_squared_error(y, q[:, 1])),
        'pinball_loss': {str(a): float(mean_pinball_loss(y, q[:, i], alpha=a)) for i, a in enumerate(QUANTILES)},
        'empirical_fraction_y_leq_quantile': {str(a): float(np.mean(y <= q[:, i])) for i, a in enumerate(QUANTILES)},
        'raw_p10_p90_coverage': float(np.mean((y >= q[:, 0]) & (y <= q[:, 2]))),
        'raw_mean_width_kWh': float(np.mean(q[:, 2]-q[:, 0])),
        'calibrated_coverage': float(np.mean((y >= lower) & (y <= upper))),
        'calibrated_mean_width_kWh': float(np.mean(upper-lower)),
        'calibration_margin_kWh': margin, 'nominal_central_coverage': 0.8,
        'coverage_guaranteed': False,
    }


def research_alert_metrics(actual, quantiles, times, threshold, margin):
    """Cảnh báo biên trên so với p95 TRAIN; không phải xác suất hay phí hợp đồng."""
    full = evaluate_alerts(actual, np.asarray(quantiles)[:, 2], times, threshold, margin)
    keys = ('rows', 'tp', 'fp', 'fn', 'tn', 'alert_rows', 'precision', 'recall', 'f1',
            'false_positive_rate', 'false_discovery_rate', 'roc_auc_ranking',
            'true_origin_episodes', 'alert_origin_episodes', 'detected_true_episodes',
            'overlap_episode_precision', 'overlap_episode_recall', 'episode_note')
    return {**{key: full[key] for key in keys}, 'threshold_kWh': threshold,
            'definition': 'actual hour total > TRAIN p95; alert = calibrated quantile upper > same threshold',
            'contract_power_limit_available': False, 'savings_measured': False,
            'lead_time_measured': False}


class ProbabilisticForecaster:
    def __init__(self, run_dir, source_sha256):
        folder = Path(run_dir).resolve()
        self.metadata = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        m = self.metadata
        if m['contract'] != CONTRACT or m['features'] != list(BASE_FEATURES):
            raise ValueError('Sai hợp đồng feature/target phân vị')
        if m['source_sha256'] != source_sha256 or m['sklearn'] != sklearn.__version__:
            raise ValueError('Nguồn hoặc sklearn không khớp model phân vị')
        path = (folder/m['model_file']).resolve()
        if not path.is_relative_to(folder) or sha256_file(path) != m['model_sha256']:
            raise ValueError('Artifact phân vị không khớp checksum/đường dẫn')
        if sha256_file(folder/'protocol.json') != m['protocol_sha256']:
            raise ValueError('Protocol không khớp checksum')
        if m['nominal_central_coverage'] != .8 or m['coverage_guaranteed'] is not False:
            raise ValueError('P10-P90 chỉ có mức danh nghĩa 80%, không bảo đảm coverage')
        for field in ('calibration_margins_kWh', 'train_research_thresholds_kWh'):
            for target in ('hour_total', 'max_interval'):
                value = m[field][target]
                if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
                    raise ValueError('Margin/ngưỡng phải hữu hạn, không âm')
        if m['train_research_thresholds_kWh']['hour_total'] <= 0:
            raise ValueError('Ngưỡng nghiên cứu phải dương')
        if m['valid_from'] != '2018-11-01T00:00:00':
            raise ValueError('Protocol này calibration tháng 10, chỉ dùng từ tháng 11')
        self.bundle = joblib.load(path)  # Chỉ artifact do nhóm tạo và tin cậy.
        self.model_path = path
        if m['point_method'] not in ('last_value', 'ridge', 'hgb') or self.bundle['point_method'] != m['point_method']:
            raise ValueError('Phương pháp point không khớp manifest')
        if m['point_method'] == 'hgb' and len(self.bundle['point_model']) != 4:
            raise ValueError('Cần bốn model point cho bốn bước')
        if m['point_method'] == 'hgb' and 'point_loss' in m:
            if m['point_loss'] not in ('squared_error', 'absolute_error') or any(model.loss != m['point_loss'] for model in self.bundle['point_model']):
                raise ValueError('Loss point không khớp model đã khai báo')
        for target in ('hour_total', 'max_interval'):
            models = self.bundle['quantile_models'][target]
            if len(models) != 3:
                raise ValueError('Cần đủ P10/P50/P90')
            for model, alpha in zip(models, QUANTILES):
                if model.loss != 'quantile' or model.quantile != alpha or model.n_features_in_ != len(BASE_FEATURES):
                    raise ValueError('Model không có đúng loss/phân vị/feature đã khai báo')
        self.valid_from = pd.Timestamp(m['valid_from'])
        self.monitoring_reference = None
        if 'monitoring_reference_file' in m:
            reference_path = (folder/m['monitoring_reference_file']).resolve()
            if not reference_path.is_relative_to(folder) or sha256_file(reference_path) != m['monitoring_reference_sha256']:
                raise ValueError('Reference giám sát không khớp checksum/đường dẫn')
            reference = json.loads(reference_path.read_text(encoding='utf-8'))
            if reference['model_sha256'] != m['model_sha256'] or reference['source_sha256'] != source_sha256:
                raise ValueError('Reference giám sát thuộc model/nguồn khác')
            self.monitoring_reference = reference
        self.thread_controller = ThreadpoolController()

    def predict_features(self, features, issue_time):
        t = pd.Timestamp(issue_time)
        if pd.isna(t) or t.tzinfo is not None or t != t.floor('15min') or t < self.valid_from:
            raise ValueError('Chỉ dùng phân vị đã calibration từ tháng 11/2018')
        if list(features.columns) != list(BASE_FEATURES) or len(features) != 1:
            raise ValueError('Cần đúng một hàng 15 feature')
        x = features.to_numpy(dtype=float)
        if not np.isfinite(x).all():
            raise ValueError('Không phát dự báo từ feature lỗi')
        with self.thread_controller.limit(limits=1):
            points = point_prediction(self.bundle, x)[0]
            result = {}
            for target in ('hour_total', 'max_interval'):
                q, crossing = quantile_prediction(self.bundle['quantile_models'][target], x)
                low, med, high = map(float, q[0])
                margin = self.metadata['calibration_margins_kWh'][target]
                result[target] = {
                    'p10_kWh': low, 'p50_kWh': med, 'p90_kWh': high,
                    'calibrated_lower_kWh': max(0., low-margin), 'calibrated_upper_kWh': high+margin,
                    'raw_width_kWh': high-low, 'crossing_reordered': bool(crossing[0]),
                    'actual_kWh': None,
                }
        threshold = self.metadata['train_research_thresholds_kWh']['hour_total']
        result.update(
            issue_time=t.isoformat(), forecast_end=(t+4*FREQUENCY).isoformat(),
            intervals=[{'time': (t+k*FREQUENCY).isoformat(), 'predicted_kWh': float(points[k-1]),
                        'actual_kWh': None} for k in range(1, 5)],
            point_method=self.metadata['point_method'], model_sha256=self.metadata['model_sha256'],
            forecast_sum_of_interval_points_kWh=float(points.sum()),
            research_alert={'alert': result['hour_total']['calibrated_upper_kWh'] > threshold,
                            'threshold_kWh': threshold, 'threshold_source': 'p95 TRAIN hour total',
                            'method': 'calibrated quantile upper bound', 'contract_power_limit': None,
                            'automatic_equipment_action': False},
            source='UCI Steel · Usage_kWh đo thật năm 2018',
            note='Phân vị ước lượng. P50 là trung vị; tổng phân vị từng bước không được gọi là phân vị tổng giờ. Không có P_limit/biểu giá hoặc số tiết kiệm thực.',
        )
        return result

    def predict(self, history, issue_time):
        return self.predict_features(latest_features(history, issue_time), issue_time)


def observed_measurements(observed, as_of):
    """Kiểm số đo đã đến; thiếu khoảng được giữ thiếu, không điền số."""
    now = pd.Timestamp(as_of)
    times = pd.to_datetime(observed.observation_time, errors='raise')
    values = pd.to_numeric(observed.Usage_kWh, errors='raise').to_numpy(dtype=float)
    if pd.isna(now) or now.tzinfo is not None or now != now.floor('15min'):
        raise ValueError('Thời điểm actual không hợp lệ')
    if (times.dt.tz is not None or times.isna().any() or not times.is_unique
            or not times.is_monotonic_increasing or not times.eq(times.dt.floor('15min')).all()
            or times.gt(now).any() or not np.isfinite(values).all() or (values < 0).any()):
        raise ValueError('Không đọc actual tương lai hoặc số đo lỗi')
    return pd.Series(values, index=pd.DatetimeIndex(times))


def attach_actuals_from_meter(saved, meter, as_of):
    """Dùng Series đã kiểm bằng observed_measurements; không sửa bản dự báo gốc."""
    result = json.loads(json.dumps(saved, allow_nan=False))
    now, issue = pd.Timestamp(as_of), pd.Timestamp(result['issue_time'])
    expected = [(issue+k*FREQUENCY).isoformat() for k in range(1, 5)]
    if [step['time'] for step in result['intervals']] != expected:
        raise ValueError('Bốn thời điểm forecast không khớp origin')
    result['hour_total']['actual_kWh'] = None
    result['max_interval']['actual_kWh'] = None
    available = []
    for step in result['intervals']:
        when = pd.Timestamp(step['time'])
        actual = float(meter.loc[when]) if when <= now and when in meter.index else None
        step['actual_kWh'] = actual
        available.append(actual)
    if all(value is not None for value in available):
        result['hour_total']['actual_kWh'] = float(sum(available))
        result['max_interval']['actual_kWh'] = float(max(available))
    return result


def attach_observed_actuals(saved, observed, as_of):
    """Chỉ gắn actual từ các hàng đã đến as_of, không ghi đè forecast đã lưu."""
    return attach_actuals_from_meter(saved, observed_measurements(observed, as_of), as_of)
