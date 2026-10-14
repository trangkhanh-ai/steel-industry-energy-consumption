"""Một hàng feature hiện tại: cùng phép tính NumPy với bản Duy đã khóa."""
import numpy as np
import pandas as pd

from .duy_forecasting import validate_history
from .preprocessing import BASE_FEATURES, FREQUENCY


def latest_features(history, issue_time):
    """Kiểm lịch sử; chỉ giảm các cửa sổ cuối, không dựng 673 hàng feature."""
    clean = validate_history(history)
    t = pd.Timestamp(issue_time)
    if pd.isna(t) or t.tzinfo is not None or t != t.floor('15min'):
        raise ValueError('Thời điểm phải đúng lưới 15 phút của dataset')
    if len(clean) < 673 or clean.observation_time.iloc[-1] != t:
        raise ValueError('Cần ít nhất 673 bản đo, kết thúc đúng issue_time')
    usage = clean.Usage_kWh.to_numpy()
    values = {f'usage_lag_{lag}': usage[-1-lag] for lag in (0, 1, 4, 96, 672)}
    for name, width, operation in (
        ('usage_sum_last_1h', 4, 'sum'), ('usage_mean_last_4h', 16, 'mean'),
        ('usage_std_last_4h', 16, 'std'), ('usage_mean_last_24h', 96, 'mean'),
        ('usage_max_last_24h', 96, 'max'),
    ):
        # Cùng trục và ddof=0 như sliding_window_view trong pipeline gốc.
        values[name] = getattr(usage[-width:][None, :], operation)(axis=1)[0]
    start = pd.DatetimeIndex([t + FREQUENCY])
    hour = start.hour.to_numpy() + start.minute.to_numpy()/60
    weekday = start.dayofweek.to_numpy()
    values.update(
        forecast_hour_sin=np.sin(2*np.pi*hour/24)[0],
        forecast_hour_cos=np.cos(2*np.pi*hour/24)[0],
        forecast_weekday_sin=np.sin(2*np.pi*weekday/7)[0],
        forecast_weekday_cos=np.cos(2*np.pi*weekday/7)[0],
        forecast_is_weekend=int(weekday[0] >= 5),
    )
    return pd.DataFrame([values], columns=list(BASE_FEATURES))
