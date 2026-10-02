# Kết quả phần Duy — lần chạy riêng

Chỉ train Jan–Aug và chọn trên validation Sep.
Test Nov–Dec đã được xem trong lịch sử; lần này không tính test hoặc calibration.

- Source: 35,040 hàng; split: {'train': 22652, 'validation': 2876, 'calibration': 2972, 'test': 5852}
- Model chọn: **hgb_31_leaves**; MAE 15.8829 kWh; RMSE 32.0475 kWh.
- MAE cải thiện 50.86% so với last_hour.
- MAE vùng cao: 68.6963 kWh, 93 mẫu; phải bàn giao Khánh kiểm cảnh báo.
- Batch/replay: 31 mốc đã kiểm, feature và dự báo khớp.
- Không có policy cảnh báo hoặc số tiết kiệm tiền điện trong lần chạy này.

## Bảng validation

| Model | MAE kWh | RMSE kWh | Peak MAE kWh |
|---|---:|---:|---:|
| hgb_31_leaves | 15.8829 | 32.0475 | 68.6963 |
| hgb_15_leaves | 16.3177 | 32.5934 | 68.2773 |
| last_hour | 32.3238 | 69.1155 | 106.5411 |
| ridge_alpha_1 | 33.0168 | 50.9669 | 84.3953 |
| ridge_alpha_100 | 33.0307 | 51.0115 | 85.5357 |
| previous_week | 43.4543 | 81.9046 | 128.8452 |
| previous_day | 47.2319 | 91.0016 | 229.3084 |
| diurnal_train_mean | 49.9719 | 78.6894 | 141.8327 |
