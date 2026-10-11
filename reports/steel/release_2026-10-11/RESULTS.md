# Đánh giá cuối — model và policy đã khóa

Test này đã được xem trong lịch sử, không phải blind test mới. Không fit lại AI hoặc chọn tham số trên test.

- Model: hgb_31_leaves; SHA256: 3de06129bd2a04a940516d6e08816df0687dfe4e56ca80eb7600fbf2c812f66e.
- Train threshold: 379.7710 kWh; October buffer: 36.5517 kWh.
- Buffer là phân vị lỗi thiếu thực nghiệm, không phải P90 có điều kiện hoặc bảo đảm độ phủ.
- Test AI MAE: 16.7351 kWh; RMSE: 32.7762 kWh; peak MAE: 80.6850 kWh.
- Test baseline last_hour MAE: 38.7626 kWh; cải thiện MAE: 56.83%.
- AI alert precision: 0.5650; recall: 0.6313; FP: 87; FN: 66.
- Empirical upper coverage test: 0.9315 (cần đo, không mặc định 90%).
- Warm local p95 model-only: 7.044 ms; complete forecast: 43.123 ms.
- Input faults rejected: 12/12.
- Có kịch bản tổng hợp +20% để đo nhạy cảm; không phải drift thực tế đã quan sát.
- Chi phí cảnh báo chỉ là điểm mô phỏng theo hàng; chưa đo tiết kiệm điện/tiền hoặc lead time sự kiện thiết bị.

## Bảng test

| Model | MAE kWh | RMSE kWh | Buffer kWh | Precision | Recall | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| last_hour | 38.7626 | 77.7645 | 90.2650 | 0.2600 | 0.6536 | 333 | 62 |
| previous_day | 53.9377 | 99.3443 | 159.2490 | 0.1125 | 0.7765 | 1097 | 40 |
| previous_week | 47.9941 | 89.8520 | 134.3690 | 0.0716 | 0.3855 | 895 | 110 |
| diurnal_train_mean | 45.3845 | 73.0328 | 84.0517 | 0.1111 | 0.0670 | 96 | 167 |
| hgb_31_leaves | 16.7351 | 32.7762 | 36.5517 | 0.5650 | 0.6313 | 87 | 66 |
