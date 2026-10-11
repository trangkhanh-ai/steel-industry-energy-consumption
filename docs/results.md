# Kết quả và những điểm đã sửa

[Trang chủ](../README.md) · [Hướng dẫn tái lập](getting_started.md)

## 1. HGB — luồng demo chính

Đối chiếu ngày 11/10/2026 bằng model đã khóa ngày 02/10. Không fit lại trong evaluation; mọi score calibration/test và hai CSV dự báo khớp chính xác với bản local ngày 07/10.

- Train 22.652, validation 2.876, calibration 2.972, test 5.852 mốc.
- Target: tổng bốn bản đo tiếp theo, đơn vị kWh. Feature implementation duy_numpy_windows_v1.
- Model SHA-256: 3de06129bd2a04a940516d6e08816df0687dfe4e56ca80eb7600fbf2c812f66e.
- Threshold train 379,771 kWh; buffer calibration 36,55168101131031 kWh.

| Chỉ số test | HGB | Giờ gần nhất |
|---|---:|---:|
| MAE kWh | 16,735149 | 38,762633 |
| RMSE kWh | 32,776163 | 77,764524 |
| TP / FP / FN / TN | 113 / 87 / 66 / 5.586 | 117 / 333 / 62 / 5.340 |
| Precision / recall / F1 | 0,5650 / 0,6313 / 0,5963 | 0,2600 / 0,6536 / 0,3720 |

F1 baseline xem giá trị đầy đủ trong JSON; số bảng chỉ là cách đọc nhanh. HGB giảm MAE 56,83%, giảm FP 246 mốc và thêm FN 4 mốc. Vùng vượt ngưỡng có 179 mẫu, MAE HGB 80,6850 kWh: vùng cao còn khó dự báo. Độ phủ biên trên 93,15% là quan sát hồi cứu, không phải xác suất P90 được bảo đảm.

**Ba mức:** model/alert metrics; system có warm latency, allocation/model size, lỗi đầu vào và stress tổng hợp +20%; industrial có khối lượng kiểm tra/điểm chi phí giả định. Chưa đo tiết kiệm kWh/tiền hoặc downtime.

- [Metrics đầy đủ](../reports/steel/release_2026-10-11/metrics.json)
- [Policy đúng HGB](../reports/steel/release_2026-10-11/locked_policy.json)
- [So sánh test từng phương pháp](../reports/steel/release_2026-10-11/test_comparison.csv)
- [Dự báo test từng hàng](../reports/steel/release_2026-10-11/test_predictions.csv)
- [Biểu đồ](../reports/steel/release_2026-10-11/figures/)

## 2. Phần Khánh — số nào thuộc model nào?

### Artifact LightGBM 15 feature lịch sử

Đã nạp model 457 cây, kiểm SHA-256 và tính lại. MAE/RMSE/precision/recall/buffer đều khớp số test đã lưu.

| Chỉ số | Giá trị |
|---|---:|
| Validation MAE / RMSE kWh | 14,961260 / 32,633229 |
| Buffer calibration kWh | 42,347034089026614 |
| Test MAE / RMSE kWh | 15,922537 / 32,581811 |
| TP / FP / FN / TN | 128 / 93 / 51 / 5.580 |
| Precision / recall / F1 | 0,5792 / 0,7151 / 0,6400 |
| Cụm mốc cảnh báo / số ngày lịch | 49 / 61 |
| Cụm mốc cảnh báo mỗi ngày | 0,8033 |

[Đối chiếu artifact](../reports/steel/khanh_lightgbm_verified_2026-10-11/verification.json). Không thay model demo bằng model này dựa vào test. Runtime latency cũ không được coi là số benchmark mới.

### Gate 7: validation 15/19 feature

Ridge và LightGBM cùng được clip prediction không âm; scaler chỉ fit train. CSV lưu độ chính xác đầy đủ và JSON có thêm score không clip để đối chiếu script lịch sử.

| Feature | Model | MAE kWh | RMSE kWh |
|---|---|---:|---:|
| 15 | Ridge | 33,016808 | 50,966902 |
| 19 | Ridge | 32,028018 | 50,548728 |
| 15 | LightGBM | 14,558653 | 31,996567 |
| 19 | LightGBM | 14,087299 | 31,552648 |

Nhóm bốn feature giúp MAE LightGBM validation giảm khoảng 3,24%. Chưa phải chứng minh từng feature hữu ích hoặc đảm bảo trên test/nhà máy mới. Bộ sensor tại t phải có sẵn; không dùng correlation làm chứng minh causality.

[Metrics](../reports/steel/khanh_gate7_2026-10-11/metrics.json) · [Protocol](../reports/steel/khanh_gate7_2026-10-11/protocol.json) · [Heatmap train](../reports/steel/khanh_gate7_2026-10-11/train_correlations_extended_19.png)

## 3. Những lỗi đã sửa trong bản source

- Notebook 02 trên origin/main có dấu xung đột ngoài JSON: sửa thành notebook evaluation HGB đọc kết quả; notebook 04 cho Khánh riêng.
- Audit GitHub làm mất API mà test import: khôi phục bản audit có tái dựng và kiểm từng giá trị/manifest.
- Requirements chung có NumPy <2 xung đột môi trường HGB đã khóa: dùng core pinned, LightGBM tùy chọn.
- Policy LightGBM cũ sai hash và không thuộc HGB: giữ inspector phát hiện, active policy mới hiệu chỉnh theo đúng HGB. Không bypass hash bằng thay số trong file cũ.
- Metadata demo dùng khóa khác giữa nhánh: bổ sung tương thích; thiếu actual được báo chờ bản đo và không đưa vào score.
- Gate 7 trước tự train khi import, lặp code và cần seaborn/tabulate chưa khai báo: gom feature chung, CLI output mới, matplotlib và CSV/JSON.
- Sửa ý nghĩa ramp/proxy/mặt nạ, thống nhất clipping giữa hai model, phân biệt cụm cảnh báo và forecast origins, số ngày lịch đúng 61.

## 4. Phạm vi kiểm chứng

Test đã được quan sát trong lịch sử; đánh giá này là hồi cứu/tái lập. Targets chồng lấp: 5.852 mốc không phải 5.852 sự kiện độc lập. Prototype có human log, chưa nối nhà máy, chưa kiểm định an toàn hay chi phí vận hành thực. Buffer không phải quantile regression; PSI/MAE là heuristic yêu cầu rà soát.

Chứng cứ kiểm source mới nằm trong [verification.json](../reports/steel/release_2026-10-11/verification.json). p95 model, complete forecast và browser request-to-render có phạm vi khác nhau; không lấy số model-only làm độ trễ UI.

## 5. Kiểm source ngày 11/10

- Môi trường .venv mới: 104 tests PASS, 0 lỗi/skip.
- Audit legacy và frozen NumPy đều FE-01…06 PASS; handoff 5 file code, offline/replay khớp.
- Train lại: model SHA-256, metric và predictions khớp; fit_seconds/latency thay đổi theo lần đo.
- Evaluation: toàn bộ metric calibration/test và CSV predictions khớp chính xác bản 07/10.
- 12/12 lỗi đầu vào bị từ chối; 4/4 kịch bản failure demo đạt.
- Edge: 24 bước đo, 0 lỗi JavaScript, desktop/mobile và actual trễ/nhật ký đạt.
- Warm p95 model-only 7.04 ms; complete forecast 43.12 ms; browser request→render 160.72 ms. UI chưa đạt mục tiêu 100 ms.
