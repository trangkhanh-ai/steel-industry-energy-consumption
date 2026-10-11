# Đọc source theo luồng

[Trang chủ](../README.md) · [Hướng dẫn chạy](getting_started.md)

## Luồng chính

| Thứ tự | Module | Vai trò |
|---|---|---|
| 1 | src/steel/preprocessing.py | Kiểm CSV, sort timestamp, target, feature và temporal split/manifest. |
| 2 | src/steel/duy_forecasting.py | Feature batch/replay NumPy, baseline, scaler Ridge, HGB và chọn model. |
| 3 | scripts/run_duy_pipeline.py | Ghép dữ liệu, EDA, train/validation và bàn giao. |
| 4 | src/steel/duy_inference.py | Kiểm hash/schema; dự báo từ 673 bản đo quá khứ. |
| 5 | src/steel/evaluation.py | Buffer, MAE/RMSE, cảnh báo/cụm mốc và điểm chi phí. |
| 6 | scripts/run_final_evaluation.py | Đánh giá ba mức; khóa protocol/policy trước test. |
| 7 | src/steel/policy.py | Policy HGB đúng hash/feature, ngưỡng kWh, PSI và theo dõi MAE. |
| 8 | src/steel/inference.py | Adapter model Duy sang API demo; không train lại. |
| 9 | src/steel/monitoring.py | SQLite, actual trễ, chất lượng và nhật ký bất biến. |
| 10 | src/steel/demo.py | Đồng hồ replay/reset/step/decision, không gửi nhãn tương lai. |
| 11 | scripts/serve_steel_demo.py | Server loopback, API và static UI. |
| 12 | web/steel_demo/ | HTML, CSS bố cục và JS gọi API/vẽ biểu đồ. |

Năm file trong handoff có hash đã khóa: preprocessing, duy_forecasting, duy_inference, run_duy_pipeline, prepare_steel_data. Đổi logic cần run mới, không sửa manifest để né kiểm tra.

## Nghiên cứu Khánh

- `feature_extension.py`: bốn proxy từ số đo tại t, dùng chung cho script/test.
- `run_feature_extension.py`: Gate 7 validation 15/19 feature; Ridge/LightGBM, scaler/heatmap chỉ train.
- `check_khanh_lightgbm.py`: đối chiếu artifact LightGBM 15 feature; không fit lại.
- `policy_handoff.py`: inspector lịch sử của Huy, chỉ đọc/báo khác biệt; không bật cảnh báo.
- Notebook `04_khanh_lightgbm_study.ipynb`: đọc kết quả study và hướng dẫn.

## Lệnh kiểm tra

- `audit_features_and_leakage`: schema, causality, availability, background, tái dựng và manifest.
- `check_duy_handoff`: hash và offline–online parity.
- `check_steel_demo_environment`: phiên bản, artifact, CSV và UI; inspector lịch sử tách active policy.
- `check_steel_demo_browser`: browser, actual trễ, nhật ký, desktop/mobile và timing.
- `check_steel_demo_failures`: model thiếu, interval thiếu, bản đo âm, policy lỗi trên bản sao.
- `check_repository`: notebook, link tài liệu, dấu merge và file được phép publish.

## Tương thích và dữ liệu

`modeling.py`, `sequences.py`, script train/tune/package/predict/replay cũ phục vụ luồng lịch sử và test hồi quy. Không cần chạy tất cả để mở demo; Duy đọc luồng chính trước.

| Đường dẫn | Vai trò |
|---|---|
| data/steel/raw/ | CSV UCI giữ nguyên byte. |
| data/steel/processed/ | Processed pandas cho nghiên cứu Khánh. |
| reports/steel/duy_2026-10-02/ | Feature NumPy/model HGB đã khóa. |
| reports/steel/release_2026-10-11/ | Evaluation mới và policy cho demo hiện tại. |
| reports/steel/khanh_gate7_2026-10-11/ | Validation 15/19 feature. |
| reports/steel/khanh_lightgbm_verified_2026-10-11/ | Đối chiếu LightGBM 15 feature. |
| outputs/ và local_* | Chạy thử local, không publish. |

Không trộn processed pandas với artifact HGB `duy_numpy_windows_v1`: cùng tên feature vẫn có thể khác số học. Metadata/handoff kiểm điều này.
