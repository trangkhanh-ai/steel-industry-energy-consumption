# Dữ liệu UCI Steel

**Nguồn:** [UCI Steel Industry Energy Consumption](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption), DOI `10.24432/C52G8C`, CC BY 4.0. File `raw/Steel_industry_data.csv` là bản CSV gốc, 2.731.389 byte, SHA-256 `9B1CEE6F9CB9CD9DF2B95814CA90A9A2FF15B7F5F1FBA0FAE3C643E82072EACC`. Không sửa file này; `.gitattributes` giữ CRLF khi checkout.

CSV có **35.040 dòng × 11 cột**, trải từ `2018-01-01 00:00` đến `2018-12-31 23:45`. Thứ tự dòng gốc không tăng dần: bản ghi `00:00` nằm cuối mỗi ngày (365 lần timestamp lùi). Pipeline parse `%d/%m/%Y %H:%M`, sắp xếp và xác nhận lưới 15 phút đầy đủ trước khi tạo nhãn/feature. Có 0 ô thiếu, 0 timestamp trùng và 1 bản ghi `Usage_kWh = 0`.

| Tệp trong `processed/` | Số dòng | Vai trò |
| --- | ---: | --- |
| `steel_next_60m_train.csv` | 22.652 | Fit mô hình/scaler. |
| `steel_next_60m_validation.csv` | 2.876 | Chọn mô hình và tham số. |
| `steel_next_60m_calibration.csv` | 2.972 | Hiệu chỉnh cảnh báo. |
| `steel_next_60m_test.csv` | 5.852 | Đánh giá cuối sau khi khóa mô hình. |
| `steel_quality_summary.json` | — | Nguồn, checksum, chất lượng, schema và split. |
| `steel_row_manifest.csv` | 35.040 | Tập, trạng thái và lý do giữ/loại của từng mốc. |
| `steel_train_column_profile.csv` | 11 | Thống kê từng cột gốc chỉ trên tháng 1–8. |

Target `target_next_60m_kWh` là **tổng bốn bản đo 15 phút ở `t+1…t+4`**. Không dùng timestamp, target hay phép đo tương lai làm feature. Xem [data dictionary](../../docs/data/steel_data_dictionary.md) và [báo cáo xử lý](../../reports/steel/data_preparation.md) để biết rõ từng quyết định.

Chạy lại từ thư mục gốc repo: `python -m scripts.prepare_steel_data`. Pipeline ghi lại `processed/` và tạo **5 biểu đồ chỉ từ giai đoạn train** trong `reports/steel/figures/`: tuần đầu, phân bố bản đo/nhãn, heatmap thứ–giờ, tổng kWh theo ngày và tương quan với các bản đo quá khứ. Các chỉ số đi kèm nằm trong khóa `train_eda` của `steel_quality_summary.json`.

Thống kê này dùng để hiểu và chuẩn bị dữ liệu. Mọi phép biến đổi cần **học tham số** khi huấn luyện mô hình, chẳng hạn scaler, phải fit riêng trên train rồi áp dụng cho các tập sau; không fit trên cả năm. Xem [báo cáo dữ liệu](../../reports/steel/data_preparation.md) và [hướng dẫn của scikit-learn về data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
