# Dự báo điện năng ngành thép — giai đoạn chuẩn bị dữ liệu

Bản này công bố **phần dữ liệu** của đồ án *Ứng dụng AI trong Công nghiệp*: kiểm tra dữ liệu gốc, phân tích 3B, tạo nhãn dự báo, tạo đặc trưng từ quá khứ, chia tập theo thời gian và trực quan hóa. Bài toán là dự báo **tổng kWh của 60 phút tiếp theo** tại một cơ sở thép để người quản lý năng lượng có thêm thông tin trước khi xem xét giờ tải cao.

**Nguồn:** [UCI Steel Industry Energy Consumption](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption), DOI [10.24432/C52G8C](https://doi.org/10.24432/C52G8C), giấy phép CC BY 4.0. Dữ liệu gốc gồm **35.040 bản ghi × 11 cột**, cách nhau 15 phút trong năm 2018. Bản CSV trong repo được giữ nguyên byte và kiểm tra SHA-256 trước khi xử lý.

## Chạy lại

Từ thư mục gốc repo với Python 3.12 trở lên:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m scripts.prepare_steel_data
.venv\Scripts\python -m unittest discover -s tests -p test_steel_preprocessing.py -v
```

Trên Linux/macOS, thay `.venv\Scripts\python` bằng `.venv/bin/python`. Lệnh chuẩn bị dữ liệu tạo lại bốn CSV, JSON kiểm toán, bảng truy vết từng dòng, hồ sơ cột train và năm hình PNG. Không cần tải thêm dữ liệu vì bản CSV gốc đã có trong repo.

## Quy trình và kết quả

| Bước | Việc đã làm | Lý do |
| --- | --- | --- |
| Đọc nguồn | Kiểm tra hash, đúng 11 cột, parse `DD/MM/YYYY HH:MM` rồi sắp xếp thời gian. | File UCI đặt bản ghi `00:00` cuối ngày; có **365** lần thứ tự thời gian lùi. |
| Kiểm tra chất lượng | Kiểm tra đủ lưới 15 phút, không trùng mốc, giá trị hữu hạn/không âm và cột lịch khớp timestamp. | Tránh tạo lag hoặc nhãn sai khi nguồn bị hỏng. Sau sort có **0 ô thiếu, 0 mốc trùng, 0 khoảng gián đoạn**. |
| Tạo nhãn | `target_next_60m_kWh(t) = Usage(t+15m) + Usage(t+30m) + Usage(t+45m) + Usage(t+60m)`. | Đúng câu hỏi vận hành: điện năng cộng dồn trong một giờ tương lai; đơn vị **kWh**, không phải kW. |
| Tạo feature | 15 biến mặc định gồm lag, thống kê quá khứ và lịch biết trước. | Tại thời điểm `t`, không dùng bản đo sau `t` làm đầu vào. `CO2(tCO2)` và `Load_Type` không dùng mặc định vì chưa rõ thời điểm có sẵn và ý nghĩa vận hành. |
| Chia tập | Train tháng 1–8, validation tháng 9, calibration tháng 10, test tháng 11–12. | Không xáo trộn thời gian; loại các nhãn vượt qua ranh giới tập. |

| Tập | Số mốc dự báo |
| --- | ---: |
| Train | 22.652 |
| Validation | 2.876 |
| Calibration | 2.972 |
| Test | 5.852 |
| **Tổng dùng được** | **34.352** |

**688 mốc không dùng:** 672 mốc đầu chưa đủ lịch sử bảy ngày, 12 mốc có nhãn chạm qua ba ranh giới tập và 4 mốc cuối chưa có đủ bốn bản đo tương lai. Lý do của **toàn bộ 35.040 mốc** nằm trong [`steel_row_manifest.csv`](data/steel/processed/steel_row_manifest.csv).

### 3B và quyết định xử lý

- **Broken:** nguồn không thiếu ô/khoảng sau sắp xếp; lỗi cần xử lý là thứ tự thời gian. Pipeline dừng khi gặp nguồn thiếu hoặc mốc đo sai, không tự nội suy.
- **Bad Quality:** có một bản ghi `Usage_kWh = 0` và các đỉnh cao. Không có nhật ký cảm biến để xác định chúng là lỗi, nên giữ nguyên, không cắt p99.
- **Background:** không có dữ liệu máy, sản lượng, ca làm, biểu giá hay hành động điều khiển. Vì vậy dữ liệu này hỗ trợ nghiên cứu dự báo, **chưa chứng minh tiết kiệm điện thực tế**.

Không scale trong CSV đã xử lý. Khi huấn luyện mô hình cần scale, scaler phải **fit trên train** rồi áp dụng nguyên vẹn cho các tập sau. Bài toán hiện là hồi quy với 15 feature; chưa cần PCA hay cân bằng lớp bằng SMOTE.

## Biểu đồ khám phá dữ liệu

**Tất cả hình bên dưới chỉ dùng dữ liệu tháng 1–8 (train)** để không nhìn vào các giai đoạn dùng chọn/đánh giá mô hình:

| Hình | Giúp trả lời |
| --- | --- |
| [Tuần đầu](reports/steel/figures/steel_train_first_week.png) | Chuỗi 15 phút thay đổi trong một tuần như thế nào? |
| [Phân bố](reports/steel/figures/steel_train_distributions.png) | Bản đo 15 phút và nhãn 60 phút có đuôi cao ra sao? |
| [Thứ × giờ](reports/steel/figures/steel_train_hour_weekday.png) | Mức sử dụng trung bình khác nhau theo lịch không? |
| [Tổng theo ngày](reports/steel/figures/steel_train_daily_totals.png) | Mức sử dụng có thay đổi giữa các tháng không? |
| [Tương quan lag](reports/steel/figures/steel_train_lag_correlations.png) | Các mốc 15 phút, 1 giờ, 1 ngày, 1 tuần trước liên hệ với hiện tại thế nào? |

![Mức điện năng trung bình theo thứ và giờ, chỉ trên train](reports/steel/figures/steel_train_hour_weekday.png)

Trên train, median điện năng 15 phút là **4,75 kWh**, p95 **102,02 kWh**. Tương quan với bản đo trước 15 phút, 1 giờ, 1 ngày, 1 tuần lần lượt khoảng **0,91; 0,69; 0,61; 0,67**. Đây là mô tả dữ liệu, **không phải điểm số dự báo**. Bản chi tiết và cách diễn giải thận trọng nằm trong [báo cáo chuẩn bị dữ liệu](reports/steel/data_preparation.md).

## Tệp chính

```text
data/steel/raw/Steel_industry_data.csv       Nguồn UCI nguyên trạng
data/steel/processed/                     Bốn tập và tệp kiểm toán
src/steel/preprocessing.py                 Kiểm tra, tạo nhãn/feature, chia tập
src/steel/sequences.py                     Cửa sổ quá khứ cho thử nghiệm DL sau này
scripts/prepare_steel_data.py              Lệnh tạo dữ liệu và biểu đồ
reports/steel/figures/                     Năm biểu đồ train
reports/steel/data_preparation.md           Giải thích 3B, số liệu và mọi quyết định
docs/data/steel_data_dictionary.md         Từ điển cột và thời điểm có sẵn
tests/test_steel_preprocessing.py           Kiểm thử nhãn, rò rỉ, ranh giới và dữ liệu lỗi
```

Tham khảo phương pháp: [scikit-learn về data leakage và tiền xử lý nhất quán](https://scikit-learn.org/stable/common_pitfalls.html), [pandas `shift` cho lag](https://pandas.pydata.org/docs/reference/api/pandas.Series.shift.html). Những số liệu và hình trong repo được tính lại từ bản CSV UCI; chúng không phải kết quả do UCI công bố.

**Phạm vi bản đẩy này:** chưa gồm model, điểm MAE/RMSE, dashboard hoặc tuyên bố hiệu quả công nghiệp. Các bước method selection → development → evaluation → deployment → monitoring thuộc các phần tiếp theo của lifecycle đồ án.
