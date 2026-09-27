# Data dictionary — UCI Steel Industry Energy Consumption

**Nguồn:** [UCI Steel Industry Energy Consumption](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption), DOI `10.24432/C52G8C`, CC BY 4.0.

**File gốc trong repo:** `data/steel/raw/Steel_industry_data.csv`; SHA-256 ghi tại [mô tả dữ liệu Steel](../../data/steel/README.md).
**Đơn vị quan sát:** một bản ghi cách bản ghi kế tiếp 15 phút sau khi sắp xếp theo `date`.

| Cột gốc | Nghĩa/đơn vị theo UCI | Biết tại thời điểm dự báo `t`? | Vai trò trong phiên bản đầu | Lưu ý |
| --- | --- | --- | --- | --- |
| `date` | Thời điểm `DD/MM/YYYY HH:MM` | Có | Parse và sắp xếp; tính lịch tương lai. | File gốc đặt `00:00` cuối mỗi ngày. Múi giờ và ý nghĩa đầu/cuối khoảng đo không được ghi rõ trong CSV; giữ timestamp như đã công bố. |
| `Usage_kWh` | Điện năng tiêu thụ, kWh | Giá trị tại `t` và trước đó: có | Mục tiêu tương lai và feature lịch sử. | Không đưa `t+1…t+4` vào feature. Có một giá trị 0; đỉnh cao chưa được chứng minh là lỗi. |
| `Lagging_Current_Reactive.Power_kVarh` | Điện năng phản kháng trễ, kVarh | Tại `t`: giả định đã nhận cùng bản đo | Feature phụ để thử nghiệm ablation. | Chỉ dùng nếu quy trình phát dự báo đợi bản đo hoàn tất. |
| `Leading_Current_Reactive_Power_kVarh` | Điện năng phản kháng sớm, kVarh | Tại `t`: giả định như trên | Feature phụ để thử nghiệm ablation. | Không dùng giá trị tương lai. |
| `CO2(tCO2)` | Trường CO₂; tên cột gợi tCO₂, trang UCI ghi mô tả đơn vị chưa nhất quán | Không xác nhận thời điểm và cách tính | **Loại khỏi mô hình chính.** | Tương quan với `Usage_kWh` trên train ≈ 0,986. Không dùng để khẳng định phát thải hoặc hiệu quả môi trường. |
| `Lagging_Current_Power_Factor` | Hệ số công suất trễ, % | Tại `t`: giả định như trên | Feature phụ để thử nghiệm ablation. | Kiểm tra nằm trong [0, 100]. |
| `Leading_Current_Power_Factor` | Hệ số công suất sớm, % | Tại `t`: giả định như trên | Feature phụ để thử nghiệm ablation. | Kiểm tra nằm trong [0, 100]. |
| `NSM` | Số giây từ nửa đêm | Có thể tính từ `date` | Chỉ dùng kiểm tra tính nhất quán. | Cùng thông tin với timestamp; không cần thêm vào model. |
| `WeekStatus` | Ngày thường/cuối tuần | Có thể tính từ `date` | Chỉ dùng kiểm tra, feature cuối tuần tính lại từ lịch. | Tất cả bản ghi khớp thứ của timestamp sau parse. |
| `Day_of_week` | Tên thứ trong tuần | Có thể tính từ `date` | Chỉ dùng kiểm tra, feature sin/cos thứ tính lại từ lịch. | Tất cả bản ghi khớp timestamp sau parse. |
| `Load_Type` | `Light_Load`, `Medium_Load`, `Maximum_Load` | **Chưa chứng minh** là lịch biết trước tại `t` | **Loại khỏi mô hình dự báo chính.** | Không suy diễn đây là khả năng dời tải, nhãn lỗi hay biểu giá điện. |

## Các trường trong dữ liệu đã xử lý

| Trường | Định nghĩa | Vai trò |
| --- | --- | --- |
| `observation_time` | Timestamp của bản đo hoàn tất gần nhất theo quy ước prototype. | Mốc phát dự báo; **không** đưa trực tiếp vào ma trận feature. |
| `forecast_start` | `observation_time + 15 phút`. | Bắt đầu của bốn mốc mục tiêu theo timestamp. |
| `forecast_end` | `observation_time + 60 phút`. | Mốc cuối nhãn và kiểm tra purge ở ranh giới split. |
| `target_next_60m_kWh` | Tổng `Usage_kWh` ở bốn mốc `t+1…t+4`. | Nhãn hồi quy; chỉ biết sau thời điểm dự báo. |
| `usage_lag_{0,1,4,96,672}` | Các bản đo điện năng đã có đến `t`; 96 = 24 giờ, 672 = 7 ngày. | Feature cơ sở cho mô hình ML dạng bảng. |
| `usage_sum_last_1h`, `usage_mean/std_last_4h`, `usage_mean/max_last_24h` | Rolling kết thúc tại `t`, không chứa tương lai. | Feature cơ sở cho mô hình ML dạng bảng. |
| `forecast_hour_sin/cos`, `forecast_weekday_sin/cos`, `forecast_is_weekend` | Lịch của mốc `forecast_start`, biết trước. | Feature cơ sở. |
| `known_*` | Bốn biến điện phản kháng/hệ số công suất tại `t`. | Chỉ dùng cho phép so sánh mở rộng; không nằm trong `BASE_FEATURES`. |

**Đầu vào deep learning:** `SteelSequenceView` tạo cửa sổ lịch sử tùy độ dài, mặc định 96 bản ghi kết thúc tại `observation_time`. Nó dùng **cùng nhãn và cùng mốc chia tập** với dữ liệu ML dạng bảng; mọi chuẩn hóa khi huấn luyện mạng phải fit trên train rồi áp dụng nguyên vẹn cho các tập sau. Chưa huấn luyện mạng hoặc chọn kiến trúc ở bước xử lý dữ liệu này.

**Tệp kiểm toán tiền xử lý:** `data/steel/processed/steel_row_manifest.csv` ghi `period`, `row_status`, `reason` cho từng mốc trong năm; `steel_train_column_profile.csv` chỉ thống kê các cột gốc từ tháng 1–8. Cả hai là tài liệu kiểm tra, không phải feature đưa vào mô hình.
