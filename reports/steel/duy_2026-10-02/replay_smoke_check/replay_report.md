# Kiểm tra phát lại dữ liệu thật và chức năng dự báo

Đã chạy 8 thời điểm validation tháng 9 theo thứ tự. Mỗi lần chỉ đưa vào 673 bản đo thật kết thúc đúng thời điểm dự báo; không truyền target, Power Factor, Load_Type hoặc các bản đo tương lai cho mô hình. Đây là phát lại offline theo thứ tự, không phải chạy chờ đủ 15 phút hoặc kết nối nhà máy.

Tất cả 15 feature khớp chính xác bảng validation của model hgb_31_leaves. Chênh lệch feature tuyệt đối lớn nhất: 0. Chênh lệch dự báo lớn nhất so với kết quả chạy theo bảng: 0 kWh.

MAE validation được tái tạo: 0.7785 kWh. Đây là đối chiếu chức năng dùng mô hình, không phải kết quả kiểm định mới trên dữ liệu chưa từng xem.

| Phép đo trên máy hiện tại | Thời gian ms |
|---|---:|
| Nạp và kiểm tra mô hình một lần | 30.716 |
| Lần gọi đầu tiên | 2380.965 |
| p50 của 8 lần gọi tiếp theo | 14.854 |
| p95 | 17.041 |
| Lớn nhất | 17.795 |

Mỗi lần đo bao gồm kiểm tra đầu vào trong bộ nhớ, tạo feature, giới hạn luồng, predict và đóng gói kết quả. Không gồm đọc file/cảm biến, lấy cửa sổ lịch sử, mạng hoặc giao diện. Mô hình được nạp một lần và gọi tuần tự; số đo này chưa đại diện cho nhiều người dùng đồng thời hay độ trễ nhà máy. Các phép đối chiếu với nhãn thật chạy ngoài đoạn đo.

`replay_predictions.csv` lưu từng thời điểm, phạm vi lịch sử, kết quả, target thật dùng để đối chiếu và latency. `replay_features.csv` lưu feature. `example_history.csv` là trích nguyên 673 bản đo thật cho lần dự báo đầu tiên, không phải dữ liệu mô phỏng. Hash và môi trường nằm trong `replay_summary.json`.

Lần replay giữ nguyên artifact/model đã chỉ định, không fit lại hoặc tuning. Chưa có cảnh báo, tự điều phối tải hoặc tuyên bố tiết kiệm điện. Khi dùng max_origins, kết quả chỉ mô tả các mốc đã chạy, không phải toàn tháng.
