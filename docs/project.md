# Bài toán và yêu cầu môn học

## Problem card

- **Đề tài:** dự báo tổng điện năng 60 phút tới tại cơ sở sản xuất thép, hỗ trợ cảnh báo khoảng tiêu thụ cao.
- **Vấn đề:** người điều độ cần thông tin sớm để ưu tiên kiểm tra tải; chỉ nhìn số đo hiện tại khó biết giờ tới.
- **Người dùng:** kỹ sư năng lượng/người điều độ; mô phỏng trong nhóm sinh viên.
- **Đầu vào:** Usage_kWh và lịch sử đã nhận tại t; lịch tương lai biết trước. Dữ liệu UCI một cơ sở, một năm 2018.
- **Đầu ra AI:** y_hat(t) ước lượng Usage(t+15)+Usage(t+30)+Usage(t+45)+Usage(t+60), đơn vị kWh.
- **Quyết định:** nếu y_hat + b > T, đề nghị người phụ trách kiểm đồng hồ/lịch vận hành và ghi nhận.
- **Ràng buộc:** bản đo tại t đã có; 673 mốc liên tục, đúng 15 phút; không dùng phép đo/nhãn tương lai, không tự điều khiển thiết bị.
- **Giới hạn nghiệp vụ:** T là p95 TRAIN theo nghiên cứu, không phải công suất kW hợp đồng. Không có giá điện/BESS/lịch máy để tối ưu hoặc xác nhận tránh phạt.

## Với và không có mô hình

- **Không AI:** đo hiện tại + tổng giờ gần nhất làm dự báo; so với T bằng buffer riêng của baseline. Đây là phương án đối chứng chạy được, không phải giả định nhà máy không có quản lý.
- **Có AI:** từ lịch sử tải, thời gian trong ngày/tuần, học quan hệ với tổng giờ tới. Demo bổ sung buffer hiệu chỉnh, kiểm độ tin cậy và nhật ký xác nhận.
- **Thông tin hữu ích:** số kWh dự kiến, khoảng đệm thực nghiệm, cờ kiểm tra, actual trễ, sai số và trạng thái dữ liệu.
- **Không suy ra được:** model tự giảm điện, tối ưu sản lượng hoặc can thiệp an toàn vào lò/cán thép. Hành động thực cần quy trình và context ngoài dataset.

## 3B và Data Preparation — làm gì, vì sao

| Yêu cầu | Quan sát / xử lý |
|---|---|
| Broken | 35.040 mốc, không thiếu/trùng/gap sau sort; CSV có 365 bước timestamp lùi. Sort trước rolling/target. Runtime thiếu mốc thì từ chối thay vì bịa số. |
| Bad Quality | Không NA/âm; một Usage=0 được giữ. Đỉnh lớn chưa chứng minh là sensor lỗi nên không cắt theo IQR. Sensor limitation chưa đo được bằng CSV. |
| Background | Thiếu sản lượng, lịch máy, giá, hợp đồng, sự cố. Load_Type/CO2 không dùng trong feature chính khi chưa rõ availability. |
| Missing values | Không impute tập gốc vì không có thiếu; loại warmup/nhãn chưa đủ và purge ranh giới với manifest. |
| Outlier | Giữ tải cao vì chính là vùng nghiệp vụ cần dự báo; báo riêng MAE trên vùng cao. |
| Scaling | StandardScaler fit TRAIN cho Ridge; HGB cây không cần scaler. |
| Feature engineering | Lag 0/1/4/96/672; rolling chỉ tới t; sin/cos lịch forecast; target chỉ dùng để train/evaluate. |
| Feature selection | 15 feature dễ giải thích/biết tại t. Study Khánh kiểm thêm bốn proxy trên validation với giả định sensor tại t có sẵn. |
| PCA | Không áp dụng: feature ít, cần giải thích và không có bằng chứng lợi ích. |
| Imbalance | Hồi quy nên không dùng SMOTE; lớp vượt T hiếm được đánh giá precision/recall/FP/FN riêng. |

## Phương pháp và chia dữ liệu

- Bốn baseline: giờ gần nhất, ngày trước, tuần trước, trung bình thứ/giờ fit train.
- Ridge: baseline ML tuyến tính dễ giải thích; scaler fit train.
- HGB: quan hệ phi tuyến, dữ liệu dạng bảng, chạy CPU; 15/31 lá, cấu hình chốt trên MAE validation.
- LightGBM: nghiên cứu Khánh riêng, cùng bài toán nhưng artifact/processed/version riêng. Không chọn lại model HGB bằng kết quả test LightGBM.
- Không bắt buộc Deep Learning. Dataset một năm/một cơ sở, mục tiêu môn yêu cầu phương pháp phù hợp và demo giải thích được.
- Train 01–08: 22.652; validation 09: 2.876; calibration 10: 2.972; test 11–12: 5.852.
- 688 mốc loại: 672 warmup, 12 target vượt ranh giới, 4 cuối năm chưa có future label. Mỗi mốc có lý do.

## Đối chiếu lifecycle 8 bước

| Bước / yêu cầu thầy | Bằng chứng source |
|---|---|
| Problem Definition | Problem card bên trên và docs/team/duy_problem_card.md. |
| Data Understanding | Data dictionary, 3B, quality summary, EDA train và notebook 01/03. |
| Data Preparation | preprocessing.py, duy_forecasting.py, row manifest, audit FE-01…06. |
| Method Selection | Bốn baseline + hai họ ML Ridge/HGB, validation leaderboard; study Khánh riêng. |
| Model Development | run_duy_pipeline, model_handoff, version/hash, offline–online parity. |
| Evaluation | MAE/RMSE/peak MAE; precision/recall/FP/FN/cụm mốc; latency/fault/stress; điểm kiểm tra nghiệp vụ. |
| Deployment | Replay local: Data → AI → Decision → nhật ký người xem; architecture trong README. |
| Monitoring & Improvement | PSI so với train; MAE trên nhãn đã đủ giờ; hai tuần xấu hơn reference yêu cầu rà soát, không auto-retrain. |

ROC-AUC chỉ là điểm xếp hạng vượt ngưỡng, không phải xác suất calibrated. Cụm mốc dự báo chồng lấp không đồng nghĩa sự cố máy hay lead time đã đo.

## Reliability – Robustness – Safety – Ethics

- **Reliability:** dễ sai khi lịch tải thay đổi mạnh, khởi động/đỉnh bất thường; MAE vùng cao lớn hơn toàn tập. Buffer có thể báo giả hoặc vẫn bỏ sót.
- **Robustness:** có test sensor thiếu/sai và stress level shift +20% tổng hợp; chưa chứng minh thích ứng nhà máy mới. PSI/MAE chỉ kích hoạt rà soát heuristic.
- **Safety:** dự báo sai có thể khiến bỏ sót hoặc đề nghị kiểm tra thừa; không dùng cờ dưới ngưỡng làm xác nhận an toàn.
- **Human-in-the-loop:** con người kiểm đồng hồ/context, ghi đã xem/kiểm tra/bỏ qua. Không giả lập hành động thiết bị thành tiết kiệm đã đo.
- **Ethics:** nguồn UCI công khai, ghi attribution; log tên/ghi chú local không publish; không bịa context hoặc kết quả.
- **Limitations:** một cơ sở/năm, target chồng lấp, timestamp/timezone chưa xác nhận; test đã quan sát; chưa có phân vị P10/P50/P90 có điều kiện, causal savings hoặc đánh giá ngoài.

## Sản phẩm môn học

Source + demo + hướng dẫn và bằng chứng thực nghiệm nằm trong repo. Báo cáo/slide theo yêu cầu thầy được nhóm quản lý riêng; không publish Word/PDF hay bài báo trong lần đưa source này. [Kế hoạch nhóm A–Z](team/ke_hoach_nhom_3_nguoi.md), [tham khảo](references.md).
