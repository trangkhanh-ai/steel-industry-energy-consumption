# Kế hoạch triển khai đồ án từ đầu — nhóm 3 người

**Học phần:** Ứng dụng AI trong Công nghiệp.
**Đề tài nhóm chọn:** Dự báo điện năng tiêu thụ 60 phút tiếp theo tại cơ sở sản xuất thép và hỗ trợ người vận hành nhận biết sớm các khoảng tiêu thụ cao.
**Dữ liệu nghiên cứu:** UCI Steel Industry Energy Consumption [R4].
**Kế hoạch triển khai chính của nhóm:** chốt dùng bản này từ 30/09/2026; mỗi thay đổi về mô hình, ngưỡng hoặc cách đánh giá phải được ba thành viên ghi vào `decision_log.md` trước khi thực hiện.
**Không gian làm việc chung:** [Notion của nhóm Duy–Khánh–Huy](https://app.notion.com/p/3eb7c2776902811896b1d75c0f11cf82?pvs=204).
**Cách dùng:** tài liệu này là quy trình triển khai từ bước đầu tiên. Mọi số liệu quan sát, mô hình thắng, biểu đồ và nhận định kết quả phải được tạo lại từ một lần chạy có nhật ký; bảng dưới đây chỉ định việc cần làm và tiêu chí nghiệm thu.

## 1. Một trang để cả nhóm cùng chốt

- **Câu hỏi nghiệp vụ:** ở thời điểm `t`, khi phép đo 15 phút gần nhất đã hoàn tất, 60 phút tiếp theo có thể tiêu thụ bao nhiêu kWh và có đáng để người điều độ kiểm tra lịch sản xuất, trạng thái thiết bị hoặc kế hoạch sử dụng điện hay không?
- **Người dùng:** kỹ sư năng lượng hoặc người điều độ ca. **Quyết định được hỗ trợ:** xem dự báo, kiểm tra nguyên nhân nếu cảnh báo, ghi nhận có chấp nhận hay bỏ qua cảnh báo. Hệ thống chỉ hỗ trợ quyết định; hành động vận hành cần người có thẩm quyền duyệt.
- **Biến mục tiêu chính:** `y_t = Usage_kWh(t+1) + Usage_kWh(t+2) + Usage_kWh(t+3) + Usage_kWh(t+4)`, với mỗi bước dự kiến là 15 phút. Đơn vị đầu ra là **kWh cho 60 phút tiếp theo**. Trước khi dùng công thức, nhóm phải xác nhận ý nghĩa timestamp và chu kỳ đo từ tệp gốc.
- **Nhiệm vụ AI:** dự báo hồi quy chuỗi thời gian. Lớp cảnh báo được suy ra từ dự báo và ngưỡng nghiên cứu; không tự biến bài toán chính thành phân loại nếu chưa xác định lại mục tiêu.
- **Ranh giới công nghiệp:** bộ UCI không cung cấp công suất hợp đồng, biểu giá, giá phạt, pin BESS, lịch cán thép, sản lượng, ca vận hành hay lịch hành động thực tế. Vì thế dashboard mô phỏng hỗ trợ ra quyết định; báo cáo không khẳng định giảm tiền điện, tránh phạt hay tiết kiệm điện thực tế nếu thiếu đo lường đối chứng.
- **Điểm phải trình thầy/nhóm duyệt sớm:** tên đề tài và lý do dùng UCI Steel thay gợi ý Appliances/Building trong ảnh ban đầu; định nghĩa `t` và đơn vị; tiêu chí dùng một ngưỡng nghiên cứu thay ngưỡng hợp đồng; chuẩn đầu ra mong muốn của báo cáo và demo. Ảnh ban đầu là bộ dữ liệu **đề xuất**, còn lựa chọn Steel phải được giải thích bằng mức phù hợp với bài toán công nghiệp và khả năng triển khai [R4].

| Thành viên | Vai trò chính | Người kiểm tra chéo | Chữ ký/tên sau khi họp |
| --- | --- | --- | --- |
| Duy — trưởng nhóm | Cổng 0–7: problem card, nguồn/EDA/3B, tiền xử lý, split, baseline, Ridge và mô hình cây; chốt tích hợp | Khánh kiểm split/model; Huy kiểm nghiệp vụ và hình | Duy |
| Khánh | Cổng 8–9: calibration, luật cảnh báo, đánh giá model/system/industrial; kiểm độc lập cổng 6–7 | Duy tái chạy metric; Huy kiểm diễn giải chi phí | Khánh |
| Huy | Cổng 10–13: dashboard, monitoring/safety, báo cáo tám chương, slide và gói nộp; kiểm chéo nguồn/hình | Duy kiểm kỹ thuật; Khánh kiểm metric/demo | Huy |

- **Quy tắc bàn giao:** mỗi đầu ra có người làm, người kiểm, đường dẫn và một dòng kết luận trong `decision_log.md`; không nhận đầu ra bằng câu “đã chạy” nếu thiếu lệnh, phiên bản dữ liệu và file bằng chứng.
- **Mốc nghiệm thu:** mỗi bước dưới đây chỉ chuyển sang bước sau khi đầu ra tương ứng có thể tái tạo trên máy khác. Nếu thiếu thông tin nghiệp vụ, ghi giả định rõ ràng và giới hạn câu kết luận.

## 2. Bản đồ 8 bước lifecycle theo đúng yêu cầu thầy

| Bước học phần | Việc nhóm phải thực hiện | Đầu ra kiểm được | Người chính |
| --- | --- | --- | --- |
| 1. Problem Definition | Xác định vấn đề, nguyên nhân/khó khăn, mục tiêu, người dùng, quyết định, ràng buộc | Problem card một trang; sơ đồ quyết định | Duy; Huy kiểm |
| 2. Data Understanding | Kiểm nguồn, cấu trúc, đơn vị, trực quan hóa, phân tích 3B | Data dictionary; audit chất lượng; biểu đồ EDA có chú thích | Duy; Huy kiểm |
| 3. Data Preparation | Xử lý thiếu, trùng, sai, ngoại lệ; tạo target/feature; giải thích scaling, chọn feature, PCA, imbalance | Pipeline và bảng lý do từng bước; kiểm thử chống leakage | Duy; Khánh kiểm |
| 4. Method Selection | Mốc naive; so sánh ít nhất Ridge và HistGradientBoosting | Bảng phương pháp, giả thiết và chi phí tính toán | Duy; Khánh kiểm |
| 5. Model Development | Chia theo thời gian; train/tune bằng validation; chọn chính sách trước test | Mã chạy; manifest split; model card; cấu hình khóa | Duy; Khánh kiểm |
| 6. Evaluation | Đánh giá model, system, industrial/business | MAE/RMSE, báo giả, độ trễ, kịch bản giá trị và giới hạn | Khánh; Duy/Huy kiểm |
| 7. Deployment | Chạy mô phỏng Data → AI → Decision trên máy cá nhân | Dashboard, nhật ký người duyệt, hướng dẫn demo | Huy; Duy tích hợp |
| 8. Monitoring and Improvement | Theo dõi lỗi dữ liệu, sai số sau khi có nhãn, drift, cải tiến có kiểm soát | Quy tắc giám sát, ngưỡng rà soát, kế hoạch cập nhật | Huy; Khánh kiểm metric |

**Điểm kiểm rubric:** kiến trúc phải đi đủ bảy khối `Data Source → Data Processing → AI Model → Prediction/Detection → Decision Support → Human/System → Industrial Action`; có mục riêng cho reliability, robustness, safety, human-in-the-loop và limitations. Chỉ số Precision/Recall/F1/ROC-AUC là **chỉ số bổ sung cho lớp cảnh báo có nhãn hai lớp**; dự báo hồi quy vẫn lấy MAE/RMSE làm chỉ số chính [R1, tr. 12–14 và 17; yêu cầu đồ án của thầy].

## 3. Kế hoạch thực thi từ số 0: 14 cổng nghiệm thu

### Cổng 0 — Khởi động và quản lý phiên bản

- **Làm:** Duy tạo một nhánh/lần chạy mới, ghi ngày, commit nguồn, phiên bản Python và thư viện. Khánh/Huy clone cùng nguồn. Lưu dữ liệu gốc ở thư mục `data/raw/` chỉ đọc; đầu ra xử lý, mô hình, biểu đồ và báo cáo ở thư mục khác. Tạo `decision_log.md` ghi mọi lựa chọn có lý do, người duyệt và nguồn.
- **Kiểm:** cả ba chạy được lệnh kiểm môi trường; không có khóa truy cập, dữ liệu riêng hoặc file tạm trong bản nộp. Đặt seed và ghi phiên bản thư viện; tái chạy cho ra cùng phân hoạch và bảng số trong sai số số học chấp nhận được.
- **Giao:** Duy cấu trúc repo và lập lịch; Khánh lập checklist tái tạo; Huy kiểm gói tài liệu. **Đầu ra:** README khởi động, requirements cố định, nhật ký quyết định.

### Cổng 1 — Định nghĩa vấn đề công nghiệp trước mô hình

- **Làm:** Duy viết problem card: hiện trạng giả định khó biết trước giờ tiêu thụ cao; nguyên nhân vận hành cần được kiểm bằng dữ liệu/context, không tự suy diễn “do mẻ phôi”. Mục tiêu kỹ thuật là dự báo kWh 60 phút; mục tiêu vận hành là giúp kiểm tra sớm. Nêu đối tượng dùng, thời điểm xem, thời gian còn lại để phản ứng, chi phí báo giả và bỏ sót, ai có quyền ra quyết định.
- **Kiểm:** đọc problem card, người ngoài nhóm trả lời được “dữ liệu vào là gì, AI trả ra gì, ai làm gì sau cảnh báo”. Bảo đảm không lẫn kWh với kW, điện năng với công suất cực đại, điểm dự báo với xác suất rủi ro.
- **Giao:** Duy soạn; Huy kiểm nghiệp vụ, Khánh kiểm tính đo được. **Đầu ra:** `problem_card.md` và một câu đề tài thống nhất.

### Cổng 2 — Tải và xác minh dữ liệu

- **Làm:** Duy tải CSV từ trang UCI chính thức [R4], ghi ngày truy cập, DOI `10.24432/C52G8C`, giấy phép CC BY 4.0 và SHA-256. Lập bảng ý nghĩa từng cột, đơn vị, cách thu thập, tần suất, giai đoạn, cột có thể biết trước ở thời điểm `t`. Trang UCI mô tả 35.040 bản ghi từ một cơ sở thép nhỏ ở Hàn Quốc và báo “no missing”; vẫn phải tự kiểm CSV tải về.
- **Kiểm:** raw bất biến; hash lưu trong manifest; số dòng/cột, dải timestamp và các trường được đối chiếu với metadata. Ghi rõ những gì UCI **không** cung cấp: hợp đồng/TOU, sản lượng, lệnh vận hành và nhãn hành động.
- **Giao:** Duy làm và kiểm script tải/đọc; Huy xác nhận lời mô tả nguồn; Khánh kiểm schema. **Đầu ra:** `data_dictionary.md`, `data_manifest.json`, CSV gốc.

### Cổng 3 — Khám phá dữ liệu và 3B

- **Làm:** Duy parse timestamp với định dạng rõ ràng, kiểm thứ tự, trùng, thiếu mốc 15 phút, giá trị âm/0/NaN, phạm vi các biến điện và số ngày. Chạy EDA theo thời gian: chuỗi 7 ngày mẫu, histogram và boxplot `Usage_kWh`, heatmap giờ × thứ, tổng ngày, lag plot/ACF đơn giản, biểu đồ đỉnh theo ngày. Với hình dùng để thiết kế model, chỉ xem giai đoạn train dự kiến; trên toàn raw chỉ kiểm tính toàn vẹn, tránh lấy pattern test để điều chỉnh mô hình [R2, tr. 8, 20–21; R6].
- **Broken:** thiếu bản ghi, nhảy giờ, thứ tự ngày sai, trùng timestamp. **Bad Quality:** giá trị bất khả thi, sensor noise, giá trị 0, đỉnh bất thường; đánh dấu để kiểm, không xóa đỉnh theo cảm giác. **Background:** thiếu lịch ca, dây chuyền, loại sản phẩm, thời tiết, lịch bảo trì và chính sách giá; điều này giới hạn giải thích nguyên nhân.
- **Kiểm:** mỗi hình có trục, đơn vị, phạm vi thời gian, nguồn; bảng 3B có bằng chứng, mức ảnh hưởng và quyết định xử lý. Nếu dữ liệu không thiếu thì ghi “không nội suy”, vẫn định nghĩa cách xử lý cho dữ liệu mới bị thiếu.
- **Giao:** Duy làm và kiểm parsing; Huy kiểm hình phục vụ câu chuyện nghiệp vụ; Khánh kiểm ranh giới train. **Đầu ra:** EDA script/notebook tái tạo, 5–6 hình, báo cáo 3B.

### Cổng 4 — Hợp đồng dự báo và kiểm soát thời điểm có dữ liệu

- **Làm:** Duy xác nhận một hàng dữ liệu biểu diễn khoảng thời gian nào; Khánh kiểm độc lập. Tại `t`, chỉ dùng phép đo hoàn tất tới `t` và thông tin lịch biết trước. Xây nhãn bốn bước sau `t`; loại bốn hàng cuối không có nhãn. Viết bảng feature availability: tên biến, thời điểm biết, lý do được dùng/loại. `Load_Type`, CO₂, công suất phản kháng hoặc power factor chỉ dùng nếu chứng minh được timestamp, tính sẵn có và ý nghĩa đo; mô hình chính ưu tiên lịch + lag/rolling `Usage_kWh` để demo replay hoạt động.
- **Kiểm:** lấy ba hàng ví dụ tính tay `y_t`; mọi timestamp dùng cho feature ≤ `t` và cho nhãn > `t`. Một kiểm thử thay các giá trị sau `t`: feature tại `t` giữ nguyên, nhãn đổi. Không tạo rolling dùng cửa sổ tương lai.
- **Giao:** Duy code, Khánh đối chiếu ví dụ, Huy ghi công thức trong báo cáo. **Đầu ra:** đặc tả target, bảng feature availability và kiểm thử leakage.

### Cổng 5 — Chuẩn bị dữ liệu có lý do cho từng biến đổi

| Hạng mục thầy hỏi | Quy tắc phải thử và giải thích | Điều kiện kiểm |
| --- | --- | --- |
| Missing values | Audit trước; không điền nếu gốc không thiếu. Với replay bị thiếu mốc/đọc lỗi: từ chối dự báo hoặc fallback theo quy tắc đã chốt; không dùng giá trị tương lai để nội suy. | Bảng số thiếu trước/sau, nhật ký các hàng bị loại |
| Outlier | Kiểm giá trị bất khả thi bằng quy tắc vật lý/đơn vị; giữ đỉnh hợp lý vì là đối tượng cảnh báo; thử độ nhạy có và không có hàng nghi vấn trên validation. | Danh sách nghi vấn, lý do giữ/loại, tác động MAE |
| Normalization/scaling | StandardScaler fit **chỉ train** cho Ridge qua Pipeline; cây tăng cường có thể không cần scale. | Test tham số scaler không thay khi thêm validation/test |
| Feature engineering | Lag 1/4/96/672 nếu đủ lịch sử; rolling quá khứ 1h/24h; giờ, thứ, cuối tuần, mùa. Chỉ giữ đặc trưng có tại `t`; số lag phải được xác nhận bằng EDA và validation. | Bảng công thức và một hàng tính tay |
| Feature selection | Bắt đầu với tập tối giản; so sánh tập lịch + lag với tập mở rộng trên validation; bỏ cột trùng/khó có lúc dự báo. | Danh sách cột cuối và lý do lựa chọn |
| Dimensionality reduction | Chưa cần PCA với số feature nhỏ, dễ giải thích; nếu thêm hàng trăm cột, thử PCA trong Pipeline train-only và kiểm lợi ích. | Nêu quyết định “không áp dụng” có lý do |
| Imbalance | Hồi quy không dùng SMOTE. Nhãn “giờ cao” chỉ dùng để đánh giá cảnh báo; nếu lớp hiếm, báo tỷ lệ lớp, PR/F1 và báo giả. | Tỷ lệ nhãn theo từng split, không tái cân bằng test |

- **Làm thêm:** lưu row manifest lý do mất hàng do thiếu lịch sử, nhãn cuối, lỗi chất lượng, ranh giới split. Tách code xử lý thuần dữ liệu khỏi code train; không sửa CSV gốc.
- **Giao:** Duy làm; Khánh kiểm từng quy tắc; Huy biên tập phần “vì sao” từ giải thích của Duy. **Đầu ra:** processed data, manifest, bảng trước/sau xử lý và kiểm thử.

### Cổng 6 — Chia thời gian và đăng ký trước cách thử nghiệm

- **Phương án nghiên cứu:** sau khi xác nhận dữ liệu một năm 2018 có nhịp 15 phút, dùng Jan–Aug để train, Sep để chọn model, Oct để hiệu chỉnh luật cảnh báo, Nov–Dec để đánh giá cuối. Ở mỗi ranh giới, không để nhãn bốn bước của tập trước chạm tập sau; feature của tập sau được dùng lịch sử có thật tới thời điểm `t`. Không shuffle [R6, R7].
- **Làm:** Duy ghi trước ngày cắt, độ dài lịch sử tối thiểu và cách purge; Khánh ghi danh sách metric, số lần mở test và tiêu chí chọn model để kiểm chéo. Duy tạo split manifest với min/max timestamp, số hàng, tỷ lệ giữ/bỏ; scaler/imputer/feature selector chỉ fit trên train. Có thể dùng walk-forward trên train để kiểm độ ổn định, nhưng final validation/calibration/test vẫn giữ vai trò riêng.
- **Lưu ý trung thực:** trong lịch sử đồ án, kết quả Nov–Dec của cùng dữ liệu đã từng được quan sát. Khi bắt đầu lại, nhóm không được gọi nó là một tập “chưa từng nhìn thấy” cho quyết định mới. Nếu tận dụng lại phân hoạch ấy, báo rõ đây là đánh giá tái lập có nguy cơ thiên lệch do hiểu biết trước; để có kiểm định thực sự độc lập cần **dữ liệu tương lai hoặc một nguồn khác** chưa từng tác động thiết kế. Không đổi sang một tháng trong cùng năm chỉ để lấy kết quả đẹp.
- **Kiểm:** không có một nhãn hoặc thống kê fit nào đi từ tương lai qua ranh giới; mọi lựa chọn trước lần chạy test được ghi trong nhật ký. **Giao:** Duy làm script/protocol, Khánh duyệt độc lập, Huy ghi giới hạn. **Đầu ra:** protocol thử nghiệm và split manifest.

### Cổng 7 — Baseline, ít nhất hai phương pháp ML và chọn mô hình

- **Baseline bắt buộc:** dự báo bằng tổng kWh giờ vừa qua; cùng giờ ngày trước; cùng giờ tuần trước nếu có đủ lịch sử. Baseline giúp biết mô hình ML có lợi ích thật so với quy tắc đơn giản.
- **Phương pháp 1 — Ridge Regression:** tuyến tính, ít tham số, dễ giải thích; đặt StandardScaler trong cùng Pipeline. **Phương pháp 2 — HistGradientBoostingRegressor:** nắm quan hệ phi tuyến giữa lag, giờ và ngày; có thể tăng chi phí, khó giải thích hơn. Hai phương pháp dùng cùng target, split, bộ feature được phép và chỉ số.
- **Làm:** Duy định trước dải tham số nhỏ, seed và ngân sách chạy; chọn mô hình theo MAE validation, kiểm RMSE/bias ở giờ tải cao và độ trễ. Nếu MAE hai mô hình chênh dưới 1% tương đối, ưu tiên mô hình đơn giản/nhanh hơn và ghi quy tắc này trước khi xem bảng. Nếu ML không vượt baseline đủ thuyết phục, demo baseline như lựa chọn hợp lý và giải thích kết quả. Không thêm LSTM/GRU chỉ để có Deep Learning; chỉ thử nếu baseline/ML chưa đạt và đủ dữ liệu, thời gian kiểm định độc lập [R1, tr. 9, 14; R5, mục 3.7].
- **Kiểm:** bảng so sánh cùng tập validation, có MAE/RMSE, thời gian train/predict, lý do chọn. **Giao:** Duy train, Khánh kiểm công bằng, Huy kiểm diễn giải. **Đầu ra:** leaderboard validation và model card. Phần train LightGBM trong Markdown của Khánh chỉ trở thành ứng viên bổ sung khi nhóm ghi quyết định; Khánh không phải code baseline/model theo phân công mới.

### Cổng 8 — Định nghĩa cảnh báo và hiệu chỉnh an toàn

- **Ngưỡng:** nếu không có giới hạn vận hành thật, định nghĩa `T` là phân vị 95% của **nhãn tương lai trên train** để mô phỏng “giờ tiêu thụ cao”. Ghi rõ đây là ngưỡng nghiên cứu, **không phải** công suất hợp đồng, giới hạn kW hay điều kiện phạt điện. Nếu doanh nghiệp cung cấp ngưỡng thật, phải kiểm đơn vị và cửa sổ đo trước khi thay.
- **Luật mẫu:** cảnh báo khi `ŷ_t + b > T`; cố định `b = max(0, Q90(y−ŷ))` tính trên residual của calibration Oct. Đây là đệm kinh nghiệm có thể làm tăng báo giả; không gọi `ŷ+b` là phân vị P90 đã hiệu chuẩn nếu chưa huấn luyện/kiểm định quantile model riêng. Cùng một khoảng tải cao có thể tạo nhiều cảnh báo do dự báo chạy mỗi 15 phút, nên định nghĩa gom các cửa sổ liên tiếp thành một sự kiện và thời gian tạm ngưng báo lặp trước khi đo chất lượng.
- **Làm:** Khánh xem trade-off báo giả và bỏ sót trên calibration, tính `T`/`b` và tạo policy; Huy góp ý chi phí kiểm tra vận hành; Duy khóa model, feature, version và policy sau kiểm chéo. Ghi quy tắc khi thiếu dữ liệu: hiển thị “không đủ tin cậy”, yêu cầu kiểm thủ công hoặc baseline dự phòng, không phát cảnh báo như thể model chắc chắn.
- **Kiểm:** ngưỡng lấy từ train, đệm từ calibration, test chưa dùng để sửa chúng. **Đầu ra:** policy card gồm công thức, đơn vị, người duyệt và giới hạn.

### Cổng 9 — Đánh giá ba mức với biểu đồ và phân tích lỗi

| Mức đánh giá | Cách đo và biểu đồ cần tạo | Ý nghĩa quyết định |
| --- | --- | --- |
| Model | MAE, RMSE, bias, sai số theo giờ/thứ/tải cao; biểu đồ actual–forecast theo thời gian, scatter y–ŷ, phân bố residual, lỗi theo phân vị tải. | Mô hình có tốt hơn baseline, sai nhiều ở giờ cần cảnh báo không? |
| System | Precision/Recall/F1 của cảnh báo nếu đủ hai lớp; detection rate/recall theo sự kiện sau khi gom cửa sổ liên tiếp; false alarms/ngày, missed high hours, thời gian dự báo end-to-end, độ trễ dashboard, CPU/RAM/model size; thử thiếu 1–4 mốc, timestamp trùng/sai thứ tự. ROC-AUC chỉ dùng nếu có score liên tục và cả hai lớp. | Người điều độ có bị báo quá nhiều, hệ thống có đủ nhanh và chịu lỗi không? |
| Industrial/business | Lập kịch bản minh họa: số cơ hội kiểm tra sớm, thời gian dẫn trước 15–60 phút, khối lượng cảnh báo/ngày và chi phí giả định có nêu công thức. Chỉ báo tiết kiệm điện hoặc giảm chi phí thực nếu có log can thiệp, giá và nhóm đối chứng. | Kết quả giúp hành động nào, bằng chứng nào còn thiếu để khẳng định giá trị? |

- **Làm:** Khánh thực hiện đánh giá theo protocol; Huy chọn ví dụ cảnh báo đúng, báo giả, bỏ sót và viết “người dùng nên làm gì” cho từng ca; Duy tái chạy metric để kiểm độc lập. Biểu đồ phải ghi giai đoạn và nguồn. Nếu alert class quá ít để tính metric ổn định, ghi số lượng và khoảng bất định thay vì một điểm số đẹp.
- **Kiểm:** báo cáo có bảng số baseline/Ridge/HGB, confusion matrix cho alert, p95 hoặc phân bố latency, ít nhất một fault-injection, và đoạn giới hạn suy luận giá trị công nghiệp. **Đầu ra:** evaluation report + hình + dữ liệu để tái tạo hình.

### Cổng 10 — Kiến trúc và demo Data → AI → Decision

[ARCHITECTURE]

- **Nguồn và xử lý:** CSV lịch sử/replay → kiểm timestamp, nhịp 15 phút, giá trị → feature đúng thời điểm `t` → model đã chọn. **Đầu ra:** dự báo tổng kWh/60 phút và trạng thái chất lượng dữ liệu.
- **Hỗ trợ quyết định:** so với `T` nghiên cứu, hiển thị cảnh báo cùng lý do có thể kiểm như giờ cao/lag cao, mức tin cậy vận hành theo quy tắc chất lượng. Dashboard gồm trạng thái hiện tại, chuỗi quá khứ, dự báo, ngưỡng, cảnh báo, nút người dùng “xác nhận/không xác nhận” và ô ghi chú.
- **Hành động mô phỏng:** kiểm đồng hồ, kiểm lịch sản xuất hoặc nhờ người có thẩm quyền xem xét điều chỉnh phụ tải; **không** tự ngắt máy, tự xả BESS hay ra lệnh PLC. Trong replay, thực tế của 60 phút sau chỉ được hiện ở bước đánh giá khi thời gian mô phỏng đã đi qua.
- **Kiểm:** demo được bốn tình huống: bình thường, cảnh báo, thiếu dữ liệu, dự báo sai. Mỗi tình huống có Data → AI → Decision → người duyệt và log. Demo phải chạy trên máy cá nhân không cần nhà máy thật.
- **Giao:** Huy thiết kế dashboard và luồng người dùng; Duy tích hợp inference; Khánh test chéo. **Đầu ra:** dashboard chạy được, ảnh màn hình, hướng dẫn demo 3–5 phút.

### Cổng 11 — Reliability, robustness, safety, ethics, monitoring

- **Reliability:** sai khi đổi lịch sản xuất, dừng máy, nghỉ lễ, lỗi công tơ hoặc pattern năm khác. Hiển thị lỗi lịch sử theo nhóm điều kiện và cảnh báo khi sai số tăng.
- **Robustness:** thử thiếu dữ liệu, 0 bất thường, timestamp lùi, dữ liệu ngoài miền train. Kết quả phải có chế độ từ chối/fallback rõ ràng; người dùng biết khi nào không nên tin dự báo.
- **Safety và ethics:** báo giả có thể làm gián đoạn kế hoạch; bỏ sót có thể làm mất cơ hội kiểm tra. Hệ thống không tự tác động máy; có quyền người vận hành, log quyết định, thông báo dữ liệu là nghiên cứu, không suy diễn “lỗi của nhân viên”.
- **Human-in-the-loop:** người điều độ kiểm nguyên nhân, xác nhận/huỷ cảnh báo; chỉ người được phân quyền mới thay đổi vận hành. AI không được tự kích hoạt hành động nguy hiểm.
- **Monitoring:** sau khi 60 phút đã trôi qua và nhãn có thật, lưu forecast/actual, MAE rolling, bias giờ cao, báo giả/bỏ sót, tỷ lệ fallback, thay đổi phân bố lag/giờ. Đặt trigger **rà soát** nếu chất lượng suy giảm; retrain chỉ sau điều tra dữ liệu, đánh giá trên dữ liệu mới và duyệt phiên bản. Monitoring không dùng actual tương lai tại thời điểm quyết định.
- **Limitations:** một cơ sở, một năm; không có sản lượng, lịch máy, công suất hợp đồng, giá điện và log can thiệp; chưa chứng minh khả năng tổng quát hay tiết kiệm thực. **Đầu ra:** risk register, monitoring plan, log mẫu và chương riêng trong báo cáo.

### Cổng 12 — Viết báo cáo tám chương theo cùng một bộ bằng chứng

| Chương thầy gợi ý | Phải có | Người nháp / kiểm |
| --- | --- | --- |
| 1. Introduction | Vấn đề, động cơ, mục tiêu, phạm vi, người dùng và quyết định | Duy nháp / Huy biên tập |
| 2. Industrial Problem & Data | Nguồn/DOI, dictionary, đặc tính, chất lượng, 3B, EDA | Duy nháp / Huy biên tập, Khánh kiểm |
| 3. Proposed Industrial AI Solution | Công thức bài toán, AI task, lý do chọn phương pháp, kiến trúc bảy khối | Duy + Huy / Khánh kiểm |
| 4. Data Preparation & Model Development | Thiếu/outlier/scaling/feature/selection/PCA/imbalance, split, train, chống leakage | Duy nháp / Khánh kiểm |
| 5. Evaluation | Bảng model/system/business, biểu đồ, lỗi và giải thích quyết định | Khánh nháp / Duy kiểm, Huy biên tập |
| 6. Prototype / Deployment Design | Data → AI → Decision, workflow, bốn ca demo, người dùng | Huy nháp / Duy + Khánh kiểm |
| 7. Reliability, Safety and Limitations | Failure, robustness, safety, human approval, ethics, giới hạn | Huy nháp / Duy + Khánh kiểm |
| 8. Conclusion | Kết quả có bằng chứng, giá trị hỗ trợ quyết định, bước tiếp theo | Huy nháp / Duy + Khánh kiểm |

- **Quy tắc viết:** mỗi bảng/hình ghi nguồn dữ liệu, split, đơn vị và lệnh tạo. Số trong slide, dashboard và báo cáo lấy từ một file kết quả đã khóa; không gõ tay số khác nhau. Phân biệt **quan sát** (đo từ CSV), **giả định** (ngưỡng p95), **suy luận** (gợi ý vận hành) và **chưa kiểm chứng** (tiết kiệm thực).
- **Đầu ra:** báo cáo Word/PDF, bảng nguồn tham khảo, phụ lục lệnh tái tạo và giới hạn.

### Cổng 13 — Slide, diễn tập, QA và nộp

- **Slide 8–12 trang:** vấn đề → dữ liệu/3B → EDA → target và chống leakage → hai phương pháp → ba mức đánh giá → kiến trúc/demo → an toàn/giới hạn → kết luận. Một slide nên trả lời một câu hỏi, hình có trục/đơn vị.
- **QA độc lập:** Khánh tái tạo dữ liệu/split và hai bảng số từ clone sạch; Duy chạy demo và fault cases; Huy đọc toàn bộ báo cáo để tìm tuyên bố vượt quá dữ liệu. Tập diễn 8–10 phút, mỗi thành viên trả lời được target, split, vì sao chọn model và hành động sau cảnh báo.
- **Gói nộp:** báo cáo; source code + hướng dẫn chạy demo; slide. README có cấu trúc thư mục, lệnh cài/chạy, phiên bản, nguồn dữ liệu, giấy phép và hướng dẫn tái tạo kết quả. File thô có thể tải từ UCI theo chỉ dẫn nếu giới hạn dung lượng; không để đường dẫn tuyệt đối máy Duy trong code.
- **Điều kiện xong:** người ngoài nhóm theo README chạy được `Data → AI → Decision`, xem được hình và metric tương ứng, biết giới hạn của cảnh báo.

## 4. Lịch làm đề xuất cho nhóm 3 người

| Tuần | Cổng chính | Duy — đầu bài/dữ liệu/model | Khánh — calibration/eval | Huy — demo/tài liệu | Điều kiện kết tuần |
| --- | --- | --- | --- | --- | --- |
| 1 | 0–3 | Repo, problem card, UCI/hash, EDA/3B | Kiểm schema, ranh giới EDA | Kiểm nguồn/hình, phác thảo người dùng | Nhóm hiểu dữ liệu và giới hạn |
| 2 | 4–6 | Target, feature, preprocessing, split | Kiểm leakage, protocol, split | Skeleton chương 1–2, kiểm giả định | Pipeline và protocol tái tạo được |
| 3 | 7–8 | Baseline, Ridge, HGB, model card | Kiểm validation, tính calibration/policy | Luồng người dùng và trade-off báo giả | Model/policy khóa trước đánh giá |
| 4 | 9–10 | Tích hợp inference, tái chạy metric | Metric, hình lỗi, fault tests | Dashboard và kiến trúc bảy khối | Demo Data → AI → Decision chạy được |
| 5 | 11–12 | Sửa bug, kiểm chương kỹ thuật | QA độc lập, kiểm số liệu | Monitoring/safety, báo cáo tám chương | Word có bằng chứng và giới hạn |
| 6 | 13 | Kiểm code/source, hợp nhất | Tái tạo trên máy khác | Slide, diễn tập, đóng gói nộp | Ba sản phẩm đồng nhất và đủ |

- **Nếu thời gian ít:** giữ target 60 phút, EDA/3B, hai phương pháp, ba mức đánh giá, demo và an toàn. Cắt bớt mở rộng trước khi bỏ một yêu cầu bắt buộc của thầy.
- **Nếu một cổng chưa đạt:** ghi lỗi và người xử lý; không dùng số liệu của cổng sau để lấp chỗ trống. Việc chốt thực nghiệm phải diễn ra trước khi mở kết quả đánh giá cuối.

## 5. Cấu trúc file và quy trình tái tạo dự kiến

```text
industrial-energy-forecasting/
  README.md                  # đề tài, lệnh chạy, nguồn
  requirements.txt           # phiên bản thư viện
  data/raw/                  # CSV UCI bất biến; kèm manifest/hash
  data/processed/            # split đã chuẩn bị, row manifest
  docs/problem_card.md       # quyết định và ràng buộc
  docs/data_dictionary.md    # cột, đơn vị, availability
  docs/decision_log.md       # lý do chọn mô hình/chính sách
  src/preprocessing.py       # đọc, kiểm, target, feature
  src/modeling.py            # baseline, Ridge, HGB
  src/inference.py           # dự báo tại t và quality gate
  src/monitoring.py          # sai số khi nhãn đã có
  scripts/                   # prepare, train, evaluate
  app.py                     # dashboard replay
  tests/                     # dữ liệu, leakage, split, inference
  reports/figures/           # EDA, đánh giá, kiến trúc
  reports/                   # Word/PDF và slide
```

- **Lệnh khởi tạo dự kiến trên Windows:** `python -m venv .venv`, `./.venv/Scripts/python -m pip install -r requirements.txt`; sau đó chạy script prepare → train → evaluate → demo theo README của phiên bản nhóm chốt. Đây là sơ đồ thư mục **đích**, không phải khẳng định thư mục hiện tại đã đúng y hệt.
- **Kiểm lặp lại:** từ clone sạch, tải raw bằng DOI/đường dẫn UCI, kiểm hash, chạy prepare, train, evaluate, kiểm tests, khởi động dashboard; lưu lệnh, stdout và version vào phụ lục. Tránh lưu model/CSV trung gian vào source nếu không cần cho việc nộp; nếu có thì ghi provenance.

## 6. Cách hiểu “Hướng 1 + Hướng 8” và phần mở rộng

- **Hướng 1 là mục tiêu nghiệp vụ chính:** dự báo tổng kWh 60 phút và cảnh báo giờ tiêu thụ cao. Chứng minh bằng mô hình, ngưỡng minh bạch, precision/recall, số báo giả và demo quyết định.
- **Hướng 8 là lớp bảo đảm:** kiểm chất lượng dữ liệu, đệm cảnh báo được hiệu chỉnh, xem sai số theo thời gian, phát hiện suy giảm để người vận hành rà soát. Human-in-the-loop là phần bắt buộc của demo an toàn.
- **P10/P50/P90 là mở rộng có điều kiện:** chỉ đưa lên dashboard với tên “phân vị” khi đã huấn luyện quantile regression cho từng horizon/target, kiểm coverage và pinball loss trên dữ liệu giữ lại. Nếu chưa làm, trình bày khoảng đệm `b` đúng tên, không đổi nhãn thành P90.
- **BESS, TOU và demand charge là nghiên cứu tiếp:** muốn đề xuất lệnh xả pin hoặc tránh phí công suất cần dữ liệu công suất kW theo cửa sổ tính phí, giá/biểu phí, giới hạn hợp đồng, trạng thái pin, sản lượng và ràng buộc an toàn. Bộ UCI hiện tại không đủ để chứng minh phần này.
- **Deep Learning là thử nghiệm phụ:** chỉ thêm LSTM/GRU nếu còn thời gian sau khi hoàn thành rubric, có protocol so sánh công bằng và dữ liệu mới để kiểm tổng quát. Không dùng độ phức tạp mô hình thay cho giá trị công nghiệp.

## 7. Câu hỏi phản biện nhóm phải tự trả lời

- “Tại sao dùng Steel thay Appliances?” — vì nhóm đặt bài toán ở cơ sở sản xuất thép cụ thể, UCI có chuỗi điện năng 15 phút và metadata nguồn; cần nói đây là quyết định nghiên cứu, xin thầy xác nhận phạm vi nếu ảnh chỉ là đề xuất.
- “Có dữ liệu gì thật?” — timestamp, `Usage_kWh` và các biến điện/nhãn lịch theo metadata [R4]; phải audit CSV. Không có giới hạn công suất hợp đồng, giá điện, lịch sản xuất hay log điều khiển.
- “Đang dự báo kWh hay kW?” — tổng bốn khoảng 15 phút là kWh/60 phút; muốn nói kW trung bình giờ có thể tính từ kWh/1h, nhưng không suy ra đỉnh 15 phút hoặc phí demand charge.
- “Sao không dùng ngay CO₂/Load_Type?” — chỉ thêm nếu chứng minh có sẵn tại `t` và nguồn gốc rõ; biến hậu nghiệm sẽ gây leakage [R2, tr. 8; R6].
- “Vì sao không điền missing/cắt outlier/PCA/SMOTE?” — mỗi quyết định phụ thuộc audit và mục tiêu; nếu không cần phải nêu vì sao, chứng minh đỉnh vẫn được đánh giá.
- “Model nào thắng?” — chỉ trả lời bằng bảng validation của lần chạy có protocol và bảng test dùng đúng vai trò. Nếu hiệu quả không hơn baseline, phải nói thật và điều chỉnh phạm vi ứng dụng.
- “AI giúp tiết kiệm bao nhiêu?” — chưa thể đo từ dữ liệu quan sát đơn thuần; cần giá, hành động, đối chứng và thời gian theo dõi. Bài này chứng minh khả năng hỗ trợ nhận biết sớm và cơ chế an toàn của demo.
- “Có thể triển khai ngay vào nhà máy?” — cần kết nối công tơ, kiểm timestamp/đơn vị, ngưỡng chính thức, thử nghiệm có giám sát, quyền người vận hành và kiểm định trên dữ liệu mới.

## 8. Nguồn tham khảo và cách dẫn trong báo cáo

- **[R0] Yêu cầu đồ án cuối kỳ do giảng viên cung cấp trong lớp:** rubric 8 bước, 3B, hai phương pháp, ba mức đánh giá, kiến trúc bảy khối, demo, an toàn và sản phẩm nộp. Dẫn là “Yêu cầu đồ án học phần”, không gán nội dung này cho sách hoặc UCI.
- **[R1] Huỳnh Thành Lộc, _01_Overview of Industrial AI_, bài giảng cập nhật 09/2026:** tr. 9 (forecasting), 11–14 (problem, 3B, đánh giá), 17 (trust/safety), 19 (energy/process). Bản PDF lớp: `01_Overview of Industrial AI.pdf`.
- **[R2] Huỳnh Thành Lộc, _02_Industrial Problems and Industrial Data_, bài giảng cập nhật 09/2026:** tr. 3–8 (problem, input/output/context, leakage), 16–21 (quality và EDA), 23 (readiness). Bản PDF lớp: `02_Industrial Problems and Industrial Data.pdf`.
- **[R3] Sathishkumar V E, Changsun Shin, Yongyun Cho, _Efficient energy consumption prediction model for a data analytic-enabled industry building in a smart city_, _Building Research & Information_ 49(1), 127–143 (xuất bản online 2020, số tạp chí 2021), DOI `10.1080/09613218.2020.1809983`:** bài báo được UCI liệt kê là tài liệu giới thiệu; dùng làm bối cảnh nghiên cứu, không lấy hiệu năng của bài báo thay cho kết quả nhóm. Trang nhà xuất bản: https://www.tandfonline.com/doi/full/10.1080/09613218.2020.1809983 .
- **[R4] Sathishkumar V E, Changsun Shin, Yongyun Cho (2021), _Steel Industry Energy Consumption_, UCI Machine Learning Repository, DOI `10.24432/C52G8C`:** nguồn dữ liệu chính để đối chiếu 35.040 bản ghi, mô tả cơ sở, danh sách biến, thông báo missing và giấy phép CC BY 4.0. Không dùng metadata thay cho audit CSV. Trang chính thức: https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption .
- **[R5] Jay Lee (2020), _Industrial AI: Applications with Sustainable Performance_, Springer, DOI `10.1007/978-981-15-2144-7`:** §2.7.2 (3B, tr. 26–27 của sách; PDF đính kèm tr. 43–44), §3.4–3.5 (yếu tố công nghệ/5C, tr. 48–54; PDF tr. 65–71), §3.7 (chọn thuật toán, tr. 58–61; PDF tr. 75–78), §4.2.3 (ví dụ quản lý năng lượng LCD, tr. 82–89; PDF tr. 98–105). Ví dụ LCD chỉ là cách tham khảo thiết kế hệ thống, **không phải** bằng chứng tiết kiệm của nhà máy thép. Trang sách: https://link.springer.com/book/10.1007/978-981-15-2144-7 .
- **[R6] scikit-learn, _Common pitfalls and recommended practices_, mục Data leakage:** dùng để lý giải fit preprocessing trên train và tránh thông tin tương lai. https://scikit-learn.org/stable/common_pitfalls.html#data-leakage .
- **[R7] scikit-learn, _TimeSeriesSplit_:** tham chiếu nguyên tắc chia theo thứ tự thời gian và tham số `gap`; việc chia bốn giai đoạn là protocol riêng của nhóm. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html .
- **[R8] scikit-learn, _Lagged features for time series forecasting_:** ví dụ chính thức về lag và kiểm định thời gian; không sao chép số liệu sang Steel. https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html .

**Quy tắc đối chiếu:** báo cáo phải đặt citation ngay cạnh tuyên bố nguồn; kết quả tự chạy ghi script, split, thời điểm và artifact. Với một nhận định chỉ suy từ biểu đồ của nhóm, viết “nhóm quan sát” và chỉ tới hình tương ứng. Không biến giả định vận hành thành thông tin mà UCI không cung cấp.
