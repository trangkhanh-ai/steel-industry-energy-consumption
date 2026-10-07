# Problem card — phần Duy, bản triển khai 01/10/2026

## 1. Đề tài và vấn đề thực tế

- **Tên:** Dự báo tổng điện năng tiêu thụ 60 phút tiếp theo tại cơ sở sản xuất thép và hỗ trợ cảnh báo sớm khoảng tiêu thụ cao.
- **Hiện trạng cần hỗ trợ:** đồng hồ cho biết mức tiêu thụ đã xảy ra; người điều độ cần thông tin dự báo để ưu tiên kiểm tra trước các khoảng tiêu thụ cao.
- **Bối cảnh:** UCI mô tả dữ liệu từ một cơ sở thép nhỏ tại Hàn Quốc. Không khẳng định cơ sở đó đã bị quá tải, bị phạt hoặc có nhu cầu điều độ được doanh nghiệp xác nhận.
- **Khó khăn quan sát được:** dữ liệu gốc không theo thứ tự thời gian; tiêu thụ biến động; thiếu lịch sản xuất và nguyên nhân vận hành. Không tự quy đỉnh điện cho nạp phôi, mẻ thép hoặc thời tiết.

## 2. Mục tiêu đo được

- Đầu vào: các phép đo `Usage_kWh` đã có tới thời điểm `t`, cùng lịch giờ/thứ biết trước.
- Đầu ra AI: `y_t = Usage(t+1)+Usage(t+2)+Usage(t+3)+Usage(t+4)`, một số **kWh trong 60 phút tiếp theo**; cập nhật mỗi 15 phút.
- Sau AI: Khánh hiệu chỉnh luật cảnh báo; Huy hiển thị dự báo, chất lượng, cảnh báo và ghi quyết định của người dùng.
- Thành công bước model: so sánh cùng split giữa baseline, Ridge và HGB; chọn MAE validation thấp nhất, đồng thời công bố RMSE, bias và sai số vùng cao.
- Thành công toàn đồ án: Data → AI → Decision chạy được; có ba mức đánh giá, kiến trúc, reliability/safety/limitations. Chỉ model chạy được chưa hoàn thành toàn bộ yêu cầu thầy.

## 3. Người dùng và quyết định

- **Người điều độ ca / kỹ sư năng lượng, giả định:** xem dự báo, kiểm lịch vận hành hoặc đồng hồ khi được cảnh báo; ghi nhận, đề nghị kiểm tra hoặc bỏ qua kèm lý do.
- **Quản lý năng lượng:** xem các ca sai, tỷ lệ báo giả/bỏ sót và yêu cầu rà soát.
- **Duy:** kiểm dữ liệu/model/version và tích hợp phiên bản được nhóm chốt.
- **Khánh:** kiểm policy và chỉ số; **Huy:** triển khai giao diện, giám sát và nhật ký.
- Không có lệnh tự ngắt máy, dời sản xuất hoặc xả pin. Dataset không có danh sách tải điều khiển được, ràng buộc sản xuất hay BESS.

## 4. Quy tắc ra quyết định dự kiến

1. Thiếu lịch sử, sai timestamp, giá trị không hữu hạn hoặc âm → từ chối dự báo, yêu cầu kiểm dữ liệu.
2. Model hợp lệ và policy cùng phiên bản đã được Khánh bàn giao → dùng luật `prediction + buffer > T` để đề nghị người vận hành kiểm tra.
3. Chưa có policy → hiển thị **chỉ dự báo, chưa đánh giá cảnh báo**; không hiển thị trạng thái an toàn.
4. Sau khi đủ bốn phép đo tương lai → tính actual và sai số. Không cho người dùng xem actual sớm trong replay.
5. Mọi hành động vận hành thật cần người có quyền kiểm điều kiện an toàn trước khi thực hiện.

## 5. Ràng buộc và giả định cần công bố

- Chỉ một cơ sở, năm 2018; không đảm bảo áp dụng được cho nhà máy khác hoặc năm khác.
- Dùng timestamp khai báo trong CSV. Nguồn chưa xác nhận rõ timestamp là đầu/cuối khoảng đo; giả định phép đo tại `t` đã hoàn tất và có sẵn khi phát dự báo. Không tự chuyển các hàng 00:00 sang ngày sau.
- Một bản đo được giả định là điện năng của một khoảng 15 phút. Đây là hợp đồng của prototype, phải xác nhận với nhà máy nếu triển khai thật.
- Tổng kWh giờ tới không cho biết đỉnh kW trong từng khoảng hoặc tháng; không chứng minh khả năng tránh demand charges.
- Không có hợp đồng, TOU, sản lượng, downtime, hành động hoặc đo đối chứng; không tính tiền điện/kWh tiết kiệm thực tế.
- Test Nov–Dec từng được xem, kể cả số LightGBM của Khánh. Lần mới là tái triển khai phục vụ học tập; chưa có test độc lập mới.
- Lần chạy Duy chỉ sử dụng target train/validation để fit và chọn; calibration/test được chuẩn bị và bàn giao, chưa tính điểm.

## 6. Bàn giao và đối chiếu yêu cầu thầy

- Problem Definition: problem card này.
- Data Understanding/3B: quality summary, profile train và 7 hình EDA.
- Data Preparation: parse/sort, quality gate, target/feature, split, manifest; giải thích các lựa chọn trong `duy_implementation_and_learning.md`.
- Method Selection/Development: bốn baseline; hai cấu hình Ridge và hai HGB; protocol lưu trước fit; kết quả validation, model và manifest.
- Evaluation: Duy đưa dữ liệu/validation/ca sai; Khánh làm model/system/industrial và calibration.
- Deployment/Monitoring: Duy giao API dự báo; Huy hoàn thiện dashboard/log/safety theo model và policy được chốt.
- Chương Duy đóng góp: 1, 2, phần phương pháp của 3 và chương 4; cung cấp số/hình cho chương 5–7.

**Nguồn:** [UCI Steel, DOI 10.24432/C52G8C](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption); yêu cầu thầy đã gửi; hai bài giảng 01/02; [Jay Lee, Industrial AI (2020)](https://doi.org/10.1007/978-981-15-2144-7).
