# Nhật ký quyết định kỹ thuật — phần Duy

Ngày 01/10/2026. Các lựa chọn dưới đây đã được triển khai trong phạm vi Duy phụ trách. Đây chưa phải chữ ký nghiệm thu của Khánh/Huy.

| Mã | Lựa chọn và lý do | Bằng chứng / người kiểm tiếp |
|---|---|---|
| D01 | Dự báo tổng kWh giờ tới; giữ một target, không gọi là bốn dự báo/kW đỉnh. | Problem card; Khánh/Huy kiểm đơn vị |
| D02 | Raw bất biến, SHA-256; parse định dạng cụ thể, sort theo ngày khai báo; không suy đoán sửa timestamp nửa đêm. | quality summary; Huy kiểm nguồn, cần nhà cung cấp xác nhận semantics khi triển khai thật |
| D03 | Jan–Aug train, Sep validation, Oct calibration, Nov–Dec test; purge bốn nhãn ở mỗi ranh giới. | row manifest; Khánh kiểm temporal protocol |
| D04 | Giữ đỉnh và zero; không nội suy vì dữ liệu gốc đầy đủ; dữ liệu mới lỗi thì từ chối. | audit và 3B; Khánh kiểm lỗi dữ liệu |
| D05 | 15 feature Usage/lịch. Không dùng CO2/Load_Type; chưa nhận các công thức điện học có đơn vị chưa đúng. Ramp là mở rộng hợp lệ nhưng không bắt buộc; chưa đưa vào bộ ứng viên lần này. | bảng availability; đề xuất Khánh được rà soát riêng |
| D06 | Dùng NumPy trên từng cửa sổ cho rolling của lần mới, để batch/replay khớp; giữ hàm cũ cho artifact lịch sử. Không gắn số cũ vào model mới. | `duy_numpy_windows_v1`; parity 31 mốc và tests |
| D07 | Bốn cấu hình chốt trước fit: Ridge alpha 1/100; HGB leaves 15/31, 180 cây, lr .08, seed 42, không early stopping ngẫu nhiên. | protocol lưu trước huấn luyện |
| D08 | Chọn MAE Sep thấp nhất trong cả bốn baseline và bốn AI; hòa ưu tiên baseline. Peak error và tuần được công bố để chẩn đoán, không mở search mới theo chúng. | bảng validation; Khánh kiểm rule |
| D09 | StandardScaler chỉ Ridge fit train; không PCA/SMOTE; clamp dự báo âm về 0 dùng thống nhất. | Pipeline và tests |
| D10 | Không calibration, không chấm test trong phần Duy; cả test cũ và test LightGBM GitHub đã từng được quan sát. | `old_test_already_observed=true`, `alert_policy=null` |
| D11 | Model bàn giao là ứng viên HGB của lần Duy, chưa thay model/policy trong dashboard Huy. LightGBM Khánh chưa bị bác bỏ; cần so sánh validation chung trước khi nhóm đổi model. | `model_handoff.json`; Khánh/Huy kiểm và ký |
| D12 | Giữ giới hạn nghiệp vụ: cảnh báo là hỗ trợ kiểm tra, không chứng minh phí demand/TOU hoặc tiết kiệm điện. | Problem card; Huy kiểm câu chữ |

Lần đầu `duy_2026-10-01` tạo đủ artifact nhưng lỗi in đường dẫn Unicode trên Windows. Đã sửa stdout UTF-8 và chạy lại cùng cấu hình tại **`duy_2026-10-01_v2`**, hoàn tất mã thoát 0. Hai lần cho cùng model hash và số validation; v2 là bản bàn giao. Lần đầu giữ làm dấu vết chẩn đoán, không phải thí nghiệm chọn thêm tham số.

## Cập nhật kỹ thuật ngày 02/10/2026

- Bản bàn giao để xuất bản: `reports/steel/duy_2026-10-02`. Giữ target, split, 15 feature và bốn cấu hình AI của Duy; không search thêm tham số.
- Duy sửa đường dẫn model đa nền tảng, hỗ trợ baseline thắng, kiểm đầu vào và xử lý trường hợp tải repo dạng ZIP không có Git.
- QA FE-01–06 kiểm pipeline và CSV thật; dữ liệu sai/thiếu phải BLOCKER và mã thoát 1. Đây là kiểm tự động, không thay chữ ký nghiệm thu của Khánh.
- Notebook 02 dùng model đã khóa của Duy; LightGBM cũ được giữ làm lịch sử, hai công tắc calibration/test mặc định False.
- Chưa gắn policy của LightGBM vào HGB; cần nhóm duyệt protocol/model/policy trước tích hợp dashboard. Các đề xuất thay đổi nghiệp vụ vẫn chờ nhóm chốt.

- Trong lúc tích hợp đã nhận commit GitHub mới `12eed4b`: giữ nguyên source nhóm và nối demo/monitoring với artifact Duy qua adapter, không đổi model hoặc tái tuning. Hai xung đột README/requirements được giải quyết có lưu README upstream tham chiếu.
