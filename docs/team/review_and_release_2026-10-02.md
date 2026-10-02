# Rà soát code và bản bàn giao ngày 02/10/2026

## 1. Phạm vi và phiên bản

- Nguồn tích hợp: GitHub `trangkhanh-ai/steel-industry-energy-consumption`, nền commit `f0404c5ab0d8a29136ace30071180cd32c19add3`, sau đó tích hợp commit mới `12eed4b201b8a69e1c6fa3e870febafe0d085b07` của nhóm.
- Không làm mất thay đổi local trước đó hoặc sửa CSV UCI. Bản bàn giao mới nằm riêng tại `reports/steel/duy_2026-10-02`.
- Luồng hiện tại: **dự báo tổng kWh của giờ tới**, sau đó lớp cảnh báo do Khánh hiệu chỉnh. Kế hoạch nhóm và problem card giữ cùng mục tiêu.
- Giữ nguyên 15 feature, lịch split và bốn cấu hình Ridge/HGB; kết quả tái chạy cùng MAE/RMSE của Duy ngày 01/10.
- Tác giả commit dùng danh tính Git của Duy: `dzyuu1612 <baoduynguyen1612@gmail.com>`.

## 2. Lỗi đã vá

| Phần | Vấn đề tìm thấy | Cách sửa |
|---|---|---|
| Inference | Đường dẫn model Windows có dấu backslash gây lỗi trên Linux | Ghi đường dẫn POSIX; loader đọc được cả manifest Windows cũ; chặn path ra ngoài run |
| Chọn/bàn giao | Cho baseline thắng nhưng loader chỉ nhận AI | Lưu state diurnal học train; inference hỗ trợ cả bốn baseline; kiểm checksum state |
| Input | Thiếu/trùng cột gây lỗi khó hiểu; tính rolling trên cả lịch sử dài | Thông báo ValueError rõ; kiểm toàn bộ lịch sử rồi tính rolling trên 673 hàng cần thiết |
| Baseline | Tính mean train lặp lại ở từng hàng | Fit state một lần; lấy giờ thật từ timestamp, gồm ca 00:45 |
| ZIP | Train yêu cầu Git dù người dùng tải source dạng ZIP | Ghi commit/dirty=null nếu không có Git; vẫn chạy |
| QA FE-02 | Test chuỗi giả không gọi pipeline thật | Perturb Usage tương lai trên CSV thật tại ba mốc; kiểm feature bất biến và target đổi đúng |
| QA FE-01/03/04 | Cột thừa/thứ tự sai hoặc kiểm dùng danh sách cố định có thể lọt | Kiểm metadata/CSV thật; availability đúng hợp đồng; CO2 correlation chỉ trên train |
| QA FE-05 | Số hàng/mean không đủ chứng minh CSV đúng | Tái tạo và đối chiếu mọi giá trị/thời gian/nhãn cả bốn split; không gọi đây là chứng minh scaler train-only |
| QA FE-06 | Đọc `status` trong khi manifest dùng `reason`; thiếu file vẫn PASS | So sánh manifest từng hàng và metadata; missing/mismatch BLOCKER; CLI thoát 1 |
| Notebook Khánh | Sai key metadata, RMSE API cũ, làm tròn sai giờ, chọn model trên test, trộn model/policy | Notebook 02 hiện dùng model khóa của Duy và module evaluation đơn giản; không fallback feature; mặc định calibration/test=False |
| Episode metric | Ghép theo index có thể nối qua gap; diễn giải chồng lấp thành lead time | Tách cụm theo lưới 15 phút; ghi rõ many-to-many overlap tại forecast origin, chưa đo lead time |
| Tái tạo | Hash nguồn thay đổi do CRLF khi clone | `.gitattributes` giữ Python LF; raw CSV giữ nguyên byte |

## 3. Minh bạch kết quả và lịch sử

- 34 unittest phần core đạt: preprocessing 10, Duy 16, QA 4, evaluation 4. Sau nhận source demo/monitoring mới: bộ đầy đủ gồm 62 tests, chạy với model Duy trong môi trường cài mới.
- QA FE-01–06 đạt trên dữ liệu processed trước đây và processed của run mới; dữ liệu bị sửa trong kiểm thử phải thất bại.
- Model chọn: HGB 31 lá; validation MAE **15,8828504916 kWh**, RMSE **32,0474942141 kWh**.
- Baseline tốt nhất: last_hour; MAE **32,3237726008 kWh**; cải thiện MAE **50,86%** trên validation.
- Vùng cao: 93 mẫu, MAE **68,6963150856 kWh**. 31 mốc batch/replay khớp feature và prediction; chưa chứng minh cảnh báo an toàn.
- Chưa chấm calibration/test mới; test cũ đã được quan sát. Chưa đo tiết kiệm thực tế hoặc latency toàn hệ thống.
- LightGBM/notebook trước sửa được giữ trong `notebooks/archive/`; artifact cũ giữ trong `notebooks/artifacts/` với ghi chú. Không lấy buffer/metric LightGBM gắn cho HGB.
- Lần 01/10 vẫn được ghi trong decision log như lịch sử; run 02/10 là bản source/artifact bàn giao hiện tại.

## 4. Việc còn lại cho nhóm

### Duy

- Giải thích được problem card, 3B, target, 15 feature, split 672/12/4, baseline/Ridge/HGB và ca sai.
- Cùng Khánh/Huy kiểm chéo artifact; mọi thay đổi mô hình hoặc nghiệp vụ ghi decision log.

### Khánh

- Nghiệm thu model/schema/source hash của run mới; công cụ QA không tự thay chữ ký người kiểm.
- Chốt model HGB hay thử nghiệm LightGBM riêng bằng validation; nếu đổi phải tạo artifact/replay/policy mới.
- Bật calibration tháng 10 trong notebook 02 sau chốt protocol; khóa buffer và threshold gắn đúng hash/model.
- Chấm test tháng 11–12 với model/policy khóa; báo MAE/RMSE, false alarms, false negatives, precision/recall/F1, FPR/FDR, episode overlap và failure cases.
- Kiểm robustness/drift và empirical coverage; buffer Q90 lỗi thiếu chưa tự trở thành P90 hoặc dải tin cậy có bảo đảm.
- Nếu phát triển quantile P10/P50/P90, phải dùng đúng định nghĩa, pinball loss, coverage, width và kiểm crossing; không gọi P50 là kỳ vọng trung bình.

### Huy

- Code dashboard/monitoring đã nhận từ commit `12eed4b`; Duy nối adapter `SteelForecaster` vào artifact Duy và đổi default demo/check/predict/replay. Huy cần nghiệm thu UI trên máy mình và tích hợp policy.
- UI chỉ dùng model và policy cùng phiên bản. Khi chưa có policy, hiện rõ forecast-only; dữ liệu lỗi/thiếu lịch sử phải yêu cầu kiểm lại.
- Tích hợp người duyệt, khuyến nghị kiểm tra, log quyết định; không tự dừng máy hoặc giảm tải.
- Đo latency từ dữ liệu qua feature/model/policy tới UI; kiểm artifact lỗi, phiên cũ, 00:00, mất mốc, NaN/Inf và user feedback.
- Theo dõi data quality, label delay 60 phút, residual sau khi có nhãn, drift, phiên bản và lịch hiệu chỉnh.
- Giá trị công nghiệp: kịch bản mô phỏng có giả định, không khẳng định tiền phạt/tiết kiệm khi chưa có hợp đồng, biểu giá và hành động thật.

## 5. Cách nghiệm thu nhanh

1. Clone repo, cài requirements và chạy 34 tests.
2. Chạy `audit_features_and_leakage` cho đúng processed; không bỏ qua BLOCKER.
3. Chạy `check_duy_handoff` để kiểm source hash, model và batch/replay tại 08/09 02:00.
4. Chạy `predict_duy_steel` từ lịch sử tại 01/09 08:00, không cung cấp actual tương lai cho API.
5. Xem notebook 03, bảng validation/ca sai và problem card rồi ký bàn giao.

Lệnh chi tiết ở README. Kiểm bản clone/ZIP trên cùng máy là kiểm tính di động của source/artifact; chưa thay việc Khánh/Huy chạy trên máy và hệ điều hành của mình.

## 6. Tài liệu đối chiếu

- Rubric của thầy: tám bước lifecycle, 3B, lý do preprocessing, ít nhất hai phương pháp, ba mức đánh giá, kiến trúc bảy khối, local demo và safety/HITL.
- [Lee — Industrial AI (2020)](https://doi.org/10.1007/978-981-15-2144-7): đối chiếu 3B và industrial context trong kế hoạch A–Z.
- [sklearn 1.8 Common pitfalls](https://scikit-learn.org/1.8/common_pitfalls.html), [RMSE API](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.root_mean_squared_error.html).
- [LightGBM Parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html): `subsample` cần `bagging_freq > 0` để hoạt động; bản cũ dùng objective regression_l1, không phải Huber như phần đầu notebook từng ghi.

## 7. Tích hợp cập nhật GitHub trong lúc rà soát

- `12eed4b` bổ sung source modeling/tuning, inference, replay, demo web và SQLite monitoring. Không bỏ thay đổi này khi đẩy phần Duy.
- Hai xung đột README/requirements được giải quyết bằng README thống nhất và các phiên bản đã chạy model Duy; README upstream giữ thành tài liệu tham chiếu lịch sử.
- Adapter nhận `model_handoff.json` của Duy, giữ nguyên model hash/dự báo, trả schema mà monitor/UI đang dùng. Adapter vẫn đọc được manifest `inference_v1` cũ khi người dùng chọn rõ đường dẫn artifact đó.
- Default demo/check/predict/replay và monitored replay dùng run Duy; đọc CSV inference bằng round_trip để giữ số học.
- Tests demo/monitoring dùng artifact công bố trong Git thay việc bị skip do model local thiếu. Test loaded inference đối chiếu bốn mốc với prediction Duy đã lưu.
- JSON baseline state ghi LF ngay khi tạo, tránh checksum đổi sau checkout; kiểm bàn giao xác minh cả hash state và source.
- Dashboard hiện là forecast/observation mode, không bật ngưỡng/buffer của model khác. SQLite ghi actual khi đủ số đo đã tới, mỗi reset tạo phiên riêng; theo dõi residual chưa tự là trigger drift.

## 8. Bằng chứng nghiệm thu tự động

- Môi trường .venv mới cài đúng requirements: 62/62 tests đạt, không skip; kiểm startup demo đạt.
- Notebook 03 thực thi 8 cell code, notebook 02 thực thi 3 cell code với calibration/test=False, không error output.
- Edge desktop/mobile: 24 bước đo, không lỗi JavaScript; kiểm actual trễ, play/pause, đổi phiên, cuối tháng, nhập sai ngày, chống ghi trùng/phiên cũ, text escape và nhật ký sau reload. Ảnh và JSON ở `reports/steel/duy_2026-10-02/demo_browser_check`.
- Đo UI cục bộ trong lúc chạy replay khác: p95 request-to-render 130,95 ms, backend p95 115,18 ms. Đây là kiểm chức năng trong điều kiện có tác vụ khác, chưa chứng minh SLA 100 ms hoặc tốc độ nhà máy. Không thay số đo model-only bằng số này.
- CLI replay đã sửa phần report để không phụ thuộc key riêng của inference_v1; kiểm nhanh 8 mốc exit 0, feature/prediction khớp. Bảng MAE của 8 mốc chỉ mô tả sample này; kết quả model chính vẫn là toàn validation tháng 9.
- Bằng chứng có cấu trúc trong `release_verification.json`; hash baseline state/code/model kiểm thêm từ bản clone/ZIP.
