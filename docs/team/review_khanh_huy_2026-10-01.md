# Ghi chú rà soát phần Khánh và Huy — 01/10/2026

## Phạm vi bằng chứng

- Đọc Notion trực tiếp: “Đề xuất gửi Duy”, “30/9/2026”, “demo_safety.md”, “steel_inference_handoff.md”. Không chỉnh nội dung gốc của các bạn.
- Đối chiếu code GitHub tại commit **f0404c5ab0d8a29136ace30071180cd32c19add3**: `scripts/audit_features_and_leakage.py`, `notebooks/02_steel_modeling_evaluation.ipynb`, `notebooks/artifacts/test_results.json`, requirements và preprocessing.
- Bản local HEAD **05e6d46** có nhiều file chưa commit. Không pull/merge đè lên bản local. Đã fetch để đọc source GitHub; phần Duy đặt ở module và thư mục kết quả riêng.
- GitHub có notebook/model LightGBM của Khánh. Code dashboard/API HGB mà Huy mô tả **chưa có trong cây file của commit GitHub đã kiểm**; hiện chỉ đánh giá được hợp đồng/tài liệu Huy. Không kết luận code Huy đã pass/fail khi chưa nhận source.
- Không chạy lại notebook Khánh để tránh ghi đè artifact/test hoặc vô tình chọn theo test. Lỗi dưới đây được đối chiếu tĩnh với source và API thư viện hiện tại.

## Khánh cần sửa trước nghiệm thu QA

### K01 — FE-02 chưa kiểm pipeline thật — cần sửa

- [Source dòng 79–100](https://github.com/trangkhanh-ai/steel-industry-energy-consumption/blob/f0404c5ab0d8a29136ace30071180cd32c19add3/scripts/audit_features_and_leakage.py#L79) tạo chuỗi ngẫu nhiên và hai rolling mẫu; không gọi `build_supervised_rows` của Duy.
- PASS hiện chỉ chứng minh ví dụ viết trong audit đúng. Nếu pipeline thực đổi sang centered rolling, check này vẫn có thể PASS.
- Sửa: lấy raw thật, tính pipeline, đổi dữ liệu sau `t`, tính lại; assert toàn bộ feature tại `t` không đổi và target đổi đúng. Có thể đọc tests của Duy để làm kiểm độc lập, không chỉ sao chép báo PASS.
- Không yêu cầu mọi rolling phải `.shift(1)`/`closed='left'`: bài toán phát dự báo **sau khi bản đo tại t đã có**, nên dùng `Usage(t)` và rolling kết thúc tại t là hợp lệ. Câu chữ hiện tại mô tả khác code thật; áp dụng cả shift và closed left còn lùi cửa sổ thêm bước.

### K02 — FE-06 đọc sai schema và bỏ qua kết quả điều kiện — cần sửa

- [Source dòng 198](https://github.com/trangkhanh-ai/steel-industry-energy-consumption/blob/f0404c5ab0d8a29136ace30071180cd32c19add3/scripts/audit_features_and_leakage.py#L198) đọc `manifest_df['status']`; manifest Duy có `row_status` và `reason`.
- `burn_in_ok`, `boundary_purged` được tính nhưng không dùng để quyết định PASS; chỉ số hàng manifest đúng là vẫn PASS. Thiếu manifest cũng trả True.
- Sửa: kiểm `reason` chính xác included=34.352, insufficient_history=672, boundary=12, future_label_unavailable=4; kiểm từng timestamp, `forecast_end < split_end`, cân bằng tổng35.040; thiếu manifest/cột thì BLOCKER.
- Khi có BLOCKER, runner phải trả exit code khác 0; hiện đoạn cuối chỉ in câu lỗi nên tự động hóa không biết audit thất bại.

### K03 — FE-01/FE-03 chưa chứng minh schema model — cần sửa

- FE-01 chỉ kiểm thiếu cột, chưa kiểm thứ tự hoặc cột dư trong **ma trận thực đưa vào model**.
- [FE-03 dòng 120](https://github.com/trangkhanh-ai/steel-industry-energy-consumption/blob/f0404c5ab0d8a29136ace30071180cd32c19add3/scripts/audit_features_and_leakage.py#L120) yêu cầu biến cấm đồng thời nằm trong danh sách BASE đã hard-code không chứa các tên đó; điều kiện không bắt được biến cấm thêm vào `ALL_FEATURES` của notebook.
- Sửa: audit danh sách feature thực dùng/`model.feature_names_in_` và metadata chính xác theo thứ tự. Phân biệt CSV có cột sensor để thử sau với model thật chỉ chọn15 feature.
- FE-05 dựa số hàng/mean không đủ chứng minh không SMOTE hoặc scaler train-only. Kiểm scaler `mean_` bằng thống kê train và không đổi sau predict validation; đối chiếu giá trị/timestamp với raw.

## Khánh cần sửa trước bàn giao Modeling & Evaluation

### K04 — API RMSE và dependency chưa tái chạy được

- Notebook dùng `mean_squared_error(..., squared=False)`. Môi trường hiện tại scikit-learn1.8 không có tham số này; dùng `root_mean_squared_error` hoặc `np.sqrt(mean_squared_error(...))`.
- [requirements ở commit kiểm](https://github.com/trangkhanh-ai/steel-industry-energy-consumption/blob/f0404c5ab0d8a29136ace30071180cd32c19add3/requirements.txt) chỉ có NumPy/pandas/Matplotlib; thiếu sklearn/LightGBM/Jupyter để tái tạo notebook.
- Không nói kết quả cũ là sai chỉ vì API mới không chạy. Cần ghi đúng phiên bản đã tạo kết quả cũ và cung cấp môi trường tái tạo.
- Không ẩn toàn bộ warnings; chỉ xử lý cảnh báo cụ thể sau khi biết nguyên nhân.
- Nguồn đối chiếu: [API RMSE scikit-learn1.8](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.root_mean_squared_error.html).

### K05 — Diurnal baseline làm tròn giờ và phần chọn model bằng test

- Cell “Step 7” giải mã sin/cos rồi `np.round`: `forecast_start=00:45` thành giờ1; `23:45` thành0 nhưng thứ vẫn giữ ngày cũ. Đây là grouping khác với hour-of-day đã mô tả.
- Sửa: parse `forecast_start`, dùng `.dt.hour` và `.dt.dayofweek`; nếu muốn slot15 phút thì dùng672slot/tuần, ghi rõ và học lookup chỉ từ train. Bản Duy dùng168slot với giờ nguyên.
- Cell “Step 8” lấy `argmin(maes)` trên test và in “Best model on test MAE”. Nếu chỉ xếp bảng hậu nghiệm, ghi rõ không dùng kết quả để chọn triển khai. Chốt model từ validation trước calibration/test; không lấy winner test làm quyết định.
- Test đã được xem nhiều lần nên không gọi là blind holdout trong báo cáo mới.

### K06 — Bagging LightGBM chưa được bật

- Notebook đặt `subsample=.8` nhưng không có `subsample_freq/bagging_freq >0`; với mặc định0, row bagging không hoạt động.
- Sửa trước một thí nghiệm mới được ghi protocol: thêm `bagging_freq=1` hoặc bỏ tuyên bố model đã dùng row subsampling. Không sửa rồi gắn số test cũ vào model mới.
- Nguồn: [LightGBM Parameters: bagging](https://lightgbm.readthedocs.io/en/stable/Parameters.html#bagging_freq).

### K07 — Thiếu policy gắn đúng artifact

- JSON test có `q_buffer_kWh`, nhưng thiếu model hash, thứ tự feature, timestamp convention, T/nguồnT, định nghĩa episode, fallback, versions và schema trạng thái.
- Sửa: xuất policy/model card có đủ nguồn T từ train, b từ Oct, toán tử `>`, quantile convention, model/data hash, seed/versions và script tái tạo. Không lấy **42.35kWh của LightGBM** để áp dụng HGB Duy/Huy.
- Thống nhất `max(0,Q90(y−prediction))` hay `Q90(max(y−prediction,0))`. Hai cách thường cho cùng số khi Q90 dương, nhưng không phải một công thức ở mọi tình huống nội suy phân vị.
- Đệm là luật kinh nghiệm toàn cục. Chưa phải quantile forecast P90, khoảng tin cậy80% hay xác suất vượt ngưỡng.

### K08 — Episode/lead time và “Level3” đang bị diễn giải quá mức

- Notebook dùng overlap nhiều–nhiều để đếm episode được phát hiện và episode cảnh báo có ích. Đây là một quy ước có thể dùng, nhưng không phải matching một–một; hai mẫu số có thể dùng hai số true-positive khác nhau.
- Episode recall22/33 và precision24/49 không mâu thuẫn theo quy ước đó. Cần công bố quy ước và không gọi F1 như F1event một–một nếu chưa matching.
- Group phải tách tại gap timestamp. Cần định nghĩa event là chuỗi **cửa sổ target giờ tới** hay sự kiện tiêu thụ15 phút; hai loại không giống nhau.
- “Phát hiện trước60phút” chưa được code đo: cửa sổ dự báo60phút không đảm bảo khoảng đệm60phút trước lúc tải thực bắt đầu cao. Đo lead-time dựa thời điểm cảnh báo và bắt đầu sự kiện thực.
- Event recall/precision vẫn là đánh giá cảnh báo. Để đáp ứng Industrial/Business level, thêm bảng hỗ trợ quyết định: tỷ lệ sự kiện cần kiểm được cảnh báo, số lượt kiểm/ngày, thời gian phản ứng **đo được**; ghi là proxy, không phải tiết kiệm/downtime thực.
- TOU là cấu trúc giá theo thời gian; demand charge là phí theo demand. Muốn mô phỏng cần hợp đồng/biểu giá và đại lượng đo phù hợp. TổngkWh giờ không chứng minh ngăn phạt đỉnhkW tháng.

## Nhận xét “Đề xuất gửi Duy” của Khánh

- Giữ BASE rồi thử extension cùng validation là hợp lý. `steel_ramp_rate=Usage(t)−Usage(t−1)` chỉ biết tới t, có thể thử; Ridge vốn đã có hai biến thành phần nên đây là tổ hợp tuyến tính dư thừa đối với Ridge, nhưng có thể hữu ích cho cây.
- `S=sqrt(P²+Q²)` với P nhập bằng **kWh**, Q bằng **kVarh** không phải công suất kVA. Nếu cùng khoảng15 phút, cần quy đổi điện năng sang công suất trung bình và nêu giả định; trường lagging/leading riêng và dạng sóng/harmonic cũng chưa đủ để xác nhận S thật. Hiện chỉ có thể gọi là feature/proxy có định nghĩa, không gắn nhãn đo công suất biểu kiến thực.
- `Q/(P+1e-5)` là tỷ số năng lượng/proxy; tại Usage gần 0 có thể phóng đại. Quy định validity-mask/tolerance với lý do, không tự lấy epsilon làm bằng chứng vật lý.
- Cờ PF<90 nên đặt tên `low_lagging_pf_flag` với ngưỡng nghiên cứu, không gọi penalty risk nếu thiếu hợp đồng/nguồn. PF trong CSV là phần trăm, cần đổi sang0–1 nếu dùng hàm lượng giác/công thức điện.
- `Load_Type` chỉ thử sau khi biết nghĩa/availability; one-hot nếu không chứng minh thứ tự. Không tự dùng nhãn Load_Type tương lai.
- Tương quan CO2 cao không tự chứng minh leakage. Điều quyết định là biến có sẵn khi dự báo hay được suy từ target tương lai. Loại CO2 ở bản chính vì semantics/availability chưa đủ rõ; metadata UCI còn ghi ppm dù tên cột tCO2.
- Cùng lý do, Qt/PFt ở thời điểm t **không tự động** là leakage nếu đã đo trước issue. FE-03 cần kiểm timestamp availability, thay vì áp quy tắc “mọi biến điện phải lag” cho mọi bài toán.

## Huy: điểm hợp lý và việc cần hoàn thiện

- Tài liệu phân biệt HGB/LightGBM, actual trễ60phút, prediction-only, người xác nhận, lỗi dữ liệu và số chưa có là N/A: giữ các nguyên tắc này.
- Huy đã chỉ ra rolling toàn chuỗi có thể lệch cửa sổ, đổi nhánh cây; đã công bố fit lại và sốvalidation mới. Duy xử lý cùng rủi ro bằng `duy_numpy_windows_v1`, có test và parity; hai artifact vẫn khác nhau, không thể thay tên rồi dùng lại số.
- **H01:** bàn giao code thật/commit cho các module/API, scripts, tests và web mà Notion nêu; hiện GitHub chưa có để Duy review runtime. Các đường dẫn `../steel-industry-energy-consumption/...` từ máy Huy cần thay bằng đường dẫn gốc repo nhất quán.
- **H02:** khi nhận model mới, adapter phải đọc model_handoff, hash, schema và feature implementation; dataset processed mới nằm trong thư mục run Duy. Không lấy feature từ CSV/pandas cũ để gọi model NumPy mới.
- **H03:** nhận policy đúng model từ Khánh mới bật cảnh báo. Không dùng q=42.35 của LightGBM cho HGB; không dùng dự báo kWh để tô trạng thái “không quá tải” kW.
- **H04:** tests nên gồm cả đoạnraw thiếu bản ghi **và** input NaN/Inf/âm và model hỏng. Lỗi giả lập trong test không sửa raw gốc và không phải ngụy tạo dữ liệu nghiên cứu; nhờ đó kiểm được các đường lỗi chưa xuất hiện trong raw sạch.
- **H05:** khi tính monitoring bằng `.dt`/giờ, kiểm label chỉ có sauforecast_end; retry và overlapping hour không phải mẫu độc lập. Nêu mẫu số và không biến MAE validation thành chứng minh drift.
- **H06:** giữ trạng thái chưa kiểm máy khác và chờ review. Vẫn cần ca cảnh báo có policy, lỗi artifact, phiên cũ/log, thời điểm00:00 và ba ca phản hồi người dùng để nghiệm thu.

## Thứ tự xử lý

1. Khánh sửa K01–K05 và xuất protocol/policy đúng artifact. Gửi code và kết quả tái tạo để nhóm kiểm, không chỉ câu PASS.
2. Duy bàn giao bản v2 HGB với15 feature, validation vàAPI prediction-only. Đây là ứng viên theo kế hoạch hiện hành; cần thống nhất với LightGBM trên cùng validation nếu nhóm muốn so thêm.
3. Huy nhận đúngmodel + policy, nối adapter và kiểm toàn luồng. Duy kiểm runtime khi code được bàn giao.
4. Khánh/Huy/Duy ghi nghiệm thu từng cổng, rồi mới cập nhật các tuyên bố hoàn tất trong tài liệu nhóm.
