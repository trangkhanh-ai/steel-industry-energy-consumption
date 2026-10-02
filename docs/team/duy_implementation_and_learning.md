# Duy — hướng dẫn code, kết quả và tài liệu học

**Bản bàn giao:** `reports/steel/duy_2026-10-02/`, tái chạy ngày 02/10/2026. Đây là kết quả train/validation mới trên dữ liệu cũ. Kế hoạch A–Z vẫn ở `ke_hoach_nhom_3_nguoi.md`; cần Khánh/Huy kiểm chéo trước khi ký nghiệm thu.

**Đọc trực quan từng bước:** [Notebook phần Duy](../../notebooks/03_duy_steel_data_and_models.ipynb), đã thực thi với hình và output. **Notion:** [code, kết quả và lộ trình học](https://app.notion.com/p/3ec7c27769028108ae9bcca9664ca756?pvs=204) · [ghi chú Khánh/Huy](https://app.notion.com/p/3ec7c277690281118c0ac50e96c2131a?pvs=204). Đã qua 34 kiểm thử dữ liệu/model/inference/QA/evaluation. Xem bản rà soát ngày 02/10 cho trạng thái xuất bản và kiểm môi trường.

## 1. Phần Duy và thứ tự đọc code

- Duy làm cổng 0–7: problem, dữ liệu, EDA/3B, target, preprocessing, split, baseline và model.
- Khánh nhận model/schema để calibration tháng 10 và đánh giá ba mức.
- Huy nhận API inference để làm Data → AI → Decision, giám sát và an toàn.
- HGB của Duy, HGB `inference_v1` của Huy và LightGBM của Khánh là các artifact khác nhau; mỗi artifact phải dùng đúng số liệu của nó.

| Thứ tự | File | Cần hiểu |
|---|---|---|
| 1 | `docs/team/duy_problem_card.md` | AI dự báo gì, ai dùng, quyết định sau cảnh báo |
| 2 | `src/steel/preprocessing.py` | Hash, parse/sort, quality gate, target, manifest và split |
| 3 | `src/steel/duy_forecasting.py` | Feature batch/replay thống nhất, baseline, Ridge/HGB |
| 4 | `scripts/run_duy_pipeline.py` | Luồng chạy, protocol trước fit, hình, bảng so sánh và artifact |
| 5 | `src/steel/duy_inference.py` | Nạp đúng model/hash/schema/version và predict từ lịch sử |
| 6 | `scripts/predict_duy_steel.py` | Ví dụ gọi API tại một mốc thời gian |
| 7 | `tests/test_duy_forecasting.py` | Kiểm leakage, split, scaler, baseline và dữ liệu lỗi |
| 8 | `docs/team/review_khanh_huy_2026-10-01.md` | Ghi chú có bằng chứng để các bạn kiểm và sửa |

Module mới giữ riêng cách tính feature để không đổi đầu vào của artifact lịch sử. Lần này không cần chạy các script train/test/report/slides cũ.

## 2. Dữ liệu và 3B

- Nguồn UCI, DOI `10.24432/C52G8C`, CC BY 4.0; ghi attribution khi sử dụng/đăng lại [S1].
- CSV thực: **35.040 hàng, 11 cột**, từ 01/01/2018 00:00 đến 31/12/2018 23:45 sau sort, nhịp 15 phút.
- Không có ô thiếu, timestamp trùng hoặc Usage âm; có một hàng Usage=0, chưa đủ bằng chứng kết luận cảm biến hỏng.
- File gốc có **365 lần timestamp lùi**. Sort theo ngày khai báo và kiểm lại lưới 15 phút; không sửa raw hay tự chuyển các hàng 00:00 sang ngày sau.
- SHA-256: `9b1cee6f9cb9cd9df2b95814ca90a9a2ff15b7f5f1fba0fae3c643e82072eacc`.
- Nhà cung cấp chưa xác nhận rõ timestamp là đầu/cuối khoảng đo. Prototype giả định bản đo tại t đã hoàn tất và có sẵn khi phát dự báo; phải xác nhận điều này trước triển khai thật.

| 3B | Bằng chứng | Quyết định và giới hạn |
|---|---|---|
| Broken | Thứ tự raw có 365 lần lùi; sau sort không gap/trùng | Parse định dạng cụ thể, sort, assert 15 phút; không nội suy |
| Bad Quality | Một zero; train có 234 giá trị trên p99; không có Usage âm | Giữ zero/đỉnh nếu chưa xác nhận lỗi; reject NaN/Inf/âm; CSV sạch chưa chứng minh sensor chính xác |
| Background | Thiếu lịch ca, sản lượng, mẻ thép, bảo trì, TOU/hợp đồng và hành động | Dự báo thống kê, hỗ trợ kiểm tra; chưa giải thích nguyên nhân hoặc đo tiết kiệm thực |

Lee (2020), §2.7.2, trang sách 26–27/PDF 43–44, là căn cứ về vấn đề dữ liệu và context [S5].

## 3. Nhãn và feature: vì sao không nhìn trước?

Tại t, sau khi bản đo mới nhất đã có, dự báo tổng bốn bản đo tiếp theo:

```python
target = sum(usage.shift(-step) for step in range(1, 5))
```

Ví dụ thật tại 08/01/2018 00:00: bốn giá trị kế tiếp 3.78, 3.38, 3.31, 3.89 kWh → target **14.36 kWh**. Ba ví dụ tính tay được lưu trong `target_examples.json`.

- `shift(-1)` chỉ dùng tạo **nhãn học**, không đưa vào input.
- `usage_lag_0` hợp lệ vì bản đo t đã có; không mặc định mọi biến tại t là leakage.
- Rolling kết thúc tại t, không centered. Đổi Usage sau t không đổi feature tại t nhưng làm đổi target.
- Lịch tương lai là thông tin biết trước: biết ngày mai là thứ mấy không có nghĩa biết Usage ngày mai.

| Nhóm | Feature | Ý nghĩa |
|---|---|---|
| Lag, 5 cột | `usage_lag_0/1/4/96/672` | Hiện tại, 15 phút, 1 giờ, 1 ngày và 1 tuần trước |
| Rolling, 5 cột | `usage_sum_last_1h`, `usage_mean_last_4h`, `usage_std_last_4h`, `usage_mean_last_24h`, `usage_max_last_24h` | Mức, độ biến động và đỉnh trong quá khứ; std dùng ddof=0 |
| Calendar, 5 cột | Hour sin/cos, weekday sin/cos, weekend | Chu kỳ giờ/thứ tại `forecast_start=t+15 phút` |

**Sửa rủi ro batch/inference:** pandas rolling trên toàn chuỗi và cửa sổ ngắn có thể khác số rất nhỏ; cây có thể đổi nhánh khi giá trị sát ngưỡng. Bản mới dùng NumPy trực tiếp trên từng cửa sổ cho cả training và inference: `duy_numpy_windows_v1`. Đã kiểm 31 mốc validation, gồm 08/09 02:00 Huy nêu: feature và dự báo khớp ở các mốc đã kiểm. Khi đổi dependency/phần cứng vẫn phải kiểm lại.

## 4. Giải thích từng bước Data Preparation theo rubric

| Mục thầy yêu cầu | Đã làm | Vì sao |
|---|---|---|
| Missing values | Không impute raw sạch; loại hàng thiếu lịch sử/nhãn bằng manifest; runtime lỗi thì reject | Không bịa lịch sử/nhãn, không nội suy bằng tương lai |
| Outliers | Kiểm giá trị bất khả thi; giữ đỉnh/zero hợp lệ | Đỉnh thật chính là vùng nghiệp vụ cần dự báo |
| Scaling | Ridge dùng StandardScaler trong Pipeline fit train; HGB không scale | Ridge regularization phụ thuộc thang feature; cây không cần z-score |
| Feature engineering | 5 lag, 5 rolling, 5 calendar | Nắm mức hiện tại, chu kỳ và biến động, chỉ đọc dữ liệu tới t |
| Feature selection | Chọn explicit 15 cột; loại CO2/Load_Type khỏi input chính | Semantics/availability chưa đủ rõ; demo chỉ cần Usage và lịch |
| Dimensionality reduction | Không PCA | 15 feature còn quản lý được; giữ ý nghĩa, chưa có bằng chứng cần PCA |
| Imbalance | Không SMOTE; báo riêng sai số vùng target > p95 train | Nhãn liên tục; không tạo mẫu tổng hợp phá cấu trúc thời gian |

CSV giữ đơn vị gốc. Scaler nằm trong model Ridge; không fit trước split hoặc fit lại trên validation [S2]. CSV có thêm cột sensor dành cho nghiên cứu sau; **ma trận model vẫn chỉ 15 feature**.

## 5. Split và số hàng giữ/bỏ

| Tập | Thời gian | Số hàng | Cách dùng |
|---|---|---:|---|
| Train | Jan–Aug 2018 | 22.652 | Fit model/scaler và học diurnal mean |
| Validation | Sep 2018 | 2.876 | Chọn trong các cấu hình đã chốt |
| Calibration | Oct 2018 | 2.972 | Khánh tính buffer đúng model; Duy chưa score |
| Test | Nov–Dec 2018 | 5.852 | Khánh đánh giá sau khóa; đã được xem trong lịch sử |

- Bỏ 672 hàng đầu vì lag tuần chưa đủ; bỏ 12 nhãn chạm ba ranh giới; bỏ 4 hàng cuối thiếu nhãn tương lai.
- **35.040 = 34.352 giữ + 672 thiếu lịch sử + 12 purge + 4 thiếu nhãn cuối.**
- Origin 31/08 23:00 có forecast_end 01/09 00:00 → không vào train. Origin train cuối là 31/08 22:45, forecast_end 23:45.
- Validation được dùng lịch sử đã xảy ra trong train; đó là thông tin đã có khi dự báo. Điều cấm là dùng nhãn/giá trị tương lai chưa quan sát.
- Không shuffle hoặc k-fold ngẫu nhiên cho bài toán này [S3].

## 6. Baseline, Ridge và HGB

| Cách | Cách dự báo | Lý do so sánh |
|---|---|---|
| Last-hour persistence | Tổng bốn bản đo vừa qua | Đối chứng đơn giản khi tải ổn định |
| Previous day | Cửa sổ cùng ngày trước: t−95…t−92 | Đối chứng chu kỳ ngày |
| Previous week | Cửa sổ tuần trước: t−671…t−668 | Đối chứng chu kỳ tuần |
| Diurnal train mean | Mean target train theo thứ và giờ nguyên của forecast_start | Chỉ học lịch train; không làm tròn sin/cos |
| Ridge alpha 1/100 | Hồi quy tuyến tính với phạt L2 | Dễ hiểu, xử lý một phần multicollinearity |
| HGB 15/31 leaves | Cộng nhiều cây học quan hệ phi tuyến | Dữ liệu dạng bảng, lag/calendar, chạy CPU |

HGB chốt 180 iterations, learning_rate=0.08, L2=1, seed=42, `early_stopping=False`. Không dùng validation ngẫu nhiên ngầm. Hai lựa chọn leaves/alpha là tập thử nhỏ; params lưu trước fit trong `protocol.json` [S4].

**Kết quả mới trên validation, đơn vị kWh:**

| Model | MAE | RMSE | MAE vùng cao |
|---|---:|---:|---:|
| HGB 31 leaves | 15.8829 | 32.0475 | 68.6963 |
| HGB 15 leaves | 16.3177 | 32.5934 | 68.2773 |
| Last hour | 32.3238 | 69.1155 | 106.5411 |
| Ridge alpha 1 | 33.0168 | 50.9669 | 84.3953 |
| Ridge alpha 100 | 33.0307 | 51.0115 | 85.5357 |
| Previous week | 43.4543 | 81.9046 | 128.8452 |
| Previous day | 47.2319 | 91.0016 | 229.3084 |
| Diurnal mean | 49.9719 | 78.6894 | 141.8327 |

- Chọn **HGB 31** theo MAE validation thấp nhất trong cả baseline/AI: giảm **50.86%** so với baseline tốt nhất last_hour.
- Ridge có RMSE thấp hơn last_hour nhưng MAE cao hơn; không nói Ridge hơn baseline về mọi chỉ số.
- HGB train MAE=14.4243, validation=15.8829; hai số này chưa chứng minh model không overfit hoặc ổn định ở nhà máy khác.
- Vùng cao: target > 379.771 kWh, p95 train. Có 93 mẫu Sep; model dự báo thấp 92/93 mẫu. MAE vùng cao 68.70 cần Khánh kiểm calibration/recall/báo giả.
- Ca sai thật 10/09 18:00: actual=26.04, model Duy=298.8629 kWh. Không dùng số 299.41 của Huy cho artifact Duy. Chưa có policy nên chưa gọi đây là cảnh báo giả.
- MAE theo tuần ISO 35–39: 15.65, 18.47, 20.00, 13.71, 11.39; số mẫu tuần đầu/cuối khác nhau. Đây là chẩn đoán, không mở search mới theo các điểm này.
- Latency trong CSV: CPU local, 50 lần single-row sau warmup, một luồng toán; chưa phải end-to-end UI hoặc nhà máy. Thời gian kiểm parity gồm assertions và hai lần predict, không dùng để chứng minh SLA100ms.

**MAE** là trung bình độ lớn sai số; **RMSE** nhạy với sai số lớn; **bias=prediction−actual**, âm nghĩa là dự báo thấp. Không chọn MAPE làm chỉ số chính vì target có thể gần 0.

## 7. Mỗi hình trả lời gì?

Có 7 hình train và 3 hình validation trong thư mục `figures/` của bản v2:

- Chuỗi tuần đầu: mức tải theo thời gian, chưa đủ context kết luận nguyên nhân.
- Histogram Usage/target: phân bố và nhu cầu đánh giá riêng vùng cao.
- Heatmap giờ–thứ: mô tả chu kỳ, hỗ trợ lựa chọn calendar features.
- Tổng ngày: biến động dài hơn; median train=2.804,96 kWh/ngày, max=7.353,84 kWh/ngày.
- Correlation với lag: khoảng 0.908/0.687/0.606/0.666 cho 15 phút/1 giờ/1 ngày/1 tuần; không phải điểm model.
- Boxplot theo giờ: độ phân tán; chỉ ẩn fliers trên hình, không xóa dữ liệu.
- Feature correlation: redundancy; correlation không chứng minh nhân quả hoặc leakage.
- Bảng MAE bằng hình: AI có cải thiện so với baseline không?
- Actual/prediction: khi nào model bám hoặc trễ biến tải?
- Scatter/residual: dự báo thấp/cao, sai lớn và bias để ưu tiên Khánh kiểm.

## 8. Chạy lại và gọi dự báo

Môi trường đã kiểm: Python3.14.3, NumPy2.4.4, pandas2.3.3, sklearn1.8.0, Matplotlib3.10.8. Chưa kiểm ma trận Python khác.

```powershell
.venv\Scripts\python -m scripts.run_duy_pipeline --output-dir reports/steel/duy_reproduction_01
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python -m scripts.predict_duy_steel --run-dir reports/steel/duy_2026-10-02 --issue-time 2018-09-01T08:00:00
```

Pipeline yêu cầu thư mục mới. Lệnh predict tại mốc ví dụ trả **150.08549 kWh**; model chỉ nhận 673 bản đo tới 08:00, không trả actual tương lai.

```python
from pathlib import Path
from src.steel.duy_inference import DuyForecaster

forecaster = DuyForecaster(Path("reports/steel/duy_2026-10-02"))
# history: >=673 hàng observation_time, Usage_kWh; cuối đúng issue_time.
prediction = forecaster.predict(history, "2018-09-01 08:00:00")
```

- Nạp model một lần. Input lỗi → ValueError để UI yêu cầu kiểm; không tự đổi model hoặc dùng buffer sai.
- Khánh nhận model, `model_handoff.json`, processed Oct và validation predictions; dự báo Oct bằng đúng 15 feature/NumPy version rồi tính policy.
- Huy nhận API + policy đúng artifact; nếu đổi sang LightGBM cần kiểm batch/replay lại.
- Chỉ nạp joblib từ nguồn nhóm tin cậy; hash xác nhận toàn vẹn, không chứng nhận người gửi.

## 9. Lộ trình học cho Duy

| Buổi | Học | Bài thực hành |
|---|---|---|
| 1 | Bài toán, kWh/kW, 3B, context; UCI, bài giảng 01/02, Lee §2.7.2 | Nói một phút input–output–decision, nêu ba điều dataset không chứng minh |
| 2 | Datetime, sort, shift, rolling, lag; sklearn ví dụ lag [S3] | Tính tay ba target; vẽ timeline t−672…t…t+4 |
| 3 | Temporal split, leakage, Pipeline/StandardScaler [S2] | Giải thích 672/12/4 hàng; đổi tương lai trong bản copy rồi chạy leakage test |
| 4 | Baseline, Ridge L2, HGB, params và metrics [S4] | Giải thích Ridge RMSE tốt hơn nhưng MAE tệ hơn; đọc cấu hình model |
| 5 | Generalization, ca sai, inference parity, model card; Lee §3.7 [S5] | Predict lúc 08:00 không actual; phân tích ca 10/09; bàn giao Khánh đúng version |

Các câu nên trả lời được:

- **Vì sao raw sạch vẫn cần 3B?** Thứ tự sai, chưa biết độ chính xác sensor và thiếu context dù không NaN.
- **Vì sao dùng Usage(t)?** Bản đo t đã có; target bắt đầu t+1.
- **Vì sao không shuffle/cắt đỉnh?** Triển khai dự báo tương lai; đỉnh thật là tình huống nghiệp vụ cần học.
- **Vì sao chưa dùng deep learning?** Rubric yêu cầu ít nhất hai phương pháp phù hợp. Ridge/HGB đáp ứng; nếu thử LSTM/GRU phải cùng target/split, scaler train-only, đo lợi ích và chi phí.
- **Vùng cao vẫn sai thì có ích không?** MAE tổng cải thiện, nhưng cảnh báo cần calibration/recall/báo giả và người kiểm; chưa bảo đảm an toàn.
- **Áp dụng sách Lee ở đâu?** 3B, chọn thuật toán theo nguồn/điều kiện, kết nối model–decision–human; không lấy case LCD làm số tiết kiệm của Steel.
- **Đã xong đồ án chưa?** Đã triển khai phần Duy; calibration/evaluation/demo/monitoring và sản phẩm nộp cần nhóm kiểm và hoàn thiện tiếp.

## 10. Tài liệu tham khảo có đối chiếu

- **[S1]** [UCI Steel](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption): nguồn, đơn vị, license và metadata; phân biệt với audit do nhóm tính.
- **[S2]** [scikit-learn1.8 — Common pitfalls](https://scikit-learn.org/1.8/common_pitfalls.html): leakage và Pipeline fit train.
- **[S3]** [Lagged features for time series forecasting](https://scikit-learn.org/1.8/auto_examples/applications/plot_time_series_lagged_features.html): chuỗi sang bảng, temporal evaluation.
- **[S4]** [Ridge](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.Ridge.html), [HGB](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html), [RMSE](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.root_mean_squared_error.html): thuật toán, params và API đúng version.
- **[S5]** [Jay Lee, Industrial AI (2020)](https://doi.org/10.1007/978-981-15-2144-7): §2.7.2 trang26–27/PDF43–44; §3.4–3.5 trang48–54/PDF65–71; §3.7 trang58–61/PDF75–78, đã đối chiếu bản PDF người dùng cung cấp.
- **[S6]** [LightGBM Parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html): kiểm bagging_fraction/freq nếu thử LightGBM.
- Rubric chính: yêu cầu thầy đã gửi và hai bài giảng 01/02. Giá trị công nghiệp phải gắn với quyết định, không chỉ bảng sai số.
