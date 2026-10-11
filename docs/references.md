# Nguồn và cách áp dụng

Nguồn ngoài cung cấp phương pháp/metadata. Số audit và model trong repo do nhóm tính, không phải số công bố của các nguồn bên dưới.

| Nguồn chính | Áp dụng trong dự án |
|---|---|
| [UCI Steel Industry Energy Consumption](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption), Sathishkumar V E, Changsun Shin, Yongyun Cho (2021), DOI 10.24432/C52G8C, CC BY 4.0 | Dataset, tên/đơn vị biến và attribution. CSV giữ nguyên byte; audit được tính lại. |
| [Jay Lee, Industrial AI (2020)](https://doi.org/10.1007/978-981-15-2144-7) | 3B, context công nghiệp và kết nối prediction với quyết định. Không lấy kết quả case study làm số tiết kiệm của nhóm. |
| [scikit-learn 1.8: common pitfalls](https://scikit-learn.org/1.8/common_pitfalls.html) | Fit scaler train, tránh leakage, tách evaluation khỏi model selection. |
| [Lagged features for time series forecasting](https://scikit-learn.org/1.8/auto_examples/applications/plot_time_series_lagged_features.html) | Lag/rolling, temporal split, baseline và đánh giá dự báo. |
| [Ridge](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.Ridge.html) và [HGB](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) | Hai họ ML chính, regularization và cấu hình cây/CPU. |
| [LightGBM parameters](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html) | regression_l1, early stopping validation, seed, bagging frequency và CPU thread trong study Khánh. |
| [Cookiecutter Data Science](https://github.com/drivendataorg/cookiecutter-data-science) | Tách raw/processed, source, scripts, numbered notebooks, docs, tests và kết quả. Không dùng toàn bộ template/framework. |
| [scikit-learn repository](https://github.com/scikit-learn/scikit-learn) | README chỉ rõ cài đặt, examples, kiểm thử và đóng góp; tests đặt riêng khỏi sản phẩm. |
| [LightGBM repository](https://github.com/lightgbm-org/LightGBM) | Tài liệu cài đặt/usage riêng, reproducibility và dependency tùy chọn. |
| [actions/checkout](https://github.com/actions/checkout) và [actions/setup-python](https://github.com/actions/setup-python) | Workflow kiểm source; action khóa vào commit v7 kiểm tại thời điểm chỉnh repo. |

Repo tham khảo được kiểm công khai ngày 11/10/2026: Cookiecutter Data Science khoảng **10,1k**, scikit-learn **67,5k**, LightGBM **18,9k stars**. Số sao chỉ hỗ trợ chọn tài liệu có cộng đồng sử dụng, không chứng minh model phù hợp hoặc kết quả của nhóm đúng.

Bài giảng và rubric do thầy cung cấp được ánh xạ trong [project.md](project.md) và [kế hoạch nhóm](team/ke_hoach_nhom_3_nguoi.md). Repo không phân phối lại PDF bài giảng, sách/bài báo hoặc Word của thầy.
