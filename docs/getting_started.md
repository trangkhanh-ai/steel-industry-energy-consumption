# Hướng dẫn chạy và tái lập

[Trang chủ](../README.md) · [Bài toán](project.md) · [Source](code_map.md)

## 1. Chuẩn bị

- Đã kiểm: Windows, Python **3.14.3**, NumPy **2.4.4**, scikit-learn **1.8.0**; gói khóa trong `requirements.txt`.
- Chạy tại thư mục có `README.md`, `src/`, `data/`. Dùng Python của `.venv` để tránh nhầm môi trường.
- Model joblib cần cùng phiên bản scikit-learn; chỉ nạp artifact từ nguồn nhóm tin cậy.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m scripts.check_steel_demo_environment
```

`requirements-demo.txt` bao gồm cùng `requirements.txt`. Khánh dùng `requirements-khanh.txt` để bổ sung LightGBM. Notebook/UI test cài riêng khi cần.

## 2. Demo có cảnh báo

```powershell
.venv\Scripts\python -X utf8 -m scripts.serve_steel_demo --port 8765 --policy reports/steel/release_2026-10-11/locked_policy.json
```

- Mở http://127.0.0.1:8765; mốc hợp lệ **01/11/2018 00:00** đến **31/12/2018 22:45**.
- Mở phiên, tiến bốn bước để thấy actual của dự báo đầu; tiếp tục xem sai số/cảnh báo.
- Nhập tên/ghi chú, chọn đã xem, kiểm tra hoặc bỏ qua. Đây là nhật ký prototype, không phải lệnh điều khiển máy.
- Reset tạo phiên riêng. Log SQLite trong `reports/steel/modeling/demo_sessions/` không đưa lên Git.
- Windows có launcher: `powershell -File scripts/start_steel_demo.ps1 -Port 8765`.
- Chỉ dự báo: bỏ `--policy` hoặc launcher với `-ForecastOnly`; replay tháng 9. Inspector báo policy LightGBM lịch sử `incompatible` không làm HGB ngừng dự báo. Policy cũ không được bật.

## 3. Kiểm source và dữ liệu

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python -m scripts.check_repository
.venv\Scripts\python -m scripts.audit_features_and_leakage
.venv\Scripts\python -m scripts.audit_features_and_leakage --processed-dir reports/steel/duy_2026-10-02/processed
.venv\Scripts\python -m scripts.check_duy_handoff --run-dir reports/steel/duy_2026-10-02
```

Audit tái dựng feature/target/manifest từ CSV gốc. Handoff kiểm hash model, nguồn và năm file code đã khóa; dự báo offline/replay phải khớp. `check_repository` kiểm JSON notebook, xung đột, link local và file bị loại khỏi gói source.

## 4. Huấn luyện lại phần Duy

```powershell
.venv\Scripts\python -m scripts.run_duy_pipeline --output-dir reports/steel/local_duy_01
.venv\Scripts\python -m scripts.audit_features_and_leakage --processed-dir reports/steel/local_duy_01/processed
.venv\Scripts\python -m scripts.check_duy_handoff --run-dir reports/steel/local_duy_01
.venv\Scripts\python -m scripts.predict_duy_steel --run-dir reports/steel/local_duy_01 --issue-time 2018-09-01T08:00:00
```

Sinh EDA train, split, bốn baseline, Ridge/HGB, bảng validation và artifact. Chỉ chọn bằng validation tháng 9. Output phải **chưa tồn tại**; chọn tên mới mỗi lần. `local_*` bị bỏ qua bởi Git.

## 5. Calibration và evaluation ba mức

```powershell
.venv\Scripts\python -m scripts.run_final_evaluation --model-run reports/steel/duy_2026-10-02 --output-dir reports/steel/local_evaluation_01
```

- Không fit lại AI; khóa protocol trước calibration, khóa buffer tháng 10 trước test.
- Sinh policy, dự báo từng hàng, MAE/RMSE/confusion/cụm mốc, benchmark và kịch bản lỗi/level shift.
- Industrial level dùng điểm kiểm tra giả định; chưa quy đổi tiền điện.
- Demo có thể dùng policy mới với `--policy reports/steel/local_evaluation_01/locked_policy.json`.
- Test đã được xem; không tiếp tục tuning rồi gọi kết quả trên cùng tập là độc lập.

## 6. Phần Khánh — tùy chọn

```powershell
.venv\Scripts\python -m pip install -r requirements-khanh.txt
.venv\Scripts\python -m scripts.run_feature_extension --output-dir reports/steel/local_gate7_01
.venv\Scripts\python -m scripts.check_khanh_lightgbm --output-dir reports/steel/local_lightgbm_check_01
```

Lệnh đầu chỉ đọc train/validation, cùng preprocessing/clipping và heatmap train. Lệnh sau nạp **LightGBM 15 feature lịch sử**, tính lại buffer và đối chiếu score cũ. Model 19 feature chưa có đánh giá test/adapter deploy; không gán số test của model 15 cho model 19.

## 7. Notebook và kiểm UI

```powershell
.venv\Scripts\python -m pip install -r requirements-notebooks.txt
```

Mở từ repo hoặc `notebooks/`, chọn kernel `.venv`. `01`: preprocessing; `02`: calibration HGB; `03`: dữ liệu/model Duy; `04`: nghiên cứu Khánh. `archive/` là lịch sử.

Máy có Microsoft Edge, terminal thứ hai khi server đang chạy:

```powershell
.venv\Scripts\python -m pip install -r requirements-ui-test.txt
.venv\Scripts\python -m scripts.check_steel_demo_browser --url http://127.0.0.1:8765 --output-dir outputs/browser_check_01
.venv\Scripts\python -m scripts.check_steel_demo_failures --policy-path reports/steel/release_2026-10-11/locked_policy.json --output outputs/failure_checks_01.json
```

Checker tạo/reset phiên kiểm thử. `outputs/`, ảnh/log local và SQLite bị bỏ qua bởi Git.

## 8. Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| Không tìm thấy `src`/CSV | Terminal tại gốc repo; dùng `python -m scripts...`. |
| Khác phiên bản gói | Tạo `.venv` riêng, cài `requirements.txt`, dùng đúng executable. |
| Output đã tồn tại | Chọn output mới, không ghi đè kết quả đã khóa. |
| Policy không khớp model | Dùng policy đúng run; không sửa hash/buffer để bỏ kiểm tra. |
| Ngày sai | Có policy: tháng 11–12; không policy: tháng 9; bước 15 phút. |
| Port bận | Đổi `--port 8766`, mở URL tương ứng. |
| Thiếu LightGBM | Cài `requirements-khanh.txt` khi chạy study. |
| Actual vẫn chờ | Chưa đủ bốn bản đo hoặc thiếu phép đo; không điền số giả. |
