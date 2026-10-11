# Artifact lịch sử của Khánh

- `lightgbm_model.txt`: LightGBM **15 feature, 457 cây**, giữ nguyên byte/hash.
- `test_results.json`: score lịch sử; đối chiếu mới khớp MAE/RMSE/precision/recall/buffer.
- `policy.json`: bàn giao lịch sử **sai hash model**, không active. Inspector hiển thị sai khác; không sửa hash để né kiểm tra.
- Calibration tính lại và score đầy đủ ở [kết quả mới](../../reports/steel/khanh_lightgbm_verified_2026-10-11/verification.json).
- Feature extension 19 biến thuộc [study riêng](../../reports/steel/khanh_gate7_2026-10-11/validation_comparison.csv).

Demo mặc định dùng HGB của Duy, policy theo đúng artifact HGB. [Giới hạn và kết quả](../../docs/results.md).
