# Scripts

Chạy từ gốc repo: `python -m scripts.<tên>`; không chạy mọi script một lượt.

- Chính: run_duy_pipeline → predict_duy_steel → run_final_evaluation → serve_steel_demo.
- QA: audit_features_and_leakage, check_duy_handoff, check_repository, check_steel_demo_environment.
- UI/đường lỗi: check_steel_demo_browser, check_steel_demo_failures.
- Khánh tùy chọn: run_feature_extension, check_khanh_lightgbm.
- Train/tune/package/replay cũ giữ cho tương thích và tests.

[Toàn bộ lệnh](../docs/getting_started.md).
