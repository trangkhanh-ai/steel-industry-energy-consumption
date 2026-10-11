# Thay đổi source

## 11/10/2026

- Tích hợp HGB evaluation/policy local với demo/monitoring của Huy từ origin/main ngày 08/10.
- Sửa notebook 02 xung đột JSON; nghiên cứu LightGBM Khánh có notebook/script riêng.
- Khôi phục audit tái dựng giá trị và API dùng bởi test.
- Tách active policy HGB và inspector LightGBM lịch sử; log bàn giao giữ theo phiên.
- Gate 7 thành CLI: output riêng, không auto-train khi import, không cần seaborn/tabulate, cùng clipping cho Ridge/LightGBM.
- Kiểm LightGBM 15 feature; không bật policy cũ sai hash, không gán test 15 feature cho model 19 feature.
- Khóa môi trường core; LightGBM/notebook/browser tùy chọn.
- Thêm README, hướng dẫn, code map, rubric, tham khảo và kiểm tự động.
- Bỏ Word khỏi cây file publish; report/paper/export local bị loại bằng .gitignore. Giữ lịch sử Git cũ.

Folder ngày cũ là bằng chứng lịch sử. [Kết quả hiện tại](docs/results.md).
