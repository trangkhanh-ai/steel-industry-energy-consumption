# Làm việc trong nhóm

- Đọc [bài toán](docs/project.md), [code map](docs/code_map.md), [hướng dẫn chạy](docs/getting_started.md).
- Duy: dữ liệu/model; Khánh: study/evaluation; Huy: demo/monitoring. Giữ credit khi sửa phần của nhau.
- Hàm ngắn, tên rõ, giải thích vì sao với quyết định về dữ liệu/đơn vị/causality.
- Không fit preprocessing trên validation/calibration/test; không random split chuỗi thời gian.
- Kết quả mới vào folder mới, ghi schema/checksum/cấu hình/split. Không sửa hash để bỏ kiểm tra.
- Chạy test/audit/handoff theo README; thay UI/policy cần kiểm thêm browser/đường lỗi.
- Commit source, hướng dẫn và số liệu đã kiểm. Không commit Word/PDF, bài báo, token, môi trường Python hoặc log người dùng.
- GitHub Actions kiểm source; timing trên máy mới có thể khác local.
