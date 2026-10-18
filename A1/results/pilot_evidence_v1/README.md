# Bằng chứng thăm dò trước khi khóa A1.1

Bản sao nguyên trạng các thư mục người học tải về từ server. Đã kiểm tra SHA-256
từng tệp sau khi sao chép; không sửa nội dung hoặc ghi đè kết quả nguồn. Đường dẫn
`output_dir` và hash mã nguồn trong JSON tiếp tục chỉ lượt chạy gốc.

| Thư mục | Cấu hình | Epoch | Batch | LR | Mục đích |
| --- | ---: | ---: | ---: | ---: | --- |
| [pilot_epoch1_batch16_seed42](pilot_epoch1_batch16_seed42/) | 8 | 1 | 16 | 1e-4 | Đo tính khả thi ban đầu |
| [pilot_epoch1_batch64_seed42](pilot_epoch1_batch64_seed42/) | 8 | 1 | 64 | 1e-4 | Đối chiếu batch và bộ nhớ |
| [pilot_epoch5_batch64_seed42](pilot_epoch5_batch64_seed42/) | 8 | 5 | 64 | 1e-4 | Đọc diễn biến học và chi phí |
| [pilot_full_epoch5_batch64_lr1e-5_seed42](pilot_full_epoch5_batch64_lr1e-5_seed42/) | 2 full | 5 | 64 | 1e-5 | Chọn LR cho hai cấu hình full |

Đây là **train/validation**, không phải test hoặc số liệu A1.2. Cùng split seed
42, train seed 42, Adam/weight decay 1e-4, FP32, ảnh 224 và pipeline resize/lật ngang.
Không có checkpoint để nối tiếp các lượt này. Mỗi thư mục giữ số liệu thô, log,
môi trường, chỉ số chia và hash nguồn; xem [phân tích](../../../docs/a1-pilot-analysis.md)
và [protocol đã duyệt](../../protocol-a1.1.md).

Mục đích lưu riêng tại `pilot_evidence_v1/` là phân biệt thăm dò với thư mục lượt
chính `A1/results/epochs30_batch64_seed42_v1/`, đồng thời giữ nguyên các folder
pilot trên server. Chỉ lượt chính theo protocol mới được dùng làm bảng A1.2 tương ứng.
