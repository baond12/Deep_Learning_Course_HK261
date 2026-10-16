# Reports

Đặt các file báo cáo cuối cùng theo đúng đường dẫn đã gắn trên landing page:

- `exercises.pdf`: báo cáo chung E1–E3, bản PDF dùng để đọc và nộp bài.
- `exercises.docx`: bản Word tương ứng để tải về và chỉnh sửa.
- `assignment1.pdf`: báo cáo Assignment 1, bản PDF dùng để đọc và nộp bài.
- `assignment1.docx`: bản Word tương ứng để tải về và chỉnh sửa.
- `assignment2.pdf`

Không cần tách báo cáo E1, E2, E3 thành nhiều file; đề yêu cầu một báo cáo chung cho
E1-E3.

Bản Word E1–E3 đã cập nhật E1 từ `E1/results/epochs100_batch256_seed42/` và
E2 từ `E2/results/tokenization_msa_v2/` (100 epoch, batch size 256, seed 42).
E2 có đủ 8 cấu hình và kiểm chứng MSA CPU/CUDA đều PASS. E3 giữ số liệu lịch sử
20 epoch, batch size 16; báo cáo nêu rõ khác biệt ngân sách huấn luyện.
Đây chưa phải báo cáo cuối của kế hoạch E3 mới. Khi có đủ 8 RNN mới, thay phần E3
lịch sử và dùng lại nguyên baseline E1; bảng chung có 19 cấu hình độc lập, không
đếm lại MLP/CNN. Quy tắc nguồn số liệu và các bước đối chiếu được ghi tại
[`docs/e1-e3-report-protocol.md`](../docs/e1-e3-report-protocol.md).
PDF do người học xuất lại từ bản Word mới; khi thay Word, cần xuất PDF tương ứng
trước khi commit hai tệp để bảo đảm nội dung đồng nhất.
Landing page dùng đường dẫn ổn định
`reports/exercises.pdf` và `reports/exercises.docx` để có thể cập nhật phiên bản
báo cáo mà không phải đổi liên kết.

Bản Assignment 1 hiện tại được lấy từ `assignment1_report_v01.pdf` và
`assignment1_report_v01.docx`. Landing page dùng đường dẫn ổn định
`reports/assignment1.pdf` và `reports/assignment1.docx`.

