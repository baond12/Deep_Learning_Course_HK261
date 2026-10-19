# Báo cáo học phần

Các đường dẫn ổn định cho báo cáo; trạng thái phiên bản E1–E3 được ghi bên dưới:

- `exercises.pdf`: PDF E1–E3 hiện hành v02, gồm 31 trang.
- `exercises.docx`: bản Word E1–E3 hiện hành v02.
- `assignment1.pdf`: PDF A1 v01 của lượt thăm dò trước, chưa phản ánh kết quả 30 epoch mới.
- `assignment1.docx`: Word v01 tương ứng, giữ nguyên cho tới khi có bản mới đồng bộ.
- `assignment2.pdf`

Không cần tách báo cáo E1, E2, E3 thành nhiều file; đề yêu cầu một báo cáo chung cho
E1-E3.

## Báo cáo E1–E3 hiện hành

Bản v02 được chốt tạm làm bản hiện hành ngày 16/10/2026. `exercises.docx` là bản
sao nguyên nội dung từ `exercises_review_v02.docx` do người học cung cấp, không
biên tập thêm sau quyết định chốt tạm.

- SHA-256 Word v02: `58d7d89fe303a142fabd2243447f03f7a6f7becce74a7eadced0a8354ab2145c`.
- SHA-256 PDF v02: `f6aacd5dc4f799a37eec5f99a7be1817ffbe1677fa841c6f554c110775a09315`.
- Phạm vi: E1 `epochs100_batch256_seed42`, E2 `tokenization_msa_v2`, E3 `sequence_v2`.
- Tổng 19 cấu hình độc lập và 1.900 bản ghi huấn luyện, 100 vòng/lô 256/hạt giống 42.
- E3 giữ nguyên hai mốc MLP/CNN E1, không tính thêm lượt huấn luyện.
- Quy tắc số liệu: [thiết kế và đối chiếu thực nghiệm](../docs/e1-e3-report-protocol.md).

**Word và PDF v02 đã đồng bộ.** PDF được người học xuất từ Word đã chốt tạm,
lấy nguyên tệp `exercises_review_v02.pdf`, gồm 31 trang. Đã đối chiếu nội dung
chữ và bảng với Word, kiểm tra mục lục và xem các trang sau khi xuất. Hai tệp
trong repo giữ nguyên nội dung người học cung cấp. Trang chủ và trang E1–E3
đều liên kết đến [PDF](exercises.pdf) và [Word](exercises.docx) hiện hành.

Việc chốt tạm không xác nhận đã nộp bài hoặc thay người học xác nhận thông tin
thành viên và khai báo hỗ trợ AI. Các gói soạn thảo và nhận xét rà soát được giữ
trong khu vực làm việc, không cần đưa vào đợt cập nhật báo cáo hiện hành.

Bản Assignment 1 hiện tại được lấy từ `assignment1_report_v01.pdf` và
`assignment1_report_v01.docx`. Landing page dùng đường dẫn ổn định
`reports/assignment1.pdf` và `reports/assignment1.docx`.

Ngày 19/10/2026 đã tổng hợp [kết quả A1 mới](../A1/results/epochs30_batch64_seed42_v1/analysis.md),
tám cấu hình/240 epoch theo protocol đã duyệt. Gói làm báo cáo cho ChatGPT Plus
đã chuẩn bị; chưa có Word v02 ở máy. Người học đồng ý xuất PDF từ Word mới trước
khi review nhóm. Chưa thay thế hai tệp v01 bằng kết quả khác hoặc ghi nhãn chúng
là báo cáo mới. Khi có Word mới, xuất PDF từ chính Word đó, kiểm tra đồng bộ,
rồi cập nhật hai đường dẫn ổn định; thao tác Git do người học tự thực hiện.

