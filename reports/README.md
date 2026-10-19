# Báo cáo học phần

Các đường dẫn ổn định cho báo cáo; trạng thái phiên bản được ghi bên dưới:

- `exercises.pdf`: PDF E1–E3 hiện hành v02, gồm 31 trang.
- `exercises.docx`: bản Word E1–E3 hiện hành v02.
- `assignment1.pdf`: PDF A1 v02, 21 trang, đã đối chiếu với Word theo kết quả 30 epoch mới.
- `assignment1.docx`: Word A1 v02 đã sửa cục bộ; Word và PDF đã đồng bộ.
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

## Báo cáo Assignment 1 v02

Word hiện hành được sửa cục bộ từ `assignment1_report_v02.docx` do người học cung
cấp, theo [kết quả tám cấu hình/240 epoch](../A1/results/epochs30_batch64_seed42_v1/analysis.md).
Bản gốc ở thư mục người học được giữ nguyên. Bản trong repo đã sửa font tiêu đề
bằng cách bỏ theme font ghi đè, đặt chữ về Times New Roman/đen; sửa Protocol ID
thành `a1_cifar100_p100_v1`; thống nhất khoảng cách R-S thành 44,02 điểm phần
trăm; cập nhật và lưu mục lục bằng Word, bỏ mục lục tự tham chiếu.

Đã dựng 21 trang PNG bằng Word để kiểm tra nội bộ, không xuất PDF. Giữ nguyên
11 bảng và 8 ảnh; bảng số liệu không đổi, chỉ một ô Protocol ID được sửa.
Công thức Word được giữ nguyên.

- SHA-256 Word nguồn v02: `73ee7340b8ce02666601824f69eca4cf55f5bfd8f158183a89f93e1082b4361a`.
- SHA-256 Word v02 đã sửa: `0f8bf22b5be4b015e8fb299fa89c620f391177d913a41180b1c47eb8ae2b558e`.
- SHA-256 PDF v02 hiện hành: `c23ac7fa3489ca6bf48e9ee31609cd0dd89c08fd91bf0c619ac67067eaeb42e5`.

**Word và PDF v02 đã đồng bộ ngày 19/10/2026.** Người học đã xuất
[PDF 21 trang](assignment1.pdf) từ [Word đã sửa](assignment1.docx). Đã đối chiếu
nội dung toàn bộ 11 bảng với Word, kiểm tra Protocol ID, mục lục và cách làm tròn;
dựng và xem cả 21 trang PDF. Tiêu đề và thân bài dùng Times New Roman, công thức
giữ font toán; chữ màu đen, không phát hiện chữ nằm ngoài khổ trang hoặc lỗi cắt
bảng. Word không thay đổi sau lần chuẩn hóa. Trang chủ và trang A1 cùng liên kết
đến hai tệp hiện hành. Chi tiết: [biên bản kiểm tra](../docs/a1-report-v02-review.md).
Thao tác Git do người học tự thực hiện; chưa xác nhận nhóm đã hoàn tất rà soát
học thuật hoặc đã nộp bài.

