# Kiểm tra và đồng bộ báo cáo A1 v02

Nguồn người học cung cấp: `assignment1_report_v02.docx` và
`assignment1_report_v02.pdf` tại thư mục `D:\master_hcmut\HK261\Deep_Learning`.
PDF hiện có 21 trang; Word có 11 bảng và 8 ảnh nhúng. Chưa thay hai tệp báo cáo
hiện hành trong `reports/` bằng bản này tại thời điểm rà soát ban đầu.

**Trạng thái hiện hành ngày 19/10/2026:** Word v02 đã sửa nằm tại
`reports/assignment1.docx`; người học đã xuất lại `reports/assignment1.pdf` từ
bản Word này. PDF 21 trang đã được kiểm tra và hai tệp đã đồng bộ. Các phát hiện
trước sửa được giữ bên dưới để truy vết; cả bốn mục đã được xử lý và kiểm tra lại
trên PDF mới.

## Những điểm đã đối chiếu

- Bảng chất lượng đủ tám cấu hình, đúng epoch chọn, validation tốt nhất/cuối,
  test accuracy, macro F1 và test loss theo kết quả chính thức.
- Các bảng tham số, thời gian, allocated/reserved và lỗi confidence cao phù hợp
  với bảng tổng hợp. Giữ riêng validation và test; không cộng pilot vào 240 epoch.
- Báo cáo nêu một seed, LR riêng của ViT full, khác biệt công thức pretrained,
  giới hạn đo thời gian/GPU, tình trạng license và test từng được quan sát.
- PDF dùng chữ màu đen; các màu trong danh mục style Word chưa đủ để kết luận
  có chữ màu trong bản in. Không phát hiện text nằm ngoài khổ trang PDF.
- Đã render và xem toàn bộ 21 trang để kiểm tra bố cục sơ bộ. Có bảng tiếp trang,
  hình và chú thích; chưa coi đây là bản đã được nhóm duyệt học thuật.

## Phát hiện trước sửa, đã xử lý

1. **Font tiêu đề:** PDF thực tế dùng `Calibri-Bold` cho các tiêu đề và đề mục,
   trong khi yêu cầu là Times New Roman. Thân bài dùng Times New Roman;
   `CambriaMath` xuất hiện ở công thức và cần được xem xét riêng theo khả năng
   hiển thị ký hiệu. Không suy ra font chỉ bằng thống kê các style chưa dùng.
2. **Protocol ID tại Bảng 10:** đang ghi
   `CIFAR-100 / highest_validation_accuracy_first_epoch_on_tie`, tức ghép tên
   dataset và tiêu chí chọn checkpoint. ID đúng trong JSON là
   `a1_cifar100_p100_v1`. Hai thông tin kia có thể giữ ở dòng riêng, không dùng
   thay ID.
3. **Mục lục Word/PDF:** cache mục lục trong DOCX ghi Tóm tắt ở trang 3, trong
   khi PDF đã cập nhật thành trang 4 và tổng 21 trang. Cần cập nhật trường mục
   lục, bỏ dòng tự tham chiếu “MỤC LỤC” nếu không cần, lưu Word rồi xuất lại PDF
   từ đúng bản đã lưu. Không thay số trang thủ công theo bản cũ.
4. **Làm tròn:** đoạn diễn giải R-S ghi khoảng cách train/validation 44,03 điểm
   phần trăm từ hai tỷ lệ đã làm tròn; bảng ghi 44,02 từ số gốc. Đây là sai khác
   làm tròn nhỏ, nên thống nhất lấy số gốc và ghi 44,02.

## Kiểm tra sau sửa

Đã sửa cục bộ trong bản sao Word và giữ nguyên nguồn người học gửi. Đã cập nhật
và lưu mục lục bằng Word; 21 trang được dựng trực tiếp thành PNG để kiểm tra,
không xuất PDF. Không còn Calibri trong các trang EMF dùng dựng ảnh; giữ nguyên
toàn bộ 8 ảnh và 11 bảng. So sánh tất cả ô bảng cho thấy chỉ ô Protocol ID đổi;
các ô số liệu khác giữ nguyên.

SHA-256 Word đã sửa: `0f8bf22b5be4b015e8fb299fa89c620f391177d913a41180b1c47eb8ae2b558e`.
Word giữ nguyên mã băm này sau khi người học xuất PDF.

SHA-256 PDF v02: `c23ac7fa3489ca6bf48e9ee31609cd0dd89c08fd91bf0c619ac67067eaeb42e5`.

- PDF gồm 21 trang, đã dựng ảnh và xem toàn bộ; không phát hiện chữ bị cắt,
  chồng lấp hoặc nằm ngoài khổ trang. Bảng tiếp trang giữ tiêu đề cột.
- Đối chiếu nội dung 376 ô thuộc 11 bảng với Word; các giá trị không đổi.
  Tiêu đề cột “Val tốt nhất (%)” xuống hai dòng, đã kiểm tra trực quan.
- Bảng 10 ghi đúng `a1_cifar100_p100_v1`; đoạn R-S và bảng cùng dùng 44,02.
- Mục lục ghi Tóm tắt ở trang 4, không còn dòng mục lục tự tham chiếu; số trang
  các phần phù hợp bản xuất 21 trang.
- Tiêu đề và thân bài dùng Times New Roman, không còn Calibri. Công thức giữ
  Cambria Math và ký hiệu hỗ trợ; Cambria còn ở khoảng trắng của mục lục.
  Mọi đoạn chữ được trích xuất đều màu đen.

Đã cập nhật trạng thái và liên kết Word/PDF v02 trên trang chủ, trang A1 và tài liệu
repo. Người học tự commit/push. Kiểm tra kỹ thuật không thay nhóm xác nhận tác giả,
rà soát học thuật hoặc nộp bài.
