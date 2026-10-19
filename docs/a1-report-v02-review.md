# Kiểm tra sơ bộ báo cáo A1 v02

Nguồn người học cung cấp: `assignment1_report_v02.docx` và
`assignment1_report_v02.pdf` tại thư mục `D:\master_hcmut\HK261\Deep_Learning`.
PDF hiện có 21 trang; Word có 11 bảng và 8 ảnh nhúng. Chưa thay hai tệp báo cáo
hiện hành trong `reports/` bằng bản này tại thời điểm rà soát ban đầu.

**Cập nhật sau khi người học duyệt sửa cục bộ:** đã đưa Word v02 đã sửa vào
`reports/assignment1.docx`. Người học tự xuất lại PDF, vì vậy PDF trong repo vẫn
giữ v01 và trang đích ghi rõ trạng thái chờ đồng bộ. Phần “cần chuẩn hóa” bên
dưới là danh sách phát hiện trước sửa; cả bốn mục đã được xử lý trong Word.

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

## Các điểm cần chuẩn hóa trước khi đồng bộ bản hiện hành

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

## Bước tiếp theo

Đã sửa cục bộ trong bản sao Word và giữ nguyên nguồn người học gửi. Đã cập nhật
và lưu mục lục bằng Word; 21 trang được dựng trực tiếp thành PNG để kiểm tra,
không xuất PDF. Không còn Calibri trong các trang EMF dùng dựng ảnh; giữ nguyên
toàn bộ 8 ảnh và 11 bảng. So sánh tất cả ô bảng cho thấy chỉ ô Protocol ID đổi;
các ô số liệu khác giữ nguyên.

SHA-256 Word đã sửa: `0f8bf22b5be4b015e8fb299fa89c620f391177d913a41180b1c47eb8ae2b558e`.
Người học mở `reports/assignment1.docx`, xuất PDF vào `reports/assignment1.pdf`.
Sau đó kiểm tra đồng bộ và cập nhật trạng thái PDF trên landing page. Việc kiểm
tra kỹ thuật không thay nhóm xác nhận tác giả, rà soát học thuật hoặc nộp bài.
