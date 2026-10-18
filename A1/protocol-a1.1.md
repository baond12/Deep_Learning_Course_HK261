# A1.1 — Dữ liệu và quy trình so sánh CNN với Transformer

Trạng thái: **người học đã duyệt quy trình dưới đây sau bốn lượt thăm dò trên P100**. Chưa có kết quả A1.2 của quy trình này. Cấu hình có thể đọc bằng chương trình ở [protocol-a1.1.json](protocol-a1.1.json); bằng chứng lựa chọn ở [phân tích thăm dò](../docs/a1-pilot-analysis.md). Bản [dự thảo cũ](protocol-a1.1-draft.md) chỉ để truy vết, không dùng thay quy trình đã duyệt.

Mốc A1.1: 26/10/2026, 23:59 GMT+7. Mốc A1.2: 09/11/2026, 23:59 GMT+7.

## Dữ liệu và lý do chọn

Giữ **CIFAR-100**, đủ 100 lớp nhãn chi tiết. Đây là tập được đề bài cho phép, có 60.000 ảnh RGB 32×32 và nhiều lớp dễ nhầm; phù hợp so sánh học từ đầu với chuyển giao trên P100 16 GB. Không thay bằng tập dữ liệu E1. Ảnh nhỏ được phóng lên 224×224 để dùng cùng đầu vào cho ResNet18 và ViT-B/16; việc phóng ảnh không tạo thêm thông tin gốc.

Nguồn: [trang CIFAR của tác giả](https://cave.cs.toronto.edu/kriz/cifar.html), tham khảo Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009. Tải bằng torchvision. **Tình trạng giấy phép:** lần kiểm tra phục vụ dự thảo chưa xác định được giấy phép chuẩn hóa trên trang gốc; [thẻ uoft-cs/cifar100](https://huggingface.co/datasets/uoft-cs/cifar100/blob/main/README.md) ghi `unknown`. Nhóm ghi đúng tình trạng này, không tự gán MIT/CC; cần bổ sung xác nhận nếu giảng viên yêu cầu tên giấy phép cụ thể.

| Tập | Nguồn | Số ảnh mỗi lớp | Tổng |
| --- | --- | ---: | ---: |
| Huấn luyện | Tập huấn luyện gốc | 400 | 40.000 |
| Xác thực | Phần còn lại của tập huấn luyện gốc | 100 | 10.000 |
| Kiểm tra | Tập kiểm tra chính thức | 100 | 10.000 |

Không lấy tập con theo lớp. Giữ thuật toán chia cân bằng của `A1/data.py`, seed chia **42**, seed huấn luyện **42**. Chỉ số train/validation không giao nhau, hợp đủ 50.000 chỉ số gốc. Mã băm chỉ số bắt buộc:

`39684f5ef7a60f4cf3da5897931830b02da9723be21241a882b93fa1d0bda956`

Chương trình xuất chỉ số và kiểm tra mã băm trước huấn luyện; tự tạo bảng số ảnh từng lớp, kích thước mảng ảnh gốc và lưới ảnh mẫu **chỉ từ tập huấn luyện**. Các tệp EDA này chỉ được xem là đã có sau khi chạy trên CIFAR-100 thật, không thay bằng hình tổng hợp. Kiểm tra chỉ số không chứng minh không có ảnh trùng nội dung.

## Tiền xử lý giữ nguyên từ thăm dò

- Train: resize 224×224, nội suy song tuyến tính với antialias; lật ngang xác suất 0,5; chuyển tensor; chuẩn hóa theo ImageNet.
- Validation/test: resize và chuẩn hóa như trên, không lật/cắt ngẫu nhiên.
- Mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`.
- Chưa thêm random crop, MixUp, CutMix hoặc thay đổi chính sách tăng cường dữ liệu trong lượt chính.

## Ma trận tám cấu hình

| Cấu hình | Khởi tạo | Phần được học | Learning rate |
| --- | --- | --- | ---: |
| `resnet18_scratch` | Ngẫu nhiên | Toàn bộ | 1e-4 |
| `resnet18_pretrained_head` | ImageNet-1K V1 | `fc` | 1e-4 |
| `resnet18_pretrained_partial` | ImageNet-1K V1 | `layer4` và `fc` | 1e-4 |
| `resnet18_pretrained_full` | ImageNet-1K V1 | Toàn bộ | 1e-4 |
| `vit_b16_scratch` | Ngẫu nhiên | Toàn bộ | 1e-4 |
| `vit_b16_pretrained_head` | ImageNet-1K V1 | `heads` | 1e-4 |
| `vit_b16_pretrained_partial` | ImageNet-1K V1 | Hai block encoder cuối và `heads` | 1e-4 |
| `vit_b16_pretrained_full` | ImageNet-1K V1 | Toàn bộ | **1e-5** |

Dùng tường minh `ResNet18_Weights.IMAGENET1K_V1` và `ViT_B_16_Weights.IMAGENET1K_V1`, thay đầu ra thành 100 lớp. Phần đóng băng không cập nhật trọng số hoặc thống kê BatchNorm; kiểm tra hash trạng thái trước/sau. Phần được học phải có thay đổi trạng thái. ResNet18 và ViT-B/16 khác số tham số và công thức tiền huấn luyện; không xem đây là phép cô lập hoàn toàn ảnh hưởng của kiến trúc.

## Ngân sách và lý do chọn

- **30 epoch mỗi cấu hình**, batch **64**, FP32; không AMP, scheduler hoặc warmup.
- Adam; weight decay **1e-4**; cross entropy; chỉ đưa tham số được học vào optimizer.
- Không dùng early stopping riêng cho từng mô hình. Cùng giới hạn epoch không có nghĩa cùng thời gian hoặc đã hội tụ.
- Kiểm tra loss/gradient hữu hạn; ghi lỗi và giữ kết quả đã hoàn tất nếu có sự cố.
- Mục tiêu P100 16 GB, ưu tiên chạy tuần tự trên GPU dành riêng. Môi trường thăm dò thực tế: Python 3.10.13, PyTorch 2.2.1+cu118, torchvision 0.17.1+cu118. Giữ môi trường server đã chạy được; không cần nâng thư viện để chạy lượt chính.

Đã thăm dò tám cấu hình một epoch batch 16, tám cấu hình một epoch batch 64, tám cấu hình năm epoch batch 64; sau đó thử hai cấu hình full năm epoch với LR `1e-5`. Hai cấu hình full có cùng hai lựa chọn LR `1e-4`/`1e-5`; sáu cấu hình khác mới khảo sát `1e-4`.

Theo validation tốt nhất trong năm epoch, ResNet full đạt 78,23% với `1e-4` so với 70,26% với `1e-5`; ViT full đạt 83,50% với `1e-4` so với 86,07% với `1e-5`. Vì vậy chọn LR khác cho ViT full. Đây là lựa chọn trong phạm vi đã khảo sát, **không phải tối ưu hyperparameter toàn cục**. Kết luận về full/partial phải công khai khác biệt LR và ngân sách lựa chọn; không quy mọi chênh lệch cho duy nhất đóng băng.

Ước tính tám cấu hình × 30 epoch khoảng **21 giờ** theo thời gian fit thăm dò. Lượt chính ghi checkpoint ra đĩa nên có thêm chi phí I/O; chưa gồm test, xuất hình và gián đoạn. Chưa cam kết thêm seed. Nếu bổ sung seed hoặc thay đổi công thức, phải công khai thay đổi và phân biệt với lượt này.

## Chọn mô hình và đánh giá cuối

1. Huấn luyện chỉ nạp train/validation, đặt lại seed và thứ tự loader trước mỗi cấu hình.
2. Lưu ngay checkpoint khi validation accuracy cao hơn giá trị tốt nhất; nếu hòa giữ epoch sớm nhất. Lưu toàn bộ 30 epoch và báo riêng validation cuối với validation tốt nhất.
3. Khóa công thức theo tài liệu này trước khi đánh giá test; không dùng test để đổi LR, số epoch hoặc chọn kiến trúc.
4. Lệnh test kiểm tra protocol, split, thứ tự lớp, mã nguồn và SHA-256 checkpoint; nạp checkpoint tốt nhất rồi chạy một lượt suy luận mỗi cấu hình.
5. Báo accuracy, macro F1 trên đủ 100 lớp, loss, tham số tổng/được học, thời gian train và validation riêng, thời gian toàn fit và bộ nhớ đỉnh. Xuất dự đoán theo chỉ số ảnh gốc, báo cáo từng lớp, ma trận nhầm lẫn dạng số/hình và tối đa 25 mẫu sai.

Các lượt cũ đã từng đánh giá test, nên không mô tả test là chưa từng được nhóm quan sát. Bảng chính mới không trộn test cũ với validation thăm dò hoặc test theo quy trình mới.

Thời gian từng pha gồm loader, chuyển dữ liệu, kiểm tra hữu hạn và tính toán. Thời gian toàn fit còn gồm callback/ghi checkpoint; setup tách riêng. MiB allocated/reserved là số đo của tiến trình PyTorch, không phải tổng VRAM GPU. Một seed chưa cho phép kết luận ý nghĩa thống kê, và 30 epoch chưa bảo đảm mọi mô hình hội tụ.

## Triển khai và kết quả cần giữ

Dùng [run_protocol.py](run_protocol.py), theo [hướng dẫn chạy](RUN_PROTOCOL.md). `train.py` là công cụ cũ; không dùng lệnh cũ thay cho protocol này. Không sửa mã chung E1–E3 hoặc thay baseline/report Exercise.

Giữ toàn bộ thư mục kết quả và checkpoint trên server cho đến khi đã đọc xong test. Checkpoint bị Git ignore; các số liệu, cấu hình, log, hình và tài liệu được đưa lên Git bằng lệnh người học tự thực hiện. Landing page công bố CIFAR-100 và protocol, ghi lượt chính chưa có kết quả; báo cáo cũ được nhận diện là thăm dò. Sau A1.2, thay liên kết báo cáo bằng bản mới đã rà soát.
