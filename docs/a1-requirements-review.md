# Đối chiếu yêu cầu Assignment 1 và hướng chuẩn hóa A1.1

Ngày rà soát: 17/10/2026. Căn cứ: [đề A1](../Exercise/assignment1-vne.pdf), mã nguồn và kết quả đang có trong repo. Người học đã chốt CIFAR-100 và xác nhận GPU dự kiến là P100 16 GB. Chưa xác nhận GPU được dành riêng khi đo chi phí.

**Kết luận:** đã có dữ liệu, hai kiến trúc, đủ tám cấu hình và báo cáo thăm dò. Phần còn thiếu chủ yếu là một quy trình A1.1 được khóa rõ ràng, cách đóng băng BatchNorm, đo bộ nhớ, lưu bằng chứng tái lập và xử lý huấn luyện không hữu hạn. Chưa nên dùng tám lượt cũ để kết luận một họ kiến trúc ưu việt hoặc làm bộ so sánh cuối được kiểm soát.

Tài liệu này cùng [bản đề xuất A1.1](../A1/protocol-a1.1-draft.md) ghi lại hiện trạng trước khi hiện thực bước thăm dò. Các bảng vấn đề bên dưới là căn cứ của lần rà soát đó, không phải danh sách khẳng định mọi lỗi vẫn tồn tại sau sửa mã.

Cập nhật ngày 17/10/2026: người học đã duyệt chuẩn bị lượt một vòng cho tám cấu hình. Đã bổ sung [chế độ thăm dò](../A1/PILOT.md), vòng lặp riêng cho A1, chính sách đóng băng, kiểm tra hữu hạn, thời gian/bộ nhớ/môi trường và lưu kết quả riêng từng lượt. Số liệu/báo cáo cũ và trang đích giữ nguyên. Các mở rộng như AdamW, random crop, 100 vòng, ba seed vẫn chờ số đo P100 để quyết định.

## Đối chiếu từng yêu cầu

| Yêu cầu | Hiện trạng | Bằng chứng và việc cần hoàn thiện |
| --- | --- | --- |
| Chọn tập hợp lệ, khó hơn bài E1 | Đạt về lựa chọn | CIFAR-100 được nêu ngay trong đề. Đủ 100 lớp, 60.000 ảnh; E1 dùng MNIST. Không đổi tập dữ liệu. |
| Công bố chủ đề và tập dữ liệu | Đã công bố | Đã kiểm tra trực tiếp [trang A1 đang công khai](https://baond12.github.io/Deep_Learning_Course_HK261/pages/assignment1.html): HTTP 200, có CIFAR-100, ResNet18, ViT-B/16. Bản nguồn ở [pages/assignment1.html](../pages/assignment1.html). |
| Nguồn, giấy phép, lý do chọn | Đạt một phần | Có mô tả nguồn và quy mô nhưng chưa có điều khoản giấy phép rõ. “Công khai” không thay thế thông tin license. Cần mô tả lý do chọn 100 lớp/ảnh màu/nhiều dạng đối tượng và giới hạn độ phân giải gốc 32×32. |
| Tập dữ liệu và DataLoader | Đã hiện thực | [A1/data.py](../A1/data.py) chia cân bằng theo lớp: 40.000 huấn luyện, 10.000 xác thực từ 50.000 ảnh huấn luyện gốc; 10.000 kiểm tra chính thức. Seed 42. Chưa lưu danh sách chỉ số và hash split trong kết quả. |
| Tăng cường dữ liệu | Có, còn đơn giản | Resize 224×224 và lật ngang ngẫu nhiên; xác thực/kiểm tra không có biến đổi ngẫu nhiên. Có chuẩn hóa ImageNet. Chưa có EDA định lượng hoặc đối chiếu phương án tăng cường mới. Không bắt buộc phải có nhiều kỹ thuật phức tạp. |
| Một CNN và một Transformer | Đạt về triển khai và chạy thử | [A1/models.py](../A1/models.py): ResNet18, ViT-B/16; có lịch sử/kết quả của cả hai. |
| Huấn luyện từ đầu và tiền huấn luyện | Đủ cấu hình thăm dò | Có ở cả hai họ, vượt mức tối thiểu một họ. Sáu lượt thiếu thông tin cấu hình riêng, chưa chứng minh cùng điều kiện chạy thực tế chỉ bằng metadata dùng chung. |
| Đóng băng toàn bộ, một phần, tinh chỉnh toàn bộ | Có về tham số; cần làm rõ trạng thái | ResNet mở `fc`, hoặc `layer4 + fc`, hoặc toàn mạng. ViT mở `heads`, hoặc hai khối cuối + `heads`, hoặc toàn mạng. `requires_grad=False` không khóa thống kê BatchNorm khi vòng lặp gọi `model.train()`. |
| Độ chính xác/F1 trên xác thực và kiểm tra | Có | [summary.csv](../A1/results/summary.csv) đủ tám dòng. Hai cấu hình có JSON nguồn; sáu cấu hình lấy kết quả cuối từ log do người học cung cấp. Cần giữ đúng nguồn và số chữ số, không coi các giá trị khôi phục là bản ghi đầy đủ. |
| Chi phí thời gian và số tham số | Có nhưng bằng chứng chưa đồng đều | Thời gian cũ chịu ảnh hưởng chạy song song, thiếu thiết bị/phân bổ tài nguyên. Số tham số sáu lượt là giá trị suy ra từ mã hiện tại. Đợt chính thức cần đếm trực tiếp và lưu môi trường riêng từng lượt. |
| Bộ nhớ GPU | Chưa có | Chưa ghi bộ nhớ đỉnh cấp phát/đặt trước hay cách đo. Cần bổ sung để đáp ứng mục tiêu phân tích chi phí đầy đủ, dù bảng chỉ số tối thiểu ở Mục 4 không liệt kê bộ nhớ. |
| Đường cong và trường hợp sai | Đạt mức minh chứng ban đầu | Có tám lịch sử, ma trận dạng ảnh và ảnh mẫu sai. Chưa có dự đoán từng ảnh/ma trận CSV trong lần nhập kết quả, chưa phân tích kỹ lớp dễ nhầm. |
| Mã nguồn, báo cáo, GitHub Pages | Đã có tổ chức | Có `A1/`, `A1/results/`, báo cáo `reports/assignment1.docx`/`.pdf`, trang A1 và liên kết trên trang chủ. Cần phân biệt rõ báo cáo thăm dò hiện có với quy trình và kết quả chính thức mới. |
| A1.1 và A1.2 liên kết được | Cần chuẩn hóa | Hiện có ví dụ 5 vòng/lô 32 bên cạnh kết quả 50 vòng/lô 16 và metadata chưa đủ. Cần một bản quy trình có phiên bản, sau đó A1.2 đối chiếu từng quyết định trong bản đó. |

## Giấy phép cần trình bày trung thực

Đã kiểm tra [trang CIFAR gốc](https://cave.cs.toronto.edu/kriz/cifar.html) và [thẻ dữ liệu của uoft-cs](https://huggingface.co/datasets/uoft-cs/cifar100/blob/main/README.md). Trang gốc cung cấp dữ liệu, mô tả và yêu cầu trích dẫn báo cáo của Alex Krizhevsky năm 2009; thẻ `uoft-cs/cifar100` ghi `license: unknown`.

Không có căn cứ từ những nguồn này để tự ghi MIT, CC-BY hoặc CC-BY-NC cho tập dữ liệu đang tải qua torchvision. Trong A1.1, cần ghi rõ nguồn, ngày truy cập, tình trạng chưa xác định được giấy phép cụ thể và yêu cầu trích dẫn. Nếu tiêu chí chấm yêu cầu một giấy phép xác định, cần xác nhận cách ghi nhận này với giảng viên. Đây là điểm cần hoàn thiện thông tin, không phải đề nghị đổi CIFAR-100 mà người học đã chọn và đề đã cho phép làm ví dụ.

## Những vấn đề của các lượt thăm dò

Đã đọc lại tám CSV lịch sử: mỗi cấu hình có 50 vòng, tổng 400 bản ghi. Vòng xác thực tốt nhất khớp bảng kết quả. Các giới hạn ở [README kết quả](../A1/results/README.md) và [danh mục nguồn](../A1/results/results_inventory.csv) vẫn còn nguyên:

- ViT từ đầu có loss không hữu hạn từ vòng 11, trạng thái xác thực tốt nhất ở vòng 10.
- ViT tinh chỉnh toàn bộ có loss không hữu hạn từ vòng 37, trạng thái tốt nhất ở vòng 3.
- Sáu cấu hình cần xác nhận thông tin chạy riêng. Không thể dùng thông tin của hai JSON để xác nhận thay sáu lượt đó.
- Thời gian các lượt cũ không phải phép so tốc độ có kiểm soát. Không dự báo tốc độ P100 mới bằng tỷ lệ trực tiếp từ những thời gian này.
- Chưa có nhiều hạt giống huấn luyện hoặc bộ nhớ GPU. Không suy ra mức độ ổn định hay ý nghĩa thống kê.

Các dữ kiện trên giúp lập giả thuyết: cần kiểm tra ổn định số học, tốc độ học, lịch học, đóng băng đúng nghĩa và cách ghi kết quả. Chưa xác định được nguyên nhân NaN; không kết luận chỉ cần tăng batch là sửa được NaN hay tăng độ chính xác.

## Các điểm mã nguồn cần xử lý trước đợt chính thức

| Mức ưu tiên | Điểm đã thấy | Hướng xử lý đề xuất |
| --- | --- | --- |
| Cao | [models.py](../A1/models.py) khóa gradient nhưng [common/engine.py](../common/engine.py) gọi `model.train()` cho toàn mạng | Khóa cả thống kê BatchNorm của phần ResNet được đóng băng; kiểm chứng tham số và buffer không đổi. Khi mở `layer4`, cho BatchNorm trong phần này hoạt động theo chính sách đã chốt. |
| Cao | Vòng lặp chưa kiểm tra loss/gradient không hữu hạn | Dừng lượt lỗi, ghi bước/vòng và chẩn đoán; không tiếp tục cập nhật trọng số hoặc tính lượt lỗi như lượt hoàn tất ổn định. Không xóa dữ liệu lỗi để làm đẹp đường cong. |
| Cao | `--no-download-weights` khiến mô hình mang tên pretrained được dựng với `weights=None` | Chỉ dùng cho kiểm tra hình dạng; lượt chính thức phải xác nhận đã nạp đúng trọng số tiền huấn luyện, tên phiên bản và nguồn. |
| Cao | Kết quả tổng hợp chỉ được ghi cuối vòng lặp tám cấu hình; thư mục đầu ra/checkpoint có tên dùng chung | Lưu kết quả từng lượt ngay khi hoàn tất, tên đường dẫn có cấu hình/hạt giống, không ghi đè. Khôi phục được trạng thái khi máy ngắt nếu chọn hỗ trợ tiếp tục. |
| Cao | `set_seed` gọi một lần trước tất cả cấu hình; bộ nạp chưa có bộ sinh thứ tự riêng | Tách seed chia dữ liệu và seed huấn luyện; đặt lại cho mỗi lượt và lưu chỉ số split. Kết quả một cấu hình không nên phụ thuộc vị trí trong danh sách `all`. |
| Trung bình | Một tốc độ học/Adam cố định cho tất cả chế độ; chưa có lịch giảm | Cho phép cấu hình riêng, cùng ngân sách chọn bằng xác thực; lưu cả cấu hình thắng và những thử nghiệm đã thực hiện. Khác công thức học phải được công khai trong diễn giải. |
| Trung bình | Chưa có đo bộ nhớ, metadata đầy đủ, CSV dự đoán hoặc ma trận | Bổ sung giống cách lưu bằng chứng E2/E3, đồng thời đếm tổng/tham số được học trực tiếp. |
| Trung bình | Bản sao trạng thái tốt nhất được giữ trên cùng thiết bị sau `model.to(device)` | Khi mở rộng batch, cân nhắc lưu bản tốt nhất trên CPU hoặc đĩa; đo bộ nhớ bằng chính cách lưu được dùng cho lượt chính thức. |
| Trung bình | Hàm phân tích lấy trung bình các chế độ trong mỗi họ | Không dùng trung bình này thay cho trung bình nhiều seed hoặc bằng chứng CNN/ViT tốt hơn. So sánh từng cặp chế độ tương ứng và báo riêng sự biến thiên qua lượt chạy. |

Những thay đổi dành riêng cho A1 nên đặt trong `A1/`; nếu cần sửa `common/`, giữ hành vi mặc định của E1–E3 và kiểm tra không làm thay đổi cơ chế đánh giá đã dùng. Không chạy lại hoặc thay số liệu E1–E3 để phục vụ A1.

## Phân biệt yêu cầu môn học với phần đề xuất thêm

Đề yêu cầu hai họ kiến trúc, so từ đầu/tiền huấn luyện, ba chế độ đóng băng, cùng split, bảng/chỉ số/phân tích lỗi và mã tái lập. Đề không bắt buộc 100 vòng, lô 256, ba seed, AdamW, warm-up, cosine hoặc AMP.

Đề xuất mở rộng nhằm tăng độ tin cậy: quy trình cố định, đo khả thi trên P100, ghi nguồn đầy đủ, tìm tốc độ học có kiểm soát và nhiều seed nếu ngân sách cho phép. Mọi thông số cụ thể trong bản dự thảo cần được xem là kế hoạch, không phải kết quả đã chạy hoặc yêu cầu của giảng viên.

## Nguồn kỹ thuật đã đối chiếu

- [BatchNorm2d của PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm2d.html): thống kê chạy được cập nhật khi huấn luyện, độc lập với việc có học tham số affine hay không.
- [ResNet18 của torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) và [ViT-B/16](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html): trọng số `IMAGENET1K_V1`, đầu vào và chuẩn hóa. Bản chạy thực tế vẫn phải khóa phiên bản thư viện của máy P100.
- [AMP của PyTorch](https://docs.pytorch.org/docs/2.14/amp.html) và [bộ nhớ đỉnh cấp phát](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.max_memory_allocated.html): cơ sở đề xuất đo độ chính xác tính toán và bộ nhớ. Chưa chạy benchmark P100 trong lần rà soát.

