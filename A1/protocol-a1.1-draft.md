# Dự thảo A1.1 về dữ liệu và quy trình so sánh CNN với Transformer

Ngày lập: 17/10/2026. Trạng thái: **đề xuất để người học review, chưa khóa làm quy trình chạy chính thức**. Hạn A1.1: 26/10/2026, 23:59 GMT+7. Hạn A1.2: 09/11/2026, 23:59 GMT+7.

**Bản lưu lịch sử:** người học đã duyệt [protocol chính thức](protocol-a1.1.md)
và [JSON cấu hình](protocol-a1.1.json) sau lượt LR bổ sung. Nội dung dự thảo bên
dưới không thay thế bản đã duyệt; dùng [hướng dẫn mới](RUN_PROTOCOL.md) để chạy.

**Cập nhật 18/10/2026:** đã đọc ba lượt thăm dò tám cấu hình (một epoch batch 16,
một epoch batch 64 và năm epoch batch 64) và lượt bổ sung hai cấu hình full,
năm epoch batch 64 với learning rate `1e-5` trên P100. Phân tích số liệu,
giới hạn và đề xuất ngân sách tiếp theo ở [phân tích thăm dò A1](../docs/a1-pilot-analysis.md).
Batch 64 đã được người học đồng ý. Đề xuất hiện tại chọn LR `1e-4` cho ResNet
full và `1e-5` cho ViT full; ngân sách 30 epoch và protocol chính thức chưa được
chốt. Mã và cách gửi kết quả ở [PILOT.md](PILOT.md). Những phương
án 100 vòng, ba seed, augmentation mới và AdamW bên dưới là các lựa chọn dự thảo
trước đó; chưa áp dụng vào pilot và không mặc nhiên là kế hoạch đã duyệt.

Mục tiêu là so sánh ResNet18 và ViT-B/16 trên CIFAR-100, xét tác động của tiền huấn luyện và phạm vi tinh chỉnh, đồng thời ghi nhận chất lượng dự đoán, tham số, thời gian, bộ nhớ và lỗi. Những lượt đã chạy là bằng chứng thăm dò để thiết kế quy trình này; không dùng thay kết quả có kiểm soát của đợt mới.

## 1. Câu hỏi và phạm vi kết luận

1. Trong từng kiến trúc, tiền huấn luyện thay đổi kết quả so với huấn luyện từ đầu thế nào?
2. Với trọng số tiền huấn luyện cố định, chỉ học đầu phân loại, mở phần cuối và tinh chỉnh toàn bộ khác nhau thế nào về chất lượng và chi phí?
3. Giữa hai kiến trúc ở cùng chế độ, sự đánh đổi độ chính xác, số tham số được học, thời gian và bộ nhớ trên P100 ra sao?
4. Những lớp nào khó phân biệt và các mô hình sai giống/khác nhau ở đâu?

Không đặt mục tiêu chứng minh CNN hoặc Transformer luôn tốt hơn. ResNet18 và ViT-B/16 khác quy mô, công thức tiền huấn luyện và chi phí. Chọn cùng nguồn ImageNet-1K không khiến hai mô hình đã được tiền huấn luyện theo cùng ngân sách. Nếu tốc độ học được chọn riêng cho mỗi cấu hình, kết luận là về các quy trình được chọn bằng xác thực, chưa phải tác động thuần túy của kiến trúc.

## 2. Dữ liệu cố định

| Thành phần | Đề xuất giữ nguyên |
| --- | --- |
| Tập dữ liệu | CIFAR-100 gốc, đủ 100 nhãn chi tiết; không đổi sau A1.1 |
| Ảnh gốc | RGB 32×32; 60.000 ảnh; không dùng nhãn nhóm lớn làm nhãn phân loại chính |
| Huấn luyện | 400 ảnh/lớp từ tập huấn luyện gốc: 40.000 ảnh |
| Xác thực | 100 ảnh/lớp còn lại từ tập huấn luyện gốc: 10.000 ảnh |
| Kiểm tra | Toàn bộ 100 ảnh/lớp của tập kiểm tra chính thức: 10.000 ảnh |
| Chia dữ liệu | Giữ thuật toán `A1/data.py` và seed chia 42; lưu chỉ số gốc và SHA-256 để xác nhận |
| Kích thước đầu vào mô hình | 224×224 cho cả hai, mọi chế độ; ghi rõ là phóng ảnh, không tăng thông tin gốc |

Đây là dùng toàn bộ dữ liệu, không phải bỏ bớt lớp hoặc bỏ 10.000 ảnh huấn luyện: 50.000 ảnh huấn luyện gốc được chia thành 40.000 để học và 10.000 để xác thực. Giữ cách chia này để liên kết được với những lượt thăm dò; không tự chuyển sang split mới chỉ vì muốn tăng điểm.

**Lý do chọn:** tập được nêu trong đề, có nhiều lớp hơn MNIST và những nhóm đối tượng dễ nhầm; đủ lớn để học chuyển giao có ý nghĩa, nhưng vẫn phù hợp phạm vi môn học. Điểm hạn chế là ảnh gốc nhỏ, nên không đồng nhất bài toán này với phân loại ảnh độ phân giải cao.

**Nguồn và giấy phép:** tải bằng torchvision từ [trang CIFAR gốc](https://cave.cs.toronto.edu/kriz/cifar.html); trích dẫn Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009. Ngày kiểm tra nguồn: 17/10/2026. Trang gốc chưa nêu rõ giấy phép chuẩn hóa; [thẻ uoft-cs/cifar100](https://huggingface.co/datasets/uoft-cs/cifar100/blob/main/README.md) ghi license `unknown`. Không gán một giấy phép chưa được xác minh. Cần hoàn thiện cách ghi nhận hoặc xác nhận với giảng viên nếu yêu cầu một tên giấy phép xác định.

**EDA cần đưa vào A1.1:** số ảnh mỗi lớp và mỗi split; phân bố kích thước/kênh ảnh; vài ảnh có nhãn của các nhóm tiêu biểu; xác nhận không trùng chỉ số giữa huấn luyện và xác thực, không đưa ảnh kiểm tra vào hai tập này. Kiểm tra bằng chỉ số không tự chứng minh hoàn toàn không có ảnh trùng nội dung; chỉ kết luận ở mức đã kiểm tra. Các bảng/ảnh EDA phải lấy từ dữ liệu thật, chưa được tạo trong đợt lập kế hoạch này.

## 3. Tiền xử lý và tăng cường dữ liệu

Phương án chính đề xuất:

- Huấn luyện: `RandomCrop(32, padding=4)` trên ảnh gốc, lật ngang với xác suất 0,5, resize 224×224 bằng nội suy song tuyến tính, chuyển tensor và chuẩn hóa theo ImageNet.
- Xác thực/kiểm tra: resize 224×224, chuyển tensor, chuẩn hóa; không cắt/lật ngẫu nhiên.
- Cố định mean `(0.485, 0.456, 0.406)` và std `(0.229, 0.224, 0.225)`, cách padding, nội suy và antialias theo cấu hình có phiên bản.
- Áp dụng cùng công thức đầu vào cho hai họ và mọi chế độ. Đề không bắt buộc phải dùng lại nguyên bộ biến đổi ImageNet; khác biệt với resize-256/crop-224 của trọng số gốc phải được mô tả, không nói là giống hoàn toàn.

Đây là thay đổi có chủ đích so với lượt cũ chỉ resize/lật ngang. Có thể đối chiếu phương án cũ và mới trong thăm dò bằng xác thực; chỉ chọn một công thức cho bảng so sánh chính. Chưa đưa MixUp, CutMix hoặc RandAugment vào ma trận bắt buộc để tránh mở quá nhiều yếu tố cùng lúc.

## 4. Ma trận tám cấu hình

| Mã cấu hình | Khởi tạo backbone | Phần được cập nhật |
| --- | --- | --- |
| `resnet18_scratch` | Ngẫu nhiên | Toàn bộ ResNet18 và đầu 100 lớp |
| `resnet18_pretrained_head` | ImageNet-1K V1 | Chỉ `fc` |
| `resnet18_pretrained_partial` | ImageNet-1K V1 | `layer4` và `fc` |
| `resnet18_pretrained_full` | ImageNet-1K V1 | Toàn bộ mạng |
| `vit_b16_scratch` | Ngẫu nhiên | Toàn bộ ViT-B/16 và đầu 100 lớp |
| `vit_b16_pretrained_head` | ImageNet-1K V1 | Chỉ `heads` |
| `vit_b16_pretrained_partial` | ImageNet-1K V1 | Hai khối encoder cuối và `heads`; các phần còn lại giữ nguyên |
| `vit_b16_pretrained_full` | ImageNet-1K V1 | Toàn bộ mạng |

Khóa tường minh `ResNet18_Weights.IMAGENET1K_V1` và `ViT_B_16_Weights.IMAGENET1K_V1`, không chỉ ghi `DEFAULT`. Thay đầu phân loại thành 100 lớp; lưu chính xác tên trọng số, nguồn tải, phiên bản thư viện và mã nguồn. Không dùng `--no-download-weights` trong lượt pretrained chính thức, trừ khi đã có cơ chế nạp và xác nhận trọng số từ tệp cục bộ. Cờ hiện tại dựng mô hình với `weights=None`, không phải chỉ ngăn tải lại tệp.

Đầu phân loại mới được khởi tạo có kiểm soát; trong cùng kiến trúc và seed, các chế độ pretrained dùng cùng trạng thái ban đầu trước khi áp dụng mặt nạ đóng băng. ViT và ResNet là hai mô hình khác nhau, không có yêu cầu khớp đầu ra như hai bản cùng MSA ở E2.

**Đóng băng đúng phạm vi:**

- Với ResNet chỉ học đầu: giữ các tầng backbone, bao gồm BatchNorm, ở chế độ đánh giá; chỉ đầu phân loại được học.
- Với ResNet mở một phần: phần trước `layer4` giữ nguyên tham số và thống kê BatchNorm; `layer4` và đầu được huấn luyện. Full/scratch được cập nhật bình thường.
- Với ViT: giữ các mô-đun bị đóng băng ở chế độ đánh giá và phần được mở ở chế độ huấn luyện. Ghi chính xác chính sách LayerNorm và các tham số ngoài hai khối cuối; không mô tả chung là “mở toàn encoder”.
- Kiểm tra một bước cập nhật: đầu ra có dạng `[B,100]`, giá trị và gradient hữu hạn, đúng danh sách tham số được học; tham số/buffer đóng băng không đổi, phần dự kiến học có cập nhật. Kiểm tra hình dạng hiện tại chưa đủ thay thế kiểm tra này.

## 5. Cấu hình học và cách ra quyết định

Không chốt tốc độ học, số vòng và batch chỉ bằng dung lượng VRAM. Đề xuất tách công việc thành ba bước sau để người học duyệt trước khi hiện thực.

### Bước 1 Đo khả thi và kiểm tra ổn định trên P100

1. Ghi đúng model GPU, VRAM khả dụng, Python/PyTorch/torchvision/CUDA, số worker và mức độ dùng chung thiết bị.
2. Dùng cùng ảnh 224×224, kiểm tra từng chế độ với batch 16, 32, 64; chỉ thử 128 nếu đã có dư bộ nhớ. Đo cả lan truyền ngược/bước tối ưu, không chỉ suy luận, và dùng đúng chính sách lưu trạng thái tốt nhất dự kiến.
3. Chọn batch vật lý chung cho cả tám cấu hình nếu khả thi. Batch 64 là ứng viên, không phải cam kết phù hợp P100. Không mặc định batch 256.
4. Chạy thử vài vòng để phát hiện lỗi số học và ghi tốc độ; chưa đánh giá tập kiểm tra. Một phép thử ngắn không chứng minh cả 100 vòng sẽ ổn định.

Ưu tiên FP32 làm cấu hình kiểm chứng đầu tiên vì các lượt cũ đã có NaN. FP16 với autocast/GradScaler là phương án thử để giảm bộ nhớ, chỉ dùng sau khi kiểm tra ổn định trên đúng môi trường P100; không mặc định dùng BF16 hoặc hứa tăng tốc. AMP không phải biện pháp tự động chữa NaN.

Nếu không có batch chung phù hợp, có thể dùng tích lũy gradient và ghi cả batch vật lý, số bước tích lũy, batch hiệu dụng. Batch hiệu dụng bằng nhau không khiến BatchNorm tương đương batch vật lý lớn hoặc làm mọi chi phí giống nhau. Ưu tiên giảm batch chung trước khi thêm nhánh này.

### Bước 2 Chọn cấu hình bằng tập xác thực

Phương án xuất phát để thử, chưa phải khẳng định tối ưu:

| Thành phần | Đề xuất |
| --- | --- |
| Bộ tối ưu | AdamW cho bảng chính, cùng cách đặt nhóm tham số và quy tắc weight decay |
| Số vòng chính | Dự kiến 100 cho mỗi lượt; xác nhận tính khả thi theo thời gian đo được trước khi khóa A1.1 |
| Lịch tốc độ học | Warm-up 5 vòng rồi cosine đến hết ngân sách; không dừng sớm trong bảng so sánh ngân sách cố định |
| Chọn trọng số | Độ chính xác xác thực cao nhất; nếu bằng nhau giữ vòng sớm nhất |
| Gradient | Kiểm tra hữu hạn trước cập nhật; dự kiến clip chuẩn gradient ở 1,0, ghi rõ nếu dùng |
| Seed chia | Luôn 42, giữ nguyên chỉ số chia |
| Seed huấn luyện | Dự kiến 42, 43, 44 cho mỗi cấu hình; đây là ba lượt lặp, không phải ba split |

Các ứng viên tốc độ học ban đầu:

| Nhóm cấu hình | Ba ứng viên tốc độ học |
| --- | --- |
| ResNet18 từ đầu | 1e-4, 3e-4, 1e-3 |
| ViT-B/16 từ đầu | 3e-5, 1e-4, 3e-4 |
| Chỉ học đầu, cả hai họ | 1e-4, 3e-4, 1e-3 |
| Mở một phần, cả hai họ | 1e-5, 3e-5, 1e-4 |
| Tinh chỉnh toàn bộ, cả hai họ | 5e-6, 1e-5, 5e-5 |

Đây là một danh sách nhỏ có kiểm soát, không quét mọi tổ hợp. Đề xuất dùng cùng ngân sách sàng lọc khoảng 20–30 vòng/ứng viên, seed huấn luyện 42, ba ứng viên mỗi cấu hình; số vòng sàng lọc phải được chốt trước khi so. So sánh bằng xác thực; không loại ViT từ đầu chỉ vì thua ở vài vòng khởi động. Kết quả ngắn là cơ sở lựa chọn hữu hạn, không chứng minh thứ hạng khi hội tụ hoàn toàn.

Sau sàng lọc, khóa cấu hình cho từng chế độ trước đợt chính; không đổi tốc độ học theo điểm kiểm tra. Weight decay khởi điểm đề xuất 1e-4 để giới hạn yếu tố thay đổi; cách xử lý bias/LayerNorm/BatchNorm phải được ghi rõ trong cấu hình. Nếu muốn tìm cả weight decay hoặc tốc độ học riêng backbone/head, cần công bố thêm ngân sách và ma trận, không âm thầm thêm cho riêng mô hình đang thua.

100 vòng, ba seed và mọi giá trị trong bảng là đề xuất bổ sung, không phải yêu cầu môn học. Trước khi khóa, ước tính tổng thời gian từ benchmark P100 thực tế: tổng thời gian các lượt sàng lọc + tám cấu hình × số seed × thời gian mỗi lượt chính + thời gian đánh giá/kiểm tra. Ba seed tạo 24 lượt chính, còn ma trận kiến trúc/chế độ vẫn là tám cấu hình. Cùng số vòng không phải cùng FLOPs hoặc cùng thời gian; đó là so sánh ngân sách số lượt duyệt dữ liệu.

Nếu ước tính không phù hợp hạn nộp, ưu tiên đủ tám cấu hình ổn định và thu đủ bằng chứng; thống nhất giảm phạm vi tìm kiếm hoặc số seed cho mọi cấu hình và công khai giới hạn. Không chỉ lặp thêm mô hình có kết quả đẹp rồi dùng làm bảng so sánh chính.

### Bước 3 Chạy chính thức và đánh giá cuối

Đặt lại seed huấn luyện và bộ sinh thứ tự DataLoader cho mỗi lượt; seed chia dữ liệu tách riêng. Giữ nguyên cấu hình đã khóa khi đổi seed. Không lấy seed tốt nhất làm kết quả đại diện.

Sau khi khóa công thức huấn luyện, mỗi lượt dùng trọng số chọn bằng xác thực để đánh giá tập kiểm tra; mọi chỉ số, ma trận và dự đoán của lượt phải lấy từ cùng trạng thái này. Không dùng điểm kiểm tra để đổi cấu hình rồi chạy lại chỉ mô hình đó. Test đã được xem trong đợt thăm dò cũ, vì vậy báo cáo cần nói rõ lịch sử này và không tuyên bố đây là tập kiểm tra chưa từng được xem.

## 6. Bộ chỉ số và bằng chứng phải lưu

| Nhóm | Bằng chứng cho mỗi lượt |
| --- | --- |
| Nhận diện lượt | Kiến trúc, chế độ, seed chia/huấn luyện, mã phiên bản quy trình, commit/hash mã, thời điểm |
| Dữ liệu | Danh sách lớp, chỉ số chia/hash, kích thước ảnh, cấu hình biến đổi, batch vật lý/hiệu dụng, worker |
| Mô hình và học | Trọng số nguồn, trạng thái khởi tạo, mặt nạ tham số được học, chính sách BatchNorm, optimizer/lịch học, độ chính xác tính toán |
| Lịch sử | Hàm mất mát, độ chính xác huấn luyện/xác thực, tốc độ học theo vòng; trạng thái hoàn tất/lỗi và chẩn đoán |
| Kết quả | Vòng tốt nhất, độ chính xác xác thực/kiểm tra, F1 trung bình theo lớp, mất mát kiểm tra, tổng/tham số được học |
| Chi phí | Thời gian huấn luyện+xác thực; thời gian đánh giá tách riêng; bộ nhớ đỉnh GPU cấp phát/đặt trước, cách đo và thiết bị |
| Phân tích lỗi | CSV chỉ số ảnh/nhãn thật/dự đoán, CSV ma trận, biểu đồ ma trận, mẫu sai và báo cáo theo lớp |
| Tái lập | Trạng thái mô hình tốt nhất lưu ngoài Git; nếu hỗ trợ tiếp tục, lưu cả optimizer/scheduler/scaler/RNG và vị trí huấn luyện |

Đo thời gian và bộ nhớ khi chạy riêng một lượt trên GPU nếu có thể. Dùng đồng bộ CUDA khi đo ranh giới thời gian, đặt lại thống kê bộ nhớ trước phần cần đo, ghi phạm vi đo có gồm xác thực và lưu trạng thái hay không. `max_memory_allocated` và `max_memory_reserved` của PyTorch không phải tổng bộ nhớ của mọi tiến trình mà `nvidia-smi` hiển thị; không trộn hai khái niệm.

Sau chạy nhiều seed, báo cáo từng lượt cùng trung bình và độ lệch chuẩn mẫu theo tám cấu hình. Có thể thể hiện chênh lệch theo seed ghép cặp trên cùng split. Ba seed vẫn là số ít, không đủ tự tuyên bố ưu thế phổ quát hoặc ý nghĩa thống kê. Không lấy trung bình bốn chế độ để thay cho trung bình các lượt lặp.

## 7. Tổ chức repo và báo cáo

Giữ `A1/` cho mã nguồn, `A1/results/` cho kết quả và `reports/assignment1.docx`/`.pdf` cho báo cáo chung A1. E1–E3 vẫn là bài khác với cùng phong cách trình bày, không gộp số liệu hoặc tái sử dụng baseline MNIST để tính chênh lệch với CIFAR-100.

Đề xuất khi hiện thực:

- Giữ nguyên bộ thăm dò đang có ở `A1/results/`; không ghi đè hoặc đổi nghĩa các số đó.
- Bộ chính thức mới ở `A1/results/protocol_v2/`, với thông tin riêng từng cấu hình/seed và bảng tổng hợp chung. Chi tiết tên tệp/thư mục phải được thống nhất cùng mã lưu kết quả; không xem đường dẫn đề xuất này là tính năng CLI hiện đã hỗ trợ đầy đủ.
- Quy trình đã duyệt lưu thành một bản có phiên bản; A1.2 tham chiếu đúng phiên bản đó. Mọi thay đổi sau khóa có nhật ký lý do và phạm vi chạy lại; không thay dataset sau A1.1 nếu chưa có đồng ý của giảng viên.
- Trang A1 công bố CIFAR-100 và hai kiến trúc như hiện tại, bổ sung trạng thái “kết quả thăm dò” cho bộ cũ và liên kết rõ bản A1.1 được duyệt. Chỉ liên kết bộ kết quả chính được dùng trong bảng cuối khi có đủ bằng chứng.
- Markdown tiếng Việt có dấu, thuật ngữ thống nhất với E1–E3. Báo cáo Word dùng Times New Roman và hình đơn sắc; PDF xuất từ đúng Word được nhóm duyệt. Không ghi trạng thái đã nộp chỉ vì đã chạy xong.

## 8. Nội dung A1.1 và sự nối tiếp sang A1.2

| A1.1 cần chốt/công bố | A1.2 cần chứng minh |
| --- | --- |
| CIFAR-100, nguồn/license, lý do và EDA | Dùng đúng tập, nhãn, kích thước và ghi nhận nguồn đã công bố |
| Chỉ số split và công thức biến đổi | Hash/chỉ số khớp, không có tăng cường ngẫu nhiên ở xác thực/kiểm tra |
| Tám cấu hình và chính sách đóng băng | Đếm trực tiếp, kiểm chứng tham số/buffer, trạng thái đầu ra hợp lệ |
| Ngân sách lựa chọn, vòng/lô, lịch học, seed và độ chính xác tính toán | Log đầy đủ, không đổi tùy điểm kiểm tra; nêu rõ sai lệch so với kế hoạch nếu có |
| Cách chọn trọng số và đánh giá | Kết quả và hình đều từ trạng thái xác thực tốt nhất, không từ vòng cuối mặc định |
| Chỉ số và cách đo chi phí | Bảng, đường cong, bộ nhớ/thời gian, ma trận và phân tích lớp sai |
| Kế hoạch nhiều seed và giới hạn | Báo tất cả seed, trung bình/độ lệch chuẩn hoặc công khai lý do chỉ có một lượt |
| Bố cục và liên kết GitHub Pages | Mã, số liệu, báo cáo và trang đích nhất quán; PDF/Word cùng phiên bản |

A1.1 chưa cần số liệu so sánh cuối. Có thể trích những bất thường thăm dò để giải thích quyết định thiết kế, nhưng không trình bày tám lượt cũ như đã xác nhận xong quy trình mới.

## 9. Lịch công việc dự kiến

- 17–19/10: rà soát, hoàn thiện nguồn dữ liệu/EDA và đo khả thi P100.
- 20–23/10: hiện thực các sửa đổi được duyệt, kiểm tra đóng băng/ổn định/lưu kết quả và chọn cấu hình bằng xác thực.
- 24–26/10: khóa A1.1, rà soát và công bố quy trình/báo cáo đúng hạn.
- 27/10–06/11: chạy ma trận chính và các seed đã chốt; theo dõi đủ bằng chứng, không tune bằng kiểm tra.
- 07–09/11: tổng hợp, viết phân tích, nhóm review, xuất PDF và người học tự push theo lệnh được chuẩn bị.

Lịch chạy chính phải được kiểm tra lại bằng thời gian P100 đo thực tế, không suy ra chỉ từ VRAM 16 GB. Các mốc khóa, lựa chọn tham số và số lượt có thể được chốt sớm hơn nếu đã đủ căn cứ.

