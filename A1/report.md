# A1 - CNN vs Transformer trên full CIFAR-100

**Tài liệu lịch sử của lượt cũ.** Quy trình hiện tại do người học duyệt ở
[protocol-a1.1.md](protocol-a1.1.md), cấu hình ở
[protocol-a1.1.json](protocol-a1.1.json). Chạy theo [RUN_PROTOCOL.md](RUN_PROTOCOL.md).
Đã có [phân tích lượt 30 epoch](results/epochs30_batch64_seed42_v1/analysis.md);
những số liệu/ví dụ bên dưới không thay thế bộ kết quả mới đó.

## 1. Chủ đề

A1 so sánh CNN và Transformer trên một tập ảnh lớn hơn E1. Dataset chọn là
**full CIFAR-100, 100 classes**, không trùng MNIST/Fashion-MNIST/CIFAR-10 đã dùng cho E1.

## 2. Dataset và license

Dataset gốc: CIFAR-100, gồm 60,000 ảnh màu `32 x 32`, 100 lớp, mỗi lớp có 500 ảnh
train và 100 ảnh test. Dataset được cung cấp công khai bởi Alex Krizhevsky cho mục đích
nghiên cứu/học tập.

Protocol A1.1 dùng đủ 100 lớp nhưng giới hạn số mẫu mỗi lớp để split cân bằng và
tái lập được:

- Train: 400 ảnh/lớp từ training set gốc, tổng 40,000 ảnh.
- Validation: 100 ảnh/lớp từ training set gốc, tổng 10,000 ảnh.
- Test: 100 ảnh/lớp từ test set gốc, tổng 10,000 ảnh.
- Split được thực hiện cân bằng theo class bằng seed `42`.

Dataset được tải một lần bằng `A1/prepare_data.py` vào `--data-dir`; các lần train sau
chỉ đọc lại cùng path đó và không tự download lại. Thư mục dataset nằm ở local/Google Drive
và không push lên GitHub.

## 3. Data pipeline và augmentation

Theo slide data pipeline, test set chỉ dùng cho đánh giá cuối; validation dùng để chọn
checkpoint/cấu hình.

Transform train:

- Resize ảnh về `224 x 224`.
- Random horizontal flip.
- ToTensor.
- Normalize theo ImageNet mean/std để phù hợp pretrained models.

Transform validation/test:

- Resize `224 x 224`.
- ToTensor.
- Normalize ImageNet mean/std.

## 4. Kiến trúc

### 4.1 CNN

CNN chọn **ResNet18** từ `torchvision`. ResNet là CNN backbone nhẹ, phổ biến, có
pretrained ImageNet weights và phù hợp để so sánh các chế độ freeze/fine-tune.

### 4.2 Transformer

Transformer chọn **ViT-B/16** từ `torchvision`. ViT chuyển ảnh thành patch tokens, thêm
classification head và dùng self-attention toàn cục trên patch sequence.

## 5. Protocol thí nghiệm

Các thí nghiệm mặc định:

| Experiment | Family | Architecture | Training mode |
| --- | --- | --- | --- |
| `resnet18_scratch` | CNN | ResNet18 | From scratch |
| `resnet18_pretrained_head` | CNN | ResNet18 | Freeze backbone, train head |
| `resnet18_pretrained_partial` | CNN | ResNet18 | Freeze một phần, train layer4 + head |
| `resnet18_pretrained_full` | CNN | ResNet18 | Full fine-tune |
| `vit_b16_scratch` | Transformer | ViT-B/16 | From scratch |
| `vit_b16_pretrained_head` | Transformer | ViT-B/16 | Freeze backbone, train head |
| `vit_b16_pretrained_partial` | Transformer | ViT-B/16 | Freeze một phần, train 2 encoder blocks cuối + head |
| `vit_b16_pretrained_full` | Transformer | ViT-B/16 | Full fine-tune |

## 6. Cấu hình train mặc định

- Epochs: 5.
- Batch size: 32.
- Num classes: 100.
- Train/validation/test per class: 400/100/100.
- Optimizer: Adam.
- Learning rate: `1e-4`.
- Weight decay: `1e-4`.
- Loss: CrossEntropyLoss.
- Metric: accuracy, macro F1, số tham số total/trainable, training time.

## 7. Lệnh chạy

Tải dataset một lần:

```bash
python A1/prepare_data.py --data-dir /path/to/cifar100
```

Train dùng lại cùng `--data-dir`:

```bash
python A1/train.py --data-dir /path/to/cifar100 --experiments all --epochs 5 --batch-size 32 --device auto
```

Trên Google Colab GPU:

```bash
python A1/train.py --data-dir /path/to/cifar100 --experiments all --epochs 5 --batch-size 32 --device cuda --progress
```

Smoke test không tải dataset/weights:

```bash
python A1/smoke_test.py
```

## 8. Kết quả thực nghiệm

Đã tổng hợp đủ tám cấu hình. Mỗi history có 50 epoch; chọn trạng thái mô hình
có độ chính xác xác thực cao nhất để đánh giá cuối trên tập kiểm tra theo mã huấn luyện.
Hai cấu hình ResNet18 partial và ViT scratch lấy chỉ số từ JSON/CSV gốc; sáu cấu hình
khác lấy từ log người học cung cấp, với chỉ số test đến bốn chữ số thập phân.
Cấu hình mặc định ở mục 6–7 là cấu hình ví dụ, không phải bản ghi đầy đủ của các lượt này.
JSON gốc của hai cấu hình ghi batch size 16 và 50 epoch; metadata chạy riêng của sáu
cấu hình bổ sung vẫn cần xác nhận. Số tham số cho sáu cấu hình đó được suy ra từ kiến
trúc mã nguồn hiện tại, cần đối chiếu nếu mã chạy khác với repo.

| Cấu hình | Epoch tốt nhất | Accuracy xác thực | Test loss | Accuracy kiểm tra | Macro F1 | Thời gian |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `resnet18_scratch` | 19 | 61.64% | 1.5821 | 60.97% | 0.6070 | 2h 27m 17s |
| `resnet18_pretrained_head` | 40 | 61.08% | 1.4319 | 60.47% | 0.6022 | 1h 24m 4s |
| `resnet18_pretrained_partial` | 7 | 72.83% | 1.050193 | 72.27% | 0.722477 | 1h 33m 4s |
| `resnet18_pretrained_full` | 11 | 77.29% | 0.9163 | 76.84% | 0.7685 | 2h 25m 35s |
| `vit_b16_scratch` | 10 | 35.37% | 2.54557 | 35.30% | 0.342563 | 18h 23m 37s |
| `vit_b16_pretrained_head` | 19 | 82.18% | 0.6376 | 81.32% | 0.8130 | 26h 8m 14s |
| `vit_b16_pretrained_partial` | 27 | 83.66% | 0.9088 | 83.25% | 0.8324 | 28h 48m 53s |
| `vit_b16_pretrained_full` | 3 | 78.73% | 0.7332 | 78.64% | 0.7859 | 40h 44m 46s |

Các bảng đầy đủ và nguồn dữ liệu ở `results/summary.csv`, `results/metrics.json`;
danh mục kiểm chứng ở `results/results_inventory.csv`. Phân tích và bốn biểu đồ so sánh
được tạo bằng `python A1/analyze_results.py --results-dir A1/results`.

## 9. Thảo luận kết quả

Trong nhóm ResNet18, full fine-tune đạt accuracy kiểm tra cao nhất 76,84%, hơn
from-scratch 15,87 điểm phần trăm. Partial đạt 72,27%; head-only đạt 60,47%,
thấp hơn from-scratch 60,97% trong lần chạy này. Kết quả cho thấy mức độ mở các
tầng để tinh chỉnh có ảnh hưởng rõ đến khả năng thích nghi trên CIFAR-100.

Trong nhóm ViT, partial đạt kết quả tốt nhất với accuracy 83,25% và macro F1 0,8324;
head-only đạt 81,32%, full fine-tune đạt 78,64%, còn from-scratch đạt 35,30%.
ViT partial hơn head-only 1,93 điểm phần trăm. Các cấu hình pretrained ViT đều cao
hơn ViT from-scratch; không thể từ các lượt này kết luận full fine-tune luôn tốt nhất.

ViT pretrained cao hơn ResNet18 ở cùng chế độ: head-only hơn 20,85 điểm phần trăm,
partial hơn 10,98 điểm phần trăm, full hơn 1,80 điểm phần trăm. Ngược lại, ở chế độ
from-scratch, ResNet18 hơn ViT 25,67 điểm phần trăm. So sánh scratch cần xét đồng thời
bất thường NaN ở ViT và giới hạn cấu hình được thử.

## 10. Giới hạn và bất thường

ViT scratch phát sinh train/validation loss NaN từ epoch 11; ViT full từ epoch 37.
Trạng thái tốt nhất tương ứng ở epoch 10 và 3, trước khi loss NaN xuất hiện. Mã nạp
lại trạng thái tốt nhất trước đánh giá test; không dùng trạng thái cuối đã lỗi làm
căn cứ giải thích số liệu test. Chưa đủ dữ liệu để xác định nguyên nhân NaN.

Người học cho biết chạy các thực nghiệm song song. Thời gian quan sát của ViT lớn
hơn ResNet18 trong các lượt này, nhưng chưa kiểm soát thiết bị và phân bổ tài nguyên;
không dùng trực tiếp các giá trị này làm bằng chứng về tỷ lệ tăng tốc giữa hai kiến trúc.
Chưa có kết quả nhiều seed hoặc kiểm định thống kê. Phân tích hiện tại chưa thay thế
báo cáo PDF A1 cuối cùng và chưa phân tích chi tiết từng lớp từ ma trận nhầm lẫn.
