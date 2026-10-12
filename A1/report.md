# A1 - CNN vs Transformer trên full CIFAR-100

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

## 8. Kết quả cần điền sau khi train

Sau khi train, kết quả nằm trong `A1/results/`:

- `summary.csv`: bảng so sánh chính.
- `metrics.json`: cấu hình và metric chi tiết.
- `analysis.md`: bảng tổng hợp và gợi ý nhận xét do `A1/analyze_results.py` sinh.
- `learning_curves.png`: loss/accuracy curves.
- `bar_test_accuracy.png`, `bar_macro_f1.png`, `bar_trainable_parameters.png`,
  `bar_training_time_seconds.png`: biểu đồ so sánh A1.2.
- `confusion_matrix_*.png`: confusion matrix từng experiment.
- `misclassified_*.png`: mẫu dự đoán sai.

## 9. A1.2 - So sánh cuối

Sau khi chạy đủ thí nghiệm, dùng:

```bash
python A1/analyze_results.py --results-dir A1/results
```

Phần báo cáo cuối cần bổ sung từ kết quả thật:

1. Bảng `summary.csv`: accuracy, macro F1, thời gian, total/trainable parameters.
2. Biểu đồ bar chart: test accuracy, macro F1, trainable parameters, training time.
3. Learning curves: xem mô hình nào hội tụ nhanh/chậm, có overfit không.
4. Confusion matrix và mẫu sai: chỉ ra các lớp CIFAR-100 dễ nhầm.
5. Thảo luận:
   - ResNet18 vs ViT-B/16 trên cùng split.
   - From-scratch vs pretrained.
   - Freeze backbone/head-only vs partial unfreeze vs full fine-tune.
   - Khi nào freeze có lợi, khi nào full fine-tune đáng chi phí.
   - ViT-B/16 có cần pretrained hơn ResNet18 không.

Checkpoint là tùy chọn. Nếu bật `--save-checkpoints`, model được lưu ở
`A1/checkpoints/`, nhưng các file model đã bị `.gitignore` chặn để không đẩy lên GitHub.

## 10. Nhận xét dự kiến

ResNet18 có inductive bias không gian mạnh nên thường học tốt hơn khi dữ liệu không quá
lớn. ViT-B/16 có attention toàn cục và thường cần pretrained weights hoặc dữ liệu lớn để
phát huy tốt. Freeze-head thường nhanh và ít tốn bộ nhớ nhất nhưng khả năng thích nghi
hạn chế; partial unfreeze cân bằng giữa chi phí và độ thích nghi; full fine-tune có khả
năng đạt kết quả tốt hơn nhưng tốn compute và dễ overfit hơn.
