# Assignment 1 - CNN vs Transformer

Mục tiêu: so sánh CNN và Transformer trên **full CIFAR-100, 100 classes** với các chế độ
from-scratch, pretrained freeze backbone/head-only, pretrained partial unfreeze và full
fine-tune.

## Bước hiện tại trước khi khóa A1.1

Chạy thăm dò **đủ tám cấu hình, một vòng, batch 16, trên P100**, chỉ huấn luyện/xác
thực và chưa đánh giá test. Dùng [hướng dẫn chạy và gửi kết quả](PILOT.md).

```bash
CUDA_VISIBLE_DEVICES=1 python -u A1/pilot.py --experiments all --epochs 1 --batch-size 16 --seed 42 --train-seed 42 --data-dir data --device cuda --progress --output-dir A1/results/pilot_epoch1_batch16_seed42
```

Thư mục đầu ra phải mới. Gửi lại toàn bộ thư mục hoặc tệp nén, gồm chỉ số, lịch sử,
thời gian từng pha, bộ nhớ GPU, cấu hình và log. Từ số đo đó mới chốt ngân sách và
quy trình chính thức. Những kết quả đang có tại `A1/results/` là bộ thăm dò trước;
không ghi đè. [Đề xuất A1.1](protocol-a1.1-draft.md) vẫn là bản dự thảo.

## Cấu trúc

- `prepare_data.py`: tải CIFAR-100 một lần vào `--data-dir`.
- `data.py`: full CIFAR-100, split cân bằng train/validation/test theo class, augmentation.
- `models.py`: ResNet18 và ViT-B/16 với các chế độ freeze/fine-tune.
- `train.py`: training loop A1, lưu metric và hình phân tích.
- `pilot.py`: chạy thăm dò tuần tự, không test, ghi môi trường/thời gian/bộ nhớ và lỗi.
- `engine.py`: vòng lặp A1, kiểm soát phần đóng băng, dừng khi loss/gradient không hữu hạn.
- `test_pilot.py`: kiểm tra đường chạy thăm dò, lưu kết quả và xử lý lỗi bằng dữ liệu tổng hợp.
- `analyze_results.py`: tổng hợp `summary.csv`, tạo bar charts và `analysis.md`.
- `smoke_test.py`: kiểm tra model shape/trainable params, không tải weights/dataset.
- `report.md`: bản nháp A1.1/A1.2.
- `COLAB.md`: hướng dẫn chạy trên Google Colab.
- `checkpoints/`: nơi lưu checkpoint tùy chọn; model files bị Git ignore.
- `results/`: thư mục sinh kết quả sau khi train.

## Dataset và protocol

Mặc định dùng đủ 100 lớp của CIFAR-100:

- Train: 400 ảnh/lớp, tổng 40,000 ảnh.
- Validation: 100 ảnh/lớp, tổng 10,000 ảnh.
- Test: 100 ảnh/lớp, tổng 10,000 ảnh.
- Resize ảnh về `224 x 224`.
- Split cân bằng theo class và tái lập bằng seed `42`.

Dataset không được push lên GitHub. Tải CIFAR-100 một lần vào local/Google Drive,
sau đó các lần train chỉ truyền lại cùng `--data-dir`.

## Smoke test

```bash
python A1/smoke_test.py
```

## Train

Tải dataset một lần:

```bash
python A1/prepare_data.py --data-dir data
```

Sau đó train bằng đúng path đó:

```bash
python A1/train.py --data-dir data --experiments all --epochs 5 --batch-size 32 --device auto --output-dir A1/results/example_new_run
```

Trên Colab GPU:

```bash
python A1/train.py --data-dir /content/drive/MyDrive/Deep_Learning_Datasets --experiments all --epochs 5 --batch-size 32 --device cuda --progress --output-dir A1/results/example_new_run
```

Các tham số `--num-classes`, `--train-per-class`, `--val-per-class`, `--test-per-class`
vẫn có thể dùng để chạy debug nhanh, nhưng protocol A1.1 chính thức là 100 classes,
400/100/100 samples per class. Số vòng, batch và chế độ học chính thức chưa được
khóa; các lệnh `train.py` ở mục này là ví dụ, có đánh giá test và không dùng thay
lệnh thăm dò ở đầu tài liệu.

Nếu muốn lưu checkpoint trên Colab/local nhưng không đẩy model lên GitHub:

```bash
python A1/train.py --data-dir /content/drive/MyDrive/Deep_Learning_Datasets --experiments all --epochs 5 --batch-size 32 --device cuda --progress --save-checkpoints --output-dir A1/results/example_new_run --checkpoint-dir A1/checkpoints/example_new_run
```

Các file `.pt`, `.pth`, `.ckpt`, `.safetensors` trong `A1/checkpoints/` đã được ignore.

## Phân tích A1.2

Đã nhập lịch sử và hình kết quả của đủ 8 cấu hình vào `A1/results/`.
`summary.csv` và `metrics.json` đã đủ kết quả cuối của 8 cấu hình:
2 cấu hình từ JSON/CSV gốc và 6 cấu hình khôi phục từ log người học cung cấp.
Best epoch và validation accuracy đã đối chiếu với history. Nguồn và mức chính xác
được ghi theo từng cấu hình; thông số chạy của 6 log bổ sung còn cần xác nhận riêng.
Xem `A1/results/README.md` và `A1/results/results_inventory.csv` trước khi
phân tích. Hai lịch sử ViT có loss NaN được giữ nguyên và ghi rõ trong danh mục.

Sau khi train xong:

```bash
python A1/analyze_results.py --results-dir A1/results
```

Script sinh:

- `A1/results/analysis.md`
- `A1/results/bar_test_accuracy.png`
- `A1/results/bar_macro_f1.png`
- `A1/results/bar_trainable_parameters.png`
- `A1/results/bar_training_time_seconds.png`

## Experiments

- `resnet18_scratch`
- `resnet18_pretrained_head`
- `resnet18_pretrained_partial`
- `resnet18_pretrained_full`
- `vit_b16_scratch`
- `vit_b16_pretrained_head`
- `vit_b16_pretrained_partial`
- `vit_b16_pretrained_full`

## Kết quả đầu ra

- `A1/results/summary.csv`
- `A1/results/metrics.json`
- `A1/results/analysis.md`
- `A1/results/learning_curves.png`
- `A1/results/bar_*.png`
- `A1/results/confusion_matrix_*.png`
- `A1/results/misclassified_*.png`
