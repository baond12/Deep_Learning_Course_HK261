# Assignment 1 - CNN vs Transformer

Mục tiêu: so sánh CNN và Transformer trên **full CIFAR-100, 100 classes** với các chế độ
from-scratch, pretrained freeze backbone/head-only, pretrained partial unfreeze và full
fine-tune.

## Quy trình chính thức đã được duyệt

Đã kết thúc thăm dò trên P100. Người học duyệt **30 epoch, batch 64, FP32**, seed
chia/huấn luyện **42**. ViT full dùng LR **1e-5**; bảy cấu hình khác dùng **1e-4**.
Giữ Adam, weight decay 1e-4 và tiền xử lý từ pilot. **Lượt chính đã hoàn tất ngày
19/10/2026:** tám cấu hình, 240 epoch, 80.000 dòng dự đoán test đã đối chiếu.
Xem [phân tích kết quả](results/epochs30_batch64_seed42_v1/analysis.md),
[bảng tổng hợp](results/epochs30_batch64_seed42_v1/report_summary.csv),
[kiểm chứng](results/epochs30_batch64_seed42_v1/report_checks.json).
ViT full đạt test accuracy 86,46%, macro F1 0,8651. [Word v02](../reports/assignment1.docx)
đã chuẩn hóa cục bộ từ bản người học cung cấp. PDF trong `reports/` vẫn là v01,
đang chờ người học xuất lại từ Word đã sửa; chưa công bố hai bản đã đồng bộ.

Đọc [A1.1 đã chốt](protocol-a1.1.md), [JSON cấu hình](protocol-a1.1.json),
[hướng dẫn chạy và gửi kết quả](RUN_PROTOCOL.md), [phân tích lựa chọn](../docs/a1-pilot-analysis.md).

```bash
CUDA_VISIBLE_DEVICES=1 python -u A1/run_protocol.py --phase train --experiments all --data-dir /workspace/cifar100 --device cuda --progress --output-dir A1/results/epochs30_batch64_seed42_v1
```

Sau khi train hoàn tất, chạy lệnh trên với `--phase test` để đánh giá checkpoint
tốt nhất theo validation. Xem hướng dẫn trước khi chạy; không dùng test để tuning.
Thư mục đầu ra phải mới. Giữ checkpoint trên server; tệp mô hình bị Git ignore.
Kết quả cũ tại `A1/results/` và báo cáo cũ là thăm dò, không ghi đè hoặc gộp vào
bảng chính. [Dự thảo cũ](protocol-a1.1-draft.md) và [hướng dẫn pilot](PILOT.md)
được giữ để truy vết.

## Cấu trúc

- `prepare_data.py`: tải CIFAR-100 một lần vào `--data-dir`.
- `data.py`: full CIFAR-100, split cân bằng train/validation/test theo class, augmentation.
- `models.py`: ResNet18 và ViT-B/16 với các chế độ freeze/fine-tune.
- `train.py`: training loop A1, lưu metric và hình phân tích.
- `run_protocol.py`: chạy quy trình đã duyệt, lưu checkpoint ngay khi validation cải thiện, tách test và kiểm tra nguồn gốc.
- `protocol-a1.1.json`: 30 epoch/batch 64 và LR từng cấu hình; không cần nhập lại trên CLI.
- `test_protocol.py`: kiểm tra tám cấu hình, chọn checkpoint, chống ghi đè, kiểm tra test và lỗi.
- `build_report_bundle.py`: đối chiếu tệp kết quả/80.000 dự đoán/checkpoint, tạo phân tích và gói 19 tệp một cấp cho ChatGPT Plus; không train hoặc suy luận lại.
- `pilot.py`: chạy thăm dò tuần tự, không test, ghi môi trường/thời gian/bộ nhớ và lỗi.
- `engine.py`: vòng lặp A1, kiểm soát phần đóng băng, dừng khi loss/gradient không hữu hạn.
- `test_pilot.py`: kiểm tra đường chạy thăm dò, lưu kết quả và xử lý lỗi bằng dữ liệu tổng hợp.
- `analyze_results.py`: tổng hợp `summary.csv`, tạo bar charts và `analysis.md`.
- `smoke_test.py`: kiểm tra model shape/trainable params, không tải weights/dataset.
- `report.md`: bản nháp A1.1/A1.2.
- `COLAB.md`: hướng dẫn chạy trên Google Colab.
- `checkpoints/`: nơi lưu checkpoint tùy chọn; model files bị Git ignore.
- `results/`: thư mục sinh kết quả sau khi train.

## Tổng hợp báo cáo từ lượt chính đã chạy

```bash
python A1/build_report_bundle.py --source /path/to/epochs30_batch64_seed42_v1
```

Công cụ này dùng matplotlib, NumPy, Pillow và reportlab để biên soạn hình/PDF;
đây là phụ thuộc hậu xử lý, không đổi `requirements.txt` hoặc môi trường đã dùng
để huấn luyện. Bản hiện tại dùng font Times New Roman của Windows. Thư mục nguồn
cần có checkpoint để kiểm tra SHA-256, nhưng bản sao trong repo/gói ChatGPT Plus
không chứa checkpoint. Giữ nguyên tệp gốc, chỉ sinh thêm các tệp `report_*`, hình
tổng hợp và phân tích. Gói nằm tại `A1_ChatGPT_Plus_30epoch/`, đủ 19 tệp một cấp,
được Git ignore vì là gói làm việc để đính kèm, không phải báo cáo đã nộp.

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

## Lệnh huấn luyện cũ — chỉ để tham khảo

**Không dùng các lệnh bên dưới thay cho lượt chính.** Lượt chính dùng
`run_protocol.py` và [RUN_PROTOCOL.md](RUN_PROTOCOL.md); `train.py` cũ đánh giá
test ngay sau từng mô hình và không ghi đầy đủ manifest như chương trình mới.

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
vẫn có thể dùng để chạy debug nhanh bằng công cụ cũ. Protocol chính thức là 100
lớp, 400/100/100 ảnh mỗi lớp, 30 epoch và batch 64 trong JSON cấu hình mới.

Nếu muốn lưu checkpoint trên Colab/local nhưng không đẩy model lên GitHub:

```bash
python A1/train.py --data-dir /content/drive/MyDrive/Deep_Learning_Datasets --experiments all --epochs 5 --batch-size 32 --device cuda --progress --save-checkpoints --output-dir A1/results/example_new_run --checkpoint-dir A1/checkpoints/example_new_run
```

Các file `.pt`, `.pth`, `.ckpt`, `.safetensors` trong `A1/checkpoints/` đã được ignore.

## Phân tích bộ kết quả cũ

Phần này mô tả công cụ và số liệu trước khi chốt protocol mới. Với lượt chính,
dùng `test/test_summary.csv`, `test/comparison.md` và `test/test_metrics.json`
do `run_protocol.py --phase test` sinh ra; không chạy bộ phân tích cũ trên bảng
train/validation mới.

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
