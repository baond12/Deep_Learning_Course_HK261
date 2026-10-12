# Chạy A1 trên Google Colab

## 1. Clone repo

```bash
!git clone https://github.com/baond12/Deep_Learning_Course_HK261.git
%cd Deep_Learning_Course_HK261
```

## 2. Cài dependencies

```bash
!pip install -r requirements.txt
```

## 3. Smoke test model/shape

```bash
!python A1/smoke_test.py
```

## 4. Chuẩn bị dataset một lần

Mount Google Drive nếu muốn giữ dataset qua nhiều phiên Colab:

```bash
from google.colab import drive
drive.mount('/content/drive')
```

Chọn một path dataset theo ý bạn. Ví dụ:

```bash
DATA_DIR = "/content/drive/MyDrive/Deep_Learning_Datasets"
```

Tải CIFAR-100 đúng một lần:

```bash
!python A1/prepare_data.py --data-dir "$DATA_DIR"
```

## 5. Train các thí nghiệm A1

Protocol mặc định dùng full CIFAR-100 gồm 100 lớp:

- Train: 400 ảnh/lớp, tổng 40,000 ảnh.
- Validation: 100 ảnh/lớp, tổng 10,000 ảnh.
- Test: 100 ảnh/lớp, tổng 10,000 ảnh.
- Ảnh được resize về `224 x 224`.

Các lần train sau chỉ đọc dataset từ `--data-dir`, không tự download lại:

```bash
!python A1/train.py --data-dir "$DATA_DIR" --experiments all --epochs 5 --batch-size 32 --device cuda --progress
```

Nếu muốn lưu checkpoint trong phiên Colab nhưng không push model lên repo:

```bash
!python A1/train.py --data-dir "$DATA_DIR" --experiments all --epochs 5 --batch-size 32 --device cuda --progress --save-checkpoints
```

Checkpoint sẽ nằm trong `A1/checkpoints/`. Các file model `.pt/.pth/.ckpt/.safetensors`
đã được `.gitignore` chặn.

Nếu muốn chạy thử nhanh một vài thí nghiệm:

```bash
!python A1/train.py --data-dir "$DATA_DIR" --experiments resnet18_scratch resnet18_pretrained_head vit_b16_pretrained_head --epochs 1 --batch-size 32 --device cuda --progress
```

## 6. Tạo phân tích A1.2

Sau khi train xong:

```bash
!python A1/analyze_results.py --results-dir A1/results
```

Script này tạo `analysis.md` và các bar chart để đưa vào báo cáo A1.2.

## 7. Kết quả

Kết quả nằm ở:

```text
A1/results/
```

Các file chính:

- `summary.csv`
- `metrics.json`
- `analysis.md`
- `learning_curves.png`
- `bar_test_accuracy.png`
- `bar_macro_f1.png`
- `bar_trainable_parameters.png`
- `bar_training_time_seconds.png`
- `confusion_matrix_*.png`
- `misclassified_*.png`
