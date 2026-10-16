# Chạy E3 trên Google Colab

## 1. Clone repo

```bash
!git clone https://github.com/baond12/Deep_Learning_Course_HK261.git
%cd Deep_Learning_Course_HK261
```

## 2. Cài dependencies

```bash
!pip install -r requirements.txt
```

## 3. Smoke test shape/logic

```bash
!python E3/smoke_test.py
```

## 4. Train ảnh MNIST dạng chuỗi

```bash
!python E3/train.py --experiments all --epochs 100 --batch-size 256 --device cuda --progress --probe-order --output-dir E3/results/sequence_v2
```

Bộ `sequence_v2` đã có kết quả trong repo; khi chạy lại sau khi clone, đổi
`--output-dir` sang thư mục mới (ví dụ `E3/results/sequence_v2_rerun`). Không xóa
kết quả đã đối chiếu để chạy đè.

Nếu không có GPU, kiểm tra offline bằng dữ liệu giả:

```bash
!python E3/smoke_test.py
```

## 5. Train sentiment analysis toy pipeline

```bash
!python E3/train_sentiment.py --rnn-type lstm --epochs 5 --device cuda
!python E3/train_sentiment.py --rnn-type gru --epochs 5 --device cuda
```

## 6. Kết quả

Kết quả ảnh nằm ở:

```text
E3/results/sequence_v2/
```

Các file chính:

- `summary.csv`
- `comparison_rnn.csv`, `comparison_baselines.csv`
- `comparison_order_probe.csv` (vì lệnh trên bật `--probe-order`)
- `metrics.json`
- `learning_curves_<tokenizer>.png`, `learning_curves_baselines_e1.png`
- `baselines_e1.json` (tái sử dụng đúng số liệu baseline E1 đã có)
- `checkpoints/best_*.pt` (giữ riêng, không đưa lên Git)
- `predictions_*.csv`, `confusion_matrix_*.csv`
- `confusion_matrix_*.png`
- `misclassified_*.png`

Chỉ train 8 cấu hình: LSTM/GRU × hàng, patch, CNN-stem, patch chồng lấn.
MLP/CNN lấy kết quả E1 ở `E1/results/epochs100_batch256_seed42/`, không chạy lại;
bảng tổng hợp có 10 hàng và ghi rõ nguồn E1/E3. Split seed 42 cố định; dùng
`--train-seed` khác và output mới để chạy lặp. Output đã có kết quả bị từ chối
ghi đè; không hỗ trợ resume. Khi chạy xong, tải cả thư mục output và checkpoint.

Kết quả sentiment nằm ở:

```text
E3/results/sentiment/
```

