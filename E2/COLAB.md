# Chạy E2 trên Colab

## Chuẩn bị

Chọn runtime GPU, clone hoặc cập nhật repo chứa phiên bản E2 mới:

```python
!git clone https://github.com/baond12/Deep_Learning_Course_HK261.git
%cd Deep_Learning_Course_HK261
!pip install -r requirements.txt
```

Nếu đã clone, chuyển vào repo và `!git pull` sau khi bảo toàn mọi kết quả local.

## Chạy đủ tám thực nghiệm

```python
!python E2/train.py --experiments all --epochs 100 --batch-size 256 --device cuda --progress --output-dir E2/results/tokenization_msa_v2
```

Lệnh tự kiểm tra MSA trên CPU float64 và GPU float32 cho cả bốn tokenizer, trước
khi tải dữ liệu/train. Mọi kiểm tra phải PASS. Không cần lệnh `--phase` hoặc
`--download`: đây là luồng E2, tự tải MNIST nếu thiếu, train/validation rồi test
best checkpoint của từng mô hình.

Chỉ muốn kiểm tra, không tải MNIST và không train:

```python
!python E2/train.py --experiments all --verify-only --device cuda --output-dir E2/results/tokenization_msa_v2
```

Có thể dùng chính thư mục chỉ chứa verification để train tiếp; chương trình
luôn kiểm tra lại trước khi train. Nếu output đã có kết quả, chọn tên khác thay
vì ghi đè. Không có resume lượt dở dang.

## Lưu lâu dài trên Drive

Colab có thể mất dữ liệu khi hết phiên. Mount Drive và truyền output trực tiếp:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
!python E2/train.py --experiments all --epochs 100 --batch-size 256 --device cuda --progress --output-dir /content/drive/MyDrive/E2_tokenization_msa_v2
```

Output bao gồm cả checkpoint. Không dùng nhiều runtime cùng ghi vào thư mục
Drive này. Tránh chạy chung GPU với tác vụ khác nếu cần so sánh thời gian.

## Xem kết quả

Đọc `verification.json` (tất cả PASS), `metrics.json` (`status=completed`, đủ 8
mô hình), `summary.csv`, `comparison_msa.csv`, `comparison_tokenization.csv`,
`comparison.md`, `comparison.png`, learning curves, confusion matrix và mẫu lỗi.

Số token ảnh mặc định: patch 16, hàng 28, CNN-stem 49, patch overlap 64; mỗi chuỗi
có thêm một CLS. Khớp số học không yêu cầu accuracy sau train giống nhau vì
dropout và kernel có thể có ngẫu nhiên/làm tròn khác nhau.

## Mang về repo local

Tải nguyên thư mục kết quả về và đặt tại `E2/results/tokenization_msa_v2/`.
Giữ nguyên kết quả E2 cũ ở thư mục cha, không copy đè các file lịch sử.
Giữ checkpoint riêng; `.gitignore` không đưa chúng lên Git. Cập nhật `E2/report.md`
và báo cáo Word/PDF bằng số liệu mới sau khi review; commit/push do người dùng tự làm.

Nếu chạy nhiều training seed, giữ split: `--seed 42 --train-seed 43` (sau đó 44)
và dùng mỗi seed một thư mục output khác nhau. Không chọn seed tốt nhất theo test.
