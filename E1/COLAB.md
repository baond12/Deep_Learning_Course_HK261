# Chạy E1 trên Google Colab

## Thực nghiệm Softmax mới (tách biệt luồng E1 cũ)

Sau khi clone repo và cài dependencies như bên dưới, xem kế hoạch trước:

```python
!python E1/run_softmax_experiments.py --phase plan
```

Nghiên cứu mặc định có 20 cấu hình OFAT, 3 seed, 45 lượt train thực tế do dùng
chung tiền tố cho 6 mốc epoch. Không phải quét toàn bộ tổ hợp. Xem thiết kế và
quy tắc chọn checkpoint trong [README.md](README.md).

Nên lưu cả kết quả lẫn checkpoint trên Google Drive ngay từ đầu để tránh mất
khi Colab hết phiên. Mount Drive bằng một cell Python:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Ví dụ chạy seed 42 (các seed còn lại chạy riêng bằng cùng cấu hình/môi trường):

```python
!python E1/run_softmax_experiments.py --phase train --seeds 42 --device cuda --download --progress --output-root /content/drive/MyDrive/E1_softmax/results --checkpoint-root /content/drive/MyDrive/E1_softmax/checkpoints
```

Chạy tiếp seed 43 và 44 bằng cách đổi `--seeds`. Dùng cùng phiên bản PyTorch,
GPU, số worker và mã nguồn; analyzer sẽ từ chối gộp các môi trường khác nhau.
Giữ nguyên source trong suốt study. Có thể chạy song song các run ID khác nhau
trong cùng một runtime/ổ lưu, nhưng cần kiểm soát tranh chấp GPU và ghi rõ cách
đo thời gian. Không xem thời gian khi dùng chung GPU là benchmark độc lập.
Không nên dùng nhiều phiên Colab ghi cùng study qua các Drive mount khác nhau:
Drive có độ trễ đồng bộ và không bảo đảm khóa liên tiến trình giữa các runtime.

Sau khi train dừng hoàn toàn:

```python
!python E1/analyze_softmax_experiments.py --study-dir /content/drive/MyDrive/E1_softmax/results/softmax_ofat_v1
```

Đọc `analysis/analysis.md` và `analysis/summary.csv`, chọn ID dựa trên validation.
Sau đó thay Sxx trong lệnh dưới bằng ID đã duyệt:

```python
!python E1/run_softmax_experiments.py --phase lock --select S00 Sxx --output-root /content/drive/MyDrive/E1_softmax/results --checkpoint-root /content/drive/MyDrive/E1_softmax/checkpoints
!python E1/run_softmax_experiments.py --phase test --device cuda --download --output-root /content/drive/MyDrive/E1_softmax/results --checkpoint-root /content/drive/MyDrive/E1_softmax/checkpoints
```

Không test trong khi tuning; khóa lựa chọn sẽ cấm train thêm trong study. Test
chỉ đánh giá checkpoint đã chọn của mọi seed, không chọn theo test. Lệnh train
mặc định không mở tập test. Không resume lượt dở dang; `--skip-completed` chỉ bỏ
qua lượt đã hoàn thành và kiểm tra checkpoint còn nguyên. Giữ study dở dang làm
bằng chứng, tạo `study_id` mới trong bản sao JSON nếu cần chạy lại.

Để đưa kết quả về repo local: copy nguyên thư mục `softmax_ofat_v1` từ Drive
results vào `E1/results/experiments/softmax/`; giữ checkpoint ở Drive hoặc copy
vào `E1/checkpoints/softmax_experiments/softmax_ofat_v1/` (không push Git).
Nếu dùng cấu hình khác mặc định, truyền `--config <file.json>` cho mọi phase và
giữ JSON đó cùng mã nguồn. Commit đúng thư mục kết quả của study và config đã
dùng, không ghi đè kết quả E1 cũ. Người dùng tự thực hiện commit/push.

## 1. Clone repo

```bash
!git clone https://github.com/baond12/Deep_Learning_Course_HK261.git
%cd Deep_Learning_Course_HK261
```

## 2. Cài dependencies

Colab thường đã có PyTorch. Nếu thiếu package, chạy:

```bash
!pip install -r requirements.txt
```

## 3. Train 3 mô hình

```bash
!python E1/train.py --models all --epochs 5 --batch-size 128 --device cuda --progress
```

Nếu runtime không có GPU:

```bash
!python E1/train.py --models all --epochs 5 --batch-size 128 --device cpu --progress
```

## 4. Xem kết quả

Kết quả nằm ở:

```text
E1/results/
```

Các file quan trọng:

- `summary.csv`
- `metrics.json`
- `learning_curves.png`
- `confusion_matrix_*.png`
- `misclassified_*.png`

## 5. Commit kết quả từ Colab

```bash
!git config user.email "ngobao.bk.tp@gmail.com"
!git config user.name "baond12"
!git add E1/results E1/report.md
!git commit -m "Add E1 experiment results"
!git push
```

Nếu Colab yêu cầu đăng nhập khi push, cách đơn giản hơn là tải thư mục `E1/results/`
về máy rồi copy vào repo local, sau đó commit/push từ PowerShell.

