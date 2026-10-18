# Chạy A1 theo quy trình đã duyệt

Mã cấu hình: `a1_cifar100_p100_v1`. Đọc [protocol A1.1](protocol-a1.1.md) và [JSON cấu hình](protocol-a1.1.json). Lệnh tự lấy 30 epoch, batch 64, seed 42 và LR riêng từng cấu hình từ JSON; không cần nhập lại từng tham số trên terminal.

## 1. Huấn luyện trên server

Chạy tại thư mục gốc repo, sau khi người học đã push và server đã nhận mã mới. Giữ môi trường CUDA/PyTorch đã chạy pilot thành công. CIFAR-100 đã nằm ở `/workspace/cifar100`, không cần tải lại.

```bash
CUDA_VISIBLE_DEVICES=1 python -u A1/run_protocol.py \
  --phase train --experiments all \
  --data-dir /workspace/cifar100 --device cuda --num-workers 2 --progress \
  --output-dir A1/results/epochs30_batch64_seed42_v1
```

`--phase train` chỉ huấn luyện và xác thực, **không nạp test**. Chọn checkpoint bằng validation; chưa tính test trong lúc lựa chọn. Mọi cấu hình được đặt lại seed và loader trước khi chạy. Chương trình tự áp dụng LR `1e-5` cho ViT full, `1e-4` cho bảy cấu hình khác, Adam/WD `1e-4`, FP32.

Thư mục đầu ra phải mới; không ghi đè kết quả cũ. Khoảng 21 giờ là ước tính cho đủ tám cấu hình, chưa tính thêm I/O checkpoint, đánh giá test hoặc gián đoạn. Checkpoint tốt nhất được ghi ngay khi validation cải thiện; nếu hòa giữ lần đầu.

**Giới hạn khi bị ngắt:** đã lưu checkpoint tốt nhất và lịch sử từng epoch, nhưng chưa lưu optimizer/RNG để nối tiếp epoch. Không có cờ resume; không dùng checkpoint tốt nhất để giả lập đã tiếp tục đúng một lượt. Nếu có cấu hình lỗi, giữ nguyên thư mục để chẩn đoán trước khi quyết định chạy lại; các cấu hình còn lại vẫn có thể hoàn tất. Khi bấm Ctrl+C, chương trình dừng và giữ kết quả đã ghi.

## 2. Đánh giá test

Chỉ chạy sau khi đủ các cấu hình trong lượt có trạng thái `completed` và công thức đã được chốt. Lệnh sau dùng đúng checkpoint đã chọn bằng validation:

```bash
CUDA_VISIBLE_DEVICES=1 python -u A1/run_protocol.py \
  --phase test --experiments all \
  --data-dir /workspace/cifar100 --device cuda --num-workers 2 --progress \
  --output-dir A1/results/epochs30_batch64_seed42_v1
```

`--phase test` không huấn luyện lại. Không tải lại trọng số ImageNet: dựng kiến trúc rồi nạp toàn bộ trạng thái checkpoint, kiểm tra nguồn gốc và SHA-256. Giữ nguyên mã nguồn từ train tới test; không `git pull` thay đổi các tệp mã đang được kiểm tra giữa hai bước. Nếu có thay đổi thật sự cần thiết, phải xem xét và ghi nhận trước khi đánh giá, không sửa tay hash để vượt kiểm tra.

Chương trình suy luận test một lần mỗi cấu hình; cùng dự đoán đó dùng cho loss, accuracy, macro F1, confusion matrix và mẫu sai. Không dùng kết quả test để chọn lại hyperparameter. Mặc định test lưu vào thư mục mới `test/` bên trong lượt train, không sửa `metrics.json` train hoặc ghi đè test đã có. Nếu cần chạy lại vì lỗi xuất tệp, giữ kết quả cũ và dùng `--test-output-dir` mới, nêu rõ lý do; không xem nhiều lần test là cơ sở tuning.

## 3. Gửi kết quả và giữ checkpoint

Gửi toàn bộ thư mục `A1/results/epochs30_batch64_seed42_v1`, có thể nén **trừ checkpoint**:

```bash
tar --exclude='epochs30_batch64_seed42_v1/checkpoints' \
  -czf A1_epochs30_batch64_seed42_v1.tar.gz \
  -C A1/results epochs30_batch64_seed42_v1
```

Giữ `checkpoints/` tại server hoặc nơi lưu riêng; không xóa trước khi test và đối chiếu xong. Mô hình lớn bị Git ignore, không cần tải lên GitHub. Tệp dữ liệu CIFAR-100/cache trọng số cũng không đưa vào Git.

| Thành phần | Ý nghĩa |
| --- | --- |
| `protocol.json`, `metrics.json` | Cấu hình đầy đủ, hash nguồn/split, môi trường, trạng thái từng lượt |
| `summary.csv` | Validation tốt nhất và cuối, epoch, LR, tham số, thời gian/bộ nhớ |
| `run_*.json`, `history_*.csv`, `run.log` | Lịch sử, thời gian từng pha và chẩn đoán lỗi |
| `learning_curves_*.png` | Đường cong train/validation |
| `split_indices.json` | Chỉ số train/validation |
| `dataset_summary.json`, `class_counts.csv`, `dataset_samples_train.png` | EDA từ dữ liệu train thật; số test ở bước này là dự kiến, chưa quan sát |
| `checkpoints/*.pt` | Mô hình tốt nhất theo validation, có hash và metadata; không có optimizer để resume |
| `test/test_metrics.json`, `test/test_summary.csv` | Kết quả test liên kết với đúng checkpoint và bản ghi train |
| `test/comparison.md` | Bảng đối chiếu bằng tiếng Việt, chỉ lấy số liệu của lượt hiện tại |
| `test/predictions_*.csv` | Nhãn thật, nhãn dự đoán, độ tin cậy, chỉ số ảnh test gốc |
| `test/confusion_matrix_*.csv/.png` | Ma trận đủ 100 lớp, dạng số và hình |
| `test/classification_report_*.json`, `test/misclassified_*.png` | Precision/recall/F1 từng lớp và mẫu sai nếu có |

## 4. Kiểm tra mã ở máy phát triển

```bash
python -m unittest A1.test_protocol A1.test_pilot -v
python A1/smoke_test.py
```

Kiểm tra dùng dữ liệu tổng hợp và mô hình nhỏ trên CPU, không tải CIFAR-100 hoặc trọng số. Đầu ra được đánh dấu `synthetic_test`; không dùng làm số liệu A1. P100 và hiệu năng thực tế được xác nhận từ các lượt thăm dò người học đã gửi, chưa phải từ một lượt 30 epoch trên máy phát triển.
