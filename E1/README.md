# E1 - Softmax, MLP, CNN

Nguồn chính của báo cáo E1 là `results/epochs100_batch256_seed42/`:
100 epoch, batch 256, seed 42. MLP đạt 98,63%, CNN đạt 99,27%; E3 tái sử dụng
nguyên hai kết quả này. Xem [quy tắc báo cáo chung](../docs/e1-e3-report-protocol.md).
Nghiên cứu tuning Softmax bên dưới là luồng bổ sung riêng, không tự thay nguồn này.

## Thực nghiệm Softmax có cấu hình

Luồng mới độc lập với `train.py` và các kết quả E1 đã có. Kiến trúc vẫn là
`SoftmaxRegression` (7.850 tham số), Adam, learning rate cố định, không scheduler
và không early stopping. Đây là thí nghiệm bổ sung cho E1 trên MNIST.

- `configs/softmax_experiments.json`: kế hoạch và các mức siêu tham số.
- `softmax_data.py`: chia dữ liệu cố định, augmentation chỉ trên train.
- `run_softmax_experiments.py`: lập kế hoạch, train, khóa lựa chọn và test.
- `analyze_softmax_experiments.py`: tổng hợp validation, biểu đồ và so sánh baseline.
- `smoke_test_softmax_experiments.py`: kiểm tra offline bằng dữ liệu giả trên CPU.

### Thiết kế so sánh

Mỗi lần chỉ thay đổi một yếu tố (OFAT), các yếu tố còn lại giữ bằng baseline.
Chưa khảo sát tổ hợp hay tương tác giữa các siêu tham số; kết quả bước này giúp
quyết định phạm vi nghiên cứu tiếp theo.

| Yếu tố | Các mức |
|---|---|
| Data augmentation | Không / Có (xoay ngẫu nhiên ±10°, nội suy bilinear, nền 0) |
| Epochs | 10, 20, 40, 80, 150, 200 |
| Batch size | 2, 4, 8, 16, 32, 64, 128, 256 |
| Learning rate | 0.001, 0.002, 0.005, 0.01 |
| Weight decay | 0.0001, 0.0002, 0.0005, 0.001 |

Baseline S00: không augmentation, 20 epochs, batch 16, lr 0.001, weight decay
0.0001. Có 20 cấu hình phân biệt × 3 seed huấn luyện (42, 43, 44) = 60 kết quả.
Train/validation luôn là 54.000/6.000 ảnh, dùng đúng thuật toán `random_split`
và split seed 42 của `common/data.py`; test là 10.000 ảnh MNIST chính thức.
Seed huấn luyện và batch size không làm đổi thành viên train/validation.
Các lượt vẫn có thể có thứ tự batch khác nhau; seed của DataLoader được lưu riêng.

Với `reuse_epoch_prefixes=true`, sáu mốc epoch được lấy từ cùng một lượt chạy
200 epochs cho mỗi seed: tổng cộng **45 lượt huấn luyện thực tế**. Best checkpoint
tại mốc N chỉ xét epochs 1…N; nếu validation accuracy bằng nhau, chọn epoch sớm
nhất, giống luồng cũ. Đồng thời lưu checkpoint và validation tại epoch cuối để
phân biệt hiệu quả tăng ngân sách với hiệu quả chọn checkpoint. Các mốc này
không phải các lần lặp độc lập. Có thể đổi `reuse_epoch_prefixes=false` trong một
study mới để chạy riêng mọi mốc (60 lượt).

Baseline được chạy lại trong cùng pipeline, không trộn số liệu cũ vào trung bình.
Không bảo đảm số liệu mới giống tuyệt đối baseline lịch sử do cách quản lý RNG
và môi trường có thể khác. Test đã được xem trong báo cáo trước; không coi nó là
tập kiểm định mới hoàn toàn cho toàn bộ quá trình nghiên cứu.

### Lệnh chạy từ thư mục gốc repo

```bash
# Kiểm tra code trước khi chạy nghiên cứu; không tải MNIST.
python E1/smoke_test_softmax_experiments.py

# Chỉ in kế hoạch, không train và không tạo kết quả.
python E1/run_softmax_experiments.py --phase plan

# Chạy nghiên cứu; --download cho phép tải MNIST nếu chưa có.
python E1/run_softmax_experiments.py --phase train --device cuda --download --progress

# Tổng hợp sau khi các tiến trình train đã dừng.
python E1/analyze_softmax_experiments.py --study-dir E1/results/experiments/softmax/softmax_ofat_v1
```

Có thể phân chia lượt chạy bằng `--seeds 42` hoặc `--run-ids <ID>` lấy từ kế hoạch.
Một run ID là một lượt huấn luyện, có thể trả về nhiều mốc epoch. Chọn seed 42
vẫn chạy mọi nhóm siêu tham số của seed đó. Mỗi lượt ghi thư mục riêng; các tiến
trình dùng chung ổ đĩa có khóa ngắn để bảo vệ manifest và bước chốt lựa chọn.
Chỉ chạy analyzer khi các lượt train đã kết thúc để tránh bản tổng hợp dở dang.
Không dùng nhiều runtime với bản sao thư mục kết quả độc lập rồi ghi đè lên nhau.

Analyzer lưu `analysis/summary.csv`, `analysis/metrics.json`, `analysis/analysis.md`
và `analysis/comparison.png`, cùng learning curves từng lượt. Có mean và độ lệch chuẩn mẫu theo seed, best/last val,
train sạch tại best checkpoint, số bước optimizer, thời gian và chênh lệch ghép
cặp với baseline. Chỉ cấu hình đủ cả 3 seed mới được xếp hạng. Lượt failed/running
bị loại toàn bộ. So sánh epochs cố định không đồng nghĩa ngân sách tính toán bằng
nhau giữa các batch size; dùng số bước và thời gian để diễn giải.

Sau khi đọc validation, chọn trước baseline và cấu hình cần đánh giá cuối cùng:

```bash
# Sxx là ID đã chọn từ plan/analysis; thay bằng ID thực tế.
python E1/run_softmax_experiments.py --phase lock --select S00 Sxx
python E1/run_softmax_experiments.py --phase test --device cuda --download
```

`train` không tạo test loader. `test` chỉ chạy sau khi có `selection.json`, dùng
best checkpoint đã khóa của mọi seed, kiểm tra hash trước khi đọc test. Không
chọn seed tốt nhất để báo cáo và không dùng test để điều chỉnh/xếp hạng cấu hình.
Kết quả test từng seed gồm loss, accuracy, macro F1 (10 lớp), dự đoán và confusion
matrix dạng CSV trong `final_test/`, kèm summary và mean/std trên mọi seed cho
các cấu hình đã khóa. Khóa lựa chọn xong thì study không nhận thêm
training. Nếu cần nghiên cứu tiếp, tạo study mới và ghi rõ test đã được xem.

### Nơi lưu và quy tắc bảo toàn kết quả

```text
E1/results/experiments/softmax/<study_id>/manifest.json
E1/results/experiments/softmax/<study_id>/<run_id>/history.csv
E1/results/experiments/softmax/<study_id>/<run_id>/metrics.json
E1/results/experiments/softmax/<study_id>/analysis/...
E1/results/experiments/softmax/<study_id>/selection.json
E1/results/experiments/softmax/<study_id>/final_test/<selection_hash>/...
E1/checkpoints/softmax_experiments/<study_id>/<run_id>/best_e<N>.pt
E1/checkpoints/softmax_experiments/<study_id>/<run_id>/last_e<N>.pt
```

Không ghi vào `E1/results/summary.csv` hoặc các kết quả lịch sử. Config, hash mã
nguồn, hash ảnh/nhãn, hash split và môi trường được lưu để truy vết. Để gộp số
liệu, phải cùng dữ liệu/split và môi trường (device, phiên bản, số worker, threads).
Chạy trên môi trường cố định; deterministic mode có thể báo lỗi nếu thiết bị
không hỗ trợ thao tác cần thiết. Không hứa tái lập bit-for-bit giữa CPU/GPU hay
giữa các phiên bản PyTorch.

Không ghi đè run đã tồn tại. `--skip-completed` chỉ bỏ qua lượt hoàn tất đã xác
minh checkpoint, không resume lượt dở dang. Nếu lượt thất bại hoặc cần sửa config,
source, môi trường: giữ lại bằng chứng và tạo `study_id` mới. Với khóa thư mục
`.study.lock` còn lại sau khi tiến trình bị kill, kiểm tra không còn tiến trình
đang chạy trước khi xóa khóa đó; không xóa một khóa đang được dùng.

Checkpoint bị `.gitignore` loại khỏi Git: phải giữ trên máy/Drive và mang theo
khi test. Chỉ kết quả nhẹ và code được commit; không dùng `git add .` để tránh
đưa dữ liệu hay tài liệu không liên quan vào commit.

Mục tiêu: xây dựng và so sánh ba bộ phân loại ảnh nhỏ gồm softmax classifier, MLP và
CNN trên MNIST.

## Cấu trúc

- `models.py`: định nghĩa SoftmaxRegression, MLPClassifier và SmallCNN.
- `train.py`: training loop, đánh giá, lưu bảng metric và hình phân tích.
- `report.md`: phân tích tiếng Việt cho bộ kết quả E1 dùng trong báo cáo chung.
- `results/`: thư mục sinh ra sau khi chạy training.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy local hoặc Colab

```bash
python E1/train.py --models all --epochs 100 --batch-size 256 --device auto --output-dir E1/results/epochs100_batch256_seed42_rerun
```

Nếu chạy Google Colab có GPU:

```bash
python E1/train.py --models all --epochs 100 --batch-size 256 --device cuda --output-dir E1/results/epochs100_batch256_seed42_rerun
```

## Kết quả đầu ra

Các lệnh trên chỉ dùng khi cần chạy lại và lưu vào output mới; không ghi đè bộ
`E1/results/epochs100_batch256_seed42/` đang làm nguồn baseline/report. Nếu thư mục
`_rerun` đã có kết quả, chọn tên thư mục mới. Tham số `--output-dir` xác định nơi lưu:

- `summary.csv`
- `metrics.json`
- `learning_curves.png`
- `history_softmax.csv`, `history_mlp.csv`, `history_cnn.csv`
- `confusion_matrix_*.png`
- `misclassified_*.png`

Sau khi có kết quả, cập nhật bảng và nhận xét vào `E1/report.md`, rồi gom vào
`reports/exercises.pdf` khi làm báo cáo chung E1-E3.

