# E3 — LSTM và GRU trên bốn cách biểu diễn ảnh

MNIST được chuyển thành chuỗi vị trí không gian để so sánh LSTM/GRU với MLP/CNN
từ E1. `--experiments all` chỉ huấn luyện **8 cấu hình**: bốn cặp LSTM–GRU.
Hai baseline được đọc nguyên kết quả E1 đã có, không huấn luyện hay test lại.
Pipeline sentiment minh họa vẫn chạy riêng, không thuộc kết quả MNIST của E3.

Bộ `results/sequence_v2/` đã hoàn tất 8 cấu hình × 100 epoch, batch 256, seed 42.
LSTM CNN-stem đạt test accuracy 99,37%; GRU CNN-stem đạt 99,27%.
Đã đối chiếu 80.000 dự đoán, ma trận nhầm lẫn, history, snapshot baseline E1,
hash split/mã nguồn và metadata tám checkpoint. Xem [phân tích kết quả](report.md).
Checkpoint giữ tại thư mục nguồn của lượt chạy, không đưa vào Git.

## Biểu diễn chuỗi

| Cách biểu diễn | Thứ tự / đặc trưng mỗi bước | T | D | Tên cấu hình |
|---|---|---:|---:|---|
| Hàng ảnh | Từ trên xuống dưới, giữ nguyên 28 pixel mỗi hàng | 28 | 28 | `lstm_row`, `gru_row` |
| Patch | Ô 7×7 rời nhau, trái sang phải rồi trên xuống dưới; flatten pixel trong mỗi ô | 16 | 49 | `lstm_patch`, `gru_patch` |
| CNN-stem | Hai conv/ReLU/pooling → feature map 64×7×7; quét vị trí theo hàng | 49 | 64 | `lstm_cnn_stem`, `gru_cnn_stem` |
| Patch chồng lấn | Cửa sổ 7×7, stride 3, lưới 8×8; quét theo hàng | 64 | 49 | `lstm_overlap_patch`, `gru_overlap_patch` |

Nhóm pixel/hàng chọn phương án hàng; nhóm biến thể chọn patch chồng lấn.
Patch dùng pixel thô như E3 cũ; khác E2, không có phép chiếu patch/row học được.
CNN-stem học đặc trưng cùng mạng hồi quy: Conv(1,32,3,pad1) → ReLU → Pool2 →
Conv(32,D,3,pad1) → ReLU → Pool2. `--stem-dim` mặc định 64.

Không thêm CLS hoặc positional embedding. Thứ tự vào LSTM/GRU xác định thứ tự
chuỗi. Readout là trạng thái ẩn cuối của lớp cuối; khi dùng `--bidirectional`,
ghép trạng thái cuối hai hướng. Đã sửa cách lấy `output[:, -1, :]` cũ vì nó không
chứa trạng thái cuối đầy đủ của hướng ngược. Với mạng một chiều, hai cách tương đương.

Mặc định hidden size 128, một lớp, một chiều, dropout head 0.1. Khi nhiều lớp,
dropout còn áp dụng giữa các lớp hồi quy. Số tham số (mặc định):

| Biểu diễn | LSTM | GRU |
|---|---:|---:|
| Hàng | 82.186 | 61.962 |
| Patch | 92.938 | 70.026 |
| CNN-stem | 119.434 | 94.602 |
| Patch chồng lấn | 92.938 | 70.026 |

Hai baseline `mlp_baseline` (235.146 tham số) và `cnn_baseline` (421.642) lấy từ
`E1/results/epochs100_batch256_seed42/`: test accuracy lần lượt **98,63%** và
**99,27%**, đúng số liệu E1 trong báo cáo. Có thể chỉ định thư mục E1 bằng
`--baseline-results`; script kiểm tra dataset, split seed/ratio, số mẫu và cấu hình
epochs/batch/lr/weight decay, dừng khi thiếu hoặc không khớp. Không tự quay về bộ
E1 cũ 20 epoch/lô 16. Nguồn và hash metrics/history được lưu để truy vết.
CNN-stem và chiều D khác nhau làm thay đổi
capacity; đây không phải ablation chỉ thay độ dài chuỗi.

## Chạy từ gốc repo

```bash
python E3/smoke_test.py
CUDA_VISIBLE_DEVICES=1 python E3/train.py --experiments all --epochs 100 --batch-size 256 --device cuda --progress --probe-order --output-dir E3/results/sequence_v2
```

Colab dùng `!python ...`, bỏ `CUDA_VISIBLE_DEVICES=1` nếu chỉ có một GPU.
Mặc định đã là 100 epoch, batch 256. Adam, lr 0.001, weight decay 0.0001,
CrossEntropyLoss nhận logits; không scheduler/early stopping/augmentation.
MNIST train/validation/test = 54.000/6.000/10.000, Normalize(0.1307,0.3081).
`--seed 42` cố định split; `--train-seed` mặc định cùng giá trị.

Mỗi cấu hình reset seed huấn luyện và DataLoader; shuffle dùng generator riêng
để cùng thứ tự batch qua mọi epoch, độc lập RNG dropout. LSTM/GRU cùng tokenizer
CNN-stem bắt đầu bằng cùng trọng số stem, nhưng không thể yêu cầu tham số/cơ chế
cổng hay accuracy của LSTM và GRU giống nhau như phép kiểm chứng hai bản MSA.

Chọn best checkpoint theo validation accuracy, bằng nhau giữ epoch sớm nhất;
test đánh giá checkpoint đó. Đổi `--train-seed 43`/`44` và output mới để chạy nhiều
lượt mà giữ `--seed 42`. Môi trường GPU, phiên bản thư viện và hash split/source
được ghi vào metrics. E1 có cách tiêu thụ RNG khác và thiếu metadata GPU, nên
thời gian baseline giữ như số đo lịch sử, không coi là phép đo tốc độ cùng thiết bị
với lượt E3 mới. Tái sử dụng kết quả E1 bảo đảm cùng số liệu baseline trong báo cáo.

Thư mục kết quả trên đã có dữ liệu: khi chạy lại, đổi `--output-dir` thành đường
dẫn mới, ví dụ `E3/results/sequence_v2_seed43` cùng `--train-seed 43`.
Output phải trống/mới; script từ chối ghi đè và không hỗ trợ resume. Mỗi tiến trình
có `.run.lock`; chỉ xóa khóa còn sót sau khi đã xác nhận tiến trình dừng.
Kết quả cũ sáu cấu hình tại `E3/results/` giữ nguyên.

## Bàn luận phụ thuộc theo thứ tự

`--probe-order` là tùy chọn đánh giá bổ sung, không huấn luyện thêm. Sau khi chọn
checkpoint, đo trên **validation** với thứ tự gốc, đảo ngược và một hoán vị cố
định theo training seed. Hoán vị áp dụng sau tokenizer, giữ nguyên ảnh và trọng số;
cùng hoán vị trong từng cặp LSTM/GRU. Tệp lưu cả hoán vị để truy vết.

Độ giảm accuracy thể hiện độ nhạy với thay đổi thứ tự so với lúc train; không
chứng minh trí nhớ dài hạn hoặc khả năng xử lý thời gian thực. Không so trực tiếp
probe này với việc xáo pixel đầu vào CNN/MLP. Để khẳng định phụ thuộc dài cần thêm
thí nghiệm kiểm soát; accuracy MNIST đơn thuần chưa đủ.

## Đầu ra và kiểm thử

`E3/results/sequence_v2/` lưu:

- `metrics.json`, `run_config.json`, `summary.csv`, history của 8 mô hình RNN.
- `baselines_e1.json`: bản chụp hai kết quả E1, nguồn, hash, `trained_in_this_run=false`.
- Checkpoint tốt nhất tại `checkpoints/best_*.pt` (Git bỏ qua, giữ riêng).
- Dự đoán từng ảnh, ma trận nhầm lẫn CSV/PNG, tối đa 25 mẫu sai đầu theo thứ tự test.
- `comparison_rnn.csv`: LSTM/GRU trong từng tokenizer; đánh dấu cặp chưa đủ.
- `comparison_baselines.csv`: tám RNN mới và hai baseline E1 tái sử dụng, có cột
  nguồn và đánh dấu mô hình có được huấn luyện trong lượt này hay không.
- `comparison_order_probe.csv` nếu bật probe; `comparison.png`, `comparison.md`.
- `learning_curves_<tokenizer>.png`; `learning_curves_baselines_e1.png` dựng từ
  history E1 gốc, không tạo history/checkpoint baseline giả như vừa train ở E3.

Metrics/summary được ghi sau từng mô hình trước khi vẽ, có trạng thái
`running`/`failed`/`completed`. Không tự trộn `E1/results` hoặc `E2/results` cũ
vào bảng so sánh như pipeline trước.

Smoke test chạy offline: kiểm tra thứ tự pixel/patch và phủ biên, CNN-stem, forward/
backward đủ tám RNN, readout hai chiều; train nhỏ tám RNN, nạp lại checkpoint,
đối chiếu dự đoán, kiểm tra order probe và bảo vệ output khi lỗi/ghi đè. Kiểm tra
chỉ có 8 lần gọi training, hai baseline nhập nguyên giá trị và từ chối protocol sai.
Dữ liệu giả chỉ kiểm tra code, không được dùng làm kết quả môn học.

Sentiment giữ nguyên: `python E3/train_sentiment.py --rnn-type lstm --epochs 5`.
