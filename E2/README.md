# E2 — MSA và bốn nhóm tokenization

Triển khai đủ bốn nhóm ví dụ của đề trên MNIST: patch, hàng, CNN-stem và biến thể
patch chồng lấn. Mỗi nhóm có hai bản MSA (tự viết và `nn.MultiheadAttention`), tạo
**8 thực nghiệm**. Trong nhóm “pixel / hàng”, chọn phương án mỗi hàng là một token;
trong nhóm “biến thể”, chọn patch overlap. Không gọi row là pixel tokenization và
không coi patch embedding bằng Conv2d là CNN-stem.

## Kiến trúc và các thực nghiệm

| Nhóm | Cách lấy token | Token ảnh | Có CLS | Tên thực nghiệm |
|---|---|---:|---:|---|
| Patch | Conv2d kernel 7, stride 7, không padding | 16 | 17 | `custom_patch`, `pytorch_patch` |
| Hàng | Mỗi hàng 28 pixel → Linear(28,64) | 28 | 29 | `custom_row`, `pytorch_row` |
| CNN-stem | Conv(1,32,3,pad1) → ReLU → Pool2 → Conv(32,64,3,pad1) → ReLU → Pool2; mỗi vị trí 7×7 là token | 49 | 50 | `custom_cnn_stem`, `pytorch_cnn_stem` |
| Patch chồng lấn | Conv2d kernel 7, stride 3, không padding; lưới 8×8 | 64 | 65 | `custom_overlap_patch`, `pytorch_overlap_patch` |

Số token trên áp dụng cho mặc định. `--patch-size` và `--overlap-stride` được kiểm
tra để các cửa sổ phủ hết ảnh 28×28, không bỏ phần biên. Overlap yêu cầu stride nhỏ
hơn patch size. Bộ encoder chung: embedding 64, 4 heads, FFN 128, depth 1,
pre-norm, residual, positional embedding học được và head lấy CLS để tạo 10 logits.

MSA tự viết dùng Linear QKV, reshape/permute, nhân ma trận, chia √head_dim, softmax,
dropout trên attention và ghép head. Không dùng module attention/Transformer có
sẵn trong nhánh này. Dropout sau attention projection nằm ở residual block cho
**cả hai backend**; đã bỏ lớp output dropout dư của custom phiên bản cũ.

## Kiểm tra bắt buộc trước khi train

`train.py` luôn kiểm tra cả cặp custom/PyTorch của từng tokenizer được chọn,
kể cả khi chỉ yêu cầu train một thành viên. Kiểm tra diễn ra **trước khi tải MNIST**:

1. Copy cùng trọng số cho toàn classifier: tokenizer, CLS, positional embedding,
   LayerNorm, FFN, head, QKV và output projection.
2. Đối chiếu riêng attention: đầu ra, gradient đầu vào và mọi tham số QKV/output.
3. Đối chiếu classifier: logits suy luận trong `eval` + `no_grad`, logits/loss
   khi train nhưng tắt dropout, gradient ảnh và mọi tensor tham số.
4. CPU float64 là kiểm tra tham chiếu, atol=1e-10, rtol=1e-8; kiểm tra thêm float32
   trên **thiết bị thực tế sẽ train**, atol=2e-5, rtol=2e-4. Với float64, kiểm tra
   thêm một bước Adam sau backward. Lưu sai số và kết quả trong `verification.json`.

Nếu có một mục FAIL, dừng trước khi train. Không có tùy chọn bỏ qua kiểm tra.
Kiểm tra dùng input tổng hợp và không thay đổi RNG của thí nghiệm. Đây là kiểm
chứng số học trên các trường hợp kiểm tra, không phải chứng minh cho mọi input.

“Khớp” là khớp phép toán tại **cùng trọng số** và trong điều kiện không có ngẫu
nhiên dropout. Hai lượt train thật không bắt buộc có accuracy giống nhau: kernel,
thứ tự làm tròn và mặt nạ dropout có thể khác dù cùng xác suất và seed.

## Lệnh chạy từ thư mục gốc repo

```bash
pip install -r requirements.txt

# Kiểm tra offline toàn bộ code, không tải MNIST.
python E2/smoke_test.py

# Chỉ kiểm tra tương đương trên máy/GPU sẽ train.
python E2/train.py --experiments all --verify-only --device cuda

# Chạy đủ 8 cấu hình, tự kiểm tra lại trước khi train.
CUDA_VISIBLE_DEVICES=1 python E2/train.py --experiments all --epochs 100 --batch-size 256 --device cuda --progress --output-dir E2/results/tokenization_msa_v2
```

Colab dùng `!python ...`, bỏ phần `CUDA_VISIBLE_DEVICES=1` nếu chỉ có một GPU.
Mặc định mới cũng là 100 epochs, batch 256; learning rate 0.001, weight decay
0.0001, Adam, dropout 0.1, không scheduler và không early stopping.

Muốn chọn riêng một cặp, dùng `--experiments custom_cnn_stem pytorch_cnn_stem` và
một `--output-dir` mới. Không chạy đồng thời các tiến trình vào cùng thư mục.
Thư mục đã có kết quả bị từ chối ghi đè; chỉ thư mục trống hoặc chỉ chứa
`verification.json` được dùng để bắt đầu train. Không hỗ trợ resume.
Nếu tiến trình bị kill để lại `.run.lock`, xác nhận tiến trình đã dừng trước khi
xóa riêng khóa đó. Giữ các kết quả dở dang và chạy lại vào một output mới.

## Kiểm soát so sánh

- Cùng MNIST như E1; train/val/test = 54.000/6.000/10.000; split seed 42;
  ToTensor + Normalize(0.1307, 0.3081), không augmentation.
- Cả hai mô hình của mỗi tokenizer bắt đầu bằng **cùng toàn bộ trọng số** và lưu
  cùng hash khởi tạo. Không chỉ đặt cùng seed cho hai constructor khác nhau.
- Reset DataLoader cho từng thực nghiệm; generator shuffle riêng tách khỏi RNG
  dropout, bảo đảm cùng thứ tự batch giữa hai backend qua mọi epoch.
- `--seed` cố định split; `--train-seed` mặc định bằng `--seed`. Khi chạy lặp,
  giữ `--seed 42`, đổi `--train-seed 43`/`44` và dùng output mới để không đổi split.
- Cùng ngân sách 100 epochs và batch 256; chọn best checkpoint theo validation
  accuracy, bằng nhau giữ epoch sớm nhất. `train.py` sau đó tự đánh giá checkpoint
  ấy trên test. Không dùng test để chọn lại cấu hình.
- Thời gian là train + validation, không gồm preflight/test/vẽ hình. GPU dùng
  chung với E1 hoặc tác vụ khác làm sai lệch phép đo; để so thời gian nên chạy
  tuần tự trên cùng GPU không có tải cạnh tranh, lưu tên GPU và phiên bản thư viện.
- Tokenization thay đổi số token và số tham số, đặc biệt CNN-stem bổ sung CNN;
  không coi đây là ablation chỉ thay đúng một đại lượng hay khẳng định nhân quả.

## Đầu ra

Mặc định mới lưu riêng tại `E2/results/tokenization_msa_v2/`:

- `verification.json`, `run_config.json`, `metrics.json`, `summary.csv`.
- `comparison_msa.csv`: custom vs PyTorch cho mỗi tokenizer, delta accuracy (điểm
  phần trăm), thời gian và tỷ số thời gian; ghi rõ cặp chưa đủ kết quả.
- `comparison_tokenization.csv`: 8 hàng để so tokenization trong cùng backend.
- `comparison.png`, `comparison.md`: biểu đồ accuracy/thời gian và nhận xét theo
  số liệu đã chạy, kèm giới hạn diễn giải.
- `learning_curves_<tokenizer>.png`: 4 hình, mỗi hình đối chiếu hai backend.
- `history_<experiment>.csv`, `predictions_<experiment>.csv`, confusion matrix
  CSV/PNG và mẫu lỗi PNG cho từng mô hình.
- `checkpoints/best_<experiment>.pt`: best checkpoint sau lựa chọn validation;
  bị `.gitignore` loại khỏi Git, cần giữ riêng để đánh giá/tái lập.

Ghi metrics/summary sau từng mô hình để bảo toàn các lượt đã hoàn tất; trạng thái
`running`, `failed`, `completed` nằm trong metrics. Chỉ đưa bộ đủ 8 lượt và đủ
kiểm tra PASS vào bảng kết quả hoàn chỉnh. Một seed chưa cho biết độ ổn định.

Kết quả lịch sử tại `E2/results/*.csv`, `*.json`, `*.png` là phiên bản ba cấu hình,
20 epochs, batch 16 với output dropout khác. Giữ nguyên để truy vết, không gộp
vào bộ mới. `E2/report.md` mô tả protocol mới và chỗ cần cập nhật sau khi bạn chạy;
báo cáo Word/PDF chung chưa thay đổi số liệu.

## Các file mã nguồn

- `models.py`: 4 tokenizer, 2 MSA, classifier và copy trọng số cặp.
- `verify_msa.py`: các kiểm tra tương đương bắt buộc.
- `train.py`: chạy 8 cấu hình, seed/data order, đánh giá và lưu kết quả.
- `comparison.py`: hai phép đối chiếu, bảng/hình/nhận xét.
- `smoke_test.py`: kiểm tra offline, gồm huấn luyện nhỏ đủ 8 mô hình và cố tình
  làm sai MSA để xác nhận training bị chặn.
