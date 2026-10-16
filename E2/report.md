# E2 — So sánh MSA tự viết, PyTorch và bốn nhóm tokenization

## 1. Mục tiêu và trạng thái

Thí nghiệm mở rộng E2 hiện thực MSA từ phép toán cơ bản và bằng
`nn.MultiheadAttention`, đối chiếu trên cùng tokenizer, đồng thời so sánh bốn nhóm
lấy token mà đề gợi ý. Đã có mã và bộ kiểm tra; **chưa có kết quả MNIST của bộ
8 cấu hình mới** để đưa vào báo cáo. Kết quả smoke test dùng dữ liệu giả, không
phải kết quả thực nghiệm khoa học.

Kết quả cũ ba cấu hình (20 epochs, batch 16) vẫn nằm ở `E2/results/` và báo cáo
Word/PDF hiện có. Bộ mới dùng thư mục `E2/results/tokenization_msa_v2/`, đồng
nhất dropout hai backend; không trộn số liệu hai phiên bản.

## 2. Dữ liệu và protocol

Dùng cùng MNIST như E1: ảnh xám 1×28×28, 10 lớp. Chia 60.000 ảnh train gốc
thành 54.000 train và 6.000 validation bằng `random_split`, split seed 42; giữ
riêng 10.000 ảnh test chính thức. Tiền xử lý ToTensor và chuẩn hóa mean 0.1307,
std 0.3081, không augmentation.

Cấu hình dự kiến chung: 100 epochs, batch size 256, Adam, lr 0.001, weight decay
0.0001, CrossEntropyLoss, dropout 0.1; không scheduler hoặc early stopping.
Chọn checkpoint theo validation accuracy cao nhất, bằng nhau giữ epoch sớm nhất,
sau đó mới đánh giá test. Thời gian đo gồm train + validation, không gồm bước
kiểm tra tương đương, test hoặc vẽ hình. Môi trường/GPU được lưu trong metrics.

## 3. Bốn cách lấy token

| Nhóm đề bài | Cách hiện thực | Token ảnh | Tổng có CLS |
|---|---|---:|---:|
| Patch | Kernel 7, stride 7, lưới 4×4 | 16 | 17 |
| Pixel / hàng | Chọn mỗi hàng 28 pixel → Linear(28,64) | 28 | 29 |
| CNN-stem | Hai conv/ReLU/pooling → feature map 64×7×7, mỗi vị trí là một token | 49 | 50 |
| Biến thể | Patch chồng lấn kernel 7, stride 3, lưới 8×8 | 64 | 65 |

Nhánh patch dùng Conv2d cho projection không chồng lấn; khác với CNN-stem gồm
nhiều tầng học đặc trưng phi tuyến. Overlap cho phép các cửa sổ chia sẻ pixel.
Tất cả chuỗi có positional embedding và CLS học được.

## 4. MSA và classifier

Với X có dạng B×N×64, một Linear tạo Q, K, V, chia 4 heads (head_dim=16):

```text
A_i = softmax(Q_i K_i^T / sqrt(16))
H_i = Dropout(A_i) V_i
Y = Concat(H_1, ..., H_4) W_o + b_o
```

Custom không gọi `nn.MultiheadAttention`/`nn.TransformerEncoder*`. Bản PyTorch
dùng `nn.MultiheadAttention(batch_first=True, need_weights=False)`. Cả hai có cùng
attention dropout và output dropout tại residual block. Một encoder pre-norm
gồm LayerNorm → MSA → residual, LayerNorm → FFN(64→128→64, GELU) → residual;
head LayerNorm + Linear(64,10) lấy CLS và trả logits cho CrossEntropyLoss.

## 5. Kiểm chứng tương đương

Trước khi train, mỗi tokenizer được kiểm tra một cặp có cùng toàn bộ trọng số
(kể cả CNN/patch projection, CLS, positional embedding, FFN và head).
`verification.json` ghi rõ config, thiết bị, dtype, tolerance và sai số từng mục:

- Attention: đầu ra, gradient input và các tham số QKV/output projection.
- Classifier: logits suy luận eval/no_grad; logits/loss, gradient ảnh và toàn bộ
  tham số trong train mode với dropout tắt.
- CPU float64 với atol=1e-10, rtol=1e-8, có thêm một bước Adam; float32 trên
  thiết bị huấn luyện với atol=2e-5, rtol=2e-4.

Một kiểm tra FAIL sẽ chặn tải dữ liệu và training. Accuracy gần nhau không chứng
minh MSA đúng. Các phép kiểm tra trên input tổng hợp là bằng chứng số học hữu
hạn; không phải chứng minh hình thức cho mọi input hoặc yêu cầu hai lượt train
có accuracy giống nhau.

Khi train thật, trong từng cặp khởi tạo từ cùng reference custom và copy toàn bộ
trọng số sang PyTorch, ghi hash khởi tạo; thứ tự batch được reset bằng generator
riêng. Dropout dùng cùng vị trí và xác suất nhưng không ép cùng mặt nạ giữa các
kernel, nên quỹ đạo học có thể khác. Split seed tách khỏi `--train-seed` để có thể
chạy lặp mà giữ nguyên train/validation.

## 6. Kết quả cần cập nhật sau khi chạy

Từ `summary.csv`, lập bảng đủ 8 hàng: `custom_patch`, `pytorch_patch`,
`custom_row`, `pytorch_row`, `custom_cnn_stem`, `pytorch_cnn_stem`,
`custom_overlap_patch`, `pytorch_overlap_patch`. Mỗi hàng có val tốt nhất, best
epoch, test loss/accuracy, macro F1, số tham số, số token và thời gian train + val.
Chỉ dùng bộ hoàn tất đủ 8 mô hình và mọi kiểm tra PASS.

Hai phép đối chiếu riêng:

1. **MSA tự viết và PyTorch:** bốn cặp cùng tokenizer, xem `comparison_msa.csv`.
   Báo chênh accuracy theo điểm phần trăm, thời gian mỗi bản và tỷ số thời gian.
2. **Tokenization:** so bốn cách trong cùng backend, xem
   `comparison_tokenization.csv`. Xét đồng thời chất lượng, token count, số tham
   số và chi phí; không quy hiệu quả CNN-stem riêng cho độ dài chuỗi.

Dùng `comparison.png`, bốn `learning_curves_<tokenizer>.png`, confusion matrix
và ảnh lỗi để giải thích kết quả. `comparison.md` tự xuất nhận xét định lượng
ban đầu; cần người học đọc lại và bổ sung diễn giải học thuật từ số liệu thật.

## 7. Giới hạn và tái lập

Một seed chưa đủ kết luận ý nghĩa thống kê. Thời gian phụ thuộc GPU/kernel và
tải cạnh tranh, không đồng nhất với latency suy luận. CNN-stem thay đổi capacity;
overlap thay đổi độ dài chuỗi và positional embedding, nên chưa tách được tác
động nhân quả riêng của từng thành phần. Không dùng test để chọn hyperparameters.

```bash
CUDA_VISIBLE_DEVICES=1 python E2/train.py --experiments all --epochs 100 --batch-size 256 --device cuda --progress --output-dir E2/results/tokenization_msa_v2
```

Trước khi tổng hợp vào `reports/exercises.docx` và `reports/exercises.pdf`, cần
đối chiếu số liệu, tên GPU/môi trường, trạng thái hoàn tất, kết quả verification
và hình minh họa. Không thay kết quả lịch sử bằng số liệu chưa chạy.
