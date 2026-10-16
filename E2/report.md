# E2 — So sánh MSA tự viết, PyTorch và bốn nhóm tokenization

## 1. Mục tiêu và trạng thái

Thí nghiệm E2 hiện thực MSA từ phép toán cơ bản và bằng
`nn.MultiheadAttention`, đối chiếu trên cùng tokenizer, đồng thời so sánh bốn nhóm
lấy token mà đề gợi ý. Bộ MNIST mới đã hoàn tất **8 cấu hình**, mỗi cấu hình
100 epoch, batch size 256; cả 8 trường hợp kiểm chứng số học đều PASS.

Kết quả cũ ba cấu hình (20 epochs, batch 16) vẫn nằm ở `E2/results/` để truy vết.
Bộ mới nằm ở `E2/results/tokenization_msa_v2/`, đồng nhất dropout hai backend.
Báo cáo Word chung đã cập nhật; người học tự xuất PDF tương ứng trước khi push.

## 2. Dữ liệu và protocol

Dùng cùng MNIST như E1: ảnh xám 1×28×28, 10 lớp. Chia 60.000 ảnh train gốc
thành 54.000 train và 6.000 validation bằng `random_split`, split seed 42; giữ
riêng 10.000 ảnh test chính thức. Tiền xử lý ToTensor và chuẩn hóa mean 0.1307,
std 0.3081, không augmentation.

Cấu hình đã chạy: 100 epochs, batch size 256, Adam, lr 0.001, weight decay
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

## 6. Kết quả thực nghiệm

| Cấu hình | Tham số | Best epoch | Best val (%) | Test loss | Test accuracy (%) | Macro F1 | Train + val (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| custom_patch | 38.602 | 98 | 97,48 | 0,078682 | 97,59 | 0,975703 | 738,8 |
| pytorch_patch | 38.602 | 55 | 97,13 | 0,096932 | 97,05 | 0,970280 | 733,0 |
| custom_row | 38.026 | 83 | 98,05 | 0,065533 | 98,04 | 0,980230 | 750,6 |
| pytorch_row | 38.026 | 97 | 98,15 | 0,057587 | 98,24 | 0,982280 | 751,3 |
| custom_cnn_stem | 56.330 | 82 | 99,32 | 0,024487 | 99,28 | 0,992767 | 763,6 |
| pytorch_cnn_stem | 56.330 | 59 | 99,28 | 0,024037 | 99,28 | 0,992781 | 827,2 |
| custom_overlap_patch | 41.674 | 97 | 98,47 | 0,050016 | 98,53 | 0,985199 | 739,7 |
| pytorch_overlap_patch | 41.674 | 82 | 98,45 | 0,056096 | 98,33 | 0,983174 | 737,4 |

**MSA tự viết so với PyTorch:** chênh test accuracy (PyTorch trừ tự viết) lần lượt
là −0,54; +0,20; 0,00; −0,20 điểm phần trăm cho patch, hàng, CNN-stem và overlap.
Tỷ số thời gian PyTorch/tự viết lần lượt là 0,9922; 1,0010; 1,0834; 0,9969.
Không có một bản luôn chính xác hoặc nhanh hơn trên cả bốn tokenizer.

**So sánh tokenization trong cùng bản MSA:** CNN-stem dẫn đầu, cùng accuracy
99,28% và 72 lỗi ở cả hai bản, nhưng có F1 và vị trí lỗi khác nhau. Patch chồng
lấn hơn patch rời 0,94 điểm phần trăm ở bản tự viết và 1,28 ở PyTorch.
CNN-stem bổ sung CNN và tăng capacity; overlap tăng số token và nhúng vị trí,
nên chưa thể quy cải thiện riêng cho một yếu tố.

Đã tính lại accuracy/F1 từ 10.000 dự đoán mỗi mô hình và đối chiếu đủ ma trận
nhầm lẫn CSV. Mỗi history đủ 100 epoch; best epoch khớp validation. Các hash
khởi tạo trong cặp khớp nhau, hash split khớp seed 42, hash nguồn khớp repo sau
chuẩn hóa xuống dòng LF. Đây là kiểm tra dữ liệu đã cung cấp, không huấn luyện lại.

Sai số tuyệt đối lớn nhất trong các mục kiểm chứng: CPU float64 ≤ 8,89×10⁻¹⁵,
CUDA float32 ≤ 3,82×10⁻⁶; tất cả đạt tiêu chuẩn kết hợp atol/rtol đã khai báo.
Môi trường: Python 3.10.13, PyTorch 2.2.1+cu118, torchvision 0.17.1+cu118,
CUDA 11.8, Tesla P100-PCIE-16GB. Không có xác nhận GPU hoàn toàn không chịu
tải cạnh tranh trong lúc chạy.

Cuối 100 epoch, CNN-stem có loss train/validation khoảng 0,0133/0,0325 (tự viết)
và 0,0116/0,0331 (PyTorch), cho thấy khoảng cách khái quát hóa. Checkpoint test
là epoch tốt nhất trong bảng, không mặc định epoch cuối. Với custom_patch,
`7 → 2` và `3 → 5` đều có 15 lỗi; CNN-stem tự viết còn 8 lỗi `4 → 9`.
Hình lỗi chỉ minh họa các mẫu sai đầu tiên theo thứ tự test.

## 7. Giới hạn và tái lập

Một seed chưa đủ kết luận ý nghĩa thống kê. Thời gian phụ thuộc GPU/kernel và
tải cạnh tranh, không đồng nhất với latency suy luận. CNN-stem thay đổi capacity;
overlap thay đổi độ dài chuỗi và positional embedding, nên chưa tách được tác
động nhân quả riêng của từng thành phần. Không dùng test để chọn hyperparameters.

```bash
CUDA_VISIBLE_DEVICES=1 python E2/train.py --experiments all --epochs 100 --batch-size 256 --device cuda --progress --output-dir E2/results/tokenization_msa_v2
```

`reports/exercises.docx` đã tổng hợp bộ mới cùng E1 (100 epoch/lô 256) và E3
lịch sử (20 epoch/lô 16). PDF trong repo chỉ được thay sau khi người học xuất
lại từ bản Word này. Chênh một ảnh giữa E2 CNN-stem (99,28%) và CNN E1 (99,27%)
chưa đủ chứng minh ưu thế có ý nghĩa thống kê.
