# E3 — Kế hoạch báo cáo LSTM và GRU trên chuỗi ảnh

## Trạng thái và mục tiêu

Pipeline mới huấn luyện tám cấu hình LSTM/GRU và đọc hai baseline MLP/CNN E1 đã có.
**Chưa có kết quả MNIST của tám cấu hình RNN mới.** Smoke test dùng dữ liệu giả chỉ
kiểm tra code. Kết quả sáu cấu hình 20 epoch/lô 16 tại `E3/results/` và chương E3
trong báo cáo Word là số liệu lịch sử, không phải kết quả của pipeline mới.

Mục tiêu là so sánh mô hình hồi quy trên cùng cách biểu diễn, so các cách biểu diễn
trong cùng họ mô hình và luôn đối chiếu với baseline không hồi quy. Các ví dụ
tokenization từ E2 có thể dùng cho E3, nhưng không bắt buộc phải dùng Transformer,
MSA hoặc CLS. Ở đây, readout là trạng thái ẩn cuối của LSTM/GRU.

## Dữ liệu và protocol đã chọn

MNIST ảnh xám 1×28×28, 10 lớp; train/validation/test = 54.000/6.000/10.000.
Split seed 42, ToTensor + Normalize(0.1307,0.3081), không augmentation.
100 epoch, batch 256, Adam, learning rate 0.001, weight decay 0.0001.
Hidden size 128, một lớp, một chiều, dropout head 0.1. Không scheduler/dừng sớm.
MLP/CNN giữ nguyên kết quả E1 ở `E1/results/epochs100_batch256_seed42/`, không train
hoặc test lại: test accuracy 98,63%/99,27%, F1 macro 0,986219/0,992631. Nguồn,
hash và cấu hình baseline được lưu riêng trong `baselines_e1.json`; không coi đó
là hai lượt huấn luyện mới của E3. Script kiểm tra protocol trước khi chạy RNN.

Reset seed huấn luyện và generator shuffle cho từng cấu hình. Checkpoint được
chọn theo validation accuracy; nếu bằng nhau giữ epoch sớm nhất. Test chỉ đánh
giá checkpoint đó, không dùng để chọn lại cấu hình. Chạy lặp giữ `--seed 42`,
đổi `--train-seed` và output để đo độ ổn định mà không đổi split.

## Bốn cách biểu diễn

| Biểu diễn | T | D | Thứ tự |
|---|---:|---:|---|
| Hàng ảnh | 28 | 28 | Từ trên xuống dưới; mỗi hàng giữ 28 pixel |
| Patch rời 7×7 | 16 | 49 | Trái sang phải, rồi trên xuống dưới; flatten trong patch |
| CNN-stem | 49 | 64 | Mỗi vị trí bản đồ 7×7 là một token, quét theo hàng |
| Patch chồng lấn 7×7, stride 3 | 64 | 49 | Lưới 8×8, quét theo hàng, phủ đủ biên |

CNN-stem: Conv(1→32,3,pad1) → ReLU → Pool2 → Conv(32→64,3,pad1) → ReLU → Pool2.
Khác E2, hàng và patch dùng pixel thô như E3 cũ, không thêm phép chiếu học được.
Kiến trúc xử lý chuỗi B×T×D bằng nn.LSTM/nn.GRU, head dropout + Linear → 10 logits
cho CrossEntropyLoss. Khi hai chiều, ghép hai trạng thái cuối của lớp cuối.

## Các bảng và hình cần cập nhật sau khi chạy

1. Bảng 10 hàng gồm tám RNN mới và hai baseline E1 tái sử dụng: test accuracy,
   F1 macro, test loss, val tốt nhất, best epoch, tham số, T/D, thời gian và nguồn.
2. Bảng LSTM–GRU trong từng tokenizer: chênh accuracy và chi phí.
3. Bảng so tokenization trong từng họ và đối chiếu nguyên giá trị MLP/CNN E1.
4. Đường cong loss/accuracy, ma trận nhầm lẫn và mẫu sai; tính lại chỉ số từ CSV
   dự đoán để tránh trộn file khi chạy nhiều lượt.
5. Nếu bật `--probe-order`, bảng độ nhạy trên validation khi giữ thứ tự gốc,
   đảo ngược hoặc hoán vị cố định sau tokenizer; checkpoint không thay đổi.

Không suy ra khả năng nhớ dài hạn chỉ từ accuracy MNIST hoặc độ giảm ở order
probe. Ảnh là chuỗi không gian, không có thời gian thực. Đảo/xáo chuỗi cũng tạo
khác biệt phân phối so với lúc train, nên probe chỉ hỗ trợ bàn luận độ nhạy thứ tự.
CNN/MLP là baseline chất lượng/chi phí, không trực tiếp nhận probe đảo token này.

## Giới hạn cần giữ khi diễn giải

LSTM/GRU có cơ chế cổng khác nhau; không có yêu cầu hai mô hình phải khớp đầu ra
như hai cách hiện thực cùng MSA. GRU ít tham số hơn ở cấu hình tương đương, nhưng
thời gian thực tế còn phụ thuộc kernel và GPU. CNN-stem bổ sung tham số CNN; D
khác nhau làm thay đổi số tham số RNN, còn overlap tăng T so với patch mà giữ D.
Vì vậy không quy toàn bộ cải thiện riêng cho chiều dài chuỗi.

Một seed chưa cho biết độ lệch chuẩn hay ý nghĩa thống kê. E1 thiếu metadata GPU
và có cách tiêu thụ RNG khác E3, nên thời gian baseline không phải số đo cùng
thiết bị với lượt RNN mới. Train + validation không phải latency suy luận. Bộ mới
lưu hash nguồn/split, môi trường và checkpoint RNN để truy vết.

```bash
python E3/smoke_test.py
CUDA_VISIBLE_DEVICES=1 python E3/train.py --experiments all --epochs 100 --batch-size 256 --device cuda --progress --probe-order --output-dir E3/results/sequence_v2
```

Kết quả mới nằm ở `E3/results/sequence_v2/`. Chỉ cập nhật số liệu vào báo cáo
Word sau khi chạy hoàn tất tám RNN, đối chiếu hai baseline E1 và kiểm tra metadata.
Pipeline sentiment toy là phần minh họa riêng, không thuộc các bảng MNIST này.
