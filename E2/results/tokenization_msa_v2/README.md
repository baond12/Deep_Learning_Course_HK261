# E2 — Tám cấu hình MSA và tokenization

Bộ kết quả nhập từ thư mục `tokenization_msa_v2` do người học cung cấp.
Giữ nguyên 52 tệp kết quả gốc; không huấn luyện lại khi cập nhật báo cáo.
Tám checkpoint vẫn được giữ trong thư mục nguồn của lượt chạy, không sao chép
vào Git theo `.gitignore`. Kết quả lịch sử ba cấu hình tại thư mục cha giữ nguyên.

MNIST: train/validation/test = 54.000/6.000/10.000, split seed và training seed 42;
100 epoch, batch 256, Adam, lr 0.001, weight decay 0.0001, dropout 0.1.
Một encoder pre-norm, embedding 64, 4 head, FFN 128, CLS học được; không augmentation,
scheduler hoặc early stopping. Chọn best checkpoint bằng validation trước khi test.

| Tokenizer | Token ảnh / có CLS | Tham số mỗi bản | Test tự viết (%) | Test PyTorch (%) | Δ PyTorch − tự viết (đpt) |
|---|---:|---:|---:|---:|---:|
| Patch 7×7, stride 7 | 16 / 17 | 38.602 | 97,59 | 97,05 | −0,54 |
| Hàng ảnh | 28 / 29 | 38.026 | 98,04 | 98,24 | +0,20 |
| CNN-stem | 49 / 50 | 56.330 | 99,28 | 99,28 | 0,00 |
| Patch chồng lấn 7×7, stride 3 | 64 / 65 | 41.674 | 98,53 | 98,33 | −0,20 |

Đã đối chiếu:

- `status=completed`, đủ 8 cấu hình và 8 trường hợp CPU float64/CUDA float32 PASS.
- 800 bản ghi history hữu hạn, epoch 1–100 liên tiếp, best epoch khớp validation.
- 80.000 dự đoán test: tính lại accuracy, F1 macro và đủ 8 ma trận CSV đều khớp.
- `summary.csv` khớp metrics; hash khởi tạo trong từng cặp khớp nhau.
- Hash split khớp phép chia seed 42; SHA-256 mã nguồn khớp repo sau chuẩn hóa LF.

Môi trường lưu trong metrics: Python 3.10.13, PyTorch 2.2.1+cu118,
torchvision 0.17.1+cu118, CUDA 11.8, Tesla P100-PCIE-16GB.
Chưa có bằng chứng về tải cạnh tranh, độ lệch chuẩn qua nhiều seed hoặc latency.
Kiểm chứng cùng trọng số không yêu cầu accuracy sau train của hai bản giống nhau.

```bash
python E2/train.py --experiments all --epochs 100 --batch-size 256 --seed 42 --train-seed 42 --device cuda --output-dir E2/results/tokenization_msa_v2_rerun
```

Lệnh tái lập dùng output mới vì pipeline từ chối ghi đè kết quả.
Báo cáo chung Word được cập nhật tại `reports/exercises.docx`; người học xuất lại
`reports/exercises.pdf` từ Word trước khi push để hai bản đồng nhất.
