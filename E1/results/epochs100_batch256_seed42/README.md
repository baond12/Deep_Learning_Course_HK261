# E1 — 100 epoch, batch size 256, seed 42

Kết quả được nhập từ thư mục `results_E1_100epoch_256_batchsize` do người học cung cấp.
Các tệp dữ liệu và hình gốc được giữ nguyên; không huấn luyện lại khi cập nhật báo cáo.
Kết quả E1 trước đây (20 epoch, batch size 16) vẫn nằm ở thư mục cha.

MNIST: 54.000 ảnh train, 6.000 ảnh validation, 10.000 ảnh test; `val_ratio=0.1`,
`seed=42`. Adam, learning rate `0.001`, weight decay `0.0001`, thiết bị CUDA.
Bộ kết quả không ghi tên GPU hoặc phiên bản CUDA/PyTorch và không kèm checkpoint.

```bash
python E1/train.py --models all --epochs 100 --batch-size 256 --learning-rate 0.001 --weight-decay 0.0001 --val-ratio 0.1 --seed 42 --device cuda --output-dir E1/results/epochs100_batch256_seed42
```

| Mô hình | Tham số | Epoch tốt nhất | Val accuracy | Test loss | Test accuracy | Macro F1 | Train + validation (giây) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Softmax | 7.850 | 32 | 0,920333 | 0,271124 | 0,9259 | 0,924782 | 718,2565 |
| MLP | 235.146 | 81 | 0,984000 | 0,060617 | 0,9863 | 0,986219 | 760,5613 |
| CNN | 421.642 | 83 | 0,993167 | 0,025890 | 0,9927 | 0,992631 | 769,9987 |

Đã đối chiếu `metrics.json`, `summary.csv` với ba history: mỗi history đủ 100 epoch
liên tiếp, giá trị hữu hạn; epoch tốt nhất khớp cực đại validation accuracy
(giữ epoch sớm nhất nếu bằng nhau). Test được đánh giá ở checkpoint đã chọn bằng validation.

Báo cáo chung: `reports/exercises.docx` và PDF xuất từ Word tại `reports/exercises.pdf`.
Hình trong báo cáo được chuyển thang xám; ảnh nguồn trong thư mục này giữ nguyên.
E2/E3 trong báo cáo vẫn dùng ngân sách lịch sử 20 epoch, batch size 16.
Lượt E1 mới thay đồng thời epochs và batch size, nên chưa tách được ảnh hưởng từng yếu tố;
một seed chưa đủ đánh giá độ ổn định của kết quả.
