# Chạy thăm dò Assignment 1 trên P100

Mục tiêu hiện tại là đo tính khả thi trước khi chốt A1.1: tám cấu hình, mỗi cấu hình một vòng, batch 16, FP32, chạy tuần tự. Chưa phải đợt so sánh cuối A1.2. Không tăng số vòng hoặc chọn cấu hình theo điểm kiểm tra ở bước này.

## Đã chuẩn bị trong mã

- `pilot.py`: chạy cả tám cấu hình tuần tự; chỉ dùng 40.000 ảnh huấn luyện và 10.000 ảnh xác thực của CIFAR-100. Không nạp hoặc đánh giá tập kiểm tra trong chương trình này.
- `engine.py`: vòng lặp riêng cho A1, Adam 1e-4 và weight decay 1e-4 theo mặc định thăm dò, FP32, kiểm tra loss/gradient hữu hạn. Chưa đổi sang AdamW, warm-up, cosine hay augmentation mới trong bản đề xuất.
- Phần ResNet được đóng băng giữ cả tham số lẫn thống kê BatchNorm; phần ViT bị đóng băng ở chế độ đánh giá. Kiểm tra hash tham số/buffer cố định trước và sau lượt; kiểm tra phần được học có thay đổi.
- Đo thời gian và số ảnh/giây cho huấn luyện và xác thực riêng, có đồng bộ CUDA. Ghi bộ nhớ đỉnh cấp phát/đặt trước của PyTorch, GPU/phần mềm, seed, mã nguồn và split.
- Khởi tạo/tải trọng số được đo riêng. Lưu trạng thái tốt nhất trên CPU, không giữ bản sao tốt nhất trên GPU.
- Lưu kết quả ngay sau từng pha/vòng/lượt. Lượt NaN/OOM/lỗi khác được ghi rõ rồi tiếp tục cấu hình kế tiếp; Ctrl+C dừng và giữ kết quả đã có.
- Không ghi đè: thư mục đầu ra phải chưa tồn tại. Cả tám lượt dùng cùng batch; không tự giảm batch sau OOM vì như vậy sẽ đổi điều kiện đo mà người học không biết.

`A1/train.py` cũng dùng vòng lặp A1 mới để áp dụng chính sách đóng băng và kiểm tra hữu hạn. Hành vi thăm dò không tự chọn lại siêu tham số hoặc biến thành quy trình chính thức đã khóa. Mã `common/` cùng E1–E3 giữ nguyên.

## Chuẩn bị trên server Linux

Chạy trong repo. Dữ liệu, trọng số tiền huấn luyện và tệp đầu ra lưu trên ổ cục bộ của server nếu có thể, để tránh ảnh hưởng I/O từ ổ mạng khi đo thời gian.

```bash
git pull --ff-only origin main
python A1/prepare_data.py --data-dir data
```

Ưu tiên dùng môi trường PyTorch/CUDA đã chạy được các bài trước trên P100. Không
cần nâng phiên bản thư viện để chạy lệnh này. Nếu môi trường mới thiếu thư viện,
cài các phụ thuộc trong `requirements.txt` với bộ PyTorch/torchvision tương thích
P100 đã được kiểm tra trên server; không thay bộ cài GPU đang hoạt động chỉ để
lấy phiên bản mới nhất. Chương trình sẽ ghi phiên bản thực tế vào kết quả.

Nếu CIFAR-100 đã ở một thư mục khác, dùng đường dẫn đó ở cả bước chuẩn bị và `--data-dir`. Chuẩn bị lại cùng đường dẫn sẽ kiểm tra dữ liệu, không buộc tải lại. Ba cấu hình pretrained của mỗi họ cần nạp trọng số ImageNet-1K V1 thực; torchvision dùng cache và có thể cần mạng ở lần đầu. Không có cơ chế âm thầm thay bằng trọng số ngẫu nhiên khi tải lỗi.

## Lệnh chạy đầu tiên

Ví dụ dùng GPU vật lý số 1, như các lần chạy trước. Đổi `CUDA_VISIBLE_DEVICES` nếu P100 có chỉ số khác. Hạn chế các tiến trình khác trên cùng GPU trong thời gian đo; chương trình chỉ bảo đảm tám cấu hình trong chính nó chạy tuần tự.

```bash
CUDA_VISIBLE_DEVICES=1 python -u A1/pilot.py \
  --experiments all --epochs 1 --batch-size 16 \
  --learning-rate 0.0001 --weight-decay 0.0001 \
  --seed 42 --train-seed 42 --num-workers 2 \
  --data-dir data --device cuda --progress \
  --output-dir A1/results/pilot_epoch1_batch16_seed42
```

Không dùng `A1/train.py` cho lệnh thăm dò này vì chương trình huấn luyện thông thường còn đánh giá test. Mã thăm dò không lưu checkpoint, không tạo ma trận hay hình lỗi từ test.

Mặc định vẫn giữ resize 224×224, lật ngang ngẫu nhiên khi huấn luyện và chuẩn hóa ImageNet như pipeline cũ. Việc thêm random crop hoặc lịch học mới sẽ được quyết định sau khi đọc số liệu thăm dò. Seed chia 42 giữ nguyên thuật toán cân bằng 400/100 ảnh mỗi lớp; seed huấn luyện được đặt lại cho mỗi cấu hình.

Một vòng vẫn xử lý đầy đủ 40.000 + 10.000 ảnh, không dùng tập con để làm số đo trông nhanh hơn. Vòng đầu bao gồm khởi động bộ nạp, cấp phát bộ nhớ và các chi phí lạnh khác; chưa thể coi thời gian đó là tốc độ ổn định của cả đợt dài. Chưa có benchmark P100 thực tế trong quá trình kiểm tra mã trên máy phát triển.

## Kết quả cần gửi lại

Gửi **toàn bộ thư mục** `A1/results/pilot_epoch1_batch16_seed42/` hoặc nén thành một tệp:

```bash
tar -czf A1_pilot_epoch1_batch16_seed42.tar.gz \
  -C A1/results pilot_epoch1_batch16_seed42
```

Chạy lệnh nén ở dòng riêng, kể cả khi chương trình trả mã lỗi: kết quả lỗi vẫn cần để lập kế hoạch. Nếu thư mục đã có, dùng tên mới cho lần chạy sau; không xóa/ghi đè số liệu đầu tiên.

Thông thường thư mục có 13 tệp, một cấp:

| Tệp | Nội dung |
| --- | --- |
| `metrics.json` | Cấu hình, môi trường, trạng thái tám lượt, lịch sử và thời gian/bộ nhớ từng pha, lỗi và kiểm chứng đóng băng |
| `summary.csv` | Bảng nhanh đủ tám cấu hình, kể cả lượt lỗi hoặc chưa chạy |
| `run_<cấu hình>.json` × 8 | Bản ghi riêng từng lượt; lưu ngay khi hoàn tất hoặc lỗi |
| `split_indices.json` | Chỉ số huấn luyện/xác thực trong tập gốc; không chứa ảnh |
| `run.log` | Diễn biến và chẩn đoán lỗi; thanh tiến độ trên terminal không được chép vào log |
| `README.md` | Ý nghĩa và giới hạn của số liệu |

Không cần gửi ảnh CIFAR-100, cache trọng số hoặc checkpoint. Chưa cần tự push kết quả thăm dò lên Git trước khi cùng đọc và đối chiếu.

Nếu có thể, kèm thông tin thực tế: GPU có dùng chung không, dataset ở SSD hay ổ mạng, có ngắt/kết nối lại phiên chạy không. Tên GPU, phiên bản thư viện và ảnh chụp `nvidia-smi` trước lượt đã được chương trình ghi nếu khả dụng.

## Cách đọc để ra quyết định

1. Xác nhận đủ tám lượt, đúng split, trọng số và chính sách đóng băng; tách lỗi tải trọng số/dữ liệu khỏi lỗi huấn luyện.
2. Xem bộ nhớ đỉnh và tốc độ theo từng chế độ. MiB cấp phát/đặt trước là số đo của tiến trình PyTorch, không đồng nhất tổng VRAM của mọi tiến trình. CPU ghi `null`, không giả lập số GPU.
3. Với lượt hoàn tất, ước tính sơ bộ chi phí huấn luyện nhiều vòng; tách thời gian tải/khởi tạo, xét độ bất định của vòng đầu. Nếu cần, chạy thêm vài vòng cho cấu hình đại diện trước khi chốt ngân sách.
4. Quyết định có thử batch 32/64 không, số vòng thực tế và phạm vi chọn tốc độ học. Chỉ chạy tăng batch sau khi đã xem bộ nhớ của các cấu hình nặng.
5. Chốt kế hoạch đủ tám cấu hình theo đề, ưu tiên các lượt ổn định và bằng chứng đầy đủ. Nhiều seed là bước bổ sung theo ngân sách, không mặc định 100 vòng hoặc ba seed chỉ vì có P100 16 GB.

Một vòng không xác nhận hội tụ, khả năng tổng quát hóa hoặc kiến trúc tốt nhất. Độ chính xác xác thực ở đây chủ yếu phục vụ kiểm tra đường đi dữ liệu/mô hình; không dùng để xếp hạng cuối.

## Kiểm tra mã không tải dữ liệu hoặc trọng số

```bash
python -m unittest A1.test_pilot -v
python A1/smoke_test.py
```

Bộ kiểm tra tự động dùng dữ liệu tổng hợp/CPU, có đánh dấu `synthetic_test` trong tệp tạm của nó. Kiểm tra mô hình thật dùng `weights=None` để kiểm tra hình dạng và số tham số, không phải xác minh hiệu năng pretrained. Lỗi OOM được giả lập để kiểm tra đường xử lý, không thay thế phép đo VRAM thật trên P100.

