# Kết quả thực nghiệm A1

Kết quả gộp từ `results_A1_ResNet` và `results_A1_ViT` bên ngoài repo, bổ sung sáu đoạn log do người học cung cấp. Hai thư mục nguồn được giữ nguyên tại máy; không tạo thư mục `source_runs`.

## Các file chính

- `summary.csv`: bảng kết quả đủ 8 cấu hình, theo thứ tự trong `A1/train.py`.
- `metrics.json`: thông tin chi tiết của 8 cấu hình, thời gian theo giây, nguồn từng chỉ số và nguyên văn sáu đoạn log được bổ sung trong `experiments.<experiment>.provenance.source_log`.
- `results_inventory.csv`: danh mục 8 cấu hình, kết quả xác thực tốt nhất từ history, nguồn dữ liệu, mức chính xác và tình trạng xác nhận cấu hình.
- 8 file `history_*.csv`, mỗi file 50 epoch; 8 file `confusion_matrix_*.png` và 8 file `misclassified_*.png`: giữ nguyên từng byte của nguồn.
- `learning_curves.png`: đường cong của đủ 8 cấu hình, dựng từ CSV. Loss NaN không được điền hoặc nội suy; đánh dấu epoch bắt đầu NaN.
- `analysis.md` và 4 file `bar_*.png`: phân tích và so sánh 8 cấu hình bằng `A1/analyze_results.py`.

## Nguồn và độ chính xác

`resnet18_pretrained_partial` và `vit_b16_scratch` giữ nguyên các giá trị số từ JSON/CSV gốc. Sáu cấu hình khác được khôi phục từ log do người học cung cấp: test loss, test accuracy và macro F1 có 4 chữ số thập phân; thời gian đến giây. Không giả lập thêm chữ số thập phân. Thời gian dạng giây của sáu cấu hình này được quy đổi từ giờ/phút/giây; hai cấu hình gốc giữ thời gian giây có phần lẻ.

Best epoch và best validation accuracy của cả sáu log bổ sung đã đối chiếu khớp với history. History chỉ có train/validation, không dùng để suy ra chỉ số test. Số tham số của hai cấu hình gốc lấy từ JSON. Số tham số của sáu cấu hình bổ sung là số dự kiến tính từ kiến trúc hiện tại trong `A1/models.py`, không phải phép đếm đã khôi phục từ log chạy.

Với 100 lớp: ResNet18 có tổng 11.227.812 tham số, head-only có (512+1)×100=51.300 tham số được huấn luyện. ViT-B/16 có tổng 85.875.556 tham số, head-only có (768+1)×100=76.900 tham số được huấn luyện; partial có 2×7.087.872+76.900=14.252.644. Mỗi khối encoder trong phép tính này dùng chiều nhúng 768 và chiều tầng ẩn FFN 3072. Các giá trị suy ra phụ thuộc việc mã đã chạy đúng với định nghĩa hiện tại.

## Cấu hình cần xác nhận thêm

Cấu hình chung trong JSON được giữ từ hai bản JSON nguồn: CIFAR-100, split 40.000/10.000/10.000, seed 42, 100 lớp, 50 epoch, batch size 16, learning rate 1e-4, weight decay 1e-4, CUDA. Sáu đoạn log mới chưa xác nhận riêng batch size, learning rate, weight decay, seed/split hoặc phiên bản mã chạy của sáu cấu hình. `training_metadata_scope` và danh mục kết quả ghi rõ giới hạn này; không coi chúng là đã được xác nhận cho tất cả các lượt chạy. Số 50 epoch được xác minh trực tiếp cho cả tám lịch sử.

Người học cho biết các thí nghiệm chạy song song. Chưa có tên GPU và thông tin phân bổ tài nguyên từng lượt; thời gian hiện có là thời gian trôi qua, không phải phép so sánh tốc độ được kiểm soát phần cứng. Theo mã hiện tại, thời gian gồm huấn luyện và xác thực, không phải độ trễ suy luận.

## Bất thường huấn luyện

- `vit_b16_scratch`: loss NaN từ epoch 11; trạng thái xác thực tốt nhất ở epoch 10 (0,3537), test accuracy từ JSON gốc là 0,3530.
- `vit_b16_pretrained_full`: loss NaN từ epoch 37; trạng thái xác thực tốt nhất ở epoch 3 (0,7873), test accuracy từ log bổ sung là 0,7864.

Theo mã huấn luyện, trạng thái tốt nhất được nạp lại trước đánh giá test. Vì vậy chỉ số test vẫn có thể hữu hạn dù các epoch sau phát sinh NaN. Không mô tả hai lượt này là huấn luyện ổn định trong cả 50 epoch; chưa xác định nguyên nhân NaN từ dữ liệu hiện có.

Chưa có dữ liệu nhiều seed, độ lệch chuẩn hoặc kiểm định thống kê. Dataset/checkpoint không nằm trong lần nhập kết quả này. Checkpoint của sáu log mới chưa được xác nhận; không tự gán trạng thái đã lưu hoặc không lưu.
