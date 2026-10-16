# E1 So sánh Softmax MLP và CNN trên MNIST

## Mục tiêu và nguồn kết quả

E1 đối chiếu bộ phân loại tuyến tính Softmax với MLP phi tuyến và CNN khai thác lân cận không gian. Bộ dùng trong báo cáo gồm đúng ba mô hình tại [thư mục kết quả đã đối chiếu](results/epochs100_batch256_seed42/README.md). Đây cũng là nguồn MLP/CNN đối chứng cho E3; không tạo hai lượt huấn luyện mới khi so sánh ở E3.

## Dữ liệu và huấn luyện

Cả ba bài dùng MNIST, ảnh xám 28×28 thuộc 10 lớp. Tập huấn luyện gốc được chia thành 54.000 ảnh huấn luyện và 6.000 ảnh xác thực với hạt giống chia dữ liệu 42; tập kiểm tra chính thức có 10.000 ảnh. Tiền xử lý gồm `ToTensor()` và `Normalize(0.1307, 0.3081)`, không tăng cường dữ liệu.

Mỗi cấu hình chạy đủ 100 lượt huấn luyện (epoch), kích thước lô 256, dùng Adam với tốc độ học 0,001 và hệ số suy giảm trọng số 0,0001. Vòng lặp PyTorch tự viết; không có lịch điều chỉnh tốc độ học hoặc dừng sớm. Trọng số được chọn theo độ chính xác xác thực cao nhất; nếu bằng nhau, giữ lượt sớm nhất. Tập kiểm tra dùng để đánh giá trọng số đã chọn, không dùng để chọn lại siêu tham số.

Độ chính xác được trình bày theo phần trăm; F1 là trung bình cộng F1 của 10 lớp, trong khoảng 0–1. Thời gian gồm huấn luyện và xác thực, không phải độ trễ suy luận. Mỗi cấu hình có một hạt giống huấn luyện; số liệu chưa cho biết độ ổn định qua nhiều lượt chạy.

## Kiến trúc và vòng lặp

- Softmax làm phẳng ảnh thành 784 phần tử, dùng `Linear(784,10)`, có 7.850 tham số.
- MLP dùng các tầng `784 → 256 → 128 → 10`, ReLU và dropout 0,2, tổng 235.146 tham số. Phép làm phẳng không loại bỏ giá trị điểm ảnh, nhưng kiến trúc không mã hóa trực tiếp quan hệ lân cận như CNN.
- CNN gồm hai khối tích chập 3×3 có 32/64 kênh, ReLU và gộp cực đại; phần phân loại `3136 → 128 → 10` có dropout 0,3. Tổng 421.642 tham số.

Các mô hình trả về điểm số chưa chuẩn hóa (logits); `CrossEntropyLoss` xử lý log-softmax bên trong. Khi dự đoán, lấy lớp có điểm số lớn nhất tương đương lấy lớp có xác suất softmax lớn nhất. Vòng lặp thực hiện xóa gradient, lan truyền thuận, tính hàm mất mát, `backward()` và `optimizer.step()`; sau mỗi lượt đánh giá trên tập xác thực.

Mã nguồn: [kiến trúc](models.py), [chương trình huấn luyện](train.py), [vòng lặp dùng chung](../common/engine.py), [chia dữ liệu](../common/data.py).

## Kết quả và nhận xét

| Mô hình | Tham số | Lượt tốt nhất | Độ chính xác xác thực (%) | Độ chính xác kiểm tra (%) | F1 trung bình các lớp | Hàm mất mát kiểm tra | Huấn luyện và xác thực (giây) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Softmax | 7.850 | 32 | 92,03 | 92,59 | 0,924782 | 0,271124 | 718,3 |
| MLP | 235.146 | 81 | 98,40 | 98,63 | 0,986219 | 0,060617 | 760,6 |
| CNN | 421.642 | 83 | 99,32 | 99,27 | 0,992631 | 0,025890 | 770,0 |

CNN đạt 99,27%, hơn MLP 0,64 điểm phần trăm và Softmax 6,68 điểm phần trăm. MLP hơn Softmax 6,04 điểm phần trăm. Kết quả phù hợp với lợi ích của đặc trưng phi tuyến và tính chất chia sẻ trọng số của tích chập, nhưng chưa phải thí nghiệm tách riêng nguyên nhân từng cải thiện.

Softmax có hàm mất mát huấn luyện giảm từ khoảng 0,539 xuống 0,228; độ chính xác xác thực tốt nhất là 92,03% tại lượt 32, còn lượt 100 là 91,65%. MLP và CNN đạt tốt nhất tại lượt 81/83. Hàm mất mát cuối trên huấn luyện/xác thực khoảng 0,0189/0,0733 với MLP và 0,0058/0,0357 với CNN. Tập kiểm tra dùng trọng số ở lượt 32, 81, 83, không mặc định lượt cuối.

## Phân tích lỗi và tệp minh chứng

Trên 10.000 ảnh kiểm tra, Softmax sai 741 ảnh, MLP sai 137, CNN sai 73. Ma trận nguồn ghi Softmax nhầm 5 thành 8 ở 37 ảnh và 5 thành 3 ở 36 ảnh; CNN còn 10 lỗi 9 thành 4 và 6 lỗi 5 thành 3. Các hình mô tả nhầm lẫn quan sát được, chưa xác định nguyên nhân từng dự đoán.

Xem [bảng tổng hợp](results/epochs100_batch256_seed42/summary.csv), [số liệu gốc](results/epochs100_batch256_seed42/metrics.json), [đường cong](results/epochs100_batch256_seed42/learning_curves.png) và [danh mục lịch sử, ma trận, mẫu sai theo mô hình](results/epochs100_batch256_seed42/README.md#tệp-theo-từng-mô-hình).

## Liên hệ E2 E3 và giới hạn

E2/E3 dùng cùng dữ liệu và ngân sách 100 lượt/lô 256. E3 giữ đúng MLP 98,63% và CNN 99,27% từ E1. Tổng ba bài có 19 cấu hình độc lập, không đếm lại hai mô hình đối chứng.

E1 chỉ ghi thiết bị CUDA, thiếu tên GPU/phiên bản thư viện và không kèm trọng số hoặc dự đoán từng ảnh. Vì vậy không tạo thêm liên kết tới tệp không có và không so tốc độ như phép đo cùng thiết bị. Một hạt giống chưa đủ kết luận ưu thế thống kê hoặc cấu hình đã tối ưu riêng cho từng mô hình.

Thực nghiệm đã hoàn tất. Báo cáo chung đang được biên tập và nhóm rà soát; PDF đã công bố là phiên bản trước, chưa phản ánh đầy đủ bộ kết quả mới. Khi đối chiếu thực nghiệm, dùng các tệp số liệu được liên kết trong trang này. Bản cuối sẽ được xuất PDF sau khi nhóm duyệt.
