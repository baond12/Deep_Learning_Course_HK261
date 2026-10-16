# E1 Kết quả thực nghiệm dùng trong báo cáo

Bộ kết quả ba mô hình Softmax, MLP và CNN dùng trong báo cáo. Nguồn nhập: `results_E1_100epoch_256_batchsize`; các tệp số liệu và hình gốc được giữ nguyên. MLP/CNN cũng là hai mô hình đối chứng được tái sử dụng trong E3.

Đọc [phân tích kết quả](../../report.md) và [thiết kế so sánh E1–E3](../../../docs/e1-e3-report-protocol.md).

## Điều kiện thực nghiệm

Cả ba bài dùng MNIST, ảnh xám 28×28 thuộc 10 lớp. Tập huấn luyện gốc được chia thành 54.000 ảnh huấn luyện và 6.000 ảnh xác thực với hạt giống chia dữ liệu 42; tập kiểm tra chính thức có 10.000 ảnh. Tiền xử lý gồm `ToTensor()` và `Normalize(0.1307, 0.3081)`, không tăng cường dữ liệu.

Mỗi cấu hình chạy đủ 100 lượt huấn luyện (epoch), kích thước lô 256, dùng Adam với tốc độ học 0,001 và hệ số suy giảm trọng số 0,0001. Vòng lặp PyTorch tự viết; không có lịch điều chỉnh tốc độ học hoặc dừng sớm. Trọng số được chọn theo độ chính xác xác thực cao nhất; nếu bằng nhau, giữ lượt sớm nhất. Tập kiểm tra dùng để đánh giá trọng số đã chọn, không dùng để chọn lại siêu tham số.

Độ chính xác được trình bày theo phần trăm; F1 là trung bình cộng F1 của 10 lớp, trong khoảng 0–1. Thời gian gồm huấn luyện và xác thực, không phải độ trễ suy luận. Mỗi cấu hình có một hạt giống huấn luyện; số liệu chưa cho biết độ ổn định qua nhiều lượt chạy.

## Bảng kết quả

| Mô hình | Tham số | Lượt tốt nhất | Độ chính xác xác thực (%) | Độ chính xác kiểm tra (%) | F1 trung bình các lớp | Hàm mất mát kiểm tra | Huấn luyện và xác thực (giây) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Softmax | 7.850 | 32 | 92,03 | 92,59 | 0,924782 | 0,271124 | 718,3 |
| MLP | 235.146 | 81 | 98,40 | 98,63 | 0,986219 | 0,060617 | 760,6 |
| CNN | 421.642 | 83 | 99,32 | 99,27 | 0,992631 | 0,025890 | 770,0 |

## Tệp số liệu và kiểm chứng

- [Số liệu gốc](metrics.json)
- [Bảng tổng hợp](summary.csv)

## Tệp theo từng mô hình

| Mô hình | Lịch sử huấn luyện | Ma trận nhầm lẫn | Mẫu dự đoán sai | Đường cong |
| --- | --- | --- | --- | --- |
| `softmax` | [Lịch sử](history_softmax.csv) | [Ảnh](confusion_matrix_softmax.png) | [Mẫu sai](misclassified_softmax.png) | [Đường cong](learning_curves.png) |
| `mlp` | [Lịch sử](history_mlp.csv) | [Ảnh](confusion_matrix_mlp.png) | [Mẫu sai](misclassified_mlp.png) | [Đường cong](learning_curves.png) |
| `cnn` | [Lịch sử](history_cnn.csv) | [Ảnh](confusion_matrix_cnn.png) | [Mẫu sai](misclassified_cnn.png) | [Đường cong](learning_curves.png) |

Các ma trận đặt nhãn thật theo hàng và nhãn dự đoán theo cột. Ảnh mẫu sai lưu tối đa 25 lỗi đầu theo thứ tự tập kiểm tra, dùng để minh họa định tính.

## Giới hạn và trạng thái báo cáo

Bộ nguồn không có CSV dự đoán từng ảnh, ma trận dạng CSV hoặc trọng số đã lưu; chỉ liên kết những tệp thực sự được cung cấp. Chưa ghi tên GPU hay phiên bản PyTorch/CUDA, nên không xếp hạng tốc độ với E2/E3 như phép đo cùng thiết bị. Mẫu sai là các lỗi đầu theo thứ tự duyệt tập kiểm tra, không lấy ngẫu nhiên. E1 đặt hạt giống một lần trước vòng lặp mô hình, khác cách đặt lại hạt giống cho từng cấu hình ở E2/E3.

Thực nghiệm đã hoàn tất. Bản báo cáo v02 được chốt tạm làm bản hiện hành ngày 16/10/2026, tổng hợp 19 cấu hình và 1.900 bản ghi huấn luyện. Bản Word và PDF v02 đã được đồng bộ; PDF gồm 31 trang và đã được đối chiếu với Word.
