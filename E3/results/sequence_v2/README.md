# E3 Kết quả thực nghiệm dùng trong báo cáo

Bộ kết quả tám mạng LSTM/GRU trên bốn cách biểu diễn ảnh. Nguồn nhập: `result_E3_sequence_v2`; 54 tệp gốc được giữ nguyên. MLP/CNN chỉ đọc lại từ E1, không huấn luyện hoặc đánh giá lại trên tập kiểm tra.

Đọc [phân tích kết quả](../../report.md) và [thiết kế so sánh E1–E3](../../../docs/e1-e3-report-protocol.md).

## Điều kiện thực nghiệm

Cả ba bài dùng MNIST, ảnh xám 28×28 thuộc 10 lớp. Tập huấn luyện gốc được chia thành 54.000 ảnh huấn luyện và 6.000 ảnh xác thực với hạt giống chia dữ liệu 42; tập kiểm tra chính thức có 10.000 ảnh. Tiền xử lý gồm `ToTensor()` và `Normalize(0.1307, 0.3081)`, không tăng cường dữ liệu.

Mỗi cấu hình chạy đủ 100 lượt huấn luyện (epoch), kích thước lô 256, dùng Adam với tốc độ học 0,001 và hệ số suy giảm trọng số 0,0001. Vòng lặp PyTorch tự viết; không có lịch điều chỉnh tốc độ học hoặc dừng sớm. Trọng số được chọn theo độ chính xác xác thực cao nhất; nếu bằng nhau, giữ lượt sớm nhất. Tập kiểm tra dùng để đánh giá trọng số đã chọn, không dùng để chọn lại siêu tham số.

Độ chính xác được trình bày theo phần trăm; F1 là trung bình cộng F1 của 10 lớp, trong khoảng 0–1. Thời gian gồm huấn luyện và xác thực, không phải độ trễ suy luận. Mỗi cấu hình có một hạt giống huấn luyện; số liệu chưa cho biết độ ổn định qua nhiều lượt chạy.

## Bảng kết quả

| Mô hình | Tham số | Lượt tốt nhất | Độ chính xác xác thực (%) | Độ chính xác kiểm tra (%) | F1 trung bình các lớp | Hàm mất mát kiểm tra | Huấn luyện và xác thực (giây) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM · Hàng ảnh | 82.186 | 95 | 98,98 | 98,93 | 0,989242 | 0,043294 | 723,7 |
| GRU · Hàng ảnh | 61.962 | 78 | 99,02 | 98,94 | 0,989358 | 0,044952 | 699,9 |
| LSTM · Ô ảnh rời | 92.938 | 93 | 98,28 | 98,20 | 0,981864 | 0,066834 | 703,9 |
| GRU · Ô ảnh rời | 70.026 | 76 | 98,18 | 98,36 | 0,983462 | 0,069608 | 688,0 |
| LSTM · Bộ trích đặc trưng tích chập | 119.434 | 58 | 99,40 | 99,37 | 0,993709 | 0,026820 | 743,1 |
| GRU · Bộ trích đặc trưng tích chập | 94.602 | 89 | 99,45 | 99,27 | 0,992663 | 0,027444 | 756,5 |
| LSTM · Ô ảnh chồng lấn | 92.938 | 47 | 99,13 | 98,98 | 0,989735 | 0,042115 | 755,0 |
| GRU · Ô ảnh chồng lấn | 70.026 | 94 | 98,93 | 99,12 | 0,991074 | 0,038366 | 749,3 |

## Tệp số liệu và kiểm chứng

- [Số liệu gốc](metrics.json)
- [Bảng tổng hợp](summary.csv)
- [Cấu hình lượt chạy](run_config.json)
- [Nguồn hai mô hình đối chứng E1](baselines_e1.json)
- [So sánh LSTM với GRU](comparison_rnn.csv)
- [Tám RNN và hai mô hình đối chứng](comparison_baselines.csv)
- [Độ nhạy thứ tự trên tập xác thực](comparison_order_probe.csv)
- [Biên bản kiểm tra kết quả](validation_report.json)
- [Đường cong hai mô hình đối chứng E1](learning_curves_baselines_e1.png)

## Tệp theo từng mô hình

| Mô hình | Lịch sử huấn luyện | Dự đoán trên tập kiểm tra | Ma trận nhầm lẫn | Mẫu dự đoán sai | Đường cong |
| --- | --- | --- | --- | --- | --- |
| `lstm_row` | [Lịch sử](history_lstm_row.csv) | [Dự đoán](predictions_lstm_row.csv) | [Ảnh](confusion_matrix_lstm_row.png) / [CSV](confusion_matrix_lstm_row.csv) | [Mẫu sai](misclassified_lstm_row.png) | [Đường cong](learning_curves_row.png) |
| `gru_row` | [Lịch sử](history_gru_row.csv) | [Dự đoán](predictions_gru_row.csv) | [Ảnh](confusion_matrix_gru_row.png) / [CSV](confusion_matrix_gru_row.csv) | [Mẫu sai](misclassified_gru_row.png) | [Đường cong](learning_curves_row.png) |
| `lstm_patch` | [Lịch sử](history_lstm_patch.csv) | [Dự đoán](predictions_lstm_patch.csv) | [Ảnh](confusion_matrix_lstm_patch.png) / [CSV](confusion_matrix_lstm_patch.csv) | [Mẫu sai](misclassified_lstm_patch.png) | [Đường cong](learning_curves_patch.png) |
| `gru_patch` | [Lịch sử](history_gru_patch.csv) | [Dự đoán](predictions_gru_patch.csv) | [Ảnh](confusion_matrix_gru_patch.png) / [CSV](confusion_matrix_gru_patch.csv) | [Mẫu sai](misclassified_gru_patch.png) | [Đường cong](learning_curves_patch.png) |
| `lstm_cnn_stem` | [Lịch sử](history_lstm_cnn_stem.csv) | [Dự đoán](predictions_lstm_cnn_stem.csv) | [Ảnh](confusion_matrix_lstm_cnn_stem.png) / [CSV](confusion_matrix_lstm_cnn_stem.csv) | [Mẫu sai](misclassified_lstm_cnn_stem.png) | [Đường cong](learning_curves_cnn_stem.png) |
| `gru_cnn_stem` | [Lịch sử](history_gru_cnn_stem.csv) | [Dự đoán](predictions_gru_cnn_stem.csv) | [Ảnh](confusion_matrix_gru_cnn_stem.png) / [CSV](confusion_matrix_gru_cnn_stem.csv) | [Mẫu sai](misclassified_gru_cnn_stem.png) | [Đường cong](learning_curves_cnn_stem.png) |
| `lstm_overlap_patch` | [Lịch sử](history_lstm_overlap_patch.csv) | [Dự đoán](predictions_lstm_overlap_patch.csv) | [Ảnh](confusion_matrix_lstm_overlap_patch.png) / [CSV](confusion_matrix_lstm_overlap_patch.csv) | [Mẫu sai](misclassified_lstm_overlap_patch.png) | [Đường cong](learning_curves_overlap_patch.png) |
| `gru_overlap_patch` | [Lịch sử](history_gru_overlap_patch.csv) | [Dự đoán](predictions_gru_overlap_patch.csv) | [Ảnh](confusion_matrix_gru_overlap_patch.png) / [CSV](confusion_matrix_gru_overlap_patch.csv) | [Mẫu sai](misclassified_gru_overlap_patch.png) | [Đường cong](learning_curves_overlap_patch.png) |

Các ma trận đặt nhãn thật theo hàng và nhãn dự đoán theo cột. Ảnh mẫu sai lưu tối đa 25 lỗi đầu theo thứ tự tập kiểm tra, dùng để minh họa định tính.

## Giới hạn và trạng thái báo cáo

Đã đối chiếu 800 bản ghi lịch sử, 80.000 dự đoán kiểm tra và các ma trận; hai mô hình đối chứng giữ nguyên E1. Tám trọng số gốc nạp được và khớp kiến trúc, số tham số, lượt tốt nhất; không chạy lại toàn bộ kiểm tra MNIST tại máy rà soát. Hash lịch sử E1 giữa Windows/Linux khớp sau chuẩn hóa CRLF/LF. Thử đảo/xáo chuỗi chỉ đo độ nhạy thứ tự không gian trên tập xác thực, không chứng minh trí nhớ dài hạn. Thông tin GPU cùng loại E2 chưa bảo đảm điều kiện đo thời gian giống nhau.

Thực nghiệm đã hoàn tất. Báo cáo chung đang được biên tập và nhóm rà soát; PDF đã công bố là phiên bản trước, chưa phản ánh đầy đủ bộ kết quả mới. Khi đối chiếu thực nghiệm, dùng các tệp số liệu được liên kết trong trang này. Bản cuối sẽ được xuất PDF sau khi nhóm duyệt.
