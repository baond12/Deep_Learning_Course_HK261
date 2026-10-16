# E2 Kết quả thực nghiệm dùng trong báo cáo

Bộ kết quả tám cấu hình: hai cách hiện thực MSA nhân với bốn cách tạo chuỗi đặc trưng ảnh. Nguồn nhập: `tokenization_msa_v2`; 52 tệp gốc được giữ nguyên. Tám trường hợp kiểm chứng CPU/CUDA đều đạt yêu cầu.

Đọc [phân tích kết quả](../../report.md) và [thiết kế so sánh E1–E3](../../../docs/e1-e3-report-protocol.md).

## Điều kiện thực nghiệm

Cả ba bài dùng MNIST, ảnh xám 28×28 thuộc 10 lớp. Tập huấn luyện gốc được chia thành 54.000 ảnh huấn luyện và 6.000 ảnh xác thực với hạt giống chia dữ liệu 42; tập kiểm tra chính thức có 10.000 ảnh. Tiền xử lý gồm `ToTensor()` và `Normalize(0.1307, 0.3081)`, không tăng cường dữ liệu.

Mỗi cấu hình chạy đủ 100 lượt huấn luyện (epoch), kích thước lô 256, dùng Adam với tốc độ học 0,001 và hệ số suy giảm trọng số 0,0001. Vòng lặp PyTorch tự viết; không có lịch điều chỉnh tốc độ học hoặc dừng sớm. Trọng số được chọn theo độ chính xác xác thực cao nhất; nếu bằng nhau, giữ lượt sớm nhất. Tập kiểm tra dùng để đánh giá trọng số đã chọn, không dùng để chọn lại siêu tham số.

Độ chính xác được trình bày theo phần trăm; F1 là trung bình cộng F1 của 10 lớp, trong khoảng 0–1. Thời gian gồm huấn luyện và xác thực, không phải độ trễ suy luận. Mỗi cấu hình có một hạt giống huấn luyện; số liệu chưa cho biết độ ổn định qua nhiều lượt chạy.

## Bảng kết quả

| Mô hình | Tham số | Lượt tốt nhất | Độ chính xác xác thực (%) | Độ chính xác kiểm tra (%) | F1 trung bình các lớp | Hàm mất mát kiểm tra | Huấn luyện và xác thực (giây) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MSA tự viết · Ô ảnh rời | 38.602 | 98 | 97,48 | 97,59 | 0,975703 | 0,078682 | 738,8 |
| MSA PyTorch · Ô ảnh rời | 38.602 | 55 | 97,13 | 97,05 | 0,970280 | 0,096932 | 733,0 |
| MSA tự viết · Hàng ảnh | 38.026 | 83 | 98,05 | 98,04 | 0,980230 | 0,065533 | 750,6 |
| MSA PyTorch · Hàng ảnh | 38.026 | 97 | 98,15 | 98,24 | 0,982280 | 0,057587 | 751,3 |
| MSA tự viết · Bộ trích đặc trưng tích chập | 56.330 | 82 | 99,32 | 99,28 | 0,992767 | 0,024487 | 763,6 |
| MSA PyTorch · Bộ trích đặc trưng tích chập | 56.330 | 59 | 99,28 | 99,28 | 0,992781 | 0,024037 | 827,2 |
| MSA tự viết · Ô ảnh chồng lấn | 41.674 | 97 | 98,47 | 98,53 | 0,985199 | 0,050016 | 739,7 |
| MSA PyTorch · Ô ảnh chồng lấn | 41.674 | 82 | 98,45 | 98,33 | 0,983174 | 0,056096 | 737,4 |

## Tệp số liệu và kiểm chứng

- [Số liệu gốc](metrics.json)
- [Bảng tổng hợp](summary.csv)
- [Cấu hình lượt chạy](run_config.json)
- [Kiểm chứng MSA](verification.json)
- [So sánh hai hiện thực MSA](comparison_msa.csv)
- [So sánh cách biểu diễn](comparison_tokenization.csv)

## Tệp theo từng mô hình

| Mô hình | Lịch sử huấn luyện | Dự đoán trên tập kiểm tra | Ma trận nhầm lẫn | Mẫu dự đoán sai | Đường cong |
| --- | --- | --- | --- | --- | --- |
| `custom_patch` | [Lịch sử](history_custom_patch.csv) | [Dự đoán](predictions_custom_patch.csv) | [Ảnh](confusion_matrix_custom_patch.png) / [CSV](confusion_matrix_custom_patch.csv) | [Mẫu sai](misclassified_custom_patch.png) | [Đường cong](learning_curves_patch.png) |
| `pytorch_patch` | [Lịch sử](history_pytorch_patch.csv) | [Dự đoán](predictions_pytorch_patch.csv) | [Ảnh](confusion_matrix_pytorch_patch.png) / [CSV](confusion_matrix_pytorch_patch.csv) | [Mẫu sai](misclassified_pytorch_patch.png) | [Đường cong](learning_curves_patch.png) |
| `custom_row` | [Lịch sử](history_custom_row.csv) | [Dự đoán](predictions_custom_row.csv) | [Ảnh](confusion_matrix_custom_row.png) / [CSV](confusion_matrix_custom_row.csv) | [Mẫu sai](misclassified_custom_row.png) | [Đường cong](learning_curves_row.png) |
| `pytorch_row` | [Lịch sử](history_pytorch_row.csv) | [Dự đoán](predictions_pytorch_row.csv) | [Ảnh](confusion_matrix_pytorch_row.png) / [CSV](confusion_matrix_pytorch_row.csv) | [Mẫu sai](misclassified_pytorch_row.png) | [Đường cong](learning_curves_row.png) |
| `custom_cnn_stem` | [Lịch sử](history_custom_cnn_stem.csv) | [Dự đoán](predictions_custom_cnn_stem.csv) | [Ảnh](confusion_matrix_custom_cnn_stem.png) / [CSV](confusion_matrix_custom_cnn_stem.csv) | [Mẫu sai](misclassified_custom_cnn_stem.png) | [Đường cong](learning_curves_cnn_stem.png) |
| `pytorch_cnn_stem` | [Lịch sử](history_pytorch_cnn_stem.csv) | [Dự đoán](predictions_pytorch_cnn_stem.csv) | [Ảnh](confusion_matrix_pytorch_cnn_stem.png) / [CSV](confusion_matrix_pytorch_cnn_stem.csv) | [Mẫu sai](misclassified_pytorch_cnn_stem.png) | [Đường cong](learning_curves_cnn_stem.png) |
| `custom_overlap_patch` | [Lịch sử](history_custom_overlap_patch.csv) | [Dự đoán](predictions_custom_overlap_patch.csv) | [Ảnh](confusion_matrix_custom_overlap_patch.png) / [CSV](confusion_matrix_custom_overlap_patch.csv) | [Mẫu sai](misclassified_custom_overlap_patch.png) | [Đường cong](learning_curves_overlap_patch.png) |
| `pytorch_overlap_patch` | [Lịch sử](history_pytorch_overlap_patch.csv) | [Dự đoán](predictions_pytorch_overlap_patch.csv) | [Ảnh](confusion_matrix_pytorch_overlap_patch.png) / [CSV](confusion_matrix_pytorch_overlap_patch.csv) | [Mẫu sai](misclassified_pytorch_overlap_patch.png) | [Đường cong](learning_curves_overlap_patch.png) |

Các ma trận đặt nhãn thật theo hàng và nhãn dự đoán theo cột. Ảnh mẫu sai lưu tối đa 25 lỗi đầu theo thứ tự tập kiểm tra, dùng để minh họa định tính.

## Giới hạn và trạng thái báo cáo

Kiểm chứng hai hiện thực xét cùng trọng số và tắt ngẫu nhiên dropout; không đòi hai lượt huấn luyện có cùng độ chính xác. Bộ kết quả ghi Tesla P100-PCIE-16GB, PyTorch 2.2.1+cu118, CUDA 11.8. Chưa kiểm soát tải GPU cạnh tranh. Bộ trích đặc trưng tích chập bổ sung tham số; ô ảnh chồng lấn thay cả số phần tử chuỗi và nhúng vị trí, nên chưa tách tác động riêng từng yếu tố.

Thực nghiệm đã hoàn tất. Bản báo cáo v02 được chốt tạm làm bản hiện hành ngày 16/10/2026, tổng hợp 19 cấu hình và 1.900 bản ghi huấn luyện. Bản Word và PDF v02 đã được đồng bộ; PDF gồm 31 trang và đã được đối chiếu với Word.
