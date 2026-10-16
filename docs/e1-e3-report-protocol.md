# Thiết kế và đối chiếu thực nghiệm E1 E2 E3

## Nguồn số liệu dùng chung

Chỉ ba bộ sau được dùng cho bảng và hình của báo cáo hiện tại:

| Bài | Bộ kết quả | Phân tích | Số cấu hình |
| --- | --- | --- | --- |
| E1 | [Số liệu và tệp minh chứng](../E1/results/epochs100_batch256_seed42/README.md) | [Phân tích E1](../E1/report.md) | 3 |
| E2 | [Số liệu và tệp minh chứng](../E2/results/tokenization_msa_v2/README.md) | [Phân tích E2](../E2/report.md) | 8 |
| E3 | [Số liệu và tệp minh chứng](../E3/results/sequence_v2/README.md) | [Phân tích E3](../E3/report.md) | 8 |

Tổng 19 cấu hình độc lập, 1.900 bản ghi huấn luyện; mỗi cấu hình chạy một lượt với 100 lượt huấn luyện. E3 hiển thị thêm hai mô hình đối chứng E1 nhưng không tính thêm lượt chạy. Các kết quả ngoài ba bộ này được giữ để truy vết và không dùng trong bảng chung.

## Dữ liệu và quy tắc đánh giá

Cả ba bài dùng MNIST, ảnh xám 28×28 thuộc 10 lớp. Tập huấn luyện gốc được chia thành 54.000 ảnh huấn luyện và 6.000 ảnh xác thực với hạt giống chia dữ liệu 42; tập kiểm tra chính thức có 10.000 ảnh. Tiền xử lý gồm `ToTensor()` và `Normalize(0.1307, 0.3081)`, không tăng cường dữ liệu.

Mỗi cấu hình chạy đủ 100 lượt huấn luyện (epoch), kích thước lô 256, dùng Adam với tốc độ học 0,001 và hệ số suy giảm trọng số 0,0001. Vòng lặp PyTorch tự viết; không có lịch điều chỉnh tốc độ học hoặc dừng sớm. Trọng số được chọn theo độ chính xác xác thực cao nhất; nếu bằng nhau, giữ lượt sớm nhất. Tập kiểm tra dùng để đánh giá trọng số đã chọn, không dùng để chọn lại siêu tham số.

Độ chính xác được trình bày theo phần trăm; F1 là trung bình cộng F1 của 10 lớp, trong khoảng 0–1. Thời gian gồm huấn luyện và xác thực, không phải độ trễ suy luận. Mỗi cấu hình có một hạt giống huấn luyện; số liệu chưa cho biết độ ổn định qua nhiều lượt chạy.

## Mạch lập luận

E1 thiết lập mốc tuyến tính, phi tuyến và tích chập. E2 chuyển ảnh thành chuỗi đặc trưng, kiểm chứng hai hiện thực cùng MSA rồi so bốn cách biểu diễn. E3 xử lý chuỗi vị trí bằng LSTM/GRU, so từng cặp cùng biểu diễn và luôn đối chiếu với MLP/CNN E1. Ba bài là ba hướng mô hình hóa cùng nhiệm vụ, không phải các giai đoạn huấn luyện nối tiếp.

Trong mỗi bài, đi từ câu hỏi đến kiến trúc/điều kiện, bảng và đường cong, phân tích lỗi, rồi nhận xét. Phần tổng hợp xét chất lượng và chi phí của cả cấu hình, không quy mọi khác biệt riêng cho một cơ chế.

## Những khác biệt phải giữ khi so sánh

- Cùng hạt giống chia dữ liệu không đồng nghĩa cùng khởi tạo hoặc quỹ đạo tối ưu. E1 đặt hạt giống một lần; E2/E3 đặt lại cho từng cấu hình, dùng bộ sinh thứ tự lô riêng. E2 còn sao chép toàn bộ trọng số giữa hai thành viên mỗi cặp MSA.
- E2 có phép chiếu đặc trưng, CLS và nhúng vị trí. E3 không có CLS/nhúng vị trí, hàng và ô giữ điểm ảnh thô, phân loại từ trạng thái ẩn cuối. Không coi so E2/E3 là chỉ thay cơ chế chú ý bằng mạng hồi tiếp.
- Kiểm chứng MSA cùng trọng số/tắt dropout khác câu hỏi độ chính xác sau huấn luyện. LSTM/GRU là hai kiến trúc khác nhau, không yêu cầu chúng khớp đầu ra.
- Bộ trích đặc trưng tích chập bổ sung CNN và tham số. Ô chồng lấn thay số phần tử, ngữ cảnh và chi phí. Đây chưa phải phép tách tác động nhân quả từng yếu tố.
- E1 thiếu thông tin GPU. E2/E3 cùng loại Tesla P100 nhưng chưa kiểm soát tải cạnh tranh; thời gian chỉ mô tả lượt đã chạy, không phải độ trễ suy luận.
- Thử đảo/xáo chuỗi E3 đánh giá trên tập xác thực với trọng số cố định. Thứ tự mới khác phân phối khi học; chỉ kết luận độ nhạy thứ tự không gian, không chứng minh trí nhớ dài hạn.

MLP E1 đạt 98,63%, F1 0,986219, 235.146 tham số; CNN E1 đạt 99,27%, F1 0,992631, 421.642 tham số. E3 giữ nguyên hai kết quả này. Hai bản MSA tích chập E2 cùng đạt 99,28%; LSTM tích chập E3 đạt 99,37%. Các chênh lệch quan sát từ một hạt giống chưa chứng minh ưu thế có ý nghĩa thống kê.

## Kiểm tra đã thực hiện và bản báo cáo

E2: tám cấu hình, tám kiểm chứng CPU/CUDA đạt yêu cầu; 80.000 dự đoán khớp độ chính xác/F1 và ma trận. E3: tám cấu hình, 800 dòng lịch sử, 80.000 dự đoán, 24 dòng kiểm tra thứ tự, hash mã nguồn và danh sách chia dữ liệu đã đối chiếu. Tám trọng số E3 nạp được, khớp kiến trúc/số tham số/lượt tốt nhất; không chạy lại toàn bộ kiểm tra MNIST tại máy rà soát. Nội dung lịch sử đối chứng E1 khớp sau chuẩn hóa CRLF/LF.

Thực nghiệm đã hoàn tất. Bản báo cáo v02 được chốt tạm làm bản hiện hành ngày 16/10/2026, tổng hợp 19 cấu hình và 1.900 bản ghi huấn luyện. Bản Word và PDF v02 đã được đồng bộ; PDF gồm 31 trang và đã được đối chiếu với Word.

Báo cáo hiện hành v02: [đọc PDF](../reports/exercises.pdf) và [tải Word](../reports/exercises.docx). Hai tệp được giữ nguyên theo bản người học cung cấp, đã đối chiếu nội dung và số trang sau khi xuất PDF. Việc chốt tạm chưa xác nhận đã nộp bài; nhóm chịu trách nhiệm về thông tin thành viên, khai báo hỗ trợ AI và nội dung báo cáo.
