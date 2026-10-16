# E2 Kiểm chứng MSA và so sánh bốn cách tạo chuỗi ảnh

## Mục tiêu và nguồn kết quả

E2 có hai phép đối chiếu: MSA tự viết với MSA PyTorch trên cùng cách tạo chuỗi; bốn cách biểu diễn trong cùng hiện thực MSA. Bộ [kết quả đã đối chiếu](results/tokenization_msa_v2/README.md) gồm tám cấu hình, đủ tám trường hợp kiểm chứng số học CPU/CUDA đạt yêu cầu.

## Dữ liệu và điều kiện so sánh

Cả ba bài dùng MNIST, ảnh xám 28×28 thuộc 10 lớp. Tập huấn luyện gốc được chia thành 54.000 ảnh huấn luyện và 6.000 ảnh xác thực với hạt giống chia dữ liệu 42; tập kiểm tra chính thức có 10.000 ảnh. Tiền xử lý gồm `ToTensor()` và `Normalize(0.1307, 0.3081)`, không tăng cường dữ liệu.

Mỗi cấu hình chạy đủ 100 lượt huấn luyện (epoch), kích thước lô 256, dùng Adam với tốc độ học 0,001 và hệ số suy giảm trọng số 0,0001. Vòng lặp PyTorch tự viết; không có lịch điều chỉnh tốc độ học hoặc dừng sớm. Trọng số được chọn theo độ chính xác xác thực cao nhất; nếu bằng nhau, giữ lượt sớm nhất. Tập kiểm tra dùng để đánh giá trọng số đã chọn, không dùng để chọn lại siêu tham số.

Độ chính xác được trình bày theo phần trăm; F1 là trung bình cộng F1 của 10 lớp, trong khoảng 0–1. Thời gian gồm huấn luyện và xác thực, không phải độ trễ suy luận. Mỗi cấu hình có một hạt giống huấn luyện; số liệu chưa cho biết độ ổn định qua nhiều lượt chạy.

Hạt giống huấn luyện và bộ sinh thứ tự lô được đặt lại cho từng cấu hình; trong mỗi cặp MSA còn sao chép toàn bộ trọng số ban đầu. Mặt nạ dropout giữa các nhân tính toán không bị ép giống nhau, nên quỹ đạo huấn luyện có thể khác.

## Bốn cách biểu diễn và kiến trúc

| Biểu diễn | Cách thực hiện | Số phần tử ảnh | Tổng có CLS | Chiều đặc trưng |
|---|---|---:|---:|---:|
| Ô ảnh rời | Cửa sổ 7×7, bước trượt 7, lưới 4×4 | 16 | 17 | 64 |
| Hàng ảnh | Mỗi hàng 28 điểm ảnh qua Linear(28,64) | 28 | 29 | 64 |
| Bộ trích đặc trưng tích chập (CNN-stem) | Hai khối tích chập/ReLU/gộp, bản đồ 64×7×7 | 49 | 50 | 64 |
| Ô ảnh chồng lấn | Cửa sổ 7×7, bước trượt 3, lưới 8×8 | 64 | 65 | 64 |

Mỗi chuỗi thêm phần tử đại diện phân loại CLS và nhúng vị trí học được. Bộ mã hóa có một khối tiền chuẩn hóa, bốn đầu chú ý, tầng truyền thẳng 64→128→64 với GELU; dropout 0,1. Đầu phân loại lấy CLS qua LayerNorm và Linear(64,10).

Với mỗi đầu chú ý rộng 16: `A = softmax(QKᵀ/√16)`, đầu ra là `Dropout(A)V`; ghép các đầu rồi chiếu tuyến tính. Nhánh tự viết dùng Linear, đổi dạng tensor, nhân ma trận và softmax; không gọi mô-đun chú ý chuẩn. Nhánh đối chiếu dùng `nn.MultiheadAttention`.

Mã nguồn: [kiến trúc](models.py), [huấn luyện](train.py), [kiểm chứng MSA](verify_msa.py), [tổng hợp so sánh](comparison.py).

## Kiểm chứng trước huấn luyện

Mỗi cặp dùng cùng toàn bộ trọng số, tắt dropout khi đối chiếu đầu ra, hàm mất mát và gradient của ảnh/tham số. CPU float64 kiểm tra thêm một bước Adam; CUDA float32 kiểm tra trên thiết bị huấn luyện. Ngưỡng CPU `atol=1e-10`, `rtol=1e-8`; CUDA `atol=2e-5`, `rtol=2e-4`. Cả tám trường hợp đều đạt yêu cầu theo [biên bản kiểm chứng](results/tokenization_msa_v2/verification.json).

Kiểm chứng là bằng chứng số học trên các đầu vào đã xét, không phải chứng minh cho mọi đầu vào. Độ chính xác sau huấn luyện không thay thế kiểm chứng này và không bắt buộc khớp giữa hai hiện thực.

## Kết quả và hai phép đối chiếu

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

Theo thứ tự ô rời, hàng, tích chập, ô chồng lấn, chênh độ chính xác kiểm tra PyTorch trừ tự viết là −0,54; +0,20; 0,00; −0,20 điểm phần trăm. Tỷ số thời gian tương ứng là 0,9922; 1,0010; 1,0834; 0,9969. Không có hiện thực luôn nhanh hoặc chính xác hơn trong cả bốn cặp. Xem [bảng đối chiếu hai hiện thực](results/tokenization_msa_v2/comparison_msa.csv).

Trong từng hiện thực, bộ trích đặc trưng tích chập đạt độ chính xác cao nhất, cùng 99,28% và 72 lỗi. Ô chồng lấn hơn ô rời 0,94 điểm phần trăm ở bản tự viết, 1,28 ở PyTorch. Tuy nhiên tích chập bổ sung tham số, còn chồng lấn thay số phần tử và nhúng vị trí; chưa tách được ảnh hưởng nhân quả từng thành phần. Xem [bảng so sánh biểu diễn](results/tokenization_msa_v2/comparison_tokenization.csv).

## Hội tụ và lỗi dự đoán

Hàm mất mát huấn luyện/xác thực cuối của hai bản tích chập khoảng 0,0133/0,0325 và 0,0116/0,0331. Khoảng cách gợi ý mức quá khớp trong lượt chạy, không xác định nguyên nhân. Số lỗi tự viết/PyTorch lần lượt là 241/295, 196/176, 72/72, 147/167. Bản tự viết với ô rời nhầm 7 thành 2 và 3 thành 5 đều 15 ảnh; bản tích chập tự viết còn 8 lỗi 4 thành 9.

[Danh mục minh chứng từng mô hình](results/tokenization_msa_v2/README.md#tệp-theo-từng-mô-hình) liên kết đủ lịch sử, dự đoán kiểm tra, đường cong, ma trận và mẫu sai. Đã tính lại độ chính xác/F1 từ 80.000 dự đoán và đối chiếu các ma trận CSV.

## Liên hệ và giới hạn

E2 cùng dữ liệu với E1/E3 nhưng có phép chiếu đặc trưng, CLS và nhúng vị trí. E3 dùng điểm ảnh thô cho hàng/ô và đọc trạng thái ẩn cuối; không coi so E2/E3 là chỉ thay MSA bằng RNN. Chênh 99,28% của E2 với 99,27% của CNN E1 chỉ tương ứng một ảnh kiểm tra, chưa chứng minh ưu thế thống kê.

Môi trường ghi Python 3.10.13, PyTorch 2.2.1+cu118, torchvision 0.17.1+cu118, CUDA 11.8, Tesla P100-PCIE-16GB. Chưa kiểm soát tải cạnh tranh; thời gian không phải độ trễ suy luận. Một hạt giống chưa đủ đánh giá độ ổn định.

Thực nghiệm đã hoàn tất. Báo cáo chung đang được biên tập và nhóm rà soát; PDF đã công bố là phiên bản trước, chưa phản ánh đầy đủ bộ kết quả mới. Khi đối chiếu thực nghiệm, dùng các tệp số liệu được liên kết trong trang này. Bản cuối sẽ được xuất PDF sau khi nhóm duyệt.
