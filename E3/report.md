# E3 So sánh LSTM và GRU trên bốn biểu diễn chuỗi ảnh

## Mục tiêu và liên hệ với E1 E2

E3 xử lý chuỗi vị trí không gian của ảnh MNIST bằng LSTM/GRU. Bộ [kết quả đã đối chiếu](results/sequence_v2/README.md) có tám RNN; MLP/CNN dùng nguyên kết quả E1. Ba bài tổng cộng 19 cấu hình độc lập: 3 E1 + 8 E2 + 8 E3. Hai mô hình đối chứng không được tính thành hai lượt huấn luyện mới.

## Dữ liệu và điều kiện thực nghiệm

Cả ba bài dùng MNIST, ảnh xám 28×28 thuộc 10 lớp. Tập huấn luyện gốc được chia thành 54.000 ảnh huấn luyện và 6.000 ảnh xác thực với hạt giống chia dữ liệu 42; tập kiểm tra chính thức có 10.000 ảnh. Tiền xử lý gồm `ToTensor()` và `Normalize(0.1307, 0.3081)`, không tăng cường dữ liệu.

Mỗi cấu hình chạy đủ 100 lượt huấn luyện (epoch), kích thước lô 256, dùng Adam với tốc độ học 0,001 và hệ số suy giảm trọng số 0,0001. Vòng lặp PyTorch tự viết; không có lịch điều chỉnh tốc độ học hoặc dừng sớm. Trọng số được chọn theo độ chính xác xác thực cao nhất; nếu bằng nhau, giữ lượt sớm nhất. Tập kiểm tra dùng để đánh giá trọng số đã chọn, không dùng để chọn lại siêu tham số.

Độ chính xác được trình bày theo phần trăm; F1 là trung bình cộng F1 của 10 lớp, trong khoảng 0–1. Thời gian gồm huấn luyện và xác thực, không phải độ trễ suy luận. Mỗi cấu hình có một hạt giống huấn luyện; số liệu chưa cho biết độ ổn định qua nhiều lượt chạy.

Hạt giống huấn luyện 42 và bộ sinh thứ tự lô được đặt lại cho từng cấu hình. E3/E2 có hash danh sách chỉ số huấn luyện/xác thực trùng nhau. E1 dùng cùng hàm chia và hạt giống, nhưng cách tiêu thụ ngẫu nhiên khi huấn luyện khác.

## Biểu diễn và kiến trúc

| Biểu diễn | Số bước T | Chiều đặc trưng D | Đặc trưng mỗi bước |
|---|---:|---:|---|
| Hàng ảnh | 28 | 28 | Một hàng điểm ảnh thô |
| Ô ảnh rời | 16 | 49 | Ô 7×7, bước trượt 7 |
| Bộ trích đặc trưng tích chập (CNN-stem) | 49 | 64 | Vị trí trên bản đồ đặc trưng 7×7 |
| Ô ảnh chồng lấn | 64 | 49 | Ô 7×7, bước trượt 3 |

Thứ tự quét từ trái sang phải rồi trên xuống dưới; hàng ảnh quét từ trên xuống dưới. Không có CLS hoặc nhúng vị trí; hàng/ô giữ điểm ảnh thô, khác phép chiếu học được của E2. Bộ trích đặc trưng tích chập gồm Conv(1→32,3,pad1), ReLU, Pool2, Conv(32→64,3,pad1), ReLU, Pool2.

Mỗi mạng có một tầng LSTM hoặc GRU một chiều, trạng thái ẩn rộng 128. Đọc `h_n[-1]`, qua dropout 0,1 và Linear(128,10). Dropout nội bộ RNN bằng 0 khi chỉ một tầng. Khối LSTM có `4H(D+H+2)` tham số; GRU có `3H(D+H+2)`; phần phân loại thêm 1.290. Bộ trích tích chập thêm 18.816 tham số.

Mã nguồn: [kiến trúc](models.py), [huấn luyện](train.py), [đọc mô hình đối chứng E1](baselines.py), [tổng hợp so sánh](comparison.py). LSTM/GRU khác kiến trúc, không có yêu cầu khớp đầu ra như hai hiện thực cùng MSA ở E2.

## Kết quả tám RNN

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

GRU trừ LSTM có chênh độ chính xác +0,01; +0,16; −0,10; +0,14 điểm phần trăm cho hàng, ô rời, tích chập, ô chồng lấn. GRU ít tham số hơn ở cả bốn cặp nhưng thời gian ngắn hơn ở ba cặp, dài hơn ở tích chập. Xem [bảng LSTM–GRU](results/sequence_v2/comparison_rnn.csv).

Tích chập cho độ chính xác cao nhất trong từng họ ở lượt này: LSTM 99,37%, GRU 99,27%. Ô chồng lấn hơn ô rời 0,78/0,76 điểm phần trăm (LSTM/GRU), giữ cùng D và số tham số tương ứng nhưng đồng thời thay T, ngữ cảnh chồng lấn và chi phí.

## Hai mô hình đối chứng từ E1

| Nguồn E1 | Độ chính xác kiểm tra (%) | F1 trung bình các lớp | Tham số |
|---|---:|---:|---:|
| MLP | 98,63 | 0,986219 | 235.146 |
| CNN | 99,27 | 0,992631 | 421.642 |

LSTM tích chập hơn CNN E1 0,10 điểm phần trăm, tương ứng 10 ảnh trên 10.000 ảnh kiểm tra; một hạt giống chưa đủ khẳng định ưu thế thống kê. GRU tích chập bằng độ chính xác CNN E1 nhưng F1 khác. E1 không có dự đoán từng ảnh trong nguồn nhận được, nên không kết luận hai mô hình sai ở cùng ảnh.

Xem [bản chụp nguồn E1](results/sequence_v2/baselines_e1.json), [bảng tám RNN và hai mô hình đối chứng](results/sequence_v2/comparison_baselines.csv) và [bộ E1 gốc](../E1/results/epochs100_batch256_seed42/README.md). Không huấn luyện hoặc đánh giá lại hai mô hình đối chứng.

## Hội tụ và phân tích lỗi

Ở lượt 100, hàm mất mát huấn luyện/xác thực của LSTM ô rời khoảng 0,0052/0,1073; GRU ô rời 0,0016/0,0837. Khoảng cách cho thấy mức quá khớp trong lượt chạy. Lượt xác thực tốt nhất của bốn cặp lần lượt là 95/78, 93/76, 58/89, 47/94; tập kiểm tra dùng đúng trọng số đó.

Số lỗi LSTM/GRU là 107/106 (hàng), 180/164 (ô rời), 63/73 (tích chập), 102/88 (ô chồng lấn). LSTM hàng nhầm 5 thành 3 ở 11 ảnh, 4 thành 9 ở 10 ảnh; LSTM tích chập còn 7 lỗi 4 thành 9. [Danh mục minh chứng](results/sequence_v2/README.md#tệp-theo-từng-mô-hình) có đủ lịch sử, dự đoán, ma trận, đường cong và mẫu sai.

## Độ nhạy với thứ tự chuỗi

Phép kiểm tra giữ nguyên trọng số đã chọn, biến ảnh thành chuỗi rồi đảo hoặc dùng hoán vị cố định hạt giống 42. Chỉ đánh giá trên tập xác thực, không huấn luyện thêm và không dùng tập kiểm tra. Độ chính xác giảm từ khoảng 98,18–99,45% xuống 12,32–40,32% khi đảo, 15,23–32,40% khi xáo. Xem [24 kết quả kiểm tra thứ tự](results/sequence_v2/comparison_order_probe.csv).

Thứ tự mới khác phân phối khi huấn luyện; đặc trưng tích chập đã chứa ngữ cảnh trước khi đảo, còn ô chồng lấn chia sẻ điểm ảnh. Đây là độ nhạy thứ tự không gian, không chứng minh trí nhớ dài hạn hoặc quan hệ thời gian thực. MLP/CNN E1 không nhận cùng thao tác đảo chuỗi này.

## Kiểm chứng và giới hạn

Đã đối chiếu 800 dòng lịch sử, 80.000 dự đoán, độ chính xác/F1, ma trận, bảng tổng hợp và nguồn E1. Tám trọng số gốc nạp được, khớp kiến trúc/tham số/lượt tốt nhất; không chạy lại toàn bộ tập kiểm tra tại máy rà soát. Hash lịch sử E1 khác CRLF/LF giữa Windows/Linux nhưng nội dung khớp sau chuẩn hóa. Xem [biên bản kiểm tra](results/sequence_v2/validation_report.json).

E3 ghi cùng loại GPU Tesla P100-PCIE-16GB và môi trường thư viện như E2; chưa kiểm soát tải cạnh tranh. E1 thiếu thông tin GPU, nên thời gian đối chứng là số đo của lượt E1, không phải phép đo tốc độ đồng nhất. Một hạt giống và một bộ dữ liệu giới hạn khả năng khái quát kết luận.

Thực nghiệm đã hoàn tất. Bản báo cáo v02 được chốt tạm làm bản hiện hành ngày 16/10/2026, tổng hợp 19 cấu hình và 1.900 bản ghi huấn luyện. Bản Word và PDF v02 đã được đồng bộ; PDF gồm 31 trang và đã được đối chiếu với Word.
