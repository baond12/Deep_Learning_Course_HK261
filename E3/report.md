# E3 Kết quả LSTM và GRU trên chuỗi ảnh

Nguồn: `E3/results/sequence_v2/`, 100 epoch, batch 256, split/training seed 42. Đã đối chiếu metrics, history, dự đoán, ma trận, nguồn baseline và metadata checkpoint. Các kết quả lịch sử 20 epoch/lô 16 tại thư mục cha được giữ riêng.

## 4.1 Biểu diễn ảnh và liên hệ với E1 E2

E3 dùng cùng MNIST và split như E1/E2, chuyển ảnh thành chuỗi theo bốn cách. Hàng ảnh và patch giữ pixel thô; CNN-stem học đặc trưng không gian rồi quét từng vị trí; patch chồng lấn tăng số bước bằng các cửa sổ giao nhau. Không thêm CLS hoặc nhúng vị trí. Thứ tự quét là từ trái sang phải rồi từ trên xuống dưới; với hàng ảnh là từ trên xuống dưới [4].

Bảng 8. Độ dài và chiều đặc trưng chuỗi E3

| Biểu diễn | Số bước T | Chiều D | Đặc trưng đầu vào |
| --- | --- | --- | --- |
| Hàng ảnh | 28 | 28 | 28 pixel mỗi hàng |
| Patch rời | 16 | 49 | 7×7 pixel, stride 7 |
| CNN-stem | 49 | 64 | Vị trí feature map 7×7 |
| Patch chồng lấn | 64 | 49 | 7×7 pixel, stride 3 |

Patch rời dùng kernel 7, stride 7; patch chồng lấn dùng kernel 7, stride 3, không padding và phủ đủ biên ảnh 28×28. CNN-stem gồm Conv(1→32,3,pad1) → ReLU → Pool2 → Conv(32→64,3,pad1) → ReLU → Pool2, tạo bản đồ 64×7×7. Khác E2, hàng/patch E3 không dùng projection lên 64 chiều; so sánh giữa hai bài phải tính cả khác biệt này [3], [4].

## 4.2 Kiến trúc hồi quy và cách đọc trạng thái cuối

Mỗi mô hình dùng nn.LSTM hoặc nn.GRU một tầng, một chiều, hidden size H = 128. Trạng thái ẩn cuối của lớp cuối h_n[-1] đi qua Dropout(0,1) và Linear(128,10) để tạo logits. Với một tầng, dropout nội bộ RNN bằng 0; dropout chỉ nằm ở head. MLP/CNN không được tạo mới hoặc huấn luyện lại: chương trình đọc nguyên metrics/history E1 và kiểm tra cấu hình trước khi chạy tám RNN [2], [4].

LSTM duy trì trạng thái ô nhớ cₜ và trạng thái ẩn hₜ: cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ gₜ; hₜ = oₜ ⊙ tanh(cₜ). GRU dùng cổng cập nhật và đặt lại để kết hợp trạng thái trước với trạng thái ứng viên, không có ô nhớ tách biệt. Hai kiến trúc khác nhau nên không đặt yêu cầu LSTM/GRU phải khớp đầu ra như hai hiện thực cùng MSA của E2 [4], [11].

Với hai vector bias theo cách đếm PyTorch, khối LSTM có 4H(D+H+2) tham số, GRU có 3H(D+H+2); head thêm 1.290. Hàng ảnh cho 82.186/61.962 tham số (LSTM/GRU); patch rời và chồng lấn cùng cho 92.938/70.026. CNN-stem thêm 18.816 tham số tích chập và dùng D = 64, tổng 119.434/94.602. GRU ít tham số hơn trong cùng biểu diễn, nhưng tốc độ còn phụ thuộc kernel và thiết bị [4].

## 4.3 Kết quả và đối chiếu baseline E1

Bảng 9. Tám RNN E3 với 100 epoch và kích thước lô 256

| Mô hình | Val tốt nhất | Accuracy test | F1 macro | Tham số | Train + val (s) | Best epoch |
| --- | --- | --- | --- | --- | --- | --- |
| LSTM · Hàng ảnh | 98,98% | 98,93% | 0,989242 | 82.186 | 723,7 | 95 |
| GRU · Hàng ảnh | 99,02% | 98,94% | 0,989358 | 61.962 | 699,9 | 78 |
| LSTM · Patch rời | 98,28% | 98,20% | 0,981864 | 92.938 | 703,9 | 93 |
| GRU · Patch rời | 98,18% | 98,36% | 0,983462 | 70.026 | 688,0 | 76 |
| LSTM · CNN-stem | 99,40% | 99,37% | 0,993709 | 119.434 | 743,1 | 58 |
| GRU · CNN-stem | 99,45% | 99,27% | 0,992663 | 94.602 | 756,5 | 89 |
| LSTM · Patch chồng lấn | 99,13% | 98,98% | 0,989735 | 92.938 | 755,0 | 47 |
| GRU · Patch chồng lấn | 98,93% | 99,12% | 0,991074 | 70.026 | 749,3 | 94 |

Bảng 10. Hai baseline dùng lại nguyên kết quả E1

| Nguồn E1 | Val tốt nhất | Accuracy test | F1 macro | Tham số | Train + val (s) | Best epoch |
| --- | --- | --- | --- | --- | --- | --- |
| MLP (tái sử dụng) | 98,40% | 98,63% | 0,986219 | 235.146 | 760,6 | 81 |
| CNN (tái sử dụng) | 99,32% | 99,27% | 0,992631 | 421.642 | 770,0 | 83 |

LSTM CNN-stem đạt 99,37%, hơn CNN E1 0,10 điểm phần trăm và MLP E1 0,74 điểm phần trăm; số tham số bằng khoảng 28,33% CNN E1. GRU CNN-stem đạt 99,27%, bằng accuracy CNN E1 nhưng F1 macro khác nhau. Cùng accuracy không chứng minh cùng tập lỗi. Các số E1 là kết quả đã có, không phải hai lượt train hoặc test mới của E3 [5].

Bảng 11. GRU trừ LSTM trên cùng cách biểu diễn

| Biểu diễn | Δ accuracy (đpt) | Tham số LSTM | Tham số GRU | Δ thời gian (s) |
| --- | --- | --- | --- | --- |
| Hàng ảnh | 0,01 | 82.186 | 61.962 | -23,8 |
| Patch rời | 0,16 | 92.938 | 70.026 | -15,9 |
| CNN-stem | -0,10 | 119.434 | 94.602 | 13,4 |
| Patch chồng lấn | 0,14 | 92.938 | 70.026 | -5,8 |

GRU hơn LSTM 0,01; 0,16; 0,14 điểm phần trăm tương ứng hàng ảnh, patch rời, patch chồng lấn; với CNN-stem, LSTM hơn GRU 0,10 điểm phần trăm. GRU ít tham số hơn ở cả bốn cặp, nhưng thời gian ngắn hơn ở ba cặp và dài hơn ở CNN-stem. Bảng chỉ mô tả một lượt; không khẳng định GRU hoặc LSTM luôn tốt hơn. CNN-stem tốt nhất trong từng họ, còn patch chồng lấn hơn patch rời 0,78/0,76 điểm phần trăm (LSTM/GRU) [5].

## 4.4 Diễn biến huấn luyện và xác thực

Hình 6. Đường cong E3: hàng ảnh và patch rời, mỗi nhóm gồm LSTM/GRU. Nét liền: huấn luyện; nét đứt: xác thực; 100 epoch [6].

Hình 7. Đường cong E3: CNN-stem và patch chồng lấn. Quy ước nét như Hình 6 [6].

Các lượt giảm loss mạnh ở giai đoạn đầu nhưng validation vẫn dao động về cuối. Epoch 100, LSTM patch có train loss 0,0052 và validation loss 0,1073; GRU patch tương ứng 0,0016 và 0,0837, cho thấy khoảng cách khái quát hóa trong lượt chạy. Với CNN-stem, validation loss cuối khoảng 0,0344/0,0331 (LSTM/GRU). Best epoch của các cặp hàng, patch, CNN-stem, chồng lấn lần lượt là 95/78, 93/76, 58/89, 47/94, nên checkpoint để test không mặc định là epoch 100 [5], [6].

## 4.5 Phân tích lỗi và độ nhạy thứ tự

Trên 10.000 ảnh test, số lỗi LSTM/GRU lần lượt là 107/106 (hàng), 180/164 (patch), 63/73 (CNN-stem), 102/88 (chồng lấn). LSTM hàng nhầm 5→3 ở 11 ảnh và 4→9 ở 10 ảnh; LSTM CNN-stem còn 7 lỗi 4→9, GRU CNN-stem còn 7 lỗi 9→4. Ma trận của đủ tám mô hình nằm ở Phụ lục A. Các thống kê xác nhận dạng nhầm lẫn, chưa xác định nguyên nhân từng lỗi [5].

Hình 8. Mười mẫu sai đầu của LSTM CNN-stem theo thứ tự test, trích từ tối đa 25 lỗi lưu bởi pipeline. true/pred là nhãn thật/dự đoán; hình không lấy mẫu ngẫu nhiên [4], [5].

Bảng 12. Accuracy validation khi giữ và thay thứ tự token

| Mô hình | Thứ tự gốc (%) | Đảo ngược (%) | Xáo cố định (%) |
| --- | --- | --- | --- |
| LSTM · Hàng ảnh | 98,98 | 40,32 | 20,40 |
| GRU · Hàng ảnh | 99,02 | 39,02 | 17,97 |
| LSTM · Patch rời | 98,28 | 18,00 | 30,05 |
| GRU · Patch rời | 98,18 | 13,20 | 31,85 |
| LSTM · CNN-stem | 99,40 | 23,20 | 31,20 |
| GRU · CNN-stem | 99,45 | 19,43 | 32,40 |
| LSTM · Patch chồng lấn | 99,13 | 15,53 | 25,07 |
| GRU · Patch chồng lấn | 98,93 | 12,32 | 15,23 |

Order probe giữ nguyên checkpoint đã chọn, biến ảnh thành token rồi đảo toàn bộ thứ tự hoặc dùng hoán vị cố định seed 42. Không huấn luyện thêm và không dùng test. Accuracy validation giảm từ khoảng 98,18–99,45% xuống 12,32–40,32% khi đảo và 15,23–32,40% khi xáo. Chuỗi raster theo không gian ảnh có ảnh hưởng rõ trong kiểm tra này [4], [5].

Tuy nhiên, thứ tự mới khác phân phối lúc train. Với CNN-stem, đặc trưng đã chứa ngữ cảnh cục bộ trước khi bị đảo; với patch chồng lấn, các token chia sẻ pixel. Vì vậy phép thử chỉ phản ánh độ nhạy thứ tự không gian của mô hình đã học, không tách được trí nhớ dài hạn và không chứng minh quan hệ thời gian thực. MLP/CNN E1 làm baseline chất lượng/chi phí, không nhận cùng thao tác đảo token này.

Các ký hiệu [2]–[6], [11] theo danh mục nguồn của `reports/exercises.docx`. Hình đã cập nhật trong Word; ảnh và CSV gốc giữ tại thư mục kết quả. Bản Word đang chờ nhóm review; PDF sẽ xuất sau khi nhóm duyệt.
