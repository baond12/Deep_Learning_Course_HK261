# Kết quả E3 sequence v2

Nguồn nhập: `D:/master_hcmut/HK261/Deep_Learning/result_E3_sequence_v2`.
54 file kết quả ở cấp trên cùng được sao chép nguyên byte; không sao chép checkpoint vào Git. Tám checkpoint gốc vẫn nằm tại thư mục nguồn `checkpoints/`.

Đã hoàn tất 8 RNN × 100 epoch, batch 256, MNIST split seed 42. Hai baseline MLP/CNN chỉ dùng lại kết quả E1, không huấn luyện/test lại. Bảng E3 có 10 dòng nhưng chỉ 8 thực nghiệm mới.

Kiểm tra: 800 dòng history, best epoch theo validation, 80.000 dự đoán test, accuracy/F1/ma trận/summary khớp; 24 dòng order probe hợp lệ; hash split khớp E2 và hash nguồn khớp repo sau chuẩn hóa LF; baseline giữ nguyên E1. Hash history baseline khác CRLF/LF trên Windows/Linux, nội dung trùng. Checkpoint gốc nạp được, đúng kiến trúc/tham số/best epoch; không chạy lại toàn bộ MNIST test tại máy rà soát.

LSTM CNN-stem đạt test accuracy 99,37%, GRU CNN-stem 99,27%. Một seed chưa đủ kết luận ưu thế thống kê. Order probe chỉ đo độ nhạy thứ tự trên validation. Xem `E3/report.md` và `docs/e1-e3-report-protocol.md` để biết giới hạn so sánh.

`comparison.md` và các CSV/PNG do lượt chạy sinh ra giữ nguyên; nhận xét tổng hợp dùng tài liệu nêu trên. Không chạy ghi đè vào thư mục này.
