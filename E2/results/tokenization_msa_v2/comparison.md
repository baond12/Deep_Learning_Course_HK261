# E2 — So sánh MSA và tokenization

Đã hoàn tất 8 thực nghiệm. Chỉ sử dụng các kết quả hoàn tất dưới đây.

## MSA tự viết và PyTorch trên cùng tokenization

- patch: custom đạt 97.59%, PyTorch đạt 97.05%; chênh lệch PyTorch − custom là -0.54 điểm phần trăm. Thời gian train + val lần lượt 738.83s và 733.04s.
- row: custom đạt 98.04%, PyTorch đạt 98.24%; chênh lệch PyTorch − custom là +0.20 điểm phần trăm. Thời gian train + val lần lượt 750.57s và 751.30s.
- cnn_stem: custom đạt 99.28%, PyTorch đạt 99.28%; chênh lệch PyTorch − custom là +0.00 điểm phần trăm. Thời gian train + val lần lượt 763.55s và 827.21s.
- overlap_patch: custom đạt 98.53%, PyTorch đạt 98.33%; chênh lệch PyTorch − custom là -0.20 điểm phần trăm. Thời gian train + val lần lượt 739.72s và 737.45s.

## Đối chiếu tokenization

Đọc comparison_tokenization.csv, so sánh bốn cách tokenize trong cùng backend (custom với custom, PyTorch với PyTorch), cùng split và cấu hình huấn luyện.

- custom/patch: 16 token ảnh, 38602 tham số; test accuracy 97.59%, train + val 738.83s.
- custom/row: 28 token ảnh, 38026 tham số; test accuracy 98.04%, train + val 750.57s. So với patch trong cùng backend: accuracy chênh +0.45 điểm phần trăm, thời gian chênh +11.74s.
- custom/cnn_stem: 49 token ảnh, 56330 tham số; test accuracy 99.28%, train + val 763.55s. So với patch trong cùng backend: accuracy chênh +1.69 điểm phần trăm, thời gian chênh +24.72s.
- custom/overlap_patch: 64 token ảnh, 41674 tham số; test accuracy 98.53%, train + val 739.72s. So với patch trong cùng backend: accuracy chênh +0.94 điểm phần trăm, thời gian chênh +0.89s.
- pytorch/patch: 16 token ảnh, 38602 tham số; test accuracy 97.05%, train + val 733.04s.
- pytorch/row: 28 token ảnh, 38026 tham số; test accuracy 98.24%, train + val 751.30s. So với patch trong cùng backend: accuracy chênh +1.19 điểm phần trăm, thời gian chênh +18.26s.
- pytorch/cnn_stem: 49 token ảnh, 56330 tham số; test accuracy 99.28%, train + val 827.21s. So với patch trong cùng backend: accuracy chênh +2.23 điểm phần trăm, thời gian chênh +94.17s.
- pytorch/overlap_patch: 64 token ảnh, 41674 tham số; test accuracy 98.33%, train + val 737.45s. So với patch trong cùng backend: accuracy chênh +1.28 điểm phần trăm, thời gian chênh +4.41s.

CNN-stem thêm phép học đặc trưng và tham số; overlap thay đổi độ dài chuỗi và positional embedding. Không quy toàn bộ khác biệt accuracy cho riêng số token.
Thời gian gồm train và validation, không gồm test, kiểm tra tương đương hoặc vẽ hình. Đây không phải latency suy luận; không diễn giải thời gian khi GPU đang chia sẻ tải như một benchmark phần cứng độc lập.

## Giới hạn

verification.json kiểm tra cùng trọng số và tắt dropout khi đối chiếu số học. Hai lượt train thật dùng dropout có cùng vị trí/xác suất nhưng không bảo đảm cùng mặt nạ hoặc kết quả bit-for-bit. Accuracy gần nhau không chứng minh MSA đúng; bằng chứng trực tiếp là kiểm tra đầu ra/gradient.
Một seed chưa đủ để kết luận khác biệt có ý nghĩa thống kê. Không dùng test để chọn lại siêu tham số. Kết quả lịch sử E2 có dropout khác không gộp vào bộ này.
