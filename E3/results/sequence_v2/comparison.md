# E3 — Kết quả của lượt chạy hiện tại

Huấn luyện tám LSTM/GRU ở E3; MLP/CNN lấy nguyên kết quả E1 đã có, không train hoặc test lại.
Baseline được kiểm tra cùng split và cấu hình epochs/batch/lr/weight decay; nguồn và SHA-256 nằm trong baselines_e1.json.

- E1 MLP: test accuracy 98.63%; nguồn `E1/results/epochs100_batch256_seed42/metrics.json`.
- E1 CNN: test accuracy 99.27%; nguồn `E1/results/epochs100_batch256_seed42/metrics.json`.
- row: GRU trừ LSTM = +0.01 điểm phần trăm accuracy test.
- patch: GRU trừ LSTM = +0.16 điểm phần trăm accuracy test.
- cnn_stem: GRU trừ LSTM = -0.10 điểm phần trăm accuracy test.
- overlap_patch: GRU trừ LSTM = +0.14 điểm phần trăm accuracy test.

Một seed chưa đủ kết luận độ ổn định. Tokenizer khác nhau còn thay đổi chiều đặc trưng và capacity; CNN-stem bổ sung CNN.
Ảnh là chuỗi vị trí không gian, không phải thời gian thực. Order probe chỉ đo độ nhạy thứ tự trên validation với checkpoint cố định, không chứng minh trí nhớ dài hạn.
E1 và E3 khác lượt chạy và cách tiêu thụ RNG; E1 thiếu metadata GPU. Thời gian E1 là số đo lịch sử, không đủ để xếp hạng tốc độ trực tiếp với E3.
Thời gian train + validation không phải latency suy luận; tải GPU cạnh tranh có thể ảnh hưởng.
