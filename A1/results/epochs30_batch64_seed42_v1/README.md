# Kết quả A1 theo protocol đã duyệt

Giai đoạn train chỉ dùng train/validation. `summary.csv` ghi validation tốt nhất và cuối riêng biệt. `metrics.json`, `run_*.json`, `history_*.csv`, `split_indices.json` là nguồn đối chiếu. `checkpoints/` chứa mô hình tốt nhất, bị Git ignore; phải giữ lại để đánh giá test. Chưa có cơ chế nối tiếp epoch; khi gián đoạn, kết quả và checkpoint đã ghi vẫn được giữ. Đánh giá test bằng lệnh riêng sau khi tất cả cấu hình trong lượt hoàn tất. Thời gian fit có tính ghi checkpoint; không đồng nhất với thời gian GPU tính toán thuần.
