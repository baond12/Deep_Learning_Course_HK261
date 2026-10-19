# So sánh A1 theo quy trình đã chốt

Protocol `a1_cifar100_p100_v1`; 30 epoch, batch 64, seed huấn luyện 42; chọn checkpoint bằng validation.

| Cấu hình | Trạng thái | LR | Epoch tốt nhất | Validation tốt nhất | Test accuracy | Macro F1 | Tham số được học | Train + validation (giây) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| resnet18_scratch | Hoàn tất | 0.0001 | 14 | 55.78% | 55.01% | 0.5451 | 11227812 | 2432.8 |
| resnet18_pretrained_head | Hoàn tất | 0.0001 | 30 | 69.07% | 68.01% | 0.6792 | 51300 | 1268.3 |
| resnet18_pretrained_partial | Hoàn tất | 0.0001 | 8 | 77.39% | 77.61% | 0.7755 | 8445028 | 1252.5 |
| resnet18_pretrained_full | Hoàn tất | 0.0001 | 8 | 78.85% | 78.97% | 0.7893 | 11227812 | 2436.9 |
| vit_b16_scratch | Hoàn tất | 0.0001 | 24 | 41.89% | 41.46% | 0.4146 | 85875556 | 23981.5 |
| vit_b16_pretrained_head | Hoàn tất | 0.0001 | 30 | 82.12% | 81.29% | 0.8128 | 76900 | 8499.7 |
| vit_b16_pretrained_partial | Hoàn tất | 0.0001 | 20 | 83.92% | 83.05% | 0.8303 | 14252644 | 11029.3 |
| vit_b16_pretrained_full | Hoàn tất | 1e-05 | 24 | 86.78% | 86.46% | 0.8651 | 85875556 | 23971.5 |

Số liệu trên chỉ thuộc các cấu hình có trong thư mục lượt chạy này. Không gộp test cũ hoặc validation thăm dò vào bảng. Một seed và giới hạn epoch chưa chứng minh khác biệt có ý nghĩa thống kê hoặc mọi mô hình đã hội tụ. ViT full dùng LR khác; không quy mọi chênh lệch cho duy nhất phạm vi đóng băng.

Thời gian toàn fit có tính callback/ghi checkpoint; xem JSON để tách thời gian từng pha. Bộ nhớ trong CSV là số đo PyTorch allocated/reserved, không phải tổng VRAM GPU.
