# Phân tích kết quả chính thức Assignment 1

Ngày đối chiếu: 19/10/2026. Lượt chạy `epochs30_batch64_seed42_v1`, protocol `a1_cifar100_p100_v1`.

## Kết luận trong phạm vi thực nghiệm

ViT-B/16 tiền huấn luyện, tinh chỉnh toàn bộ với learning rate 1e-5 đạt test accuracy 86,46% và macro F1 0,8651, cao nhất trong tám cấu hình đã chạy. ResNet18 tiền huấn luyện, tinh chỉnh toàn bộ đạt 78,97% và 0,7893, với thời gian toàn fit thấp hơn rõ rệt. Hai mô hình chỉ học đầu phân loại tiết kiệm tham số được cập nhật và bộ nhớ nhưng không đạt accuracy cao nhất. Kết quả là so sánh các công thức cụ thể trên một split/một seed, chưa chứng minh ưu thế phổ quát hoặc ý nghĩa thống kê.

## Nguồn và kiểm chứng

Đã tính lại accuracy, macro F1, cả tám ma trận nhầm lẫn và 800 bản ghi chỉ số từng lớp từ 80.000 dòng dự đoán. Mỗi cấu hình có cùng 10.000 chỉ số ảnh kiểm tra và cùng thứ tự nhãn thật; mỗi lớp có 100 ảnh. Đủ 240 bản ghi epoch hữu hạn; CSV lịch sử khớp JSON; epoch tốt nhất là lần đầu đạt validation accuracy lớn nhất. Hash protocol, bản ghi train/test, nguồn mã và cả tám checkpoint local đã khớp. Không chạy lại suy luận để tạo bản phân tích này. Test loss được đối chiếu giữa các bản ghi, không tính lại độc lập vì CSV chỉ lưu xác suất lớp dự đoán, không lưu xác suất của mọi lớp.

## Bảng chất lượng

| Cấu hình | LR | Epoch chọn | Validation tốt nhất | Validation cuối | Test accuracy | Macro F1 | Test loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| resnet18_scratch | 0.0001 | 14 | 55.78% | 53.38% | 55.01% | 0.5451 | 1.7004 |
| resnet18_pretrained_head | 0.0001 | 30 | 69.07% | 69.07% | 68.01% | 0.6792 | 1.1073 |
| resnet18_pretrained_partial | 0.0001 | 8 | 77.39% | 76.13% | 77.61% | 0.7755 | 0.8478 |
| resnet18_pretrained_full | 0.0001 | 8 | 78.85% | 76.25% | 78.97% | 0.7893 | 0.7875 |
| vit_b16_scratch | 0.0001 | 24 | 41.89% | 41.61% | 41.46% | 0.4146 | 3.1633 |
| vit_b16_pretrained_head | 0.0001 | 30 | 82.12% | 82.12% | 81.29% | 0.8128 | 0.6397 |
| vit_b16_pretrained_partial | 0.0001 | 20 | 83.92% | 83.61% | 83.05% | 0.8303 | 0.9169 |
| vit_b16_pretrained_full | 1e-05 | 24 | 86.78% | 86.71% | 86.46% | 0.8651 | 0.6475 |

Test dùng checkpoint chọn bằng validation, không dùng trạng thái epoch 30 một cách mặc định. Các tỷ lệ trong bảng là tỷ lệ phần trăm; chênh lệch được diễn đạt bằng điểm phần trăm.

## Tham số và chi phí

| Cấu hình | Tham số tổng | Tham số được học | Tỷ lệ được học | Train (phút) | Validation (phút) | Toàn fit (giờ) | Allocated (GiB) | Reserved (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| resnet18_scratch | 11,227,812 | 11,227,812 | 100.000% | 36.14 | 4.32 | 0.68 | 1.566 | 2.271 |
| resnet18_pretrained_head | 11,227,812 | 51,300 | 0.457% | 16.59 | 4.40 | 0.35 | 0.477 | 0.500 |
| resnet18_pretrained_partial | 11,227,812 | 8,445,028 | 75.215% | 16.48 | 4.32 | 0.35 | 0.572 | 0.885 |
| resnet18_pretrained_full | 11,227,812 | 11,227,812 | 100.000% | 36.18 | 4.35 | 0.68 | 1.565 | 2.217 |
| vit_b16_scratch | 85,875,556 | 85,875,556 | 100.000% | 371.12 | 28.19 | 6.66 | 8.632 | 9.002 |
| vit_b16_pretrained_head | 85,875,556 | 76,900 | 0.090% | 112.65 | 28.33 | 2.36 | 0.843 | 0.990 |
| vit_b16_pretrained_partial | 85,875,556 | 14,252,644 | 16.597% | 155.29 | 28.27 | 3.06 | 1.953 | 2.176 |
| vit_b16_pretrained_full | 85,875,556 | 85,875,556 | 100.000% | 370.98 | 28.21 | 6.66 | 8.632 | 9.002 |

Tổng thời gian toàn fit của tám cấu hình: **20.80 giờ**; thời gian lịch của tiến trình train: **20.80 giờ**. Fit tính callback/ghi checkpoint; số đo train/validation từng pha không bao gồm các phần này. Lượt chạy tuần tự trong một tiến trình trên Tesla P100 16 GB; ảnh chụp GPU trước lượt không chứng minh GPU được cách ly suốt hơn 20 giờ. Bộ nhớ là peak allocated/reserved của tiến trình PyTorch, không phải tổng VRAM của mọi tiến trình. Không cộng thời gian các lượt cũ chạy song song để diễn giải thành GPU-hour.

ViT full cao hơn ResNet full **7.49 điểm phần trăm**, nhưng toàn fit khoảng **9.84 lần**. ViT full/ResNet full có LR khác nhau và quy mô mô hình khác nhau, nên tỷ lệ này mô tả hai công thức trên máy đã đo, không cô lập ảnh hưởng kiến trúc.

## Tiền huấn luyện và phạm vi tinh chỉnh

### ResNet18

Full pretrained cao hơn scratch **23.96 điểm phần trăm**. Full cao hơn head **10.96 điểm phần trăm** và partial **1.36 điểm phần trăm**. Partial dùng khoảng **51.4%** thời gian fit của full; head dùng khoảng **52.0%**. Đóng băng giúp giảm tham số cập nhật và bộ nhớ gradient/optimizer, nhưng vẫn cần forward qua backbone; giảm trainable parameters không đồng nghĩa giảm tổng kích thước mô hình hoặc chi phí về không.

### ViT-B/16

Full pretrained cao hơn scratch **45.00 điểm phần trăm**. Full cao hơn head **5.17 điểm phần trăm** và partial **3.41 điểm phần trăm**. Partial dùng khoảng **46.0%** thời gian fit của full; head dùng khoảng **35.5%**. Đóng băng giúp giảm tham số cập nhật và bộ nhớ gradient/optimizer, nhưng vẫn cần forward qua backbone; giảm trainable parameters không đồng nghĩa giảm tổng kích thước mô hình hoặc chi phí về không.

ResNet partial và head có thời gian gần nhau; không suy ra partial luôn nhanh hơn head từ chênh lệch nhỏ trong một lượt. Trong ViT, full dùng LR 1e-5 còn partial/head dùng 1e-4; không gán mọi chênh lệch cho số block được mở. ViT scratch thấp hơn ResNet scratch trong ngân sách này; kết quả không chứng minh ViT không thể học từ đầu khi có dữ liệu/công thức/ngân sách khác.

## Đường cong và dấu hiệu quá khớp

| Cấu hình | Epoch loss validation nhỏ nhất | Loss nhỏ nhất | Loss validation cuối | Train accuracy cuối | Validation accuracy cuối | Khoảng cách cuối (điểm %) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| resnet18_scratch | 10 | 1.6532 | 2.2460 | 97.41% | 53.38% | 44.02 |
| resnet18_pretrained_head | 30 | 1.1119 | 1.1119 | 73.74% | 69.07% | 4.67 |
| resnet18_pretrained_partial | 4 | 0.7907 | 1.0594 | 99.56% | 76.13% | 23.43 |
| resnet18_pretrained_full | 5 | 0.7518 | 1.0233 | 99.09% | 76.25% | 22.84 |
| vit_b16_scratch | 11 | 2.3305 | 3.3213 | 95.69% | 41.61% | 54.07 |
| vit_b16_pretrained_head | 30 | 0.6197 | 0.6197 | 89.64% | 82.12% | 7.52 |
| vit_b16_pretrained_partial | 3 | 0.5755 | 0.9262 | 99.39% | 83.61% | 15.78 |
| vit_b16_pretrained_full | 5 | 0.5040 | 0.6566 | 99.75% | 86.71% | 13.04 |

Hai cấu hình head đạt validation accuracy tốt nhất ở epoch 30 và validation loss thấp nhất cũng ở epoch 30: chưa có bằng chứng chúng đã hội tụ. Scratch có khoảng cách train/validation lớn, nhất là ViT: train cuối 95,69% nhưng validation cuối 41,61%. ResNet scratch tốt nhất epoch 14, full và partial tốt nhất epoch 8; chạy dài hơn không bảo đảm tăng validation. ViT full có loss validation thấp nhất epoch 5 nhưng accuracy tốt nhất epoch 24; ViT partial có loss thấp nhất epoch 3 nhưng accuracy tốt nhất epoch 20. Accuracy chỉ xét lớp thắng, cross entropy còn nhạy với xác suất gán cho nhãn thật nên hai tiêu chí có thể chọn epoch khác nhau. Giữ tiêu chí chọn checkpoint đã khóa là accuracy; không đổi sau khi xem test.

Khoảng cách train/validation trên bảng là dấu hiệu mô tả, không phải ước lượng thuần của generalization gap: train được ghi trong chế độ huấn luyện với tăng cường ngẫu nhiên, validation trong chế độ đánh giá. Dù các lượt cũ từng có NaN, lượt mới có đủ 240 bản ghi hữu hạn; điều này không chứng minh mọi cấu hình dài hơn đều ổn định.

## Accuracy và độ tự tin khi sai

ViT head có test loss 0,6397, thấp hơn ViT full 0,6475, dù accuracy head 81,29% thấp hơn full 86,46%. ViT partial đạt accuracy 83,05% nhưng loss 0,9169. Không được viết mô hình accuracy cao nhất đồng thời tốt nhất ở mọi metric.

| Cấu hình | Tổng lỗi test | Lỗi có confidence ≥ 0,9 | Tỷ lệ trong các lỗi |
| --- | ---: | ---: | ---: |
| resnet18_scratch | 4499 | 223 | 4.96% |
| resnet18_pretrained_head | 3199 | 56 | 1.75% |
| resnet18_pretrained_partial | 2239 | 374 | 16.70% |
| resnet18_pretrained_full | 2103 | 370 | 17.59% |
| vit_b16_scratch | 5854 | 1018 | 17.39% |
| vit_b16_pretrained_head | 1871 | 127 | 6.79% |
| vit_b16_pretrained_partial | 1695 | 846 | 49.91% |
| vit_b16_pretrained_full | 1354 | 636 | 46.97% |

Confidence là softmax của lớp dự đoán. Đây là chẩn đoán bổ sung từ cùng dự đoán test, không phải thực nghiệm tuning hoặc phép đo hiệu chuẩn đầy đủ. Không khẳng định đã xác định nguyên nhân của loss cao; những lỗi rất tự tin phù hợp với khả năng cross entropy bị phạt mạnh khi mô hình gán ít xác suất cho nhãn thật.

## Lỗi theo lớp và cặp nhầm lẫn

### resnet18_scratch

Năm lớp có F1 thấp: otter 0.2065; seal 0.2525; lizard 0.2533; boy 0.2535; possum 0.2564.

Cặp lỗi có hướng: oak_tree → maple_tree: 34/100 ảnh của lớp thật; willow_tree → maple_tree: 28/100 ảnh của lớp thật; bed → couch: 27/100 ảnh của lớp thật; woman → girl: 21/100 ảnh của lớp thật; bus → streetcar: 21/100 ảnh của lớp thật.

### resnet18_pretrained_head

Năm lớp có F1 thấp: girl 0.4041; otter 0.4456; shrew 0.4520; seal 0.4590; boy 0.4600.

Cặp lỗi có hướng: maple_tree → oak_tree: 21/100 ảnh của lớp thật; girl → woman: 21/100 ảnh của lớp thật; pine_tree → oak_tree: 19/100 ảnh của lớp thật; oak_tree → maple_tree: 16/100 ảnh của lớp thật; bed → couch: 16/100 ảnh của lớp thật.

### resnet18_pretrained_partial

Năm lớp có F1 thấp: girl 0.4390; shrew 0.5514; boy 0.5672; otter 0.5775; possum 0.5829.

Cặp lỗi có hướng: plate → bowl: 21/100 ảnh của lớp thật; girl → woman: 21/100 ảnh của lớp thật; maple_tree → oak_tree: 17/100 ảnh của lớp thật; boy → man: 16/100 ảnh của lớp thật; worm → snake: 15/100 ảnh của lớp thật.

### resnet18_pretrained_full

Năm lớp có F1 thấp: girl 0.5128; seal 0.5422; otter 0.5500; oak_tree 0.5755; woman 0.5902.

Cặp lỗi có hướng: oak_tree → maple_tree: 32/100 ảnh của lớp thật; pine_tree → oak_tree: 20/100 ảnh của lớp thật; otter → seal: 19/100 ảnh của lớp thật; woman → man: 18/100 ảnh của lớp thật; boy → baby: 18/100 ảnh của lớp thật.

### vit_b16_scratch

Năm lớp có F1 thấp: girl 0.1379; bowl 0.1531; turtle 0.1556; seal 0.1685; snake 0.2200.

Cặp lỗi có hướng: willow_tree → maple_tree: 25/100 ảnh của lớp thật; bus → pickup_truck: 24/100 ảnh của lớp thật; oak_tree → maple_tree: 22/100 ảnh của lớp thật; girl → woman: 19/100 ảnh của lớp thật; rose → tulip: 18/100 ảnh của lớp thật.

### vit_b16_pretrained_head

Năm lớp có F1 thấp: willow_tree 0.5556; maple_tree 0.5758; boy 0.5911; mouse 0.6019; man 0.6161.

Cặp lỗi có hướng: maple_tree → oak_tree: 20/100 ảnh của lớp thật; willow_tree → maple_tree: 18/100 ảnh của lớp thật; oak_tree → maple_tree: 16/100 ảnh của lớp thật; girl → woman: 15/100 ảnh của lớp thật; woman → man: 14/100 ảnh của lớp thật.

### vit_b16_pretrained_partial

Năm lớp có F1 thấp: willow_tree 0.5912; seal 0.6321; boy 0.6326; girl 0.6378; shrew 0.6419.

Cặp lỗi có hướng: maple_tree → oak_tree: 23/100 ảnh của lớp thật; tulip → rose: 19/100 ảnh của lớp thật; willow_tree → oak_tree: 18/100 ảnh của lớp thật; snake → worm: 18/100 ảnh của lớp thật; pine_tree → oak_tree: 18/100 ảnh của lớp thật.

### vit_b16_pretrained_full

Năm lớp có F1 thấp: girl 0.6486; boy 0.6542; seal 0.6609; willow_tree 0.6698; oak_tree 0.6789.

Cặp lỗi có hướng: maple_tree → oak_tree: 20/100 ảnh của lớp thật; man → boy: 18/100 ảnh của lớp thật; girl → woman: 18/100 ảnh của lớp thật; plate → bowl: 14/100 ảnh của lớp thật; tulip → poppy: 13/100 ảnh của lớp thật.

Các cặp liên quan cây và người xuất hiện lặp lại ở nhiều mô hình. Khác biệt tinh tế và ảnh gốc 32×32 là giả thuyết giải thích hợp lý từ ngữ nghĩa/hình mẫu, chưa phải kết luận nhân quả vì chưa có ablation độ phân giải hoặc kiểm tra gán nhãn. Không tự quy lỗi cho dữ liệu sai nhãn. Hình mẫu sai là tối đa 25 lỗi đầu theo thứ tự loader đã cố định, không phải mẫu ngẫu nhiên hay đại diện thống kê.

## Đối chiếu đề và liên kết A1.1 với A1.2

Đủ CNN/Transformer, scratch/pretrained cho cả hai họ và head/partial/full cho từng họ; cùng dữ liệu/split, cùng preprocessing và ngân sách epoch/batch. Đã có validation/test accuracy, macro F1, tham số, thời gian, bộ nhớ, đường cong, ma trận và mẫu sai, EDA thật và dữ liệu dự đoán gốc. A1.1 đã công khai cách chọn LR bằng validation và A1.2 thực hiện đúng cấu hình đó. Tám cấu hình/240 epoch thuộc lượt chính; 26 lượt cấu hình thăm dò (66 epoch tổng cộng) chỉ là căn cứ chọn kế hoạch, không cộng vào bảng chính.

Tình trạng giấy phép CIFAR-100 chưa xác định rõ từ trang gốc/thẻ dữ liệu; phải ghi đúng, không tự gán MIT/CC hoặc coi công khai đồng nghĩa có giấy phép cụ thể. Một seed, khác LR của ViT full, khác quy mô/công thức tiền huấn luyện và chưa kiểm soát tải GPU suốt lượt là các giới hạn cần nêu. Cùng ImageNet-1K không đồng nghĩa ResNet và ViT có cùng công thức/ngân sách tiền huấn luyện. Test từng được quan sát trong các lượt cũ; không gọi đây là test hoàn toàn chưa từng thấy.

E1–E3 dùng MNIST, 19 cấu hình độc lập, 1.900 epoch, batch 256; A1 dùng CIFAR-100, tám cấu hình, 240 epoch, batch 64. Sự nhất quán nằm ở cách tách train/validation/test, truy nguồn, chọn checkpoint và diễn đạt giới hạn; không ép A1 dùng số epoch/batch của Exercise hoặc so accuracy trực tiếp giữa hai dataset.

## Trạng thái báo cáo và công bố

Đã hoàn tất thực nghiệm và tổng hợp số liệu; đang chuẩn bị bản Word A1 theo kết quả mới. Word/PDF A1 v01 hiện có là lượt cũ. Không đổi nhãn bản PDF cũ thành kết quả mới. Sau khi ChatGPT Plus tạo Word, xuất PDF từ đúng Word đó theo yêu cầu người học; kiểm tra tính đồng bộ rồi cập nhật đường dẫn báo cáo ổn định. Chưa xác nhận đã nộp bài, chưa thay nhóm xác nhận tác giả hoặc khai báo hỗ trợ AI.
