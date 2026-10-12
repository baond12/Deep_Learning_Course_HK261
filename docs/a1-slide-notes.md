# A1 slide notes

Các điểm từ slide được dùng để thiết kế A1:

- `03-cnn/09-Augmentation.pdf`, trang 5-8: chất lượng mô hình phụ thuộc data pipeline;
  train dùng để học tham số, validation để tune/chọn checkpoint, test chỉ đánh giá cuối.
- `03-cnn/09-Augmentation.pdf`, trang 9-11: tránh data leakage; augmentation mô phỏng
  biến thiên hợp lý của dữ liệu thật.
- `03-cnn/08-CNN-Architectures.pdf`, trang 5-8: CNN gồm stem, stages, downsampling,
  head; lựa chọn kiến trúc là trade-off giữa accuracy, latency, memory và deployment.
- `03-cnn/08-CNN-Architectures.pdf`: ResNet là họ CNN quan trọng, có residual learning
  và pretrained models, phù hợp làm CNN baseline/fine-tuning.
- `05-transformers/03-Model-Families-BERT-GPT-T5-ViT.pdf`, trang 29-32: ViT chuyển ảnh
  thành patch tokens, thêm class token/position embedding và dùng encoder-only
  Transformer cho image classification.
- `05-transformers/04-Practical-Finetuning-and-Efficiency.pdf`, trang 5-7: fine-tuning
  workflow gồm chọn checkpoint, head, data contract, optimizer và metric; các chế độ
  gồm head-only/linear probing, partial unfreezing và full fine-tuning.

Thiết kế A1:

- Dataset: full CIFAR-100 100 lớp, split cân bằng theo class với 40,000 train, 10,000 validation và 10,000 test samples.
- CNN: ResNet18.
- Transformer: ViT-B/16.
- Chế độ: from-scratch, pretrained freeze backbone/head-only, pretrained partial
  unfreeze, pretrained full fine-tune.
- Metric: accuracy, macro F1, total/trainable params, training time, confusion matrix.

