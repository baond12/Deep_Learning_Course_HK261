# E3 slide notes

Các điểm từ slide được dùng để thiết kế E3:

- `04-rnn/RNN-Summary.pdf`, trang 6: batch sequence thường có shape `(B, T, D)` với
  `B` là batch, `T` là số bước thời gian và `D` là số đặc trưng mỗi bước.
- `04-rnn/RNN-Summary.pdf`, trang 10-11: bài phân loại một nhãn cho cả chuỗi là
  many-to-one; có thể gắn classification head vào hidden state cuối hoặc pooling qua
  các hidden states.
- `04-rnn/RNN-Summary.pdf`, trang 20-22: training RNN dùng backpropagation through
  time; chuỗi dài có thể gặp vanishing/exploding gradients, nên gradient clipping là
  thực hành tốt.
- `04-rnn/RNN-Summary.pdf`, trang 24-29: LSTM dùng cell state và các gate để kiểm
  soát quên/ghi/đọc thông tin, giúp giữ phụ thuộc dài tốt hơn vanilla RNN.
- `04-rnn/RNN-Summary.pdf`, trang 30-34: GRU dùng update/reset gates, ít tham số hơn
  LSTM trong cấu hình tương đương và thường là baseline mạnh khi cần train nhanh.
- `04-rnn/RNN-Summary.pdf`, trang 37-39: PyTorch `nn.LSTM` và `nn.GRU` nhận input
  `(B, T, D)` khi `batch_first=True`; ví dụ many-to-one classifier có thể lấy
  `out[:, -1, :]` khi một chiều. Với hai chiều, cần ghép trạng thái ẩn cuối của
  hai hướng (`h_n` ở lớp cuối), không lấy riêng output tại timestep cuối.

Thiết kế E3 theo các điểm trên:

- Ảnh MNIST row sequence: `T=28`, `D=28`.
- Ảnh MNIST patch sequence `7 x 7`: `T=16`, `D=49`.
- CNN-stem: feature map 64×7×7, `T=49`, `D=64`, quét từng vị trí theo hàng.
- Patch chồng lấn 7×7, stride 3: `T=64`, `D=49`, không bỏ biên.
- Có tám mô hình ảnh: LSTM/GRU × hàng, patch, CNN-stem, patch chồng lấn.
- Baseline không hồi quy: tái sử dụng kết quả MLP/CNN từ bộ E1 100 epoch/lô 256,
  cùng split; giữ accuracy 98,63%/99,27% trong bảng, không train hoặc test lại.
- Không thêm CLS; dùng trạng thái ẩn cuối cho bài many-to-one. Row/patch giữ
  pixel thô như E3 cũ; khác E2 dùng projection học được trước attention.
- Tùy chọn đảo/xáo token sau tokenizer trên validation chỉ đo độ nhạy thứ tự,
  không chứng minh khả năng nhớ dài hạn. Pipeline hiện tại không bật gradient clipping.
- Sentiment analysis dùng text tokens độ dài thay đổi, padding theo batch, rồi dùng
  `pack_padded_sequence` và `pad_packed_sequence` để RNN bỏ qua padding.

