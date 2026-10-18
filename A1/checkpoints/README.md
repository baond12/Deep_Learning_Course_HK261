# A1 Checkpoints

Quy trình chính thức dùng `A1/run_protocol.py`, tự lưu checkpoint tốt nhất vào
`<output-dir>/checkpoints/<experiment>.pt` ngay khi validation accuracy cải thiện.
Đọc [hướng dẫn chạy](../RUN_PROTOCOL.md). Các tệp `.pt` bị Git ignore ở mọi thư mục;
giữ chúng trên server để chạy test. Chưa lưu optimizer/RNG để nối tiếp epoch.

Lệnh `train.py` bên dưới thuộc công cụ cũ, không thay thế quy trình đã duyệt.

Checkpoint files can be saved here with:

```bash
python A1/train.py --save-checkpoints ...
```

Model files such as `.pt`, `.pth`, `.ckpt`, and `.safetensors` are ignored by Git, so
they will not be pushed to GitHub. Keep large checkpoints on Colab, Google Drive, or
local storage.

