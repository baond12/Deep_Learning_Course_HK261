# CO5085 Học sâu và ứng dụng trong thị giác máy tính

Kho mã nguồn học phần **Học sâu và ứng dụng trong thị giác máy tính** - HK261
(2026-2027).

## Trang giới thiệu học phần

Trang GitHub Pages được đặt ở thư mục gốc:

- `index.html`: trang chủ nhóm, giảng viên, lịch nộp và liên kết các bài.
- `pages/exercises.html`: kết quả, phân tích và minh chứng E1, E2, E3.
- `pages/assignment1.html`: khung bài tập lớn A1.
- `pages/assignment2.html`: khung bài tập lớn A2.
- `assets/css/styles.css`: định dạng giao diện dùng chung.
- `assets/css/exercises-results.css`: định dạng bảng và danh mục kết quả E1–E3.
- `assets/img/course-roadmap.png`: ảnh nền cho trang chủ.

Khi đẩy lên GitHub, bật GitHub Pages từ nhánh `main`, chọn thư mục gốc (`/root`).

## Cấu trúc thư mục

```text
.
├── A1/                  # Mã nguồn, cấu hình, sổ tay thực nghiệm bài tập lớn 1
├── A2/                  # Mã nguồn, cấu hình, sổ tay thực nghiệm bài tập lớn 2
├── E1/                  # Mã nguồn và kết quả E1
├── E2/                  # Mã nguồn và kết quả E2
├── E3/                  # Mã nguồn và kết quả E3
├── Exercise/            # Đề bài gốc dạng PDF
├── assets/              # CSS và ảnh cho GitHub Pages
├── pages/               # Trang chi tiết từng hạng mục
└── reports/             # PDF báo cáo nộp cuối cùng
```

## Đường dẫn báo cáo

- `reports/exercises.pdf`: báo cáo chung E1-E3.
- `reports/assignment1.pdf`: báo cáo A1.
- `reports/assignment2.pdf`: báo cáo A2.

Nguồn kết quả E1/E2/E3, cách tái sử dụng mô hình đối chứng và quy tắc chốt báo cáo:
[Liên kết thực nghiệm và báo cáo](docs/e1-e3-report-protocol.md).
E1/E2/E3 đã có đủ 19 cấu hình độc lập, cùng 100 lượt huấn luyện và kích thước lô 256.
Số liệu đã đối chiếu; báo cáo chung đang được biên tập và nhóm rà soát.
PDF đã công bố là phiên bản trước, sẽ xuất lại từ bản Word được nhóm duyệt.

## Kết quả E1 E2 E3 dùng trong báo cáo

| Bài | Số liệu và minh chứng | Phân tích |
| --- | --- | --- |
| E1 | [Bộ kết quả E1](E1/results/epochs100_batch256_seed42/README.md) | [Đọc phân tích](E1/report.md) |
| E2 | [Bộ kết quả E2](E2/results/tokenization_msa_v2/README.md) | [Đọc phân tích](E2/report.md) |
| E3 | [Bộ kết quả E3](E3/results/sequence_v2/README.md) | [Đọc phân tích](E3/report.md) |

MNIST, 100 lượt huấn luyện, kích thước lô 256; tổng 19 cấu hình và 1.900 bản ghi.
Hai mô hình đối chứng MLP/CNN trong E3 giữ nguyên kết quả E1, không tính thêm lượt chạy.
Báo cáo đang được nhóm rà soát; PDF đã công bố là phiên bản trước.

## Hạn nộp chính

| Hạng mục | Hạn nộp |
| --- | --- |
| GitHub Pages khung trang | 19/10/2026 23:59 GMT+7 |
| A1.1 Dataset + protocol | 26/10/2026 23:59 GMT+7 |
| E1 Softmax / MLP / CNN | 02/11/2026 23:59 GMT+7 |
| E2 MSA + tokenization | 02/11/2026 23:59 GMT+7 |
| A2.1 Chủ đề + paper + dữ liệu | 02/11/2026 23:59 GMT+7 |
| E3 LSTM / GRU | 09/11/2026 23:59 GMT+7 |
| A1.2 So sánh cuối | 09/11/2026 23:59 GMT+7 |
| A2.2 Pipeline + phân tích paper | 16/11/2026 23:59 GMT+7 |
