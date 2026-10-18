# Phân tích thăm dò A1 và đề xuất kế hoạch tiếp theo

Ngày đối chiếu: 18/10/2026. Trạng thái: **đã kết thúc thăm dò; người học duyệt kế hoạch chính sau mục 8**: batch 64, FP32, 30 epoch/seed 42, ViT full LR `1e-5`, bảy cấu hình khác LR `1e-4`. Các đoạn “đề xuất/chưa duyệt” bên dưới ghi lại quá trình ra quyết định trước đó. Bản thực thi hiện tại ở [protocol đã chốt](../A1/protocol-a1.1.md), [JSON cấu hình](../A1/protocol-a1.1.json) và [hướng dẫn chạy](../A1/RUN_PROTOCOL.md). Chưa có số liệu lượt 30 epoch.

## 1. Nguồn và phạm vi kiểm tra

Đọc ba thư mục người học tải về tại `D:\master_hcmut\HK261\Deep_Learning\`:

- `pilot_epoch1_batch16_seed42`
- `pilot_epoch1_batch64_seed42`
- `pilot_epoch5_batch64_seed42` — nguồn chính của phân tích này.

Thư mục năm epoch có đủ 13 tệp, tám cấu hình hoàn tất và 40 bản ghi epoch. Đã đối chiếu tám `run_*.json` với `metrics.json`, các cột số liệu chính của `summary.csv` với bản ghi riêng, lịch sử với số đo từng pha và epoch tốt nhất với giá trị validation lớn nhất. Không có giá trị loss/accuracy không hữu hạn trong 40 bản ghi này.

Chỉ số train có 40.000 phần tử riêng biệt, validation có 10.000 phần tử riêng biệt; hai tập không giao nhau và hợp đúng toàn bộ chỉ số 0–49.999. SHA-256 của chỉ số chia khớp cả ba lượt:

`39684f5ef7a60f4cf3da5897931830b02da9723be21241a882b93fa1d0bda956`

Sáu mã băm nguồn được lưu trong kết quả khớp mã hiện tại sau khi chuẩn hóa xuống dòng LF: `A1/pilot.py`, `A1/engine.py`, `A1/models.py`, `A1/data.py`, `common/utils.py`, `requirements.txt`. Commit ghi nhận: `1ade7324ca18b7ed80ff268eb8d556fb6a467f3f`.

Mỗi epoch ghi đủ 40.000 ảnh/625 batch train và 10.000 ảnh/157 batch validation. Đã kiểm tra sự nhất quán của các tệp chỉ số và metadata; chưa đọc lại ảnh/nhãn CIFAR-100 trên server để kiểm chứng độc lập nội dung hoặc số ảnh từng lớp.

Metadata xác nhận dữ liệu thật, không nạp và không đánh giá test trong ba lượt thăm dò. Bốn cấu hình head/partial có hash trạng thái đóng băng trước/sau giống nhau; tám cấu hình đều ghi nhận phần được học đã thay đổi. Scratch/full không có phần đóng băng, nên không tính phép so sánh trạng thái rỗng là một kiểm chứng đóng băng bổ sung. Sáu cấu hình pretrained ghi trọng số `IMAGENET1K_V1`.

Bản sao nguyên trạng của cả bốn bộ thăm dò, có đối chiếu hash từng tệp, đã được
lưu tại [kho bằng chứng A1.1](../A1/results/pilot_evidence_v1/README.md). Phần dưới
ghi lại phân tích từng bước; mục 9 nêu trạng thái triển khai sau khi người học duyệt.

## 2. Điều kiện chạy

- GPU Tesla P100-PCIE-16GB; CUDA_VISIBLE_DEVICES=1.
- PyTorch 2.2.1+cu118, torchvision 0.17.1+cu118, Python 3.10.13.
- Batch 64; seed chia 42; seed huấn luyện 42; hai worker.
- Adam, learning rate 0,0001; weight decay 0,0001; FP32, không AMP.
- Ảnh RGB được resize 224×224; train có lật ngang ngẫu nhiên; chuẩn hóa ImageNet. Chưa có random crop, scheduler hoặc warmup.
- Tám cấu hình chạy tuần tự trong một tiến trình. Ảnh chụp trước lượt cho thấy GPU được chọn dùng 3 MiB; không đủ chứng minh GPU dành riêng suốt hơn ba giờ.

Các phần mở để học giữ nguyên theo mã: ResNet partial mở `layer4` và `fc`; ViT partial mở hai block encoder cuối và `heads`. Phần đóng băng được đưa về chế độ đánh giá, bao gồm thống kê BatchNorm.

## 3. Kết quả năm epoch

Accuracy dưới đây là **validation**, không phải test. Thời gian là toàn bộ quá trình train và validation năm epoch của từng cấu hình, có chi phí lưu trạng thái tốt nhất trong RAM; chưa gồm khởi tạo mô hình hoặc đánh giá test.

| Cấu hình | Validation tốt nhất | Epoch tốt nhất | Validation cuối | Thời gian 5 epoch | Tham số được học |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet18 từ đầu | 45,60% | 5 | 45,60% | 6,77 phút | 11.227.812 |
| ResNet18 chỉ head | 62,07% | 5 | 62,07% | 3,58 phút | 51.300 |
| ResNet18 một phần | 77,36% | 5 | 77,36% | 3,41 phút | 8.445.028 |
| ResNet18 toàn bộ | 78,23% | 4 | 78,18% | 6,75 phút | 11.227.812 |
| ViT-B/16 từ đầu | 31,72% | 5 | 31,72% | 66,62 phút | 85.875.556 |
| ViT-B/16 chỉ head | 78,60% | 5 | 78,60% | 23,51 phút | 76.900 |
| ViT-B/16 một phần | 83,90% | 5 | 83,90% | 30,63 phút | 14.252.644 |
| ViT-B/16 toàn bộ | 83,50% | 2 | 81,36% | 66,59 phút | 85.875.556 |

**Lưu ý khi tổng hợp:** `summary.csv` hiện lưu `val_accuracy` của epoch cuối. Muốn lấy kết quả tốt nhất phải dùng `best_val_accuracy` và `best_epoch` trong JSON hoặc tính lại từ lịch sử. Không gán 81,36% là validation tốt nhất của ViT full.

Tổng thời gian thực tế: **3 giờ 28 phút 10 giây**. Tổng thời gian fit của tám cấu hình: 12.470,48 giây. Thời gian train+validation của từng epoch ViT scratch khoảng 798,56–799,74 giây; ViT full khoảng 798,65–799,19 giây. Đây là cơ sở ước lượng tốt hơn lượt một epoch, vẫn không phải cam kết thời gian tương lai.

Bộ nhớ PyTorch đỉnh lớn nhất: cấp phát 8.839,48 MiB (8,63 GiB), đặt trước 9.218 MiB (9,00 GiB). Không đồng nhất số này với tổng VRAM của GPU hay lượng bộ nhớ có thể tăng batch an toàn trong mọi điều kiện.

## 4. Những điều có thể rút ra

### Huấn luyện từ đầu chưa đủ dài

ResNet scratch tăng validation từ 20,57% lên 45,60%; ViT scratch từ 17,16% lên 31,72%. Cả hai vẫn tốt hơn ở epoch cuối; chưa có căn cứ gọi chúng đã hội tụ hoặc kết luận ViT không thể học từ đầu. Khoảng cách pretrained/scratch hiện thể hiện lợi ích khởi tạo trong ngân sách năm epoch và công thức đang dùng.

### Fine-tune toàn bộ ViT có dấu hiệu quá khớp sớm

ViT full đạt 83,50% tại epoch 2, sau đó giảm xuống 81,36%; train accuracy vẫn tăng lên 94,11%. Validation loss tăng từ 0,5664 tại epoch 2 lên 0,7246 tại epoch 5. Mẫu hình này phù hợp với quá khớp trong lượt hiện tại. Chưa xác định riêng nguyên nhân do learning rate, augmentation hay yếu tố khác; không gọi đó là bằng chứng quên tri thức tiền huấn luyện.

ViT partial đạt 83,90%, với chi phí khoảng 46% ViT full. Tuy nhiên, validation loss thấp nhất tại epoch 3 rồi tăng trong khi accuracy vẫn tăng nhẹ. Cần theo dõi cả hai đại lượng; chưa đủ căn cứ nói partial không quá khớp. Chênh lệch 0,40 điểm phần trăm so với full trên một seed chưa chứng minh ưu thế thống kê.

ResNet full tốt hơn partial 0,87 điểm phần trăm theo validation tốt nhất, với thời gian gần gấp đôi. Tốc độ head/partial của ResNet gần nhau, thậm chí partial hơi nhanh hơn trong lượt này; không suy diễn một quy luật tốc độ từ chênh lệch nhỏ đó.

### Batch 64 phù hợp để chốt quy mô

Lượt một epoch batch 64 giảm tổng thời gian các pha khoảng 10,1% so với batch 16, trong khi bộ nhớ đặt trước lớn nhất tăng từ khoảng 3,13 lên 9,00 GiB. Không có bằng chứng tăng batch bốn lần làm ViT nhanh bốn lần hoặc cải thiện accuracy. Batch 64 đã chạy được năm epoch cho mọi cấu hình; chưa có lý do cần tiếp tục khảo sát batch 128/256 để đáp ứng đề bài.

### Tái lập có giới hạn đã quan sát được

Năm cấu hình khớp hoàn toàn bốn số liệu train/validation của epoch đầu với lượt một epoch batch 64. Hai cấu hình ViT khác có sai khác loss rất nhỏ nhưng accuracy giữ nguyên. ViT full có validation epoch đầu 81,23% thay vì 81,44% (lệch −0,21 điểm phần trăm). Vì vậy, cùng seed/split/source chưa cho phép khẳng định các lượt chạy đồng nhất từng bit. Chưa xác định nguyên nhân của sai khác này.

## 5. Đối chiếu mục tiêu môn học

| Mục tiêu | Đã có bằng chứng | Còn phải hoàn tất |
| --- | --- | --- |
| CIFAR-100 và protocol dữ liệu | 100 lớp được khai báo; split 40.000/10.000 nhất quán; đường nạp và augmentation chạy được | EDA từ dữ liệu thật; mô tả nguồn, lý do chọn và tình trạng giấy phép trong A1.1 |
| CNN và Transformer | ResNet18 và ViT-B/16 đều chạy thành công | Giữ nguyên hai mô hình trong kế hoạch chính |
| Scratch và pretrained; ba chế độ đóng băng | Đủ tám cấu hình; kiểm tra trạng thái đóng băng | Chốt công thức và ngân sách chính thức trước lượt cuối |
| Chất lượng và chi phí | Validation, lịch sử, tham số, thời gian, bộ nhớ cho năm epoch | Test accuracy, macro F1, dự đoán gốc, ma trận nhầm lẫn và mẫu lỗi của checkpoint được chọn |
| Tái lập và báo cáo | Mã nguồn, môi trường, seed, split và log thăm dò | Checkpoint, cấu hình có phiên bản, bảng/hình chính thức, báo cáo và landing page phân biệt lượt thăm dò với lượt chính |

Năm epoch đáp ứng mục đích thăm dò; chưa đủ để gọi là so sánh cuối A1.2. Đề không bắt buộc 100 epoch, batch 256, ba seed hoặc một lưới tuning lớn. Việc chưa gặp NaN trong năm epoch không đảm bảo lượt dài sẽ không lỗi.

## 6. Đề xuất có giới hạn ngân sách

### Phương án chính được đề xuất

1. **Chốt batch 64, FP32, CIFAR-100 và split hiện tại; giữ tám cấu hình.** Không mở thêm dataset, kiến trúc hoặc lưới batch. Giữ pipeline hiện tại làm mốc để đối chiếu; mọi thay đổi augmentation/optimizer phải được ghi thành công thức mới và không trộn số liệu.
2. **Nếu muốn cải thiện chế độ full trước khi khóa A1.1:** thử thêm đúng hai lượt ResNet full và ViT full, mỗi lượt năm epoch, learning rate `1e-5`; giữ nguyên mọi điều kiện khác và seed 42. So với hai lượt `1e-4` đã có, dùng validation tốt nhất trong cùng ngân sách năm epoch để quyết định. Chi phí dự kiến **1 giờ 13 phút**, không phải chạy lại cả tám lượt. Đây là kiểm tra giả thuyết learning rate thấp hơn có thể giúp, chưa phải bảo đảm tăng điểm. Năm epoch cũng có thể đánh giá bất lợi learning rate thấp học chậm; ghi rõ giới hạn đó. Có thể bỏ bước này nếu ưu tiên hoàn thành nhanh.
3. **Khóa A1.1 rồi chạy tám cấu hình × 30 epoch, trước mắt seed 42.** Đây là đề xuất ngân sách vừa phải, không phải kết luận rằng 30 epoch đủ hội tụ. Dùng checkpoint có validation accuracy cao nhất; nếu hòa giữ epoch sớm nhất, đúng quy tắc hiện tại. Giữ cùng giới hạn 30 epoch cho bảng chính và báo cả epoch tốt nhất, thời gian toàn lượt. Không dừng sớm riêng một cấu hình rồi gọi ngân sách bằng nhau.
4. **Chỉ đánh giá test sau khi đã chốt công thức và checkpoint theo validation.** Xuất accuracy, macro F1, confusion matrix dạng số/hình, lịch sử, mẫu lỗi và dự đoán theo ảnh; không dùng test để chọn learning rate hoặc kéo dài số epoch. Các lượt cũ đã từng xem test nên không mô tả đây là một test chưa từng được nhóm quan sát.
5. **Bổ sung seed sau khi có lượt chính đầu tiên và kiểm tra chi phí.** Nếu cần kết luận vững hơn và ngân sách cho phép, lặp thêm train seed 43 và 44 cho đủ cả tám cấu hình, giữ split seed 42. Nếu chỉ lặp một vài cấu hình thì ghi rõ phạm vi, không gán độ biến thiên của chúng cho toàn bộ bảng. Một seed vẫn có thể đáp ứng số liệu môn học nhưng phải công khai giới hạn; không khẳng định ý nghĩa thống kê.

Nếu lựa chọn learning rate riêng cho chế độ full, báo cáo phải công bố bảng learning rate và ngân sách lựa chọn. Khi đó kết luận nói về các công thức huấn luyện đã chọn; không quy toàn bộ chênh lệch cho riêng việc đóng băng. Nếu muốn so sánh thuần hơn dưới cùng công thức, giữ `1e-4` cho bảng tám cấu hình và đưa hai lượt `1e-5` vào phân tích bổ sung.

Không đồng thời thêm AdamW, scheduler, random crop và nhiều learning rate chỉ để tìm điểm cao: lượt thăm dò hiện tại chưa xác định hiệu quả của từng thay đổi đó. Có thể nghiên cứu tiếp sau khi hoàn tất bảng chính.

### Ước tính thời gian

Ngoại suy tuyến tính từ tổng thời gian fit năm epoch; chưa gồm các lượt chọn hyperparameter, đánh giá test, xuất hình, gián đoạn hoặc thay đổi tải GPU. Với công thức hoặc số ảnh khác, cần đo lại.

| Ngân sách cho cả tám cấu hình | Thời gian ước tính, một seed |
| --- | ---: |
| 20 epoch | 13,86 giờ |
| **30 epoch — đề xuất** | **20,78 giờ** |
| 50 epoch | 34,64 giờ |
| 100 epoch | 69,28 giờ |

Ba seed × 30 epoch ước tính **62,35 giờ**, chưa cộng hai lượt kiểm tra learning rate nếu thực hiện. Khởi đầu bằng một seed giúp kiểm tra thực tế trước khi cam kết toàn bộ ngân sách. Chạy tuần tự trên GPU dành riêng khi có thể để thời gian và bộ nhớ dễ diễn giải hơn.

## 7. Chuẩn bị mã trước lượt chính

Chưa chỉnh sửa mã trong lần phân tích này. Trước lượt chính cần bổ sung/chuẩn hóa cách lưu dữ liệu, không mặc nhiên coi `A1/train.py` đang có đầy đủ protocol cuối:

- Lưu trạng thái tốt nhất bền vững theo từng cấu hình và thư mục lượt chạy, tránh checkpoint chung bị ghi đè; lưu trạng thái tiếp tục huấn luyện nếu có yêu cầu resume.
- Lưu ngay cấu hình, source hash, split hash, môi trường, chính sách đóng băng, lịch sử và số đo thời gian/bộ nhớ từng pha như pilot; giữ kết quả đã xong khi lượt sau lỗi.
- Tách lựa chọn bằng validation khỏi đánh giá test cuối; nếu chạy learning rate riêng, manifest ghi đúng giá trị từng cấu hình.
- Lưu dự đoán, confusion matrix dạng số, F1 và các tệp phân tích lỗi để báo cáo truy về số liệu gốc.
- Giữ E1–E3 và báo cáo Exercise hiện tại; A1 dùng cùng cách trình bày nguồn số liệu, seed, giới hạn và checkpoint được chọn.

**Pilot hiện không lưu checkpoint ra đĩa.** Không thể nối tiếp năm epoch vừa chạy bằng các tệp JSON; lượt chính phải khởi tạo lại theo protocol đã chốt. Chỉ cơ chế resume được bổ sung và kiểm tra sau này mới cho phép tiếp tục một lượt đang chạy.

A1.1 phải khóa dataset/protocol trước hạn 26/10/2026; A1.2 triển khai đúng bản đã khóa và công khai mọi thay đổi cần thiết trước hạn 09/11/2026. Kết quả thăm dò và kết quả cũ phải được ghi đúng vai trò, không gộp thành kết quả chính thức của công thức mới.

## 8. Kết quả bổ sung: learning rate 1e-5 cho hai cấu hình full

Nguồn: `D:\master_hcmut\HK261\Deep_Learning\pilot_full_epoch5_batch64_lr1e-5_seed42`. Thư mục có bảy tệp, đủ hai cấu hình hoàn tất, mỗi cấu hình năm epoch. Tổng thời gian thực tế **1 giờ 13 phút 30 giây**.

Đã đối chiếu JSON riêng với JSON tổng hợp, số liệu chính trong CSV với JSON, lịch sử với số đo từng pha và epoch tốt nhất với validation accuracy lớn nhất. Mười epoch có loss/accuracy hữu hạn; mỗi epoch đủ 40.000/10.000 mẫu train/validation. Trọng số ghi nhận là `IMAGENET1K_V1`; phần được học đều thay đổi. Không nạp hoặc đánh giá test; không dùng dữ liệu tổng hợp.

Chỉ số split trùng hoàn toàn với lượt năm epoch `1e-4`; mã băm nguồn khớp lượt trước và mã hiện tại. Batch, số epoch, seed chia/huấn luyện, weight decay, worker, thiết bị, precision, optimizer và biến đổi dữ liệu không đổi. Môi trường ghi cùng P100, PyTorch và torchvision. Không có bằng chứng giám sát tải GPU suốt lượt để khẳng định thời gian đo trong điều kiện hoàn toàn cách ly.

| Cấu hình | LR | Validation tốt nhất | Epoch tốt nhất | Validation cuối | Validation loss cuối | Thời gian 5 epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet18 full | 1e-4 | 78,23% | 4 | 78,18% | 0,7518 | 6,75 phút |
| ResNet18 full | 1e-5 | 70,26% | 5 | 70,26% | 1,2229 | 6,78 phút |
| ViT-B/16 full | 1e-4 | 83,50% | 2 | 81,36% | 0,7246 | 66,59 phút |
| ViT-B/16 full | 1e-5 | **86,07%** | 5 | **86,07%** | **0,5040** | 66,58 phút |

### Nhận xét trong phạm vi bằng chứng

- **ResNet full:** `1e-4` cao hơn `1e-5` **7,97 điểm phần trăm** theo validation tốt nhất trong cùng năm epoch. Với `1e-5`, validation accuracy vẫn tăng từ 39,82% lên 70,26%; train accuracy cuối 71,54%. Đây là mẫu hình học chậm hơn trong ngân sách hiện tại, chưa chứng minh learning rate này luôn kém khi huấn luyện dài hơn.
- **ViT full:** `1e-5` cao hơn `1e-4` **2,57 điểm phần trăm** theo validation tốt nhất. Validation loss giảm qua cả năm epoch: 1,0450 → 0,6502 → 0,5502 → 0,5253 → 0,5040. Accuracy lần lượt 79,44% → 83,60% → 85,60% → 85,31% → 86,07%. Chưa xuất hiện diễn biến giảm kéo dài như lượt `1e-4`; vẫn chưa bảo đảm tránh quá khớp ở 30 epoch. Train accuracy cuối 95,37%, nên không gọi mô hình đã hết khoảng cách train/validation.
- ViT full `1e-5` cao hơn ViT partial `1e-4` (83,90%) **2,17 điểm phần trăm** theo kết quả tốt nhất của các lượt thăm dò năm epoch; thời gian full khoảng 2,17 lần partial. Các lượt dùng learning rate khác nhau, một seed và validation đã phục vụ lựa chọn, nên chưa thể kết luận full luôn tốt hơn partial hoặc có ưu thế thống kê.
- Chi phí ViT full và đỉnh bộ nhớ gần như không đổi khi đổi learning rate: cấp phát 8.839,48 MiB, đặt trước 9.218 MiB. Không có bằng chứng việc giảm learning rate làm huấn luyện nhanh hơn.

### Đề xuất chốt sau bước bổ sung

1. Giữ batch 64, FP32, toàn bộ split và preprocessing hiện tại.
2. Chọn `1e-4` cho ResNet full, `1e-5` cho ViT full, dựa trên kiểm tra validation có cùng ngân sách. Sáu cấu hình còn lại giữ `1e-4` làm công thức dự kiến; chưa khảo sát learning rate riêng cho sáu cấu hình đó, không mô tả đây là bộ hyperparameter tối ưu toàn cục.
3. Đề xuất chạy bảng chính tám cấu hình × 30 epoch, seed 42, ước tính khoảng 21 giờ. **Chưa thực hiện hoặc coi bước này đã được phê duyệt.** Trước khi chạy cần chốt A1.1 và chuẩn bị lưu checkpoint bền vững, dữ liệu tái lập và đánh giá test tách khỏi lựa chọn như mục 7.
4. Công khai ma trận learning rate và ngân sách thăm dò trong A1.1/A1.2: hai cấu hình full đã khảo sát hai giá trị, sáu cấu hình khác mới dùng một giá trị. Đánh giá các công thức đã chọn; không quy mọi chênh lệch của ViT full/partial cho duy nhất phạm vi đóng băng. Không dùng test để chọn lại công thức.

Đề xuất này thay cho nhu cầu tiếp tục khảo sát learning rate toàn bộ tám cấu hình. Chưa cần thêm lưới tuning hoặc thay đổi optimizer/augmentation để tiến tới yêu cầu môn học. Kết quả pilot không có checkpoint lưu ra đĩa và không thể dùng để nối tiếp huấn luyện.

## 9. Trạng thái sau khi người học duyệt

Đã chốt [A1.1](../A1/protocol-a1.1.md) và [JSON cấu hình](../A1/protocol-a1.1.json):
30 epoch, batch 64, FP32, seed chia/huấn luyện 42; ViT full LR `1e-5`, bảy cấu hình
khác LR `1e-4`. Adam/weight decay `1e-4`, split và preprocessing giữ nguyên.

Đã chuẩn bị [chương trình chạy](../A1/run_protocol.py) và [hướng dẫn](../A1/RUN_PROTOCOL.md):
lưu checkpoint tốt nhất ngay khi validation cải thiện, lưu đầy đủ metadata và số
đo từng pha, tự xuất EDA từ dữ liệu train thật; test dùng lệnh riêng, kiểm tra hash
và nguồn gốc, xuất dự đoán/ma trận/F1/mẫu sai cùng bảng so sánh tiếng Việt.

Đã đạt **23 kiểm tra tự động trên CPU** bằng dữ liệu tổng hợp/mô hình nhỏ, gồm
chọn checkpoint khi accuracy giảm hoặc hòa, tách test, đúng LR, chống ghi đè,
phát hiện checkpoint/nguồn/protocol sai, xử lý lỗi, EDA và đối chiếu dự đoán với
accuracy. Kiểm tra forward tám mô hình thật và số tham số cũng đạt. Windows chặn
quyền thư mục tạm khi chạy trong sandbox; đã chạy lại bộ kiểm tra ngoài sandbox
với quyền được cấp và đạt. Không dùng kết quả kiểm tra tổng hợp làm số liệu môn học.

Chưa chạy 30 epoch trên P100 hoặc sinh test theo quy trình này. Checkpoint mới
lưu mô hình để đánh giá, chưa lưu optimizer/RNG để nối tiếp epoch bị ngắt.
Trang chủ và trang A1 được cập nhật trong bản làm việc để công bố protocol, phân
biệt báo cáo cũ với lượt chính. Người học tự commit/push để công bố lên Git/Pages.
