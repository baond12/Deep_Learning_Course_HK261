# E1 - Phan loai anh: Softmax, MLP, CNN

## 1. Muc tieu

Bai E1 xay dung va so sanh ba bo phan loai tren cung mot tap anh nho:

- Softmax classifier tren vector anh flatten.
- MLP classifier tren vector anh flatten.
- CNN classifier dung convolution va pooling.

Tat ca mo hinh deu dung PyTorch va tu viet training loop theo chuoi:

```text
forward -> loss -> backward() -> optimizer.step()
```

Muc tieu chinh la so sanh su khac nhau ve kha nang bieu dien cua mo hinh tuyen tinh,
MLP phi tuyen va CNN co inductive bias phu hop voi anh.

## 2. Du lieu va split

Dataset chon cho E1-E3 la **MNIST**, gom anh xam kich thuoc `1 x 28 x 28` va 10 lop chu so
tu 0 den 9.

Theo nguyen tac data pipeline:

- Train set dung de hoc tham so cua mo hinh.
- Validation set dung de theo doi qua trinh train, chon epoch tot nhat va tuning hyperparameters.
- Test set chi dung de danh gia cuoi cung.

Split trong thuc nghiem:

| Split | So mau |
|---|---:|
| Train | 54,000 |
| Validation | 6,000 |
| Test | 10,000 |

Validation set duoc tach tu training set voi `val_ratio = 0.1` va seed co dinh `42`.
Tien xu ly gom `ToTensor()` va normalize bang thong ke chuan cua MNIST:
`mean = 0.1307`, `std = 0.3081`.

## 3. Mo hinh

### 3.1 Softmax classifier

Anh duoc flatten thanh vector 784 chieu roi dua qua mot lop tuyen tinh `784 -> 10`.
Model tra ve logits. Trong PyTorch, `nn.CrossEntropyLoss` nhan logits truc tiep va tu tinh
`log_softmax` ben trong, nen khong can dat softmax o cuoi model.

So tham so:

```text
784 * 10 + 10 = 7,850
```

### 3.2 MLP classifier

MLP dung hai tang an:

```text
784 -> 256 -> 128 -> 10
```

Cac tang an dung ReLU va Dropout `0.2`. MLP manh hon Softmax vi cac hidden layers va
activation phi tuyen giup hoc feature transformation truoc khi phan loai. Tuy nhien, MLP van
flatten anh nen lam mat cau truc khong gian giua cac pixel lan can.

So tham so:

```text
784 * 256 + 256 = 200,960
256 * 128 + 128 = 32,896
128 * 10 + 10 = 1,290
Total = 235,146
```

### 3.3 CNN classifier

CNN dung hai block convolution:

```text
Conv2d(1, 32, 3, padding=1) -> ReLU -> MaxPool2d(2)
Conv2d(32, 64, 3, padding=1) -> ReLU -> MaxPool2d(2)
Flatten -> Linear(64*7*7, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 10)
```

CNN phu hop voi anh hon vi giu input o dang `B x C x H x W`, khai thac locality, weight
sharing va feature maps. Convolution hoc cac dac trung cuc bo nhu canh, net cong va hinh dang
chu so.

So tham so:

```text
Conv1: 32 * 1 * 3 * 3 + 32 = 320
Conv2: 64 * 32 * 3 * 3 + 64 = 18,496
FC1: 64 * 7 * 7 * 128 + 128 = 401,536
FC2: 128 * 10 + 10 = 1,290
Total = 421,642
```

## 4. Cau hinh huan luyen

Thuc nghiem cap nhat chay voi thiet bi CUDA. Bo ket qua khong ghi ten GPU hay phien ban CUDA/PyTorch.

| Hyperparameter | Gia tri |
|---|---:|
| Epochs | 100 |
| Batch size | 256 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Loss | CrossEntropyLoss |
| Device | CUDA |

Lenh chay:

```bash
python E1/train.py --models all --epochs 100 --batch-size 256 --learning-rate 0.001 --weight-decay 0.0001 --val-ratio 0.1 --seed 42 --device cuda --output-dir E1/results/epochs100_batch256_seed42
```

Ket qua moi duoc luu trong `E1/results/epochs100_batch256_seed42/`, gom `summary.csv`, `metrics.json`,
`learning_curves.png`, cac confusion matrix va cac mau du doan sai.
Ket qua cu 20 epoch, batch size 16 van duoc giu o goc `E1/results/` de truy vet.

## 5. Ket qua thuc nghiem

| Model | Parameters | Best epoch | Best val accuracy | Test loss | Test accuracy | Macro F1 | Training time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Softmax | 7,850 | 32 | 0.920333 | 0.271124 | 0.9259 | 0.924782 | 11m 58s |
| MLP | 235,146 | 81 | 0.984000 | 0.060617 | 0.9863 | 0.986219 | 12m 40s |
| CNN | 421,642 | 83 | 0.993167 | 0.025890 | 0.9927 | 0.992631 | 12m 49s |

Thu tu ket qua tren test set la:

```text
CNN (99.27%) > MLP (98.63%) > Softmax (92.59%)
```

Softmax co it tham so nhat va tao baseline tuyen tinh. MLP co nhieu tham so hon va co ReLU,
nen hoc duoc quan he phi tuyen tot hon. CNN dat ket qua cao nhat vi kien truc convolution phu
hop hon voi du lieu anh.

## 6. Phan tich learning curves

Softmax dat validation accuracy tot nhat 92.03% o epoch 32; o epoch 100 con 91.65%.
Training loss giam tu 0.539 xuong 0.228, nhung validation khong cai thien on dinh.
Dieu nay chua chung minh da tim duoc gioi han toi uu cua mo hinh.

MLP dat best validation accuracy 98.40% o epoch 81. Loss cuoi tren train/validation
lan luot khoang 0.0189/0.0733. CNN dat best validation accuracy 99.32% o epoch 83,
loss cuoi khoang 0.0058/0.0357. Khoang cach train-validation van ton tai.

Test dung checkpoint tot nhat theo validation (32, 81, 83), khong mac dinh epoch 100.
So voi lan cu, test accuracy tang 0.50, 0.81 va 0.10 diem phan tram.
Vi thay doi dong thoi epochs va batch size, chua the tach rieng tac dong cua tung yeu to.

## 7. Phan tich confusion matrix va mau sai

Tren 10,000 anh test, Softmax sai 741 anh, MLP sai 137 anh va CNN sai 73 anh.
Ma tran nham lan moi ghi nhan:

- Softmax: `5 -> 8` 37 lan, `5 -> 3` 36 lan, `2 -> 8` 35 lan.
- MLP: `9 -> 4` va `4 -> 9` deu 8 lan.
- CNN: `9 -> 4` 10 lan, `5 -> 3` 6 lan.

Hinh mau sai dung de minh hoa dinh tinh, khong dai dien cho tan suat tung kieu net viet.
Bao cao Word hien thi muoi loi dau cua Softmax theo thu tu test; anh nguon luu toi da 25 loi.
Khong suy luan nguyen nhan chac chan cua tung loi chi tu hinh anh.

## 8. Ket luan

Ket qua thuc nghiem phu hop voi ky vong ly thuyet:

- Softmax la baseline tuyen tinh, don gian va it tham so nhat, nhung bi gioi han ve kha nang
  hoc dac trung phi tuyen.
- MLP cai thien dang ke nho hidden layers va ReLU, nhung van lam mat cau truc khong gian do
  flatten anh.
- CNN dat ket qua tot nhat vi convolution khai thac locality, weight sharing va feature maps,
  phu hop voi du lieu anh hon Softmax va MLP.

Do do, tren MNIST, CNN la mo hinh phu hop nhat trong ba kien truc duoc so sanh.
