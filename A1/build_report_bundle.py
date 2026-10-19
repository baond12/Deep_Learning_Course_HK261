"""Audit completed A1 outputs and prepare a flat 19-file Vietnamese reporting bundle.

No training, test inference, checkpoint rewriting, or Git mutations are performed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
import numpy as np
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

NAMES = [f"{a}_{m}" for a in ("resnet18", "vit_b16") for m in ("scratch", "pretrained_head", "pretrained_partial", "pretrained_full")]
LABELS = ["Từ đầu", "Chỉ đầu phân loại", "Mở một phần", "Tinh chỉnh toàn bộ"]
CODES = dict(zip(NAMES, [f"{a}-{m}" for a in ("R", "V") for m in ("S", "H", "P", "F")]))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(source):
    train, test = read_json(source/"metrics.json"), read_json(source/"test/test_metrics.json")
    protocol = read_json(source/"protocol.json")
    assert train["status"] == test["status"] == "completed"
    assert not train["synthetic_test"] and not test["synthetic_test"] and test["test_evaluated"]
    assert set(train["experiments"]) == set(test["experiments"]) == set(NAMES)
    assert train["protocol"] == protocol == read_json(ROOT/"A1/protocol-a1.1.json")
    assert train["protocol_sha256"] == test["protocol_sha256"] == digest(protocol)
    assert test["training_metrics_sha256"] == sha(source/"metrics.json")
    assert train["environment"]["source_sha256_lf"] == test["environment"]["source_sha256_lf"]
    for name, expected in train["environment"]["source_sha256_lf"].items():
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == expected, name
    split, tsplit = read_json(source/"split_indices.json"), read_json(source/"test/split_indices.json")
    assert split == {k: tsplit[k] for k in ("train", "validation")}
    assert len(set(split["train"])) == len(split["train"]) == 40000
    assert len(set(split["validation"])) == len(split["validation"]) == 10000
    assert set(split["train"]).isdisjoint(split["validation"])
    assert set(split["train"] + split["validation"]) == set(range(50000))
    assert set(tsplit["test"]) == set(range(10000)) and len(tsplit["test"]) == 10000
    assert digest(split) == protocol["split_indices_sha256"] == train["split"]["indices_sha256"]
    summary = {r["experiment"]: r for r in read_csv(source/"summary.csv")}
    test_summary = {r["experiment"]: r for r in read_csv(source/"test/test_summary.csv")}
    classes = train["split"]["class_names"]
    assert len(classes) == len(set(classes)) == 100
    all_predictions, all_classes, confusions, records, matrices, error_sets = [], [], [], [], {}, {}
    reference_truth = None
    checkpoints = {}
    for name in NAMES:
        r, t = train["experiments"][name], test["experiments"][name]
        assert r == read_json(source/f"run_{name}.json")
        assert r["status"] == t["status"] == "completed" and r["epochs_completed"] == 30
        assert r["learning_rate"] == protocol["learning_rates"][name]
        assert r["trainable_state_changed"]
        if r["frozen_state_checked"]:
            assert r["frozen_state_unchanged"] and r["frozen_state_sha256_before"] == r["frozen_state_sha256_after"]
        history = r["history"]
        assert [h["epoch"] for h in history] == list(range(1, 31))
        assert all(math.isfinite(h[k]) for h in history for k in ("train_loss", "val_loss", "train_accuracy", "val_accuracy"))
        csv_history = read_csv(source/f"history_{name}.csv")
        assert len(csv_history) == 30
        for a, b in zip(history, csv_history):
            assert all(float(b[k]) == v for k, v in a.items())
        best = max(history, key=lambda h: h["val_accuracy"])
        lowest = min(history, key=lambda h: h["val_loss"])
        assert best["epoch"] == r["best_epoch"] == t["best_epoch"]
        assert best["val_accuracy"] == r["best_val_accuracy"] == t["best_val_accuracy"]
        for phases in r["phase_measurements"]:
            assert phases["train"]["samples"] == 40000 and phases["train"]["batches"] == 625
            assert phases["validation"]["samples"] == 10000 and phases["validation"]["batches"] == 157
        assert len(r["phase_measurements"]) == 30
        assert abs(sum(p["train"]["seconds"] for p in r["phase_measurements"]) - r["train_seconds"]) < 1e-6
        assert abs(sum(p["validation"]["seconds"] for p in r["phase_measurements"]) - r["validation_seconds"]) < 1e-6
        assert sha(source/r["checkpoint"]) == r["checkpoint_sha256"] == t["checkpoint_sha256"]
        checkpoints[name] = r["checkpoint_sha256"]
        predicted = read_csv(source/f"test/predictions_{name}.csv")
        assert len(predicted) == 10000
        assert [int(x["original_test_index"]) for x in predicted] == tsplit["test"]
        truth = [int(x["true_label"]) for x in predicted]
        if reference_truth is None:
            reference_truth = truth
        assert truth == reference_truth
        matrix = np.zeros((100, 100), dtype=np.int64)
        for row in predicted:
            y, q = int(row["true_label"]), int(row["predicted_label"])
            assert row["true_class"] == classes[y] and row["predicted_class"] == classes[q]
            assert (row["correct"].lower() == "true") == (y == q)
            assert math.isfinite(float(row["confidence"])) and 0 <= float(row["confidence"]) <= 1
            matrix[y, q] += 1
            all_predictions.append({"experiment": name, **row})
        assert np.all(matrix.sum(1) == 100)
        matrices[name] = matrix
        precision = np.divide(matrix.diagonal(), matrix.sum(0), out=np.zeros(100), where=matrix.sum(0) != 0)
        recall = matrix.diagonal()/matrix.sum(1)
        f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(100), where=precision+recall != 0)
        accuracy = matrix.trace()/matrix.sum()
        assert abs(accuracy-t["test_accuracy"]) < 1e-12 and abs(float(f1.mean())-t["macro_f1"]) < 1e-12
        csv_matrix = read_csv(source/f"test/confusion_matrix_{name}.csv")
        assert all(row["true_class"] == classes[i] and [int(row[label]) for label in classes] == matrix[i].tolist() for i, row in enumerate(csv_matrix))
        report = read_json(source/f"test/classification_report_{name}.json")
        for i, label in enumerate(classes):
            assert report[label]["support"] == 100
            for k, value in [("precision", precision[i]), ("recall", recall[i]), ("f1-score", f1[i])]:
                assert abs(report[label][k]-value) < 1e-12
            all_classes.append({"experiment": name, "class_id": i, "class_name": label, "support": 100,
                                "true_positive": int(matrix[i,i]), "false_positive": int(matrix[:,i].sum()-matrix[i,i]),
                                "false_negative": int(matrix[i,:].sum()-matrix[i,i]),
                                "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])})
        pairs = sorted([(int(matrix[i,j]), i, j) for i in range(100) for j in range(100) if i != j and matrix[i,j]], reverse=True)
        for rank, (count, i, j) in enumerate(pairs[:20], 1):
            confusions.append({"experiment": name, "rank": rank, "true_class": classes[i], "predicted_class": classes[j],
                               "count": count, "true_class_support": 100})
        mistakes = [row for row in predicted if row["true_label"] != row["predicted_label"]]
        error_sets[name] = {int(row["original_test_index"]) for row in mistakes}
        rec = {"experiment": name, "code": CODES[name], "learning_rate": r["learning_rate"], "epochs": 30,
               "best_epoch": r["best_epoch"], "best_val_accuracy": r["best_val_accuracy"],
               "final_val_accuracy": history[-1]["val_accuracy"], "test_accuracy": float(accuracy), "macro_f1": float(f1.mean()),
               "test_loss": t["test_loss"], "total_parameters": r["total_parameters"], "trainable_parameters": r["trainable_parameters"],
               "trainable_fraction": r["trainable_parameters"]/r["total_parameters"], "train_seconds": r["train_seconds"],
               "validation_seconds": r["validation_seconds"], "fit_seconds_including_callbacks": r["training_validation_seconds"],
               "test_seconds": t["test_seconds"], "peak_allocated_mib": r["peak_allocated_mib"], "peak_reserved_mib": r["peak_reserved_mib"],
               "minimum_val_loss_epoch": lowest["epoch"], "minimum_val_loss": lowest["val_loss"],
               "final_train_accuracy": history[-1]["train_accuracy"], "final_val_loss": history[-1]["val_loss"],
               "final_train_val_gap_pp": 100*(history[-1]["train_accuracy"]-history[-1]["val_accuracy"]),
               "test_errors": len(mistakes), "wrong_with_confidence_ge_0_9": sum(float(row["confidence"]) >= .9 for row in mistakes)}
        for k in ("best_epoch", "best_val_accuracy", "total_parameters", "trainable_parameters"):
            assert float(summary[name][k]) == r[k]
        for k in ("test_accuracy", "macro_f1", "test_loss"):
            assert float(test_summary[name][k]) == t[k]
        records.append(rec)
    paired = []
    for a, b in itertools.combinations(NAMES, 2):
        ea, eb = error_sets[a], error_sets[b]
        paired.append({"experiment_a": a, "experiment_b": b, "both_wrong": len(ea & eb),
                       "a_wrong_b_correct": len(ea-eb), "a_correct_b_wrong": len(eb-ea), "both_correct": 10000-len(ea|eb)})
    checks = {"status": "passed", "checked_on_local_date": "2026-10-19", "configurations": 8,
              "finite_history_epochs": 240, "prediction_rows": 80000, "classes": 100,
              "test_support_per_class": 100, "train_validation_disjoint": True,
              "train_validation_cover_original_50000": True, "test_indices_cover_original_10000": True,
              "accuracy_and_macro_f1_recomputed": True, "all_8_confusion_matrices_recomputed": True,
              "all_800_class_reports_recomputed": True, "checkpoint_sha256_verified": checkpoints,
              "train_metrics_sha256": sha(source/"metrics.json"), "test_metrics_sha256": sha(source/"test/test_metrics.json"),
              "protocol_sha256": digest(protocol), "source_hashes_match_training": True,
              "limitations": ["No independent rerun of inference", "No access to original server GPU workload monitoring", "One training seed only",
                              "Test loss checked for consistency of records, not recomputed from unavailable true-class probabilities"]}
    return train, test, records, all_classes, confusions, all_predictions, paired, matrices, checks


def copy_results(source, target):
    copied = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if not path.is_file() or "checkpoints" in relative.parts or path.suffix in (".pt", ".pth", ".tmp"):
            continue
        dest = target/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        expected = sha(path)
        if dest.exists():
            assert sha(dest) == expected, f"Existing original artifact differs: {dest}"
        else:
            shutil.copy2(path, dest)
        assert sha(dest) == expected
        copied.append({"relative_path": str(relative).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": expected})
    return copied


def configure_fonts():
    for name in ("times.ttf", "timesbd.ttf", "timesi.ttf"):
        fontManager.addfont(str(Path("C:/Windows/Fonts")/name))
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11, "text.color": "black",
                         "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
                         "savefig.facecolor": "white", "figure.facecolor": "white"})
    pdfmetrics.registerFont(TTFont("TNR", "C:/Windows/Fonts/times.ttf"))
    pdfmetrics.registerFont(TTFont("TNR-Bold", "C:/Windows/Fonts/timesbd.ttf"))


def curves(train, names, path, title):
    fig, axes = plt.subplots(4, 2, figsize=(12, 12.8))
    for i, name in enumerate(names):
        r = train["experiments"][name]; history = r["history"]
        for col, metric, ylabel in [(0, "loss", "Hàm mất mát"), (1, "accuracy", "Độ chính xác")]:
            ax = axes[i,col]
            x = [h["epoch"] for h in history]
            ax.plot(x, [h[f"train_{metric}"] for h in history], "k--", label="Huấn luyện", linewidth=1.5)
            ax.plot(x, [h[f"val_{metric}"] for h in history], "k-", label="Xác thực", linewidth=1.5)
            epoch = r["best_epoch"]
            ax.axvline(epoch, color="black", linestyle=":", linewidth=.7)
            ax.scatter([epoch], [history[epoch-1][f"val_{metric}"]], color="black", s=18, zorder=4)
            ax.set(xlabel="Epoch", ylabel=ylabel, title=f"{LABELS[i]} | LR {r['learning_rate']:g} | chọn epoch {epoch}", xlim=(1,30))
            ax.legend(fontsize=9, loc="best", frameon=False)
    fig.suptitle(title, fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, .015, 1, .975))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def comparison(records, path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2))
    panels = [("test_accuracy", "Độ chính xác kiểm tra (%)", 100), ("macro_f1", "F1 trung bình theo lớp", 1),
              ("fit_seconds_including_callbacks", "Toàn lượt huấn luyện và xác thực (giờ)", 1/3600),
              ("trainable_parameters", "Tham số được học (triệu)", 1e-6), ("peak_allocated_mib", "Bộ nhớ PyTorch cấp phát đỉnh (GiB)", 1/1024),
              ("peak_reserved_mib", "Bộ nhớ PyTorch đặt trước đỉnh (GiB)", 1/1024)]
    x = np.arange(4)
    for ax, (key, title, factor) in zip(axes.ravel(), panels):
        for family, start, offset, hatch, color in [("ResNet18",0,-.2,"", "white"),("ViT-B/16",4,.2,"///", "0.82")]:
            values = [r[key]*factor for r in records[start:start+4]]
            bars = ax.bar(x+offset, values, width=.38, color=color, edgecolor="black", hatch=hatch, linewidth=.8, label=family)
            ax.bar_label(bars, labels=[f"{v:.2f}" if key not in ("macro_f1",) else f"{v:.3f}" for v in values], fontsize=8, padding=2)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x, ["Từ đầu", "Chỉ đầu", "Một phần", "Toàn bộ"])
        ax.margins(y=.18)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("A1 trên CIFAR-100 | 30 epoch, batch 64, một seed", fontsize=17, fontweight="bold")
    fig.text(.5, .015, "LR: ViT toàn bộ 1e-5; bảy cấu hình khác 1e-4. Không coi thời gian fit là thời gian GPU tính toán thuần.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0,.045,1,.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def atlas(source, path, records, class_rows, pairs):
    c = canvas.Canvas(str(path), pagesize=(900, 1000), pageCompression=1)
    c.setTitle("Phân tích lỗi A1 trên CIFAR-100")
    for number, record in enumerate(records):
        name = record["experiment"]
        top = [p for p in pairs if p["experiment"] == name][:4]
        low = sorted([r for r in class_rows if r["experiment"] == name], key=lambda r:r["f1"])[:5]
        for kind, label in [("confusion_matrix", "Ma trận nhầm lẫn đủ 100 lớp"), ("misclassified", "Các mẫu sai đầu tiên theo thứ tự bộ nạp kiểm tra")]:
            c.setFont("TNR-Bold", 20)
            c.drawString(34, 965, f"{CODES[name]} | {name}")
            c.setFont("TNR", 14)
            c.drawString(34, 940, f"{label} | accuracy {record['test_accuracy']:.2%}, F1 {record['macro_f1']:.4f}")
            image = Image.open(source/f"test/{kind}_{name}.png").convert("L")
            c.drawImage(ImageReader(image), 35, 116, width=830, height=800, preserveAspectRatio=True, anchor="c")
            c.setFont("TNR", 12)
            if kind == "confusion_matrix":
                c.drawString(34, 90, "Một số nhầm lẫn có hướng: " + "; ".join(f"{x['true_class']} → {x['predicted_class']}: {x['count']}" for x in top[:2]))
                c.drawString(34, 71, "F1 thấp: " + "; ".join(f"{x['class_name']} {x['f1']:.3f}" for x in low))
                c.drawString(34, 52, "Mỗi lớp có 100 ảnh. Hàng: nhãn thật; cột: dự đoán. Đọc số chính xác từ CSV, không ước lượng theo sắc độ.")
            else:
                c.drawString(34, 82, "Đây là tối đa 25 mẫu sai gặp đầu tiên, không phải lấy mẫu ngẫu nhiên hoặc đại diện toàn bộ lỗi.")
                c.drawString(34, 62, "Tên lớp giữ theo CIFAR-100; true là nhãn thật, pred là nhãn dự đoán. Ảnh chỉ chuyển sang thang xám khi trình bày.")
                c.drawString(34, 42, "Nhận xét về hình dạng/nền chỉ là giả thuyết từ hình minh họa, chưa được kiểm chứng bằng thực nghiệm loại trừ.")
            c.setFont("TNR", 11)
            c.drawRightString(865, 20, f"{number*2+(1 if kind=='confusion_matrix' else 2)}/16")
            c.showPage()
    c.save()


def hours(seconds):
    return f"{seconds/3600:.2f}"


def analysis_text(train, test, records, classes, pairs):
    by = {r["experiment"]: r for r in records}
    fullv, fullr = by["vit_b16_pretrained_full"], by["resnet18_pretrained_full"]
    elapsed = (datetime.fromisoformat(train["finished_at_utc"])-datetime.fromisoformat(train["started_at_utc"])).total_seconds()
    total_fit = sum(r["fit_seconds_including_callbacks"] for r in records)
    lines = ["# Phân tích kết quả chính thức Assignment 1", "", "Ngày đối chiếu: 19/10/2026. Lượt chạy `epochs30_batch64_seed42_v1`, protocol `a1_cifar100_p100_v1`.", "",
             "## Kết luận trong phạm vi thực nghiệm", "",
             "ViT-B/16 tiền huấn luyện, tinh chỉnh toàn bộ với learning rate 1e-5 đạt test accuracy 86,46% và macro F1 0,8651, cao nhất trong tám cấu hình đã chạy. "
             "ResNet18 tiền huấn luyện, tinh chỉnh toàn bộ đạt 78,97% và 0,7893, với thời gian toàn fit thấp hơn rõ rệt. "
             "Hai mô hình chỉ học đầu phân loại tiết kiệm tham số được cập nhật và bộ nhớ nhưng không đạt accuracy cao nhất. "
             "Kết quả là so sánh các công thức cụ thể trên một split/một seed, chưa chứng minh ưu thế phổ quát hoặc ý nghĩa thống kê.", "",
             "## Nguồn và kiểm chứng", "",
             "Đã tính lại accuracy, macro F1, cả tám ma trận nhầm lẫn và 800 bản ghi chỉ số từng lớp từ 80.000 dòng dự đoán. "
             "Mỗi cấu hình có cùng 10.000 chỉ số ảnh kiểm tra và cùng thứ tự nhãn thật; mỗi lớp có 100 ảnh. "
             "Đủ 240 bản ghi epoch hữu hạn; CSV lịch sử khớp JSON; epoch tốt nhất là lần đầu đạt validation accuracy lớn nhất. "
             "Hash protocol, bản ghi train/test, nguồn mã và cả tám checkpoint local đã khớp. Không chạy lại suy luận để tạo bản phân tích này. "
             "Test loss được đối chiếu giữa các bản ghi, không tính lại độc lập vì CSV chỉ lưu xác suất lớp dự đoán, không lưu xác suất của mọi lớp.", "",
             "## Bảng chất lượng", "", "| Cấu hình | LR | Epoch chọn | Validation tốt nhất | Validation cuối | Test accuracy | Macro F1 | Test loss |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in records:
        lines.append(f"| {r['experiment']} | {r['learning_rate']:g} | {r['best_epoch']} | {r['best_val_accuracy']:.2%} | {r['final_val_accuracy']:.2%} | {r['test_accuracy']:.2%} | {r['macro_f1']:.4f} | {r['test_loss']:.4f} |")
    lines += ["", "Test dùng checkpoint chọn bằng validation, không dùng trạng thái epoch 30 một cách mặc định. Các tỷ lệ trong bảng là tỷ lệ phần trăm; chênh lệch được diễn đạt bằng điểm phần trăm.", "",
              "## Tham số và chi phí", "", "| Cấu hình | Tham số tổng | Tham số được học | Tỷ lệ được học | Train (phút) | Validation (phút) | Toàn fit (giờ) | Allocated (GiB) | Reserved (GiB) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in records:
        lines.append(f"| {r['experiment']} | {r['total_parameters']:,} | {r['trainable_parameters']:,} | {r['trainable_fraction']:.3%} | {r['train_seconds']/60:.2f} | {r['validation_seconds']/60:.2f} | {hours(r['fit_seconds_including_callbacks'])} | {r['peak_allocated_mib']/1024:.3f} | {r['peak_reserved_mib']/1024:.3f} |")
    lines += ["", f"Tổng thời gian toàn fit của tám cấu hình: **{hours(total_fit)} giờ**; thời gian lịch của tiến trình train: **{hours(elapsed)} giờ**. "
              "Fit tính callback/ghi checkpoint; số đo train/validation từng pha không bao gồm các phần này. "
              "Lượt chạy tuần tự trong một tiến trình trên Tesla P100 16 GB; ảnh chụp GPU trước lượt không chứng minh GPU được cách ly suốt hơn 20 giờ. "
              "Bộ nhớ là peak allocated/reserved của tiến trình PyTorch, không phải tổng VRAM của mọi tiến trình. "
              "Không cộng thời gian các lượt cũ chạy song song để diễn giải thành GPU-hour.", "",
              f"ViT full cao hơn ResNet full **{100*(fullv['test_accuracy']-fullr['test_accuracy']):.2f} điểm phần trăm**, "
              f"nhưng toàn fit khoảng **{fullv['fit_seconds_including_callbacks']/fullr['fit_seconds_including_callbacks']:.2f} lần**. "
              "ViT full/ResNet full có LR khác nhau và quy mô mô hình khác nhau, nên tỷ lệ này mô tả hai công thức trên máy đã đo, không cô lập ảnh hưởng kiến trúc.", "",
              "## Tiền huấn luyện và phạm vi tinh chỉnh", ""]
    for family, names in [("ResNet18",NAMES[:4]),("ViT-B/16",NAMES[4:])]:
        s,h,p,f = [by[n] for n in names]
        lines += [f"### {family}", "",
                  f"Full pretrained cao hơn scratch **{100*(f['test_accuracy']-s['test_accuracy']):.2f} điểm phần trăm**. "
                  f"Full cao hơn head **{100*(f['test_accuracy']-h['test_accuracy']):.2f} điểm phần trăm** và partial **{100*(f['test_accuracy']-p['test_accuracy']):.2f} điểm phần trăm**. "
                  f"Partial dùng khoảng **{p['fit_seconds_including_callbacks']/f['fit_seconds_including_callbacks']:.1%}** thời gian fit của full; "
                  f"head dùng khoảng **{h['fit_seconds_including_callbacks']/f['fit_seconds_including_callbacks']:.1%}**. "
                  "Đóng băng giúp giảm tham số cập nhật và bộ nhớ gradient/optimizer, nhưng vẫn cần forward qua backbone; giảm trainable parameters không đồng nghĩa giảm tổng kích thước mô hình hoặc chi phí về không.", ""]
    lines += ["ResNet partial và head có thời gian gần nhau; không suy ra partial luôn nhanh hơn head từ chênh lệch nhỏ trong một lượt. "
              "Trong ViT, full dùng LR 1e-5 còn partial/head dùng 1e-4; không gán mọi chênh lệch cho số block được mở. "
              "ViT scratch thấp hơn ResNet scratch trong ngân sách này; kết quả không chứng minh ViT không thể học từ đầu khi có dữ liệu/công thức/ngân sách khác.", "",
              "## Đường cong và dấu hiệu quá khớp", "", "| Cấu hình | Epoch loss validation nhỏ nhất | Loss nhỏ nhất | Loss validation cuối | Train accuracy cuối | Validation accuracy cuối | Khoảng cách cuối (điểm %) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in records:
        lines.append(f"| {r['experiment']} | {r['minimum_val_loss_epoch']} | {r['minimum_val_loss']:.4f} | {r['final_val_loss']:.4f} | {r['final_train_accuracy']:.2%} | {r['final_val_accuracy']:.2%} | {r['final_train_val_gap_pp']:.2f} |")
    lines += ["", "Hai cấu hình head đạt validation accuracy tốt nhất ở epoch 30 và validation loss thấp nhất cũng ở epoch 30: chưa có bằng chứng chúng đã hội tụ. "
              "Scratch có khoảng cách train/validation lớn, nhất là ViT: train cuối 95,69% nhưng validation cuối 41,61%. "
              "ResNet scratch tốt nhất epoch 14, full và partial tốt nhất epoch 8; chạy dài hơn không bảo đảm tăng validation. "
              "ViT full có loss validation thấp nhất epoch 5 nhưng accuracy tốt nhất epoch 24; ViT partial có loss thấp nhất epoch 3 nhưng accuracy tốt nhất epoch 20. "
              "Accuracy chỉ xét lớp thắng, cross entropy còn nhạy với xác suất gán cho nhãn thật nên hai tiêu chí có thể chọn epoch khác nhau. Giữ tiêu chí chọn checkpoint đã khóa là accuracy; không đổi sau khi xem test.", "",
              "Khoảng cách train/validation trên bảng là dấu hiệu mô tả, không phải ước lượng thuần của generalization gap: train được ghi trong chế độ huấn luyện với tăng cường ngẫu nhiên, validation trong chế độ đánh giá. "
              "Dù các lượt cũ từng có NaN, lượt mới có đủ 240 bản ghi hữu hạn; điều này không chứng minh mọi cấu hình dài hơn đều ổn định.", "",
              "## Accuracy và độ tự tin khi sai", "",
              "ViT head có test loss 0,6397, thấp hơn ViT full 0,6475, dù accuracy head 81,29% thấp hơn full 86,46%. "
              "ViT partial đạt accuracy 83,05% nhưng loss 0,9169. Không được viết mô hình accuracy cao nhất đồng thời tốt nhất ở mọi metric.", "",
              "| Cấu hình | Tổng lỗi test | Lỗi có confidence ≥ 0,9 | Tỷ lệ trong các lỗi |", "| --- | ---: | ---: | ---: |"]
    for r in records:
        lines.append(f"| {r['experiment']} | {r['test_errors']} | {r['wrong_with_confidence_ge_0_9']} | {r['wrong_with_confidence_ge_0_9']/r['test_errors']:.2%} |")
    lines += ["", "Confidence là softmax của lớp dự đoán. Đây là chẩn đoán bổ sung từ cùng dự đoán test, không phải thực nghiệm tuning hoặc phép đo hiệu chuẩn đầy đủ. "
              "Không khẳng định đã xác định nguyên nhân của loss cao; những lỗi rất tự tin phù hợp với khả năng cross entropy bị phạt mạnh khi mô hình gán ít xác suất cho nhãn thật.", "",
              "## Lỗi theo lớp và cặp nhầm lẫn", ""]
    for r in records:
        name=r["experiment"]
        low=sorted([x for x in classes if x["experiment"]==name],key=lambda x:x["f1"])[:5]
        top=[x for x in pairs if x["experiment"]==name][:5]
        lines += [f"### {name}", "", "Năm lớp có F1 thấp: " + "; ".join(f"{x['class_name']} {x['f1']:.4f}" for x in low) + ".", "",
                  "Cặp lỗi có hướng: " + "; ".join(f"{x['true_class']} → {x['predicted_class']}: {x['count']}/100 ảnh của lớp thật" for x in top) + ".", ""]
    lines += ["Các cặp liên quan cây và người xuất hiện lặp lại ở nhiều mô hình. Khác biệt tinh tế và ảnh gốc 32×32 là giả thuyết giải thích hợp lý từ ngữ nghĩa/hình mẫu, "
              "chưa phải kết luận nhân quả vì chưa có ablation độ phân giải hoặc kiểm tra gán nhãn. Không tự quy lỗi cho dữ liệu sai nhãn. "
              "Hình mẫu sai là tối đa 25 lỗi đầu theo thứ tự loader đã cố định, không phải mẫu ngẫu nhiên hay đại diện thống kê.", "",
              "## Đối chiếu đề và liên kết A1.1 với A1.2", "",
              "Đủ CNN/Transformer, scratch/pretrained cho cả hai họ và head/partial/full cho từng họ; cùng dữ liệu/split, cùng preprocessing và ngân sách epoch/batch. "
              "Đã có validation/test accuracy, macro F1, tham số, thời gian, bộ nhớ, đường cong, ma trận và mẫu sai, EDA thật và dữ liệu dự đoán gốc. "
              "A1.1 đã công khai cách chọn LR bằng validation và A1.2 thực hiện đúng cấu hình đó. Tám cấu hình/240 epoch thuộc lượt chính; "
              "26 lượt cấu hình thăm dò (66 epoch tổng cộng) chỉ là căn cứ chọn kế hoạch, không cộng vào bảng chính.", "",
              "Tình trạng giấy phép CIFAR-100 chưa xác định rõ từ trang gốc/thẻ dữ liệu; phải ghi đúng, không tự gán MIT/CC hoặc coi công khai đồng nghĩa có giấy phép cụ thể. "
              "Một seed, khác LR của ViT full, khác quy mô/công thức tiền huấn luyện và chưa kiểm soát tải GPU suốt lượt là các giới hạn cần nêu. "
              "Cùng ImageNet-1K không đồng nghĩa ResNet và ViT có cùng công thức/ngân sách tiền huấn luyện. Test từng được quan sát trong các lượt cũ; không gọi đây là test hoàn toàn chưa từng thấy.", "",
              "E1–E3 dùng MNIST, 19 cấu hình độc lập, 1.900 epoch, batch 256; A1 dùng CIFAR-100, tám cấu hình, 240 epoch, batch 64. "
              "Sự nhất quán nằm ở cách tách train/validation/test, truy nguồn, chọn checkpoint và diễn đạt giới hạn; không ép A1 dùng số epoch/batch của Exercise hoặc so accuracy trực tiếp giữa hai dataset.", "",
              "## Trạng thái báo cáo và công bố", "",
              "Đã hoàn tất thực nghiệm và tổng hợp số liệu; đang chuẩn bị bản Word A1 theo kết quả mới. Word/PDF A1 v01 hiện có là lượt cũ. "
              "Không đổi nhãn bản PDF cũ thành kết quả mới. Sau khi ChatGPT Plus tạo Word, xuất PDF từ đúng Word đó theo yêu cầu người học; kiểm tra tính đồng bộ rồi cập nhật đường dẫn báo cáo ổn định. "
              "Chưa xác nhận đã nộp bài, chưa thay nhóm xác nhận tác giả hoặc khai báo hỗ trợ AI.", ""]
    return "\n".join(lines)


def package(bundle, source, target, train, test, records, classes, pairs, predictions, analysis, checks, originals):
    filenames = [
        "00_HUONG_DAN.txt", "01_PROMPT_CHATGPT_PLUS.txt", "02_DE_BAI_GOC_A1.pdf",
        "03_PROTOCOL_VA_DOI_CHIEU.md", "04_PHAN_TICH_KET_QUA.md", "05_KET_QUA_8_CAU_HINH.csv",
        "06_LICH_SU_240_EPOCH.csv", "07_METRICS_VA_KIEM_CHUNG.json", "08_CHI_SO_800_LOP.csv",
        "09_DU_DOAN_TEST_80000.csv", "10_MA_NGUON_TAI_LAP.txt", "11_DUONG_CONG_RESNET.png",
        "12_DUONG_CONG_VIT.png", "13_SO_SANH_CHAT_LUONG_CHI_PHI.png", "14_PHAN_TICH_LOI.pdf",
        "15_EDA_ANH_HUAN_LUYEN.png", "16_LY_THUYET_VA_NGUON.md", "17_THAM_DO_CHON_CAU_HINH.csv",
        "18_DANH_MUC_TEP.csv"]
    assert all(p.is_file() and p.name in filenames for p in bundle.iterdir()), "Unexpected files in bundle; choose a clean output"
    def text(name, value):
        (bundle/name).write_text(value, encoding="utf-8")
    text(filenames[0], """GÓI LÀM BÁO CÁO ASSIGNMENT 1 THEO KẾT QUẢ 30 EPOCH

Gồm đúng 19 tệp, tất cả ở cùng một cấp. Đính kèm cả 19 tệp trong một cuộc trò chuyện ChatGPT Plus, rồi gửi nguyên nội dung 01_PROMPT_CHATGPT_PLUS.txt để thực hiện ngay. Nếu giao diện đang giới hạn số tệp mỗi lần chọn, đính kèm theo nhiều đợt trong cùng cuộc trò chuyện và chỉ gửi prompt sau khi đủ tệp.

Chỉ dùng số liệu chính thức epochs30_batch64_seed42_v1: tám cấu hình, 240 epoch, 80.000 dòng dự đoán test. Tệp 17 là thăm dò chọn protocol, không gộp vào bảng A1.2. Gói không chứa báo cáo v01, checkpoint hoặc ảnh CIFAR-100 đầy đủ, để tránh nhầm phiên bản và tệp mô hình lớn.

Thứ tự đọc: 02 đề gốc; 03 protocol/đối chiếu; 04 phân tích; 05 bảng kết quả; 06–09 dữ liệu gốc/kiểm chứng; 10 mã; 11–15 hình; 16 nguồn; 17 thăm dò. 18 là danh mục/mã băm tệp, không phải số liệu thí nghiệm.

Thông tin có nguồn từ trang bìa báo cáo cũ: học viên Ngô Duy Bảo, mã số 2570395; môn CO5085; Học sâu và ứng dụng trong thị giác máy tính; HK261 (2026–2027); Trường Đại học Bách Khoa, ĐHQG TP.HCM; giảng viên Lê Thành Sách, Nguyễn Quốc Minh. Không tự thêm thành viên hoặc xác nhận nộp bài/đăng ký LMS.

Yêu cầu đầu ra: assignment1_report_v02.docx, Times New Roman, tiếng Việt học thuật tự nhiên, đen trắng. Nếu môi trường có công cụ xuất PDF, xuất assignment1_report_v02.pdf từ chính Word đó. Không dựng một bản PDF riêng có nội dung/bố cục khác. Chưa có Word v02 tại thời điểm chuẩn bị gói; báo cáo trong reports/ của repo vẫn là v01 trước lượt chính.

Repo: https://github.com/baond12/Deep_Learning_Course_HK261
Thư mục kết quả: A1/results/epochs30_batch64_seed42_v1/
Protocol: A1/protocol-a1.1.md và A1/protocol-a1.1.json

Sau khi nhận Word/PDF mới, lưu vào D:\\master_hcmut\\HK261\\Deep_Learning và gửi đường dẫn. Người học đã cho phép xuất PDF trước vòng review nhóm; việc xuất không có nghĩa nhóm đã duyệt nội dung hoặc đã nộp bài. Bản Word/PDF mới phải được đối chiếu trước khi thay đường dẫn ổn định reports/assignment1.docx và reports/assignment1.pdf. Git do người học tự commit/push.
""")
    prompt = """THỰC HIỆN NGAY VIỆC BIÊN SOẠN BÁO CÁO VÀ TẠO TỆP, KHÔNG CHỈ TRẢ LỜI KẾ HOẠCH HAY CHỜ TÔI XÁC NHẬN LẠI.

Bạn là người hỗ trợ biên soạn báo cáo học thuật cho Assignment 1 của môn CO5085. Tôi cung cấp 19 tệp và đã chốt protocol, hoàn tất thực nghiệm. Hãy đọc, đối chiếu và tạo báo cáo hoàn chỉnh assignment1_report_v02.docx. Không huấn luyện thêm, không đổi cấu hình, không bịa số liệu. Câu chữ là bản thảo hỗ trợ nhóm, không tự xác nhận nhóm đã rà soát hoặc đã nộp bài.

1. THỨ TỰ ƯU TIÊN NGUỒN
02_DE_BAI_GOC_A1.pdf là yêu cầu giảng viên. 03 mô tả protocol đã duyệt; 04 phân tích độc lập; 05 kết quả; 06 lịch sử; 07 metadata/kiểm chứng; 08 từng lớp; 09 dự đoán; 10 mã tái lập. Tệp 11–15 là hình từ đúng lượt mới; 16 là nguồn lý thuyết; 17 chỉ là pilot. Nếu phát hiện mâu thuẫn, kiểm tra dữ liệu/mã gốc, ghi nhận sai khác thay vì tự chọn con số có lợi. 18 là inventory, không phải kết quả.

2. PHẠM VI VÀ SỐ LIỆU CỐ ĐỊNH
CIFAR-100 đủ 100 lớp. Train/validation/test: 40.000/10.000/10.000, split seed 42, train seed 42. Ảnh gốc RGB 32×32 được phóng lên 224×224. Giữ resize, lật ngang p=0,5 cho train, chuẩn hóa ImageNet; validation/test không augmentation ngẫu nhiên. Adam, WD=1e-4, FP32, batch 64, 30 epoch mỗi cấu hình, không AMP/scheduler/warmup. LR=1e-5 chỉ cho ViT full, bảy cấu hình khác 1e-4. Tổng chính thức tám cấu hình, 240 epoch, một seed; mỗi mô hình có 10.000 dự đoán test, 80.000 dòng tổng.
ResNet partial: layer4 và fc; ViT partial: hai block encoder cuối và heads. Phần còn lại giữ nguyên, bao gồm thống kê BatchNorm; không tự mở encoder final LayerNorm, CLS hay embedding. Dùng ImageNet-1K V1, đầu ra 100 lớp. Lưu checkpoint có validation accuracy cao nhất, hòa giữ epoch sớm nhất; test từ đúng checkpoint đó. Không chọn lại bằng test.

3. QUY CÁCH WORD
Times New Roman cho toàn bộ văn bản, tiêu đề, chú thích, bảng, mục lục, số trang. Thân bài 12–13 pt, giãn dòng khoảng 1,3–1,5, bố cục A4 nếu công cụ hỗ trợ. Chỉ dùng chữ/đường màu đen, nền trắng; hình đen trắng/thang xám, không nhấn màu hoặc hyperlink xanh. Giữ tên mô hình, mã cấu hình, tên lớp CIFAR-100 theo nguồn; thuật ngữ kỹ thuật giải thích bằng tiếng Việt ở lần đầu. Văn phong tự nhiên, mạch lạc, chính xác và tiết chế. Không viết như nhật ký chat, không để chỉ dẫn nội bộ của gói trong thân báo cáo.
Dùng Heading Word có phân cấp và mục lục tự động, số trang, caption/bảng/hình đúng thứ tự. Không bịa số trang mục lục. Công thức trình bày thành công thức Word, có giải thích biến: cross entropy, accuracy, precision/recall/F1 từng lớp và macro F1. Tránh bảng quá rộng; tách bảng chất lượng với bảng chi phí và bảng protocol. Hình phải đủ lớn để đọc; không nhét cả tám ma trận 100 lớp vào một trang. Dùng tập PDF lỗi để lựa chọn hình/phụ lục, có thể trích tiểu ma trận những lớp khó và ghi rõ đây là phần được chọn, không thay thế ma trận đủ 100 lớp. Không biến các ảnh gốc dùng minh họa thành ảnh do AI sinh.

4. THÔNG TIN BÌA
Ngô Duy Bảo, mã số học viên 2570395; Trường Đại học Bách Khoa, ĐHQG TP.HCM; Khoa Khoa học và Kỹ thuật Máy tính; môn Học sâu và ứng dụng trong thị giác máy tính, CO5085, HK261 (2026–2027); giảng viên Lê Thành Sách và Nguyễn Quốc Minh. Lấy từ trang bìa v01 đã cung cấp trong thông tin gói, không tự thêm đồng tác giả. Tiêu đề báo cáo: So sánh ResNet18 và ViT B16 trên CIFAR 100 với các chế độ tiền huấn luyện và tinh chỉnh. Tên kiến trúc trong nội dung vẫn viết chính xác ViT-B/16.

5. CẤU TRÚC ĐỀ NGHỊ
Bìa; mục lục/danh mục hình bảng nếu cần; tóm tắt; mục tiêu và câu hỏi nghiên cứu; dữ liệu/EDA/nguồn/giấy phép; kiến trúc và phạm vi đóng băng; protocol A1.1 và căn cứ chọn từ pilot; kết quả A1.2; phân tích đường cong/chi phí/lỗi; thảo luận giới hạn; kết luận; tài liệu tham khảo; phụ lục tái lập và thông tin kiểm chứng cần thiết. Không đặt mục tiêu số trang cứng khiến nội dung bị kéo dài hoặc co chữ; ưu tiên đủ bằng chứng và hình dễ đọc. Phụ lục không chép toàn bộ 80.000 dự đoán hay hàng chục trang mã; dẫn repo và các tệp nguồn.
Giới thiệu CNN/residual learning, ViT patch embedding/self attention, transfer learning và đóng băng đủ để đọc thực nghiệm, không viết tổng quan lý thuyết dài tách rời số liệu. Liên kết với E1–E3 về cách thiết kế/đánh giá: Exercise dùng MNIST, 19 cấu hình và 1.900 epoch; A1 là bài riêng dùng CIFAR-100. Không trộn dataset, bảng số liệu hoặc buộc epoch/batch A1 bằng Exercise.

6. PHÂN TÍCH BẮT BUỘC
So sánh scratch/pretrained trong mỗi họ; head/partial/full; CNN/ViT ở cùng chế độ. Ghi đủ validation tốt nhất, epoch được chọn, test accuracy/F1/loss. Phân biệt tham số tổng và tham số được học. Báo riêng train, validation, toàn fit, bộ nhớ allocated/reserved với đơn vị đúng. Thời gian fit chứa callback/ghi checkpoint, chưa phải GPU compute thuần; chỉ có ảnh chụp GPU trước lượt, không chứng minh cách ly GPU toàn thời gian. Tổng fit khoảng 20,80 giờ.
ViT full cao nhất test accuracy 86,46%, F1 0,8651; ResNet full 78,97%, F1 0,7893. Không chỉ nêu accuracy: ViT head loss 0,6397 thấp hơn full 0,6475 dù accuracy thấp hơn; partial loss 0,9169 và nhiều lỗi rất tự tin. Phân tích bằng dữ liệu confidence trong 04, tránh tự gọi đó là kiểm định hiệu chuẩn đầy đủ.
Hai head đạt best epoch 30, chưa chứng minh hội tụ. Scratch có train/validation chênh lệch lớn; loss và accuracy có thể chọn epoch khác nhau. Quy tắc chọn bằng accuracy đã khóa vẫn giữ nguyên. Phân tích F1 theo lớp, cặp lỗi có hướng, ít nhất vài trường hợp từ hình thật. Mẫu sai là 25 lỗi đầu theo thứ tự loader, không phải mẫu ngẫu nhiên. Giải thích do lớp gần nhau/độ phân giải nhỏ phải viết là giả thuyết, không kết luận nhân quả hoặc tự quy sai nhãn.

7. GIỚI HẠN VÀ TÍNH TRUNG THỰC
Chỉ một seed; không viết khác biệt có ý nghĩa thống kê, không thêm độ lệch chuẩn/CI hoặc p-value chưa tính. Hai họ khác quy mô và công thức pretrained, cùng ImageNet-1K không đồng nghĩa cùng budget. Full ViT có LR khác, hai full đã được khảo sát hai LR trong khi sáu cấu hình khác chỉ một LR; không cô lập hoàn toàn ảnh hưởng freeze/kiến trúc. Một ngân sách 30 epoch không bảo đảm hội tụ, không nói LR đã tối ưu toàn cục. Không khẳng định ViT không thể học từ đầu. Test từng được xem ở các lượt cũ, không gọi đây là tập chưa từng thấy. Các lượt cũ NaN không được trộn với 240 epoch hữu hạn của lượt mới hoặc suy ra lỗi đã được chữa trong mọi tình huống.
License: trang CIFAR gốc chưa nêu rõ giấy phép chuẩn hóa, thẻ uoft-cs/cifar100 ghi unknown. Ghi trung thực nguồn, mục đích học tập, yêu cầu trích dẫn; không tự gán MIT/CC hoặc phát biểu đã có sự chấp thuận chưa được cung cấp.

8. TRÍCH DẪN VÀ TÁI LẬP
Dùng nguồn sơ cấp ở tệp 16: trang CIFAR/tác giả, bài ResNet/ViT, tài liệu torchvision 0.17, Adam, PyTorch và scikit-learn. Đánh số trích dẫn nhất quán, có tác giả/tựa/năm/URL, không dựng DOI hoặc tài liệu không có trong gói. Code repo https://github.com/baond12/Deep_Learning_Course_HK261 ; kết quả A1/results/epochs30_batch64_seed42_v1 ; protocol A1/protocol-a1.1.json ; chương trình A1/run_protocol.py ; lệnh và giải thích train/test trong tệp 03. Môi trường thật lấy từ 07, không ghi nhầm môi trường CPU máy biên soạn thành môi trường huấn luyện.
Khai báo hỗ trợ AI: chỉ mô tả công cụ hỗ trợ tổng hợp/biên soạn theo quy định môn học, không tự xác nhận nhóm đã kiểm tra hết hoặc tự ký cam kết thay tác giả.

9. TỆP ĐẦU RA VÀ XUẤT PDF
Tạo assignment1_report_v02.docx có thể tải về. Nếu có công cụ chuyển đổi, xuất tiếp assignment1_report_v02.pdf TỪ CHÍNH DOCX đó; kiểm tra chữ tiếng Việt, font, bảng/hình không cắt, mục lục/số trang và nội dung đồng bộ. Người học đồng ý xuất PDF trước review nhóm. Nếu không có công cụ xuất DOCX thành PDF, nói rõ giới hạn và giao DOCX; không dựng PDF riêng bằng một trình sinh khác rồi gọi đó là bản xuất từ Word. Không dừng ở câu trả lời hứa tạo file hoặc chờ câu “thực hiện prompt 01”.

10. TỰ KIỂM TRƯỚC KHI GIAO
Đủ tám cấu hình, 240 epoch; bảy LR 1e-4, một LR 1e-5; train/validation/test 40k/10k/10k; bảng/hình cùng lượt; accuracy/F1 khớp CSV; params đủ hai loại; loss không bị gán là tốt nhất chỉ vì accuracy cao; giới hạn một seed/LR/pretrain/GPU/license đầy đủ; Times New Roman/đen trắng; không còn chỉ dẫn soạn thảo hoặc chỗ trống bịa dữ liệu. Kết thúc bằng liên kết tải tệp và vài ghi chú ngắn thực chất.
"""
    text(filenames[1],prompt)
    shutil.copy2(ROOT/"Exercise/assignment1-vne.pdf",bundle/filenames[2])
    protocol_text=(ROOT/"A1/protocol-a1.1.md").read_text(encoding="utf-8")
    guide=(ROOT/"A1/RUN_PROTOCOL.md").read_text(encoding="utf-8")
    # The archived approved text remains untouched in the repo. Clarify current status in this merged reading copy.
    text(filenames[3], "# Hồ sơ yêu cầu và protocol\n\nTrạng thái hiện tại: đủ tám cấu hình đã train/test xong, đủ EDA thật. Bản protocol bên dưới được giữ như khi duyệt trước lượt chạy; các câu 'chưa có kết quả' là trạng thái lúc khóa protocol, không phải hiện trạng.\n\n" + protocol_text + "\n\n# Lệnh tái lập và ý nghĩa các giai đoạn\n\n" + guide + "\n\n# Đối chiếu yêu cầu giảng viên\n\n| Yêu cầu | Bằng chứng lượt mới |\n| --- | --- |\n| CNN và Transformer | ResNet18 và ViT-B/16 |\n| Scratch/pretrained cùng split | Có cho cả hai họ, SHA-256 split giữ nguyên |\n| Head, partial, full pretrained | Đủ ba chế độ cho từng họ |\n| Validation/test accuracy, F1 | Bảng 05 và dữ liệu 07–09 |\n| Thời gian, tham số, bộ nhớ | 05 và metadata GPU/pha trong 07 |\n| Đường cong và phân tích lỗi | 06, 08–09, 11–14 |\n| EDA | Mảng gốc 50.000×32×32×3; 400 train/100 val mỗi lớp; hình 15; test 100 mỗi lớp đối chiếu từ dự đoán |\n| License | Chưa xác định tên chuẩn hóa, phải công khai giới hạn, không tự gán |\n| Báo cáo Word/PDF | Đang biên soạn từ lượt mới, chưa xác nhận nộp bài |\n\nNgười học: Ngô Duy Bảo, 2570395; thông tin lấy từ bìa v01, không sao chép số liệu v01.\n")
    text(filenames[4],analysis)
    write_csv(bundle/filenames[5],records)
    histories=[{"experiment":n,**h} for n in NAMES for h in train["experiments"][n]["history"]]
    write_csv(bundle/filenames[6],histories)
    write_json(bundle/filenames[7],{"training":train,"test":test,"checks":checks,
                                 "dataset_summary":read_json(source/"dataset_summary.json"),
                                 "class_counts":read_csv(source/"class_counts.csv"),"top_confusions":pairs,
                                 "original_artifact_inventory":originals})
    write_csv(bundle/filenames[8],classes)
    write_csv(bundle/filenames[9],predictions)
    code_files=["A1/run_protocol.py","A1/engine.py","A1/data.py","A1/models.py","A1/pilot.py","common/utils.py","common/plots.py","A1/protocol-a1.1.json","requirements.txt"]
    text(filenames[10],"MÃ NGUỒN DÙNG CHO LƯỢT CHÍNH\n\nMã đã đối chiếu SHA-256 với metadata của lượt huấn luyện; A1/pilot.py cung cấp helper metadata/hash, không được gọi để train lượt chính.\n\n" + "\n\n".join(f"===== {n} =====\n{(ROOT/n).read_text(encoding='utf-8')}" for n in code_files))
    for src,index in [("figure_resnet_curves.png",11),("figure_vit_curves.png",12),("figure_comparison.png",13),("error_atlas.pdf",14)]:
        shutil.copy2(target/src,bundle/filenames[index])
    # This is a presentation copy only; preserve the original RGB artifact in the result directory.
    Image.open(source/"dataset_samples_train.png").convert("L").save(bundle/filenames[15])
    text(filenames[16],"""# Cơ sở lý thuyết và nguồn sơ cấp

Ngày đối chiếu nguồn: 19/10/2026. Dùng các nguồn dưới đây để hỗ trợ diễn giải, không mượn số liệu benchmark công bố làm số liệu thực nghiệm nhóm.

1. Alex Krizhevsky, 2009, Learning Multiple Layers of Features from Tiny Images. Trang dữ liệu: https://cave.cs.toronto.edu/kriz/cifar.html ; báo cáo tác giả liên kết ở cuối trang. CIFAR-100 có 100 lớp chi tiết, mỗi lớp 500 train/100 test gốc; 20 nhóm lớn không phải 100 nhãn chính được dùng ở đây. Ảnh RGB 32×32. Trang yêu cầu trích dẫn báo cáo. Không thấy giấy phép chuẩn hóa trên trang; không tự diễn giải thành MIT/CC.
2. Thẻ dữ liệu do tổ chức uoft-cs cung cấp: https://huggingface.co/datasets/uoft-cs/cifar100/blob/main/README.md ; license được ghi unknown. Đây là mô tả trạng thái, không phải một giấy phép mới được cấp. Không lấy điều khoản sử dụng website Hugging Face làm giấy phép dataset.
3. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, Deep Residual Learning for Image Recognition, CVPR 2016 (preprint 2015): https://arxiv.org/abs/1512.03385 . Residual learning dùng đường nối tắt để học phần hiệu chỉnh; ResNet18 là CNN có các khối residual. Bài gốc giải thích kiến trúc, không chứng minh ResNet luôn tốt hơn ViT trên CIFAR-100 của nhóm.
4. Alexey Dosovitskiy và cộng sự, An Image is Worth 16x16 Words Transformers for Image Recognition at Scale, ICLR 2021 (preprint 2020): https://arxiv.org/abs/2010.11929 . ViT dùng chuỗi patch nhúng với thông tin vị trí, encoder attention và đầu phân loại. B/16 dùng patch 16; ảnh 224×224 tạo 14×14=196 patch, thêm CLS thành 197 token. Không nhầm với token hàng/CNN-stem E2 hoặc khẳng định nhóm tự viết MSA cho A1: A1 dùng torchvision ViT.
5. Torchvision 0.17, resnet18: https://docs.pytorch.org/vision/0.17/models/generated/torchvision.models.resnet18.html ; vit_b_16: https://docs.pytorch.org/vision/0.17/models/generated/torchvision.models.vit_b_16.html . Hai trọng số IMAGENET1K_V1 có công thức tiền huấn luyện khác nhau. Số tham số trong tài liệu là đầu 1.000 lớp; số tham số báo cáo nhóm phải lấy sau khi thay head 100 lớp, tức 11.227.812/85.875.556. Transform chính thức inference gồm resize 256/crop 224; nhóm dùng resize trực tiếp 224 và phải mô tả đúng khác biệt.
6. Diederik P. Kingma, Jimmy Ba, Adam A Method for Stochastic Optimization, ICLR 2015: https://arxiv.org/abs/1412.6980 . Vòng lặp nhóm dùng torch.optim.Adam với weight_decay=1e-4, không phải AdamW. Giải thích cập nhật thích nghi ở mức cần thiết; không tự thêm scheduler, warmup, label smoothing hoặc gradient clipping hữu hạn. Mã chỉ kiểm tra norm gradient bằng ngưỡng vô hạn để phát hiện nonfinite.
7. PyTorch, CrossEntropyLoss: https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html . Đầu vào là logits; vòng lặp không cần thêm softmax trước cross entropy. Loss trung bình và accuracy đánh giá khía cạnh khác nhau. Softmax chỉ dùng ở bước tạo confidence cho dự đoán. Việc accuracy tốt hơn không kéo theo cross entropy thấp hơn.
8. Scikit-learn, f1_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html . Macro F1 là trung bình không trọng số của F1 từng lớp; F1 mỗi lớp là trung bình điều hòa precision/recall, không phải accuracy hoặc trung bình F1 của các batch. Tính trên đủ 100 nhãn, zero_division=0. Confusion matrix hàng thật/cột dự đoán.
9. PyTorch, bộ nhớ CUDA: https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html và https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_reserved.html . Allocated và reserved là khái niệm khác nhau, phản ánh tiến trình PyTorch; chia MiB cho 1024 ra GiB. Không dùng các số này thay tổng VRAM nvidia-smi hoặc bộ nhớ mọi tiến trình.

# Công thức cần trình bày

Accuracy = số dự đoán đúng / tổng số ảnh. Precision_c = TP_c/(TP_c+FP_c); recall_c = TP_c/(TP_c+FN_c); F1_c = 2*precision_c*recall_c/(precision_c+recall_c), quy ước bằng 0 khi mẫu số 0; macro F1 = (1/100)*tổng F1_c. Cross entropy của một ảnh là -log p(nhãn thật); loss test là trung bình trên 10.000 ảnh. Viết thành công thức Word, không để nguyên chuỗi LaTeX khó đọc.

# Diễn giải được phép

Có thể nhận xét đóng băng giảm gradient/optimizer và số tham số cập nhật, nhưng backbone vẫn forward. Phạm vi mở khác nhau và LR khác nhau là một phần công thức đã chạy. Đường cong train/validation gợi ý quá khớp, không chứng minh nguyên nhân. Các kết luận chất lượng/chi phí lấy trực tiếp từ 04–09; không suy ra ý nghĩa thống kê từ một seed. Giữ giới hạn nguồn và dữ liệu giống cách trình bày E1–E3.
""")
    pilots=[]
    for folder in sorted((ROOT/"A1/results/pilot_evidence_v1").iterdir()):
        if not folder.is_dir():continue
        m=read_json(folder/"metrics.json")
        for name,r in m["experiments"].items():
            pilots.append({"pilot_folder":folder.name,"experiment":name,"epochs":m["config"]["epochs"],
                           "batch_size":m["config"]["batch_size"],"learning_rate":m["config"]["learning_rate"],
                           "best_val_accuracy":r["best_val_accuracy"],"best_epoch":r["best_epoch"],
                           "final_val_accuracy":r["val_accuracy"],"fit_seconds":r["training_validation_seconds"],
                           "peak_allocated_mib":r["peak_allocated_mib"],"peak_reserved_mib":r["peak_reserved_mib"],
                           "test_evaluated":False,"purpose":"protocol_selection_only"})
    assert len(pilots)==26 and sum(r["epochs"] for r in pilots)==66
    write_csv(bundle/filenames[17],pilots)
    inventory=[]
    for filename in filenames[:-1]:
        path=bundle/filename
        inventory.append({"filename":filename,"bytes":path.stat().st_size,"sha256":sha(path),
                          "source_scope":"pilot_only" if filename.startswith("17_") else "instructions" if filename[:2] in ("00","01") else "current_run_or_protocol"})
    write_csv(bundle/filenames[18],inventory)
    assert len(list(bundle.iterdir()))==19 and all(p.is_file() for p in bundle.iterdir())


def update_landing(records):
    base = "../A1/results/epochs30_batch64_seed42_v1"
    labels = [f"{family} · {label}" for family in ("ResNet18", "ViT-B/16") for label in LABELS]
    lines = ['      <section id="ket-qua" class="section experiment-section">',
             '        <div class="section-heading"><p class="section-kicker">Kết quả đã đối chiếu</p><h2>Tám cấu hình theo quy trình A1.1</h2></div>',
             '        <p>Cùng CIFAR-100, split 40.000/10.000/10.000 ảnh, một seed 42. Mỗi cấu hình chạy đủ 30 epoch; checkpoint được chọn bằng accuracy trên tập xác thực, sau đó đánh giá trên đủ 10.000 ảnh kiểm tra.</p>',
             '        <div class="result-table-scroll" tabindex="0" role="region" aria-label="Bảng kết quả Assignment 1 có thể cuộn ngang">',
             '          <table class="result-table"><caption>Chất lượng và chi phí của lượt epochs30_batch64_seed42_v1</caption>',
             '            <thead><tr><th scope="col">Cấu hình</th><th scope="col">LR</th><th scope="col">Epoch chọn</th><th scope="col">Xác thực tốt nhất</th><th scope="col">Accuracy test</th><th scope="col">Macro F1</th><th scope="col">Toàn fit (giờ)</th></tr></thead><tbody>']
    for label,r in zip(labels,records):
        vals = [label, f"{r['learning_rate']:g}",str(r["best_epoch"]),f"{r['best_val_accuracy']:.2%}".replace(".",","),
                f"{r['test_accuracy']:.2%}".replace(".",","),f"{r['macro_f1']:.4f}".replace(".",","),
                f"{r['fit_seconds_including_callbacks']/3600:.2f}".replace(".",",")]
        lines.append('            <tr><th scope="row">'+vals[0]+'</th>'+''.join(f'<td>{v}</td>' for v in vals[1:])+'</tr>')
    lines += ['          </tbody></table>','        </div>',
              '        <p class="result-note">ViT toàn bộ đạt accuracy cao nhất 86,46%, nhưng thời gian fit khoảng 9,84 lần ResNet toàn bộ trong lượt này. ViT chỉ học đầu phân loại có test loss 0,6397, thấp hơn ViT toàn bộ 0,6475 dù accuracy thấp hơn. Một seed, LR khác nhau và quy mô/công thức tiền huấn luyện khác nhau chưa cho phép kết luận ưu thế phổ quát hoặc ý nghĩa thống kê. Thời gian fit gồm ghi checkpoint; bộ nhớ trong số liệu là allocated/reserved của tiến trình PyTorch.</p>',
              '        <div class="artifact-actions">']
    artifacts = [("report_summary.csv","Bảng tổng hợp chất lượng và chi phí"),("test/test_summary.csv","Số liệu test gốc"),
                 ("metrics.json","Metadata và lịch sử train"),("test/test_metrics.json","Metadata test"),
                 ("report_checks.json","Biên bản đối chiếu dữ liệu"),("report_class_metrics.csv","Chỉ số của 100 lớp cho tám mô hình"),
                 ("report_top_confusions.csv","Các cặp nhầm lẫn có hướng"),("error_atlas.pdf","Tập hình phân tích lỗi PDF"),
                 ("dataset_summary.json","EDA và quy mô dữ liệu"),("class_counts.csv","Số ảnh từng lớp"),
                 ("dataset_samples_train.png","Ảnh mẫu từ tập huấn luyện"),("figure_comparison.png","Biểu đồ chất lượng và chi phí"),
                 ("figure_resnet_curves.png","Đường cong ResNet18"),("figure_vit_curves.png","Đường cong ViT-B/16")]
    for filename,label in artifacts:
        lines.append(f'          <a href="{base}/{filename}">{label}</a>')
    lines += ['        </div>',
              '        <details class="artifact-details"><summary>Minh chứng riêng của từng cấu hình</summary>',
              '          <div class="result-table-scroll" tabindex="0" role="region" aria-label="Liên kết minh chứng từng mô hình có thể cuộn ngang">',
              '          <table class="result-table"><caption>Lịch sử, đường cong, ma trận và dự đoán từ đúng lượt mới</caption>',
              '            <thead><tr><th scope="col">Cấu hình</th><th scope="col">Huấn luyện</th><th scope="col">Phân tích lỗi</th><th scope="col">Dự đoán và chỉ số lớp</th></tr></thead><tbody>']
    for label,r in zip(labels,records):
        n=r["experiment"]
        lines.append(f'            <tr><th scope="row">{label}</th><td><a href="{base}/history_{n}.csv" download>Lịch sử</a> · <a href="{base}/learning_curves_{n}.png">Đường cong</a></td><td><a href="{base}/test/confusion_matrix_{n}.png">Ma trận hình</a> · <a href="{base}/test/confusion_matrix_{n}.csv" download>Ma trận số</a> · <a href="{base}/test/misclassified_{n}.png">Mẫu sai</a></td><td><a href="{base}/test/predictions_{n}.csv" download>Dự đoán</a> · <a href="{base}/test/classification_report_{n}.json">Chỉ số từng lớp</a></td></tr>')
    lines += ['          </tbody></table></div>','        </details>',
              '        <p class="result-note">Mẫu sai là các lỗi đầu tiên theo thứ tự bộ nạp kiểm tra, không phải mẫu ngẫu nhiên. Các ảnh/CSV gốc được giữ nguyên; tập PDF tổng hợp dùng thang xám để phục vụ báo cáo. Hoàn tất thực nghiệm không đồng nghĩa đã nộp bài hoặc đã có báo cáo mới được nhóm duyệt.</p>',
              '      </section>','']
    page = ROOT/"pages/assignment1.html"
    content = page.read_text(encoding="utf-8")
    start = content.find('      <section id="ket-qua"')
    end = content.index('      <section class="section two-column">')
    if start < 0:
        start = end
    page.write_text(content[:start] + "\n".join(lines) + "\n" + content[end:],encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=ROOT/"A1_ChatGPT_Plus_30epoch")
    args = parser.parse_args()
    source = args.source.resolve()
    target = ROOT/"A1/results/epochs30_batch64_seed42_v1"
    print("Auditing source artifacts and all eight checkpoint hashes...", flush=True)
    train,test,records,classes,pairs,predictions,paired,matrices,checks = audit(source)
    originals = copy_results(source,target)
    checks["copied_original_files"] = originals
    write_json(target/"report_checks.json", checks)
    write_csv(target/"report_summary.csv",records)
    write_csv(target/"report_class_metrics.csv",classes)
    write_csv(target/"report_top_confusions.csv",pairs)
    write_csv(target/"report_pairwise_errors.csv",paired)
    analysis = analysis_text(train,test,records,classes,pairs)
    (target/"analysis.md").write_text(analysis,encoding="utf-8")
    (ROOT/"docs/a1-final-analysis.md").write_text(analysis,encoding="utf-8")
    configure_fonts()
    curves(train,NAMES[:4],target/"figure_resnet_curves.png","ResNet18 trên CIFAR-100")
    curves(train,NAMES[4:],target/"figure_vit_curves.png","ViT-B/16 trên CIFAR-100")
    comparison(records,target/"figure_comparison.png")
    atlas(source,target/"error_atlas.pdf",records,classes,pairs)
    bundle=args.bundle.resolve()
    assert bundle.is_relative_to(ROOT),"Bundle must remain inside the workspace"
    bundle.mkdir(exist_ok=True)
    package(bundle,source,target,train,test,records,classes,pairs,predictions,analysis,checks,originals)
    update_landing(records)
    print(f"Completed: {target}\nBundle: {bundle}",flush=True)


if __name__ == "__main__":
    main()
