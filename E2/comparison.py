"""Derive the two E2 comparisons from completed experiments, without inventing results."""
from __future__ import annotations

from pathlib import Path

from common.plots import plot_learning_curves, save_summary_csv
from E2.models import TOKENIZATIONS


def write_comparisons(experiments: dict, histories: dict, out_dir: Path) -> None:
    rows = []
    for name, result in experiments.items():
        rows.append({"experiment": name, "attention": result["attention"],
                     "tokenization": result["tokenization"], "image_tokens": result["image_tokens"],
                     "total_tokens": result["total_tokens"], "parameters": result["parameters"],
                     "best_val_accuracy": result["best_val_accuracy"],
                     "test_accuracy": result["test_accuracy"], "macro_f1": result["macro_f1"],
                     "training_time_seconds": result["training_time_seconds"],
                     "best_epoch": result["best_epoch"]})
    if not rows:
        return
    save_summary_csv(rows, out_dir / "comparison_tokenization.csv")
    pairs, notes = [], []
    for tokenizer in TOKENIZATIONS:
        custom = experiments.get(f"custom_{tokenizer}")
        pytorch = experiments.get(f"pytorch_{tokenizer}")
        if custom is None and pytorch is None:
            continue
        complete = custom is not None and pytorch is not None
        delta = 100 * (pytorch["test_accuracy"] - custom["test_accuracy"]) if complete else None
        pair = {"tokenization": tokenizer, "pair_complete": complete,
                "custom_test_accuracy": custom["test_accuracy"] if custom else None,
                "pytorch_test_accuracy": pytorch["test_accuracy"] if pytorch else None,
                "pytorch_minus_custom_accuracy_pp": delta,
                "custom_train_val_seconds": custom["training_time_seconds"] if custom else None,
                "pytorch_train_val_seconds": pytorch["training_time_seconds"] if pytorch else None,
                "pytorch_over_custom_time": (pytorch["training_time_seconds"] / custom["training_time_seconds"])
                if complete and custom["training_time_seconds"] > 0 else None}
        pairs.append(pair)
        if complete:
            notes.append(f"- {tokenizer}: custom đạt {100 * custom['test_accuracy']:.2f}%, "
                         f"PyTorch đạt {100 * pytorch['test_accuracy']:.2f}%; chênh lệch "
                         f"PyTorch − custom là {delta:+.2f} điểm phần trăm. Thời gian train + val "
                         f"lần lượt {custom['training_time_seconds']:.2f}s và "
                         f"{pytorch['training_time_seconds']:.2f}s.")
        else:
            notes.append(f"- {tokenizer}: chưa hoàn tất cả hai bản, chưa so sánh cặp.")
        group = {name: history for name, history in histories.items()
                 if name in (f"custom_{tokenizer}", f"pytorch_{tokenizer}")}
        if group:
            plot_learning_curves(group, out_dir / f"learning_curves_{tokenizer}.png")
    save_summary_csv(pairs, out_dir / "comparison_msa.csv")
    token_notes = []
    for backend in ("custom", "pytorch"):
        baseline = experiments.get(f"{backend}_patch")
        for tokenizer in TOKENIZATIONS:
            result = experiments.get(f"{backend}_{tokenizer}")
            if result is None:
                continue
            statement = (f"- {backend}/{tokenizer}: {result['image_tokens']} token ảnh, "
                         f"{result['parameters']} tham số; test accuracy "
                         f"{100 * result['test_accuracy']:.2f}%, train + val "
                         f"{result['training_time_seconds']:.2f}s.")
            if baseline is not None and tokenizer != "patch":
                delta = 100 * (result["test_accuracy"] - baseline["test_accuracy"])
                difference = result["training_time_seconds"] - baseline["training_time_seconds"]
                statement += (f" So với patch trong cùng backend: accuracy chênh {delta:+.2f} "
                              f"điểm phần trăm, thời gian chênh {difference:+.2f}s.")
            token_notes.append(statement)
    text = ["# E2 — So sánh MSA và tokenization", "",
            f"Đã hoàn tất {len(experiments)} thực nghiệm. Chỉ sử dụng các kết quả hoàn tất dưới đây.",
            "", "## MSA tự viết và PyTorch trên cùng tokenization", "", *notes, "",
            "## Đối chiếu tokenization", "",
            "Đọc comparison_tokenization.csv, so sánh bốn cách tokenize trong cùng backend "
            "(custom với custom, PyTorch với PyTorch), cùng split và cấu hình huấn luyện.",
            "", *token_notes, "",
            "CNN-stem thêm phép học đặc trưng và tham số; overlap thay đổi độ dài chuỗi và "
            "positional embedding. Không quy toàn bộ khác biệt accuracy cho riêng số token.",
            "Thời gian gồm train và validation, không gồm test, kiểm tra tương đương hoặc vẽ hình. "
            "Đây không phải latency suy luận; không diễn giải thời gian khi GPU đang chia sẻ tải "
            "như một benchmark phần cứng độc lập.",
            "", "## Giới hạn", "",
            "verification.json kiểm tra cùng trọng số và tắt dropout khi đối chiếu số học. "
            "Hai lượt train thật dùng dropout có cùng vị trí/xác suất nhưng không bảo đảm "
            "cùng mặt nạ hoặc kết quả bit-for-bit. Accuracy gần nhau không chứng minh MSA đúng; "
            "bằng chứng trực tiếp là kiểm tra đầu ra/gradient.",
            "Một seed chưa đủ để kết luận khác biệt có ý nghĩa thống kê. Không dùng test "
            "để chọn lại siêu tham số. Kết quả lịch sử E2 có dropout khác không gộp vào bộ này.", ""]
    (out_dir / "comparison.md").write_text("\n".join(text), encoding="utf-8")
    plot_comparison(rows, out_dir / "comparison.png")


def plot_comparison(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    indexed = {row["experiment"]: row for row in rows}
    for backend, offset, color in (("custom", -0.19, "black"), ("pytorch", 0.19, "0.65")):
        for ax, metric, scale in ((axes[0], "test_accuracy", 100),
                                   (axes[1], "training_time_seconds", 1)):
            positions, values = [], []
            for index, tokenizer in enumerate(TOKENIZATIONS):
                row = indexed.get(f"{backend}_{tokenizer}")
                if row is not None:
                    positions.append(index + offset)
                    values.append(scale * row[metric])
            ax.bar(positions, values, width=.36, color=color, label=backend)
            ax.set_xticks(range(len(TOKENIZATIONS)), TOKENIZATIONS, rotation=20)
            ax.legend()
    axes[0].set_ylabel("Test accuracy (%)")
    axes[1].set_ylabel("Train + validation (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
