from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))

import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze A1 experiment results.")
    parser.add_argument("--results-dir", type=str, default="A1/results")
    return parser.parse_args()


def load_rows(results_dir: Path) -> list[dict[str, object]]:
    summary_path = results_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path}. Run A1/train.py first, then run this analyzer."
        )

    with summary_path.open(newline="", encoding="utf-8") as f:
        rows: list[dict[str, object]] = list(csv.DictReader(f))

    metrics_path = results_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for row in rows:
            exp = str(row["experiment"])
            experiment_metrics = metrics.get("experiments", {}).get(exp, {})
            row["provenance"] = experiment_metrics.get("provenance", {})
            row["training_diagnostics"] = experiment_metrics.get("training_diagnostics", {})
            seconds = experiment_metrics.get("training_time_seconds")
            if seconds is not None:
                row["training_time_seconds"] = float(seconds)

    for row in rows:
        for key in [
            "total_parameters",
            "trainable_parameters",
            "best_epoch",
            "best_val_accuracy",
            "test_loss",
            "test_accuracy",
            "macro_f1",
            "training_time_seconds",
        ]:
            if key in row and row[key] not in ("", None):
                row[key] = float(row[key])
    return rows


def plot_bar(rows: list[dict[str, object]], key: str, title: str, ylabel: str, out_path: Path) -> None:
    labels = [str(row["experiment"]) for row in rows]
    values = [float(row.get(key, 0.0) or 0.0) for row in rows]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    colors = ["#0f766e" if str(row["family"]) == "CNN" else "#1d4ed8" for row in rows]
    ax.bar(range(len(rows)), values, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def format_float(value: object, digits: int = 4) -> str:
    if value in ("", None):
        return "-"
    return f"{float(value):.{digits}f}"


def make_markdown(rows: list[dict[str, object]]) -> str:
    sorted_by_acc = sorted(rows, key=lambda row: float(row["test_accuracy"]), reverse=True)
    best_acc = sorted_by_acc[0]
    sorted_by_f1 = sorted(rows, key=lambda row: float(row["macro_f1"]), reverse=True)
    best_f1 = sorted_by_f1[0]

    cnn_rows = [row for row in rows if row["family"] == "CNN"]
    transformer_rows = [row for row in rows if row["family"] == "Transformer"]

    def avg(group: list[dict[str, object]], key: str) -> float:
        return sum(float(row[key]) for row in group) / max(len(group), 1)

    lines = [
        "# A1.2 Results Analysis",
        "",
        "## Best Results",
        "",
        f"- Best test accuracy: `{best_acc['experiment']}` with {format_float(best_acc['test_accuracy'])}.",
        f"- Best macro F1: `{best_f1['experiment']}` with {format_float(best_f1['macro_f1'])}.",
        f"- CNN average test accuracy: {avg(cnn_rows, 'test_accuracy'):.4f}.",
        f"- Transformer average test accuracy: {avg(transformer_rows, 'test_accuracy'):.4f}.",
        "",
        "## Summary Table",
        "",
        "| Experiment | Family | Mode | Trainable params | Best val acc | Test acc | Macro F1 | Time |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in rows:
        lines.append(
            "| "
            f"{row['experiment']} | {row['family']} | {row['mode']} | "
            f"{int(float(row['trainable_parameters'])):,} | "
            f"{format_float(row['best_val_accuracy'])} | "
            f"{format_float(row['test_accuracy'])} | "
            f"{format_float(row['macro_f1'])} | "
            f"{row.get('training_time', '-')} |"
        )

    restored = [
        str(row["experiment"])
        for row in rows
        if row.get("provenance", {}).get("final_metrics_source") == "user_supplied_console_log"
    ]
    if restored:
        lines.extend(
            [
                "",
                "## Data Provenance and Limits",
                "",
                "Restored from user-supplied console logs: " + ", ".join(f"`{exp}`" for exp in restored) + ".",
                "These test metrics have four decimal places; durations are reported to whole seconds. "
                "Their parameter counts are expected counts derived from the current model definition, "
                "rather than recovered measurements. Their full runtime configurations still need confirmation.",
                "The user reported concurrent execution. GPU models and resource allocation are not recorded, "
                "so elapsed durations are not a controlled speed comparison.",
                "Family averages above combine different training modes; they are not averages over repeated seeds.",
                "See `README.md`, `metrics.json` and `results_inventory.csv` for the complete provenance.",
            ]
        )
    for row in rows:
        first_nonfinite = row.get("training_diagnostics", {}).get("first_nonfinite_loss_epoch")
        if first_nonfinite is not None:
            lines.append(
                f"- `{row['experiment']}`: nonfinite loss from epoch {first_nonfinite}; "
                f"best validation state at epoch {int(float(row['best_epoch']))}. "
                "Do not interpret this as stable training throughout the recorded history."
            )

    lines.extend(
        [
            "",
            "## Interpretation Prompts",
            "",
            "- Compare from-scratch vs pretrained within ResNet18 and ViT-B/16.",
            "- Check whether head-only freeze is fast but less adaptive than partial/full fine-tuning.",
            "- Discuss whether ViT-B/16 benefits more from pretrained weights than ResNet18 on full CIFAR-100.",
            "- Use confusion matrices and misclassified samples to name classes that are frequently confused.",
            "",
            "## Generated Figures",
            "",
            "- `bar_test_accuracy.png`",
            "- `bar_macro_f1.png`",
            "- `bar_trainable_parameters.png`",
            "- `bar_training_time_seconds.png`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    rows = load_rows(results_dir)

    plot_bar(rows, "test_accuracy", "A1 Test Accuracy", "Accuracy", results_dir / "bar_test_accuracy.png")
    plot_bar(rows, "macro_f1", "A1 Macro F1", "Macro F1", results_dir / "bar_macro_f1.png")
    plot_bar(
        rows,
        "trainable_parameters",
        "A1 Trainable Parameters",
        "Trainable parameters",
        results_dir / "bar_trainable_parameters.png",
    )
    if "training_time_seconds" in rows[0]:
        plot_bar(
            rows,
            "training_time_seconds",
            "A1 Training Time",
            "Seconds",
            results_dir / "bar_training_time_seconds.png",
        )

    analysis = make_markdown(rows)
    output_path = results_dir / "analysis.md"
    output_path.write_text(analysis, encoding="utf-8")
    print(f"Saved A1 analysis to {output_path}")


if __name__ == "__main__":
    main()
