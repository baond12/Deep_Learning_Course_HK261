"""Aggregate validation results only. Never reads final_test for model selection."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from E1.run_softmax_experiments import (
    FACTORS, atomic_text, data_identity, digest, read_json, write_csv, write_json,
)

METRICS = ("best_val_accuracy", "last_val_accuracy", "best_val_loss", "last_val_loss",
           "clean_train_accuracy", "clean_train_loss", "train_val_seconds", "optimizer_steps")


def mean_std(values: list[float]) -> dict:
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None}


def analyze(study: Path, make_plot=True) -> dict:
    manifest = read_json(study / "manifest.json")
    config = manifest["config"]
    if digest(config) != manifest["config_sha256"]:
        raise ValueError("Manifest config hash mismatch.")
    rows, identities, environments, statuses = [], [], [], {}
    experiments = {e["experiment_id"]: e for e in manifest["experiments"]}
    for run in manifest["runs"]:
        path = study / run["run_id"] / "metrics.json"
        if not path.exists():
            statuses[run["run_id"]] = "missing"
            continue
        report = read_json(path)
        statuses[run["run_id"]] = report["status"]
        # Exclude all prefixes of interrupted runs until the complete trajectory is valid.
        if report["status"] != "completed":
            continue
        if report["run"] != run or report["config_sha256"] != digest(config):
            raise ValueError(f"Run does not match the study manifest: {run['run_id']}")
        identities.append(data_identity(report["data"]))
        environments.append(report["environment"])
        for item in run["experiments"]:
            budget = report["budgets"][str(item["epochs"])]
            if not all(math.isfinite(budget[key]) for key in METRICS):
                raise ValueError(f"Non-finite result: {run['run_id']}")
            experiment = experiments[item["experiment_id"]]
            rows.append({"experiment_id": item["experiment_id"], "factor": experiment["factor"],
                         **{key: experiment[key] for key in FACTORS},
                         "train_seed": run["train_seed"], "run_id": run["run_id"],
                         "best_epoch": budget["best_epoch"],
                         **{key: budget[key] for key in METRICS}})
    if identities and any(value != identities[0] for value in identities):
        raise ValueError("Cannot compare different datasets or train/validation splits.")
    if environments and any(value != environments[0] for value in environments):
        raise ValueError("Cannot pool runs from different environments. Use a consistent runtime.")
    baseline = {row["train_seed"]: row["best_val_accuracy"] for row in rows
                if row["experiment_id"] == "S00"}
    aggregates = []
    for experiment in manifest["experiments"]:
        subset = [row for row in rows if row["experiment_id"] == experiment["experiment_id"]]
        seeds = {row["train_seed"] for row in subset}
        if len(seeds) != len(subset):
            raise ValueError("Duplicate seed results.")
        deltas = [row["best_val_accuracy"] - baseline[row["train_seed"]]
                  for row in subset if row["train_seed"] in baseline]
        aggregates.append({**experiment, "completed_seeds": sorted(seeds),
                           "missing_seeds": sorted(set(config["train_seeds"]) - seeds),
                           "complete": seeds == set(config["train_seeds"]),
                           "metrics": {key: mean_std([row[key] for row in subset]) for key in METRICS},
                           "paired_delta_vs_baseline": mean_std(deltas)})
    complete = [item for item in aggregates if item["complete"]]
    ranking = sorted(complete, key=lambda item: (-item["metrics"]["best_val_accuracy"]["mean"],
                                                item["experiment_id"]))
    result = {"schema_version": 1, "config_sha256": digest(config), "run_statuses": statuses,
              "data": identities[0] if identities else None,
              "environment": environments[0] if environments else None,
              "validation_ranking_complete_only": [item["experiment_id"] for item in ranking],
              "aggregates": aggregates}
    derived = study / "analysis"
    derived.mkdir(exist_ok=True)
    if rows:
        write_csv(derived / "summary.csv", rows)
    write_json(derived / "metrics.json", result)
    lines = ["# Softmax OFAT: phân tích validation", "",
             "Chỉ thay đổi một yếu tố mỗi lần; chưa đánh giá tương tác giữa các siêu tham số.",
             "Không sử dụng kết quả test để xếp hạng. Chỉ xếp hạng cấu hình đủ mọi seed.",
             "Độ lệch chuẩn là độ lệch chuẩn mẫu (ddof=1); một seed chưa đủ để ước lượng.",
             "So sánh theo seed ghép cặp với baseline S00 chạy lại trong chính study này.",
             "Các mốc epoch dùng chung tiền tố huấn luyện khi reuse_epoch_prefixes=true; "
             "không phải các lần lặp độc lập. Thời gian train_val_seconds gồm train và validation, "
             "không gồm xuất checkpoint, đánh giá train sạch và ghi tệp.",
             "Các lượt chưa hoàn tất/failed bị loại toàn bộ, kể cả prefix đã lưu.",
             "Test MNIST đã xuất hiện trong thực nghiệm cũ; không xem đây là tập kiểm định mới hoàn toàn.",
             "", "| ID | Yếu tố | Số seed | Best val (mean ± std) | Last val (mean) | Δ baseline |",
             "|---|---|---:|---:|---:|---:|"]
    for item in aggregates:
        best = item["metrics"]["best_val_accuracy"]
        last = item["metrics"]["last_val_accuracy"]["mean"]
        delta = item["paired_delta_vs_baseline"]["mean"]
        formatted = "—" if best["mean"] is None else f"{best['mean']:.6f}"
        if best["std"] is not None:
            formatted += f" ± {best['std']:.6f}"
        lines.append(f"| {item['experiment_id']} | {item['factor']} | "
                     f"{best['n']}/{len(config['train_seeds'])} | {formatted} | "
                     f"{last if last is not None else '—'} | {delta if delta is not None else '—'} |")
    lines += ["", "Xếp hạng validation (chỉ cấu hình đủ seed): " +
              ", ".join(result["validation_ranking_complete_only"]), "",
              "Chốt ID bằng --phase lock --select S00 <ID đã chọn>, sau đó mới chạy --phase test.", ""]
    atomic_text(derived / "analysis.md", "\n".join(lines))
    if make_plot:
        plot_comparison(aggregates, derived / "comparison.png")
        for run_id, status in statuses.items():
            if status == "completed":
                plot_history(study / run_id / "history.csv", derived / f"history_{run_id}.png")
    return result


def plot_history(history: Path, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with history.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [int(row["epoch"]) for row in rows]
    for ax, metric in zip(axes, ("loss", "accuracy")):
        for split, style in (("train", "-"), ("val", "--")):
            ax.plot(epochs, [float(row[f"{split}_{metric}"]) for row in rows],
                    color="black", linestyle=style, label=split)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_comparison(aggregates: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(12, 11), squeeze=False)
    baseline = aggregates[0]
    for factor, ax in zip(FACTORS, axes.flat):
        items = sorted([item for item in aggregates if item["factor"] == factor] + [baseline],
                       key=lambda item: item[factor])
        for metric, marker, style in (("best_val_accuracy", "o", "-"),
                                      ("last_val_accuracy", "s", "--")):
            # Missing results remain gaps; partial-seed means are explicitly unranked.
            means = [item["metrics"][metric]["mean"] for item in items]
            ys = [float("nan") if value is None else value for value in means]
            stds = [item["metrics"][metric]["std"] or 0 for item in items]
            ax.errorbar(range(len(items)), ys, yerr=stds, color="black", marker=marker,
                        linestyle=style, capsize=3, label=metric)
        ax.set_xticks(range(len(items)), [str(item[factor]) for item in items], rotation=30)
        ax.set_title(factor)
        ax.set_ylabel("Validation accuracy")
        ax.legend(fontsize=8)
    axes.flat[-1].axis("off")
    fig.suptitle("Softmax OFAT — mean ± sample std; consult CSV for seed coverage")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    try:
        result = analyze(args.study_dir, make_plot=not args.no_plot)
        print("Completed runs:", sum(s == "completed" for s in result["run_statuses"].values()))
        print("Validation ranking:", result["validation_ranking_complete_only"])
        print("Analysis:", args.study_dir / "analysis")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
