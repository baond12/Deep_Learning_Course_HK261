"""Run the approved A1 protocol; train/validation and final test are separate commands."""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm.auto import tqdm

from A1.data import build_cifar100_balanced_loaders, reset_loader_seeds
from A1.engine import NonFiniteTrainingError, fit_classifier, synchronize
from A1.models import build_a1_model, count_parameters, parse_experiment_name, set_a1_train_mode
from A1.pilot import EXPERIMENTS, Tee, environment, fingerprint, frozen_names, timestamp, write_json
from common.utils import set_seed

PROTOCOL_FILE = ROOT / "A1/protocol-a1.1.json"
DEFAULT_OUTPUT = "A1/results/epochs30_batch64_seed42_v1"
SUMMARY_FIELDS = ["experiment", "status", "learning_rate", "epochs_completed", "best_epoch",
                  "best_val_accuracy", "final_val_accuracy", "total_parameters", "trainable_parameters",
                  "train_seconds", "validation_seconds", "training_validation_seconds",
                  "peak_allocated_mib", "peak_reserved_mib", "checkpoint_sha256", "error"]


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(path, rows, fields):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_checkpoint(path, model, metadata):
    temporary = path.with_suffix(".pt.tmp")
    torch.save({**metadata, "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}, temporary)
    temporary.replace(path)
    return file_hash(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["train", "test"], default="train")
    parser.add_argument("--experiments", nargs="+", choices=["all", *EXPERIMENTS], default=["all"])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--test-output-dir", help="Optional new directory for final test artifacts")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    if args.num_workers < 0:
        parser.error("num-workers must be nonnegative")
    if args.phase == "train" and args.test_output_dir:
        parser.error("test-output-dir is only valid with --phase test")
    return args


def run_environment(device):
    result = environment(device)
    for name in ["A1/run_protocol.py", "common/plots.py"]:
        result["source_sha256_lf"][name] = hashlib.sha256((ROOT/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return result


def get_data(args, protocol, include_test):
    return build_cifar100_balanced_loaders(
        args.data_dir, protocol["image_size"], protocol["num_classes"],
        protocol["train_per_class"], protocol["val_per_class"], protocol["test_per_class"],
        protocol["batch_size"], protocol["split_seed"], args.num_workers,
        include_test=include_test, train_seed=protocol["train_seed"])


def split_record(data, protocol):
    indices = {k: v for k, v in (data.split_indices or {}).items() if k in ("train", "validation")}
    return {"train": data.train_size, "validation": data.val_size, "image_size": data.image_size,
            "class_names": data.class_names, "indices_sha256": digest_json(indices),
            "split_seed": protocol["split_seed"]}, indices


def verify_data(data, protocol, synthetic):
    split, indices = split_record(data, protocol)
    if not synthetic:
        if (split["indices_sha256"] != protocol["split_indices_sha256"] or
                len(data.class_names) != protocol["num_classes"] or
                data.train_size != protocol["num_classes"] * protocol["train_per_class"] or
                data.val_size != protocol["num_classes"] * protocol["val_per_class"] or
                data.image_size != protocol["image_size"]):
            raise ValueError("Data/split does not match the approved protocol")
    return split, indices


def export_dataset_summary(data, protocol, output):
    """Use only the original training set; test counts are explicitly expected, not observed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selection = data.train_loader.dataset
    raw = selection.dataset
    train_counts = np.bincount([raw.targets[i] for i in data.split_indices["train"]], minlength=len(data.class_names))
    val_counts = np.bincount([raw.targets[i] for i in data.split_indices["validation"]], minlength=len(data.class_names))
    if not np.all(train_counts == protocol["train_per_class"]) or not np.all(val_counts == protocol["val_per_class"]):
        raise ValueError("Observed per-class counts do not match protocol")
    write_json(output/"dataset_summary.json", {
        "dataset": "CIFAR-100", "original_training_array_shape": list(raw.data.shape),
        "train_samples_observed": int(train_counts.sum()), "validation_samples_observed": int(val_counts.sum()),
        "original_training_label_counts": np.bincount(raw.targets, minlength=len(data.class_names)).tolist(),
        "test_loaded": False, "test_samples_expected_from_protocol": len(data.class_names) * protocol["test_per_class"],
        "train_validation_indices_disjoint": set(data.split_indices["train"]).isdisjoint(data.split_indices["validation"])})
    rows = [{"class_id": i, "class_name": label, "train_observed": int(train_counts[i]),
             "validation_observed": int(val_counts[i]), "test_expected_not_observed": protocol["test_per_class"]}
            for i, label in enumerate(data.class_names)]
    atomic_csv(output/"class_counts.csv", rows, list(rows[0]))
    first = {}
    for index in data.split_indices["train"]:
        first.setdefault(int(raw.targets[index]), index)
    fig, axes = plt.subplots(10, 10, figsize=(13, 13))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(raw[first[i]][0])
        ax.set_title(data.class_names[i], fontsize=7)
        ax.axis("off")
    fig.suptitle("CIFAR-100: một ảnh gốc từ tập huấn luyện cho mỗi lớp")
    fig.tight_layout()
    fig.savefig(output/"dataset_samples_train.png", dpi=150)
    plt.close(fig)


def persist_training(output, payload):
    write_json(output/"metrics.json", payload)
    atomic_csv(output/"summary.csv", payload["experiments"].values(), SUMMARY_FIELDS)
    for name, record in payload["experiments"].items():
        write_json(output/f"run_{name}.json", record)
        if record.get("history"):
            atomic_csv(output/f"history_{name}.csv", record["history"], list(record["history"][0]))


def plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, label in zip(axes, ("loss", "accuracy"), ("Hàm mất mát", "Độ chính xác")):
        x = [r["epoch"] for r in history]
        ax.plot(x, [r[f"train_{metric}"] for r in history], color="black", linestyle="--", label="Huấn luyện")
        ax.plot(x, [r[f"val_{metric}"] for r in history], color="black", label="Xác thực")
        ax.set(xlabel="Epoch", ylabel=label)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def train(args, device, protocol, data_bundle=None, model_factory=None):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output/"checkpoints").mkdir()
    synthetic = data_bundle is not None or model_factory is not None
    factory = model_factory or build_a1_model
    names = EXPERIMENTS if "all" in args.experiments else list(dict.fromkeys(args.experiments))
    set_seed(protocol["train_seed"])
    payload = {"schema_version": 1, "purpose": "A1 approved protocol: training/validation only",
               "status": "preparing", "started_at_utc": timestamp(), "synthetic_test": synthetic,
               "test_evaluated": False, "protocol": protocol, "protocol_sha256": digest_json(protocol),
               "config": vars(args), "environment": run_environment(device),
               "timing_scope": "Full train/validation fit, including callbacks and checkpoint writes; setup separate",
               "memory_scope": "PyTorch peak allocated/reserved per phase; not whole GPU memory",
               "experiments": {n: {"experiment": n, "status": "not_run", "learning_rate": protocol["learning_rates"][n],
                                   "history": [], "phase_measurements": []} for n in names}}
    write_json(output/"protocol.json", protocol)
    (output/"README.md").write_text(
        "# Kết quả A1 theo protocol đã duyệt\n\n"
        "Giai đoạn train chỉ dùng train/validation. `summary.csv` ghi validation tốt nhất và cuối riêng biệt. "
        "`metrics.json`, `run_*.json`, `history_*.csv`, `split_indices.json` là nguồn đối chiếu. "
        "`checkpoints/` chứa mô hình tốt nhất, bị Git ignore; phải giữ lại để đánh giá test. "
        "Chưa có cơ chế nối tiếp epoch; khi gián đoạn, kết quả và checkpoint đã ghi vẫn được giữ. "
        "Đánh giá test bằng lệnh riêng sau khi tất cả cấu hình trong lượt hoàn tất. "
        "Thời gian fit có tính ghi checkpoint; không đồng nhất với thời gian GPU tính toán thuần.\n", encoding="utf-8")
    persist_training(output, payload)
    with (output/"run.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(Tee(sys.stdout, log)):
        try:
            data = data_bundle or get_data(args, protocol, False)
            payload["split"], indices = verify_data(data, protocol, synthetic)
            payload["transforms"] = {n: str(getattr(loader.dataset, "transform", "synthetic"))
                                     for n, loader in [("train", data.train_loader), ("validation", data.val_loader)]}
            write_json(output/"split_indices.json", indices)
            if not synthetic:
                export_dataset_summary(data, protocol, output)
            payload["status"] = "running"
            persist_training(output, payload)
        except BaseException as exc:
            payload.update(status="preparation_failed", error=f"{type(exc).__name__}: {exc}", finished_at_utc=timestamp())
            persist_training(output, payload)
            raise
        interrupted = False
        for name, record in payload["experiments"].items():
            model = None
            record.update(status="running", started_at_utc=timestamp(), epochs_completed=0)
            persist_training(output, payload)
            try:
                print(f"\n=== {name} | LR {record['learning_rate']:g} ===", flush=True)
                start = time.perf_counter()
                set_seed(protocol["train_seed"])
                architecture, mode = parse_experiment_name(name)
                model = factory(architecture=architecture, mode=mode, num_classes=len(data.class_names), download_weights=True)
                if not synthetic and mode.startswith("pretrained") and model._a1_weights != protocol["pretrained_weights"]:
                    raise RuntimeError("Actual pretrained weights were not verified")
                set_a1_train_mode(model)
                frozen = frozen_names(model)
                learned = {n for n, p in model.named_parameters() if p.requires_grad}
                frozen_before, learned_before = fingerprint(model, frozen), fingerprint(model, learned)
                total, trainable = count_parameters(model)
                record.update(total_parameters=total, trainable_parameters=trainable, weights=getattr(model, "_a1_weights", None),
                              frozen_state_checked=bool(frozen), frozen_tensor_names=sorted(frozen),
                              frozen_state_sha256_before=frozen_before, trainable_parameter_names=sorted(learned))
                model.to(device)
                synchronize(device)
                record["setup_seconds"] = time.perf_counter() - start
                set_seed(protocol["train_seed"])
                reset_loader_seeds(data, protocol["train_seed"])

                def on_phase(epoch, phase, measurements):
                    record["last_completed_phase"] = {"epoch": epoch, "phase": phase, **measurements}
                    persist_training(output, payload)

                def on_epoch(row, phases):
                    record["history"].append(row)
                    record["phase_measurements"].append(phases)
                    record["epochs_completed"] = row["epoch"]
                    record["final_val_accuracy"] = row["val_accuracy"]
                    record["final_val_loss"] = row["val_loss"]
                    record["train_seconds"] = sum(p["train"]["seconds"] for p in record["phase_measurements"])
                    record["validation_seconds"] = sum(p["validation"]["seconds"] for p in record["phase_measurements"])
                    for key in ("peak_allocated_mib", "peak_reserved_mib"):
                        values = [p[phase][key] for p in record["phase_measurements"] for phase in ("train", "validation") if p[phase][key] is not None]
                        record[key] = max(values) if values else None
                    if row["val_accuracy"] > record.get("best_val_accuracy", -1):
                        metadata = {"experiment": name, "protocol_sha256": payload["protocol_sha256"],
                                    "split_sha256": payload["split"]["indices_sha256"], "class_names": data.class_names,
                                    "learning_rate": record["learning_rate"], "best_epoch": row["epoch"],
                                    "best_val_accuracy": row["val_accuracy"], "synthetic_test": synthetic}
                        checkpoint = output/"checkpoints"/f"{name}.pt"
                        sha = save_checkpoint(checkpoint, model, metadata)
                        record.update(best_epoch=row["epoch"], best_val_accuracy=row["val_accuracy"],
                                      checkpoint=f"checkpoints/{name}.pt", checkpoint_sha256=sha)
                    persist_training(output, payload)

                result = fit_classifier(model, data.train_loader, data.val_loader, protocol["epochs"],
                                        record["learning_rate"], protocol["weight_decay"], device,
                                        args.progress, on_epoch, on_phase)
                after = fingerprint(model, frozen)
                changed = fingerprint(model, learned) != learned_before
                record.update(frozen_state_sha256_after=after, frozen_state_unchanged=after == frozen_before,
                              trainable_state_changed=changed, training_validation_seconds=result.training_time_seconds)
                if after != frozen_before or not changed:
                    raise RuntimeError("Freeze/update invariant failed")
                record["status"] = "completed"
                persist_training(output, payload)
                plot_history(result.history, output/f"learning_curves_{name}.png")
            except KeyboardInterrupt:
                record.update(status="interrupted", error="KeyboardInterrupt")
                interrupted = True
            except Exception as exc:
                status = "nonfinite" if isinstance(exc, NonFiniteTrainingError) else "out_of_memory" if isinstance(exc, torch.cuda.OutOfMemoryError) else "failed"
                record.update(status=status, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
                print(record["traceback"], flush=True)
            finally:
                record["finished_at_utc"] = timestamp()
                persist_training(output, payload)
                model = None
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            if interrupted:
                break
        payload.update(status="interrupted" if interrupted else "completed" if all(r["status"] == "completed" for r in payload["experiments"].values()) else "completed_with_failures",
                       finished_at_utc=timestamp())
        persist_training(output, payload)
        print(f"\nTraining: {payload['status']} | {output.resolve()}", flush=True)
    return payload


def plot_confusion(matrix, names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    size = max(6, len(names) * 0.23)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(matrix, cmap="Greys")
    ax.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names, yticklabels=names,
           xlabel="Nhãn dự đoán", ylabel="Nhãn thật", title="Ma trận nhầm lẫn trên tập kiểm tra")
    ax.tick_params(axis="both", labelsize=6)
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def evaluate_once(model, loader, device, raw_indices, class_names, show_progress):
    if len(raw_indices) != len(loader.dataset):
        raise ValueError("Test indices do not match loader length")
    model.eval()
    rows, mistakes, targets, predictions = [], [], [], []
    count, loss_sum = 0, 0.0
    synchronize(device)
    start = time.perf_counter()
    for images, labels in tqdm(loader, desc="test", disable=not show_progress):
        logits = model(images.to(device))
        loss = torch.nn.functional.cross_entropy(logits, labels.to(device), reduction="sum")
        if not torch.isfinite(logits).all().item() or not torch.isfinite(loss).item():
            raise NonFiniteTrainingError("Nonfinite logits/loss during test")
        confidence, predicted = logits.softmax(1).max(1)
        for i, (truth, pred, prob) in enumerate(zip(labels.tolist(), predicted.cpu().tolist(), confidence.cpu().tolist())):
            rows.append({"original_test_index": raw_indices[count+i], "true_label": truth, "predicted_label": pred,
                         "true_class": class_names[truth], "predicted_class": class_names[pred],
                         "confidence": prob, "correct": truth == pred})
            targets.append(truth)
            predictions.append(pred)
            if truth != pred and len(mistakes) < 25:
                mistakes.append((images[i].cpu().numpy(), truth, pred))
        count += len(labels)
        loss_sum += loss.item()
    synchronize(device)
    seconds = time.perf_counter() - start
    if count != len(raw_indices) or not count:
        raise ValueError("Test loader sample count mismatch")
    return {"test_samples": count, "test_loss": loss_sum/count,
            "test_accuracy": float(np.mean(np.array(targets) == predictions)),
            "macro_f1": float(f1_score(targets, predictions, labels=list(range(len(class_names))), average="macro", zero_division=0)),
            "test_seconds": seconds}, rows, targets, predictions, mistakes


def write_comparison(destination, payload, protocol):
    lines = ["# So sánh A1 theo quy trình đã chốt", "",
             f"Protocol `{protocol['protocol_id']}`; {protocol['epochs']} epoch, batch {protocol['batch_size']}, "
             f"seed huấn luyện {protocol['train_seed']}; chọn checkpoint bằng validation.", "",
             "| Cấu hình | Trạng thái | LR | Epoch tốt nhất | Validation tốt nhất | Test accuracy | Macro F1 | Tham số được học | Train + validation (giây) |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, r in payload["experiments"].items():
        if r["status"] != "completed":
            lines.append(f"| {name} | Chưa hoàn tất | — | — | — | — | — | — | — |")
            continue
        lines.append(f"| {name} | Hoàn tất | {r['learning_rate']:g} | {r['best_epoch']} | "
                     f"{r['best_val_accuracy']:.2%} | {r['test_accuracy']:.2%} | {r['macro_f1']:.4f} | "
                     f"{r['trainable_parameters']} | {r['training_validation_seconds']:.1f} |")
    lines += ["", "Số liệu trên chỉ thuộc các cấu hình có trong thư mục lượt chạy này. "
              "Không gộp test cũ hoặc validation thăm dò vào bảng. Một seed và giới hạn epoch chưa chứng minh "
              "khác biệt có ý nghĩa thống kê hoặc mọi mô hình đã hội tụ. ViT full dùng LR khác; "
              "không quy mọi chênh lệch cho duy nhất phạm vi đóng băng.", "",
              "Thời gian toàn fit có tính callback/ghi checkpoint; xem JSON để tách thời gian từng pha. "
              "Bộ nhớ trong CSV là số đo PyTorch allocated/reserved, không phải tổng VRAM GPU.", ""]
    (destination/"comparison.md").write_text("\n".join(lines), encoding="utf-8")


def test(args, device, protocol, data_bundle=None, model_factory=None):
    output = Path(args.output_dir)
    training = json.loads((output/"metrics.json").read_text(encoding="utf-8"))
    synthetic = data_bundle is not None or model_factory is not None
    if training["status"] != "completed" or training["protocol"] != protocol or training["protocol_sha256"] != digest_json(protocol):
        raise ValueError("Test requires a completed training run with the current approved protocol")
    if training["synthetic_test"] != synthetic:
        raise ValueError("Cannot mix synthetic tests and actual experiment results")
    current_env = run_environment(device)
    if training["environment"]["source_sha256_lf"] != current_env["source_sha256_lf"]:
        raise ValueError("Source changed after training; review the change before test")
    requested = list(training["experiments"]) if "all" in args.experiments else list(dict.fromkeys(args.experiments))
    if not set(requested).issubset(training["experiments"]):
        raise ValueError("Requested experiment was not trained in this run")
    # Verify every requested file before touching the test dataset.
    for name in requested:
        r = training["experiments"][name]
        expected = f"checkpoints/{name}.pt"
        if r["status"] != "completed" or r["epochs_completed"] != protocol["epochs"] or r["checkpoint"] != expected:
            raise ValueError(f"Incomplete or invalid training record: {name}")
        if file_hash(output/expected) != r["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {name}")
    destination = Path(args.test_output_dir) if args.test_output_dir else output/"test"
    destination.mkdir(parents=True, exist_ok=False)
    payload = {"status": "preparing", "started_at_utc": timestamp(), "synthetic_test": synthetic,
               "protocol_sha256": digest_json(protocol), "environment": current_env, "test_evaluated": False,
               "training_metrics_sha256": file_hash(output/"metrics.json"), "experiments": {}}
    write_json(destination/"test_metrics.json", payload)
    with (destination/"test.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(Tee(sys.stdout, log)):
        try:
            set_seed(protocol["train_seed"])
            data = data_bundle or get_data(args, protocol, True)
            split, _ = verify_data(data, protocol, synthetic)
            if split != training["split"]:
                raise ValueError("Training/test split or class order mismatch")
            if data.test_loader is None or (not synthetic and data.test_size != protocol["num_classes"] * protocol["test_per_class"]):
                raise ValueError("Invalid test loader")
            write_json(destination/"split_indices.json", data.split_indices)
            payload.update(status="running", test_samples=data.test_size)
            write_json(destination/"test_metrics.json", payload)
            for name in requested:
                model = None
                record = {"experiment": name, "status": "running", "checkpoint_sha256": training["experiments"][name]["checkpoint_sha256"]}
                payload["experiments"][name] = record
                try:
                    path = output/"checkpoints"/f"{name}.pt"
                    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                    r = training["experiments"][name]
                    expected = {"experiment": name, "protocol_sha256": digest_json(protocol),
                                "split_sha256": split["indices_sha256"], "class_names": data.class_names,
                                "learning_rate": r["learning_rate"], "best_epoch": r["best_epoch"],
                                "best_val_accuracy": r["best_val_accuracy"], "synthetic_test": synthetic}
                    if any(checkpoint[k] != v for k, v in expected.items()):
                        raise ValueError("Checkpoint provenance mismatch")
                    architecture, mode = parse_experiment_name(name)
                    model = (model_factory or build_a1_model)(architecture=architecture, mode=mode,
                                                              num_classes=len(data.class_names), download_weights=False)
                    model.load_state_dict(checkpoint["state_dict"], strict=True)
                    del checkpoint
                    model.to(device)
                    print(f"\n=== test {name} | best epoch {r['best_epoch']} ===", flush=True)
                    payload["test_evaluated"] = True
                    write_json(destination/"test_metrics.json", payload)
                    stats, rows, targets, preds, mistakes = evaluate_once(
                        model, data.test_loader, device, data.split_indices["test"], data.class_names, args.progress)
                    record.update(stats, best_epoch=r["best_epoch"], best_val_accuracy=r["best_val_accuracy"],
                                  learning_rate=r["learning_rate"], total_parameters=r["total_parameters"],
                                  trainable_parameters=r["trainable_parameters"], training_validation_seconds=r["training_validation_seconds"],
                                  train_seconds=r["train_seconds"], validation_seconds=r["validation_seconds"],
                                  peak_allocated_mib=r["peak_allocated_mib"], peak_reserved_mib=r["peak_reserved_mib"])
                    atomic_csv(destination/f"predictions_{name}.csv", rows, list(rows[0]))
                    matrix = confusion_matrix(targets, preds, labels=list(range(len(data.class_names))))
                    matrix_rows = [{"true_class": label, **dict(zip(data.class_names, row.tolist()))}
                                   for label, row in zip(data.class_names, matrix)]
                    atomic_csv(destination/f"confusion_matrix_{name}.csv", matrix_rows, ["true_class", *data.class_names])
                    report = classification_report(targets, preds, labels=list(range(len(data.class_names))),
                                                   target_names=data.class_names, output_dict=True, zero_division=0)
                    write_json(destination/f"classification_report_{name}.json", report)
                    plot_confusion(matrix, data.class_names, destination/f"confusion_matrix_{name}.png")
                    if mistakes:
                        from common.plots import plot_misclassified
                        plot_misclassified(mistakes, data.class_names, destination/f"misclassified_{name}.png", "Mẫu dự đoán sai trên tập kiểm tra")
                    record["status"] = "completed"
                    print(f"accuracy {record['test_accuracy']:.4f} | macro F1 {record['macro_f1']:.4f}", flush=True)
                except KeyboardInterrupt:
                    record.update(status="interrupted", error="KeyboardInterrupt")
                    payload["status"] = "interrupted"
                    break
                except Exception as exc:
                    record.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
                    print(record["traceback"], flush=True)
                finally:
                    record["finished_at_utc"] = timestamp()
                    write_json(destination/"test_metrics.json", payload)
                    model = None
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            if payload["status"] != "interrupted":
                payload["status"] = "completed" if all(r["status"] == "completed" for r in payload["experiments"].values()) else "completed_with_failures"
        except BaseException as exc:
            payload.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            write_json(destination/"test_metrics.json", payload)
            raise
        payload["finished_at_utc"] = timestamp()
        fields = ["experiment", "status", "learning_rate", "best_epoch", "best_val_accuracy", "test_accuracy", "macro_f1", "test_loss",
                  "total_parameters", "trainable_parameters", "train_seconds", "validation_seconds", "training_validation_seconds", "test_seconds", "peak_allocated_mib", "peak_reserved_mib", "error"]
        atomic_csv(destination/"test_summary.csv", payload["experiments"].values(), fields)
        write_comparison(destination, payload, protocol)
        write_json(destination/"test_metrics.json", payload)
    return payload


def run(args, *, data_bundle=None, model_factory=None, protocol_override=None):
    # Overrides are only for offline synthetic tests, not a CLI bypass of the agreed protocol.
    protocol = protocol_override or json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    if protocol_override is not None and (data_bundle is None or model_factory is None):
        raise ValueError("Protocol overrides are restricted to injected synthetic tests")
    supported = {"dataset": "CIFAR-100", "optimizer": "Adam", "precision": "float32", "amp": False,
                 "scheduler": None, "augmentation": "resize224_horizontal_flip_p0.5_imagenet_normalize",
                 "checkpoint_selection": "highest_validation_accuracy_first_epoch_on_tie"}
    if any(protocol.get(k) != v for k, v in supported.items()):
        raise ValueError("Protocol requests a training method this runner does not implement")
    if set(protocol["learning_rates"]) != set(EXPERIMENTS) or any(not math.isfinite(lr) or lr <= 0 for lr in protocol["learning_rates"].values()):
        raise ValueError("Protocol must specify positive finite learning rates for all eight configurations")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    return (train if args.phase == "train" else test)(args, device, protocol, data_bundle, model_factory)


def main():
    payload = run(parse_args())
    return 0 if payload["status"] == "completed" else 130 if payload["status"] == "interrupted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
