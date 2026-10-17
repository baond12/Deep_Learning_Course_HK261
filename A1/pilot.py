"""Sequential train/validation-only feasibility run; never evaluates CIFAR-100 test."""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torchvision

from A1.data import build_cifar100_balanced_loaders, reset_loader_seeds
from A1.engine import NonFiniteTrainingError, fit_classifier, memory_snapshot
from A1.models import build_a1_model, count_parameters, parse_experiment_name, set_a1_train_mode
from common.utils import set_seed

EXPERIMENTS = [f"{architecture}_{mode}" for architecture in ("resnet18", "vit_b16")
               for mode in ("scratch", "pretrained_head", "pretrained_partial", "pretrained_full")]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", choices=["all", *EXPERIMENTS], default=["all"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="A1/results/pilot_epoch1_batch16_seed42")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42, help="Fixed data split seed")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("epochs/batch-size must be positive; num-workers must be nonnegative")
    if not (0 < args.learning_rate < float("inf")) or not (0 <= args.weight_decay < float("inf")):
        parser.error("learning-rate must be finite and positive; weight-decay finite and nonnegative")
    if not all(0 <= x < 2**32 - 3 for x in (args.seed, args.train_seed)):
        parser.error("seeds must be in [0, 2**32-3)")
    return args


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    # Atomic replacement, strict JSON: failures must not introduce NaN/Infinity.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def fingerprint(model, names):
    if not names:
        return None
    digest = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(names):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def frozen_names(model):
    names = {name for name, parameter in model.named_parameters() if not parameter.requires_grad}
    for prefix, module in model.named_modules():
        if not module.training:
            names.update(f"{prefix}.{name}" if prefix else name for name, _ in module.named_buffers(recurse=False))
    return names


def environment(device):
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
              "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
              "device": str(device), "torch_threads": torch.get_num_threads(),
              "cudnn_deterministic": torch.backends.cudnn.deterministic,
              "cudnn_benchmark": torch.backends.cudnn.benchmark,
              "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "execution": "sequential within this process; competing GPU workloads not controlled",
              "source_sha256_lf": {}}
    for name in ("A1/pilot.py", "A1/engine.py", "A1/models.py", "A1/data.py", "common/utils.py", "requirements.txt"):
        result["source_sha256_lf"][name] = hashlib.sha256((ROOT/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    try:
        result["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        result["git_commit"] = None
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result.update(gpu_name=props.name, gpu_total_memory_mib=props.total_memory / 2**20,
                      gpu_compute_capability=list(torch.cuda.get_device_capability(device)))
        try:
            snapshot = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                                       "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
            result["nvidia_smi_before_run"] = snapshot.stdout.strip() if snapshot.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            result["nvidia_smi_before_run"] = None
    return result


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


SUMMARY_FIELDS = ["experiment", "status", "epochs_completed", "total_parameters", "trainable_parameters",
                  "train_seconds", "validation_seconds", "train_images_per_second", "validation_images_per_second",
                  "peak_allocated_mib", "peak_reserved_mib", "train_loss", "train_accuracy", "val_loss", "val_accuracy",
                  "frozen_state_unchanged", "setup_seconds", "training_validation_seconds", "error"]


def persist(output, payload):
    write_json(output/"metrics.json", payload)
    for name, record in payload["experiments"].items():
        write_json(output/f"run_{name}.json", record)
    temporary = output/"summary.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for name, record in payload["experiments"].items():
            row = {key: record.get(key) for key in SUMMARY_FIELDS}
            writer.writerow({**row, "experiment": name})
    temporary.replace(output/"summary.csv")


def run_pilot(args, *, data_bundle=None, model_factory=None):
    """The injectable objects are for synthetic tests, never CLI substitutes for weights."""
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    experiments = EXPERIMENTS if "all" in args.experiments else list(dict.fromkeys(args.experiments))
    synthetic = data_bundle is not None or model_factory is not None
    factory = model_factory or build_a1_model
    set_seed(args.train_seed)
    payload = {"schema_version": 1, "purpose": "A1 feasibility pilot, not final comparison",
               "status": "preparing", "started_at_utc": timestamp(), "test_evaluated": False,
               "synthetic_test": synthetic, "config": vars(args), "environment": environment(device),
               "precision": "float32; AMP disabled", "optimizer": "Adam",
               "freeze_policy": "frozen stages in eval, including BatchNorm running statistics",
               "timing_scope": "phase includes loader/startup, transfers, finite checks, forward/backward/optimizer; setup and CPU best-state copies excluded from phase times",
               "memory_scope": "PyTorch process peak allocated/reserved per phase, including resident model/gradients/optimizer; best state kept on CPU",
               "limitations": "Epoch 1 includes cold-start overhead; no convergence or stability claim; no controlled speedup unless GPU dedicated",
               "experiments": {name: {"experiment": name, "status": "not_run", "history": [], "phase_measurements": []} for name in experiments}}
    (output/"README.md").write_text(
        "# Kết quả chạy thăm dò A1\n\nChỉ huấn luyện/xác thực, không đánh giá test. "
        "Đây chưa phải kết quả A1.2.\n\nGửi lại toàn bộ thư mục này hoặc tệp nén. "
        "`metrics.json` chứa cấu hình, môi trường, kết quả, lịch sử, đo từng pha và lỗi; "
        "`summary.csv` là bảng nhanh; `run_*.json` lưu từng lượt; `split_indices.json` ghi chỉ số chia; "
        "`run.log` ghi diễn biến. Đơn vị bộ nhớ MiB; CPU dùng null. "
        "Thời gian khởi tạo/tải trọng số tách riêng; vòng đầu không đại diện chắc chắn cho tốc độ ổn định.\n",
        encoding="utf-8")
    persist(output, payload)
    interrupted = False
    with (output/"run.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(Tee(sys.stdout, log)):
        try:
            set_seed(args.train_seed)
            data = data_bundle or build_cifar100_balanced_loaders(
                args.data_dir, 224, 100, 400, 100, 100, args.batch_size, args.seed, args.num_workers,
                include_test=False, train_seed=args.train_seed)
            indices = data.split_indices or {}
            write_json(output/"split_indices.json", indices)
            payload["split"] = {"train": data.train_size, "validation": data.val_size,
                                "test_loaded": False, "split_seed": args.seed,
                                "indices_sha256": hashlib.sha256(json.dumps(indices, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                                "class_names": data.class_names, "image_size": data.image_size}
            payload["transforms"] = {"train": str(data.train_loader.dataset.transform) if hasattr(data.train_loader.dataset, "transform") else "synthetic",
                                     "validation": str(data.val_loader.dataset.transform) if hasattr(data.val_loader.dataset, "transform") else "synthetic"}
            payload["status"] = "running"
            persist(output, payload)
        except BaseException as exc:
            payload.update(status="preparation_failed", error=f"{type(exc).__name__}: {exc}", finished_at_utc=timestamp())
            print(traceback.format_exc())
            persist(output, payload)
            raise
        for name in experiments:
            record = payload["experiments"][name]
            record.update(status="running", started_at_utc=timestamp(), epochs_completed=0)
            persist(output, payload)
            model = None
            stage = "model_initialization"
            try:
                print(f"\n=== {name} ===", flush=True)
                setup_start = time.perf_counter()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                set_seed(args.train_seed)
                architecture, mode = parse_experiment_name(name)
                model = factory(architecture=architecture, mode=mode, num_classes=len(data.class_names), download_weights=True)
                if not synthetic and mode.startswith("pretrained") and model._a1_weights != "IMAGENET1K_V1":
                    raise RuntimeError("Pretrained weights not verified")
                set_a1_train_mode(model)
                frozen = frozen_names(model)
                learned = {key for key, p in model.named_parameters() if p.requires_grad}
                frozen_before, learned_before = fingerprint(model, frozen), fingerprint(model, learned)
                total, trainable = count_parameters(model)
                record.update(total_parameters=total, trainable_parameters=trainable,
                              weights=getattr(model, "_a1_weights", None), frozen_state_checked=bool(frozen), frozen_tensor_names=sorted(frozen),
                              trainable_parameter_names=sorted(learned), frozen_state_sha256_before=frozen_before)
                model.to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                record["setup_seconds"] = time.perf_counter() - setup_start
                stage = "training_validation"
                set_seed(args.train_seed)
                reset_loader_seeds(data, args.train_seed)

                def on_phase(epoch, phase, measurements):
                    record["last_completed_phase"] = {"epoch": epoch, "phase": phase, **measurements}
                    persist(output, payload)

                def on_epoch(row, phases):
                    record["history"].append(row)
                    record["phase_measurements"].append(phases)
                    record["epochs_completed"] = len(record["history"])
                    for phase, prefix in ((phases["train"], "train"), (phases["validation"], "val")):
                        record[prefix+"_loss"], record[prefix+"_accuracy"] = phase["loss"], phase["accuracy"]
                    record["train_seconds"] = sum(p["train"]["seconds"] for p in record["phase_measurements"])
                    record["validation_seconds"] = sum(p["validation"]["seconds"] for p in record["phase_measurements"])
                    record["train_images_per_second"] = sum(p["train"]["samples"] for p in record["phase_measurements"]) / record["train_seconds"]
                    record["validation_images_per_second"] = sum(p["validation"]["samples"] for p in record["phase_measurements"]) / record["validation_seconds"]
                    for key in ("peak_allocated_mib", "peak_reserved_mib"):
                        values = [p[phase][key] for p in record["phase_measurements"] for phase in ("train", "validation") if p[phase][key] is not None]
                        record[key] = max(values) if values else None
                    persist(output, payload)

                result = fit_classifier(model, data.train_loader, data.val_loader, args.epochs,
                                        args.learning_rate, args.weight_decay, device, args.progress, on_epoch, on_phase)
                stage = "freeze_verification"
                frozen_after = fingerprint(model, frozen)
                record.update(frozen_state_sha256_after=frozen_after,
                              frozen_state_unchanged=frozen_after == frozen_before,
                              trainable_state_changed=fingerprint(model, learned) != learned_before,
                              best_epoch=result.best_epoch, best_val_accuracy=result.best_val_accuracy,
                              training_validation_seconds=result.training_time_seconds)
                if frozen_after != frozen_before or not record["trainable_state_changed"]:
                    raise RuntimeError("Freeze/update invariant failed")
                record["status"] = "completed"
            except KeyboardInterrupt:
                record.update(status="interrupted", error="KeyboardInterrupt", error_stage=stage)
                interrupted = True
            except Exception as exc:
                status = "nonfinite" if isinstance(exc, NonFiniteTrainingError) else "out_of_memory" if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower() else "failed"
                record.update(status=status, error=f"{type(exc).__name__}: {exc}", error_stage=stage,
                              traceback=traceback.format_exc())
                if device.type == "cuda":
                    try:
                        record["memory_at_failure"] = memory_snapshot(device)
                    except RuntimeError as memory_error:
                        record["memory_read_error"] = str(memory_error)
                print(record["traceback"], flush=True)
            finally:
                record["finished_at_utc"] = timestamp()
                persist(output, payload)
                model = None
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            print(f"{name}: {record['status']}", flush=True)
            if interrupted:
                break
        payload.update(status="interrupted" if interrupted else "completed" if all(r["status"] == "completed" for r in payload["experiments"].values()) else "completed_with_failures",
                       finished_at_utc=timestamp())
        persist(output, payload)
        print(f"\nSaved pilot: {output.resolve()} | {payload['status']}", flush=True)
    return payload


def main():
    payload = run_pilot(parse_args())
    return 130 if payload["status"] == "interrupted" else 0 if payload["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
