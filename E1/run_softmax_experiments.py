"""Isolated OFAT Softmax studies. Default command only prints the plan."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torchvision
from torch import nn
from sklearn.metrics import confusion_matrix, f1_score

from common.engine import collect_predictions, evaluate, train_one_epoch
from common.utils import count_parameters, get_device, set_seed
from E1.models import SoftmaxRegression
from E1.softmax_data import build_softmax_loaders, build_softmax_test_loader

FACTORS = ("augmentation", "epochs", "batch_size", "learning_rate", "weight_decay")
DEFAULT_CONFIG = ROOT / "E1/configs/softmax_experiments.json"
DEFAULT_OUTPUT = ROOT / "E1/results/experiments/softmax"
DEFAULT_CHECKPOINTS = ROOT / "E1/checkpoints/softmax_experiments"


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot write a CSV without rows.")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def validate_config(config: dict) -> dict:
    expected = {"schema_version", "study_id", "split_seed", "train_seeds", "val_ratio",
                "reuse_epoch_prefixes", "baseline", "levels", "rotation"}
    if set(config) != expected or config["schema_version"] != 1:
        raise ValueError("Unsupported config schema or unexpected/missing keys.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", config["study_id"]):
        raise ValueError("study_id must be a safe directory name (letters, digits, _, -).")
    seeds = config["train_seeds"]
    if not seeds or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("train_seeds must contain integers in [0, 2**32).")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate training seeds.")
    if type(config["split_seed"]) is not int or not 0 <= config["split_seed"] < 2**32:
        raise ValueError("Invalid split_seed.")
    if type(config["val_ratio"]) not in (int, float) or not 0 < config["val_ratio"] < 1:
        raise ValueError("val_ratio must be between zero and one.")
    if type(config["reuse_epoch_prefixes"]) is not bool:
        raise ValueError("reuse_epoch_prefixes must be boolean.")
    if set(config["baseline"]) != set(FACTORS) or set(config["levels"]) != set(FACTORS):
        raise ValueError("baseline and levels must specify exactly the five supported factors.")
    for key in FACTORS:
        levels = config["levels"][key]
        if not isinstance(levels, list) or not levels or len(set(levels)) != len(levels):
            raise ValueError(f"Empty or duplicate levels: {key}")
        for value in [config["baseline"][key], *levels]:
            if key == "augmentation":
                valid = type(value) is bool
            elif key in ("epochs", "batch_size"):
                valid = type(value) is int and value > 0
            else:
                valid = type(value) in (int, float) and math.isfinite(value) and value > 0
            if not valid:
                raise ValueError(f"Invalid {key}: {value!r}")
        if config["baseline"][key] not in levels:
            raise ValueError(f"levels.{key} must include the baseline value.")
    rotation = config["rotation"]
    if set(rotation) != {"degrees", "interpolation", "fill"}:
        raise ValueError("Invalid rotation keys.")
    if (type(rotation["degrees"]) not in (int, float)
            or not 0 <= rotation["degrees"] <= 180
            or rotation["interpolation"] not in ("bilinear", "nearest")
            or type(rotation["fill"]) is not int or not 0 <= rotation["fill"] <= 255):
        raise ValueError("Invalid rotation settings.")
    return config


def build_plan(config: dict) -> dict:
    validate_config(config)
    experiments = [{"experiment_id": "S00", "factor": "baseline", **config["baseline"]}]
    for factor in FACTORS:
        for value in config["levels"][factor]:
            if value != config["baseline"][factor]:
                experiments.append({"experiment_id": f"S{len(experiments):02d}",
                                    "factor": factor, **config["baseline"], factor: value})
    grouped = {}
    for experiment in experiments:
        params = {key: experiment[key] for key in FACTORS}
        if config["reuse_epoch_prefixes"]:
            params.pop("epochs")
        grouped.setdefault(digest(params), {"params": params, "experiments": []})["experiments"].append(
            {"experiment_id": experiment["experiment_id"], "epochs": experiment["epochs"]})
    runs = []
    for group in grouped.values():
        budgets = sorted({item["epochs"] for item in group["experiments"]})
        identity = digest(group)[:12]
        for seed in config["train_seeds"]:
            runs.append({"run_id": f"r_{identity}_seed{seed}", "train_seed": seed,
                         "params": group["params"], "budgets": budgets,
                         "epochs": max(budgets), "experiments": group["experiments"]})
    return {"experiments": experiments, "runs": runs}


def source_hashes() -> dict:
    names = ["E1/run_softmax_experiments.py", "E1/softmax_data.py", "E1/models.py",
             "common/engine.py", "common/data.py", "common/utils.py"]
    return {name: file_hash(ROOT / name) for name in names}


def environment(device: torch.device, workers: int) -> dict:
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "platform": platform.platform(),
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda": torch.version.cuda, "num_workers": workers,
            "torch_num_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


@contextlib.contextmanager
def study_guard(study: Path):
    study.mkdir(parents=True, exist_ok=True)
    lock = study / ".study.lock"
    deadline = time.monotonic() + 10
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Study is locked: {lock}. Check active processes before removing a stale lock.")
            time.sleep(0.05)
    try:
        yield
    finally:
        lock.rmdir()


def ensure_manifest(study: Path, config: dict) -> dict:
    """Caller holds study_guard; config and training source cannot drift within a study."""
    path = study / "manifest.json"
    hashes = source_hashes()
    if path.exists():
        manifest = read_json(path)
        if manifest["config"] != config or manifest["source_sha256"] != hashes:
            raise ValueError("Config/source changed. Use a new study_id; never mix studies.")
        return manifest
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    manifest = {"schema_version": 1, "config": config, "config_sha256": digest(config),
                "source_sha256": hashes, "git_head": head, **build_plan(config)}
    write_json(path, manifest)
    return manifest


class FiniteCrossEntropy(nn.CrossEntropyLoss):
    def forward(self, inputs, targets):
        loss = super().forward(inputs, targets)
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Non-finite loss. This run is failed, not a valid result.")
        return loss


def cpu_state(model) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def save_checkpoint(path: Path, state: dict, metadata: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    torch.save({"state_dict": state, "metadata": metadata}, temporary)
    os.replace(temporary, path)
    return {"filename": path.name, "sha256": file_hash(path)}


def train_run(config: dict, run: dict, study: Path, checkpoints: Path, *, data_dir=ROOT / "data",
              device=torch.device("cpu"), num_workers=2, download=False, progress=False,
              skip_completed=False, raw_train=None) -> dict:
    run_dir = study / run["run_id"]
    metrics_path = run_dir / "metrics.json"
    with study_guard(study):
        manifest = ensure_manifest(study, config)
        if (study / "selection.json").exists():
            raise ValueError("Selection is locked; no more training is allowed in this study.")
        if run not in manifest["runs"]:
            raise ValueError("Run is not in the immutable study plan.")
        if run_dir.exists():
            if skip_completed and metrics_path.exists():
                old = read_json(metrics_path)
                if old["status"] == "completed" and old["run"] == run:
                    for budget in old["budgets"].values():
                        for kind in ("best_checkpoint", "last_checkpoint"):
                            cp = checkpoints / run["run_id"] / budget[kind]["filename"]
                            if file_hash(cp) != budget[kind]["sha256"]:
                                raise ValueError(f"Checkpoint changed: {cp}")
                    return old
            raise FileExistsError(f"Run already exists: {run_dir}. Use a new study_id for retries.")
        run_dir.mkdir()
        report = {"status": "running", "run": run, "config_sha256": digest(config),
                  "budgets": {}, "environment": environment(device, num_workers)}
        write_json(metrics_path, report)
    try:
        cp_dir = checkpoints / run["run_id"]
        cp_dir.mkdir(parents=True, exist_ok=False)
        set_seed(run["train_seed"])
        model = SoftmaxRegression().to(device)
        if count_parameters(model) != 7850:
            raise ValueError("Unexpected Softmax architecture.")
        params = run["params"]
        data = build_softmax_loaders(
            data_dir, params["batch_size"], config["val_ratio"], config["split_seed"],
            run["train_seed"], params["augmentation"], config["rotation"],
            num_workers, download, raw_train=raw_train)
        report.update({"data": data.metadata, "parameter_count": count_parameters(model)})
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"],
                                     weight_decay=params["weight_decay"])
        criterion = FiniteCrossEntropy()
        history = []
        best_accuracy, best_epoch, best_loss = -1.0, 0, None
        best_state = None
        training_seconds = 0.0
        started = time.perf_counter()
        for epoch in range(1, run["epochs"] + 1):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            epoch_start = time.perf_counter()
            train = train_one_epoch(model, data.train_loader, criterion, optimizer, device, progress)
            val = evaluate(model, data.val_loader, criterion, device)
            if not all(math.isfinite(x) for x in (train.loss, train.accuracy, val.loss, val.accuracy)):
                raise FloatingPointError("Non-finite epoch metrics.")
            if val.accuracy > best_accuracy:
                best_accuracy, best_epoch, best_loss = val.accuracy, epoch, val.loss
                best_state = cpu_state(model)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            training_seconds += time.perf_counter() - epoch_start
            row = {"epoch": epoch, "train_loss": train.loss, "train_accuracy": train.accuracy,
                   "val_loss": val.loss, "val_accuracy": val.accuracy,
                   "learning_rate": params["learning_rate"],
                   "optimizer_steps": epoch * len(data.train_loader),
                   "samples_seen": epoch * data.metadata["train_size"],
                   "train_val_seconds": training_seconds}
            history.append(row)
            write_csv(run_dir / "history.csv", history)
            if epoch in run["budgets"]:
                last_state = cpu_state(model)
                model.load_state_dict(best_state)
                clean_train = evaluate(model, data.train_eval_loader, criterion, device)
                model.load_state_dict(last_state)
                metadata = {"run_id": run["run_id"], "budget": epoch, "best_epoch": best_epoch,
                            "config_sha256": digest(config), "data": data.metadata}
                report["budgets"][str(epoch)] = {
                    "epochs": epoch, "best_epoch": best_epoch, "best_val_accuracy": best_accuracy,
                    "best_val_loss": best_loss, "last_val_accuracy": val.accuracy,
                    "last_val_loss": val.loss, "clean_train_accuracy": clean_train.accuracy,
                    "clean_train_loss": clean_train.loss, "train_val_seconds": training_seconds,
                    "optimizer_steps": row["optimizer_steps"], "samples_seen": row["samples_seen"],
                    "best_checkpoint": save_checkpoint(cp_dir / f"best_e{epoch}.pt", best_state, metadata),
                    "last_checkpoint": save_checkpoint(cp_dir / f"last_e{epoch}.pt", last_state, metadata),
                }
                write_json(metrics_path, report)
            print(f"{run['run_id']} | epoch {epoch}/{run['epochs']} | val {val.accuracy:.4f}", flush=True)
        report.update(status="completed", wall_seconds=time.perf_counter() - started)
        write_json(metrics_path, report)
        return report
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(metrics_path, report)
        raise


def data_identity(metadata: dict) -> dict:
    return {key: metadata[key] for key in ("dataset", "dataset_sha256", "train_indices_sha256",
                                         "validation_indices_sha256")}


def lock_selection(config: dict, study: Path, checkpoints: Path, experiment_ids: list[str]) -> dict:
    if not experiment_ids or len(set(experiment_ids)) != len(experiment_ids):
        raise ValueError("Select one or more distinct experiment IDs, using validation results only.")
    with study_guard(study):
        manifest = ensure_manifest(study, config)
        if (study / "selection.json").exists():
            raise FileExistsError("Selection already locked. It cannot be replaced.")
        if not set(experiment_ids) <= {e["experiment_id"] for e in manifest["experiments"]}:
            raise ValueError("Unknown experiment ID.")
        for path in study.glob("r_*/metrics.json"):
            if read_json(path)["status"] == "running":
                raise ValueError("A training run is still running. Finish it before locking selection.")
        entries, identities, environments = [], [], []
        for run in manifest["runs"]:
            wanted = [e for e in run["experiments"] if e["experiment_id"] in experiment_ids]
            if not wanted:
                continue
            metrics_path = study / run["run_id"] / "metrics.json"
            report = read_json(metrics_path)
            if report["status"] != "completed" or report["run"] != run:
                raise ValueError(f"Selected run is incomplete: {run['run_id']}")
            if report["config_sha256"] != digest(config):
                raise ValueError("Run config hash mismatch.")
            identities.append(data_identity(report["data"]))
            environments.append(report["environment"])
            for experiment in wanted:
                budget = report["budgets"][str(experiment["epochs"])]
                checkpoint = budget["best_checkpoint"]
                path = checkpoints / run["run_id"] / checkpoint["filename"]
                if file_hash(path) != checkpoint["sha256"]:
                    raise ValueError(f"Checkpoint hash mismatch: {path}")
                entries.append({**experiment, "run_id": run["run_id"], "train_seed": run["train_seed"],
                                "checkpoint": checkpoint, "metrics_sha256": file_hash(metrics_path),
                                "best_epoch": budget["best_epoch"],
                                "best_val_accuracy": budget["best_val_accuracy"]})
        if not identities or any(item != identities[0] for item in identities):
            raise ValueError("Dataset or train/validation split differs between selected runs.")
        if any(item != environments[0] for item in environments):
            raise ValueError("Selected runs use different environments. Use a consistent runtime.")
        selection = {"config_sha256": digest(config), "manifest_sha256": file_hash(study / "manifest.json"),
                     "experiment_ids": experiment_ids, "data": identities[0], "entries": entries,
                     "selection_rule": "Manual selection using validation only; all configured seeds retained."}
        selection["sha256"] = digest(selection)
        write_json(study / "selection.json", selection)
        return selection


def validate_selection(study: Path, checkpoints: Path) -> dict:
    selection = read_json(study / "selection.json")
    if digest({k: v for k, v in selection.items() if k != "sha256"}) != selection["sha256"]:
        raise ValueError("Selection hash mismatch.")
    if file_hash(study / "manifest.json") != selection["manifest_sha256"]:
        raise ValueError("Manifest changed after selection.")
    manifest = read_json(study / "manifest.json")
    if manifest["source_sha256"] != source_hashes():
        raise ValueError("Training/evaluation source changed after selection.")
    for entry in selection["entries"]:
        path = study / entry["run_id"] / "metrics.json"
        cp = checkpoints / entry["run_id"] / entry["checkpoint"]["filename"]
        if file_hash(path) != entry["metrics_sha256"] or file_hash(cp) != entry["checkpoint"]["sha256"]:
            raise ValueError("Selected metrics/checkpoint changed after selection.")
    return selection


def test_selected(study: Path, checkpoints: Path, *, data_dir=ROOT / "data",
                  device=torch.device("cpu"), num_workers=2, download=False,
                  skip_completed=False, raw_test=None) -> list[dict]:
    # Validate every selected artifact before even constructing a test dataset.
    selection = validate_selection(study, checkpoints)
    reports = []
    for entry in selection["entries"]:
        name = f"{entry['experiment_id']}_seed{entry['train_seed']}"
        out = study / "final_test" / selection["sha256"][:12] / name
        metrics_path = out / "metrics.json"
        with study_guard(study):
            if out.exists():
                if skip_completed and metrics_path.exists():
                    old = read_json(metrics_path)
                    if old["status"] == "completed" and old["selection_sha256"] == selection["sha256"]:
                        reports.append(old)
                        continue
                raise FileExistsError(f"Test output already exists: {out}")
            out.mkdir(parents=True)
            write_json(metrics_path, {"status": "running", "entry": entry})
        try:
            loader, metadata = build_softmax_test_loader(data_dir, 256, num_workers, download, raw_test=raw_test)
            model = SoftmaxRegression().to(device)
            cp = checkpoints / entry["run_id"] / entry["checkpoint"]["filename"]
            payload = torch.load(cp, map_location="cpu", weights_only=True)
            model.load_state_dict(payload["state_dict"])
            metric = evaluate(model, loader, FiniteCrossEntropy(), device)
            truth, prediction, _, _ = collect_predictions(model, loader, device, max_misclassified=0)
            report = {"status": "completed", "selection_sha256": selection["sha256"], "entry": entry,
                      "data": metadata, "environment": environment(device, num_workers),
                      "test_loss": metric.loss, "test_accuracy": metric.accuracy,
                      "macro_f1": float(f1_score(truth, prediction, labels=list(range(10)),
                                                average="macro", zero_division=0))}
            write_csv(out / "predictions.csv", [{"target": int(t), "prediction": int(p)}
                                                for t, p in zip(truth, prediction)])
            matrix = confusion_matrix(truth, prediction, labels=list(range(10)))
            write_csv(out / "confusion_matrix.csv", [{"true_label": i, **{f"pred_{j}": int(v)
                       for j, v in enumerate(row)}} for i, row in enumerate(matrix)])
            write_json(metrics_path, report)
            reports.append(report)
        except BaseException as error:
            write_json(metrics_path, {"status": "failed", "error": f"{type(error).__name__}: {error}"})
            raise
    final_root = study / "final_test" / selection["sha256"][:12]
    rows = [{"experiment_id": r["entry"]["experiment_id"], "train_seed": r["entry"]["train_seed"],
             **{key: r[key] for key in ("test_loss", "test_accuracy", "macro_f1")}} for r in reports]
    aggregates = []
    for experiment_id in selection["experiment_ids"]:
        subset = [row for row in rows if row["experiment_id"] == experiment_id]
        aggregates.append({"experiment_id": experiment_id, "n": len(subset), "metrics": {
            key: {"mean": statistics.mean(row[key] for row in subset),
                  "std": statistics.stdev(row[key] for row in subset) if len(subset) > 1 else None}
            for key in ("test_loss", "test_accuracy", "macro_f1")}})
    write_csv(final_root / "summary.csv", rows)
    write_json(final_root / "metrics.json", {"selection_sha256": selection["sha256"],
                                             "aggregates": aggregates})
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("plan", "train", "lock", "test"), default="plan")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--run-ids", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--select", nargs="+", help="Logical experiment IDs to freeze before final test")
    args = parser.parse_args()
    try:
        if args.num_workers < 0:
            raise ValueError("num-workers must be nonnegative.")
        if args.select and args.phase != "lock":
            raise ValueError("--select is only valid with --phase lock.")
        if (args.run_ids or args.seeds) and args.phase not in ("plan", "train"):
            raise ValueError("--run-ids and --seeds only apply to plan/train, never final test.")
        config = validate_config(read_json(args.config))
        plan = build_plan(config)
        if args.run_ids and not set(args.run_ids) <= {r["run_id"] for r in plan["runs"]}:
            raise ValueError("Unknown run IDs. Inspect --phase plan first.")
        if args.seeds and not set(args.seeds) <= set(config["train_seeds"]):
            raise ValueError("--seeds must be a subset of configured train_seeds.")
        if args.phase == "plan":
            print(json.dumps({"logical_configurations": len(plan["experiments"]),
                              "logical_seed_results": len(plan["experiments"]) * len(config["train_seeds"]),
                              "physical_training_runs": len(plan["runs"]), **plan}, indent=2))
            return
        study = args.output_root / config["study_id"]
        checkpoints = args.checkpoint_root / config["study_id"]
        if args.phase == "lock":
            lock_selection(config, study, checkpoints, args.select or [])
            print(f"Selection locked: {study / 'selection.json'}")
            return
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        device = get_device(args.device)
        if args.phase == "test":
            if read_json(study / "manifest.json")["config"] != config:
                raise ValueError("Config does not match the locked study.")
            reports = test_selected(study, checkpoints, data_dir=args.data_dir, device=device,
                                    num_workers=args.num_workers, download=args.download,
                                    skip_completed=args.skip_completed)
            for report in reports:
                print(report["entry"]["experiment_id"], report["entry"]["train_seed"],
                      report["test_accuracy"], report["macro_f1"])
        else:
            for run in plan["runs"]:
                if args.run_ids and run["run_id"] not in args.run_ids:
                    continue
                if args.seeds and run["train_seed"] not in args.seeds:
                    continue
                train_run(config, run, study, checkpoints, data_dir=args.data_dir, device=device,
                          num_workers=args.num_workers, download=args.download,
                          progress=args.progress, skip_completed=args.skip_completed)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
