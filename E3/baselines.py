"""Import fixed E1 results with provenance; never construct or train a baseline."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASELINES = ROOT_DIR / "E1/results/epochs100_batch256_seed42"


def load_e1_baselines(args, dataset: str = "MNIST") -> tuple[dict, dict]:
    directory = args.baseline_results
    if not directory.is_absolute():
        directory = ROOT_DIR / directory
    directory = directory.resolve()
    path = directory / "metrics.json"
    raw = path.read_bytes()
    source = json.loads(raw)
    if source.get("dataset") != dataset:
        raise ValueError(f"Baseline dataset mismatch: expected {dataset}, got {source.get('dataset')}")
    split = source["split"]
    if split["seed"] != args.seed or split["val_ratio"] != args.val_ratio:
        raise ValueError("E1 baseline split seed/ratio differs from E3. Use the same split.")
    for key in ("epochs", "batch_size", "learning_rate", "weight_decay"):
        if source["training"][key] != getattr(args, key):
            raise ValueError(f"E1 baseline {key}={source['training'][key]} differs from E3 {getattr(args, key)}.")
    try:
        location = path.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        location = str(path)
    digest = hashlib.sha256(raw).hexdigest()
    baselines, histories = {}, {}
    for name in ("mlp", "cnn"):
        metric = source["models"][name]
        for key in ("parameters", "best_epoch", "best_val_accuracy", "test_loss", "test_accuracy", "macro_f1", "training_time_seconds"):
            if not math.isfinite(metric[key]):
                raise ValueError(f"Non-finite E1 baseline {name}.{key}")
        if not all(0 <= metric[k] <= 1 for k in ("test_accuracy", "macro_f1", "best_val_accuracy")):
            raise ValueError(f"Invalid E1 baseline accuracy/F1: {name}")
        history_path = directory / f"history_{name}.csv"
        with history_path.open(encoding="utf-8", newline="") as stream:
            history = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
        if [r["epoch"] for r in history] != list(range(1, args.epochs + 1)):
            raise ValueError(f"Incomplete E1 baseline history: {name}")
        if not all(math.isfinite(v) for row in history for v in row.values()):
            raise ValueError(f"Non-finite E1 baseline history: {name}")
        best = max(history, key=lambda row: row["val_accuracy"])
        if int(best["epoch"]) != metric["best_epoch"] or abs(best["val_accuracy"] - metric["best_val_accuracy"]) > 5.1e-7:
            raise ValueError(f"E1 baseline best epoch/validation mismatch: {name}")
        key = f"{name}_baseline"
        baselines[key] = {**metric, "experiment": key, "family": name.upper(),
                          "sequence": "flatten" if name == "mlp" else "spatial",
                          "sequence_length": None, "feature_dim": None,
                          "source": "E1_reused", "trained_in_this_run": False,
                          "source_metrics": location, "source_metrics_sha256": digest,
                          "epochs": source["training"]["epochs"], "batch_size": source["training"]["batch_size"],
                          "split": split, "source_history_sha256": hashlib.sha256(history_path.read_bytes()).hexdigest()}
        histories[key] = history
    return baselines, histories
