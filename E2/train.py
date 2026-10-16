from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import torch
import torchvision
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from common.data import DataBundle, build_mnist_loaders
from common.engine import collect_predictions, evaluate, fit_classifier
from common.plots import plot_confusion_matrix, plot_misclassified, save_history_csv, save_summary_csv
from common.utils import get_device, save_json, seconds_to_text, set_seed
from E2.comparison import write_comparisons
from E2.models import EXPERIMENT_CONFIGS, build_model, copy_custom_to_pytorch, tokenization_metadata
from E2.verify_msa import VerificationError, pytorch_name, verify_experiments

EXPERIMENTS = list(EXPERIMENT_CONFIGS)
CLASS_NAMES = [str(i) for i in range(10)]
DEFAULT_OUTPUT = "E2/results/tokenization_msa_v2"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eight paired E2 MSA/tokenization experiments; verification is mandatory.")
    parser.add_argument("--experiments", nargs="+", default=["all"], choices=["all", *EXPERIMENTS])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=42, help="Fixed split seed; also default training seed")
    parser.add_argument("--train-seed", type=int, help="Optional independent training seed; does not change the split")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=7)
    parser.add_argument("--overlap-stride", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--verify-only", action="store_true", help="Check every selected tokenizer pair without loading MNIST")
    args = parser.parse_args(argv)
    args.train_seed = args.seed if args.train_seed is None else args.train_seed
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or not 0 <= args.seed < 2**32:
        parser.error("epochs/batch-size must be positive, num-workers nonnegative, seed in [0, 2**32).")
    if not 0 <= args.train_seed < 2**32:
        parser.error("train-seed must be in [0, 2**32).")
    if not 0 < args.val_ratio < 1:
        parser.error("val-ratio must be between zero and one.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive.")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay must be finite and nonnegative.")
    if "all" in args.experiments and len(args.experiments) > 1:
        parser.error("Use all alone, or a list of explicit experiment names.")
    if len(set(args.experiments)) != len(args.experiments):
        parser.error("Duplicate experiments are not allowed.")
    return args


def selected_experiments(values: list[str]) -> list[str]:
    return list(EXPERIMENTS) if values == ["all"] else values


def model_kwargs(args) -> dict:
    return {"embed_dim": args.embed_dim, "num_heads": args.num_heads, "ffn_dim": args.ffn_dim,
            "depth": args.depth, "patch_size": args.patch_size,
            "overlap_stride": args.overlap_stride, "dropout": args.dropout}


def experiment_description(name: str, kwargs: dict | None = None) -> dict:
    config = EXPERIMENT_CONFIGS[name]
    options = kwargs or {}
    kind = config["tokenizer_type"]
    patch = options.get("patch_size", 7)
    tokenization = {"patch": f"patch_{patch}x{patch}", "row": "row_tokens",
                    "cnn_stem": "cnn_stem_7x7",
                    "overlap_patch": f"overlap_patch_{patch}x{patch}_stride{options.get('overlap_stride', 3)}"}[kind]
    return {"attention": "custom_msa" if config["attention_type"] == "custom" else "pytorch_multihead_attention",
            "tokenization": tokenization, "tokenizer_type": kind,
            "purpose": f"{config['attention_type']} MSA, tokenization={kind}"}


def state_hash(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def paired_model(experiment: str, kwargs: dict, seed: int):
    """Always start from the SAME complete custom reference within each tokenizer pair."""
    set_seed(seed)
    kind = EXPERIMENT_CONFIGS[experiment]["tokenizer_type"]
    reference = build_model(f"custom_{kind}", **kwargs)
    initial_hash = state_hash(reference)
    if EXPERIMENT_CONFIGS[experiment]["attention_type"] == "custom":
        return reference, initial_hash
    model = build_model(experiment, **kwargs)
    copy_custom_to_pytorch(reference, model)
    state = model.state_dict()
    for name, tensor in reference.state_dict().items():
        if not torch.equal(tensor, state[pytorch_name(name)]):
            raise AssertionError(f"Initial state mismatch: {name}")
    return model, initial_hash


def seed_worker(_worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def reset_loaders(data: DataBundle, args) -> DataBundle:
    # Shuffling RNG is independent of backend-specific dropout RNG consumption.
    def loader(dataset, shuffle, seed):
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                          generator=torch.Generator().manual_seed(seed), worker_init_fn=seed_worker)
    return replace(data,
                   train_loader=loader(data.train_loader.dataset, True, args.train_seed),
                   val_loader=loader(data.val_loader.dataset, False, args.train_seed + 1),
                   test_loader=loader(data.test_loader.dataset, False, args.train_seed + 2))


@contextlib.contextmanager
def output_guard(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = out_dir / ".run.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        occupied = [p for p in out_dir.iterdir() if p.name not in (".run.lock", "verification.json")]
        if occupied:
            raise FileExistsError(f"Output contains results: {out_dir}. Choose a NEW --output-dir; no overwriting/resume.")
        yield
    finally:
        lock.unlink()


def environment(device) -> dict:
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "cuda_version": torch.version.cuda,
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "num_threads": torch.get_num_threads(), "platform": platform.platform(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32}


def run_experiments(args, *, data_bundle: DataBundle | None = None) -> dict:
    device = get_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available in this Python environment.")
    experiments = selected_experiments(args.experiments)
    kwargs = model_kwargs(args)
    tokenizers = list(dict.fromkeys(EXPERIMENT_CONFIGS[name]["tokenizer_type"] for name in experiments))
    out_dir = args.output_dir
    with output_guard(out_dir):
        print(f"device: {device}; selected experiments: {len(experiments)}", flush=True)
        try:
            verification = verify_experiments(tokenizers, kwargs, device, args.train_seed)
        except VerificationError as error:
            save_json(error.report, out_dir / "verification.json")
            raise
        save_json(verification, out_dir / "verification.json")
        if args.verify_only:
            return {"verification": verification}
        # No data download or training is possible until every check has passed.
        set_seed(args.train_seed)
        injected = data_bundle is not None
        data = data_bundle if injected else build_mnist_loaders(
            args.data_dir, args.batch_size, args.val_ratio, args.seed, args.num_workers)
        print(f"MNIST split: train={data.train_size}, val={data.val_size}, test={data.test_size}", flush=True)
        sources = ["E2/models.py", "E2/train.py", "E2/verify_msa.py", "E2/comparison.py",
                   "common/data.py", "common/engine.py", "common/utils.py", "common/plots.py"]
        train_indices = getattr(data.train_loader.dataset, "indices", None)
        val_indices = getattr(data.val_loader.dataset, "indices", None)
        payload = {"schema_version": 2, "status": "running", "dataset": "synthetic_smoke" if injected else "MNIST",
                   "split": {"train": data.train_size, "validation": data.val_size, "test": data.test_size,
                             "val_ratio": args.val_ratio, "seed": args.seed,
                             "train_indices_sha256": hashlib.sha256(json.dumps(list(train_indices)).encode()).hexdigest()
                             if train_indices is not None else None,
                             "validation_indices_sha256": hashlib.sha256(json.dumps(list(val_indices)).encode()).hexdigest()
                             if val_indices is not None else None},
                   "training": {"epochs": args.epochs, "batch_size": args.batch_size,
                                "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                                "seed": args.train_seed, "split_seed": args.seed,
                                "num_workers": args.num_workers, "optimizer": "Adam",
                                "scheduler": None, "early_stopping": False, "device": str(device)},
                   "architecture": kwargs, "environment": environment(device),
                   "source_sha256": {name: hashlib.sha256((ROOT_DIR / name).read_bytes()).hexdigest() for name in sources},
                   "planned_experiments": experiments, "verification_passed": verification["passed"],
                   "experiments": {}}
        save_json(payload, out_dir / "metrics.json")
        save_json({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  out_dir / "run_config.json")
        histories, rows = {}, []
        try:
            for experiment in experiments:
                print(f"\n=== {experiment.upper()} ===", flush=True)
                payload["active_experiment"] = experiment
                save_json(payload, out_dir / "metrics.json")
                model, initial_hash = paired_model(experiment, kwargs, args.train_seed)
                token_info = tokenization_metadata(model)
                loaders = reset_loaders(data, args)
                # Reset training RNG after construction, identically within each pair.
                set_seed(args.train_seed)
                result = fit_classifier(model=model, train_loader=loaders.train_loader,
                                        val_loader=loaders.val_loader, epochs=args.epochs,
                                        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                                        device=device, show_progress=args.progress)
                if not all(math.isfinite(value) for row in result.history for value in row.values()):
                    raise FloatingPointError(f"Non-finite training history: {experiment}")
                save_history_csv(result.history, out_dir / f"history_{experiment}.csv")
                cp_dir = out_dir / "checkpoints"
                cp_dir.mkdir(exist_ok=True)
                torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            "experiment": experiment, "model_kwargs": kwargs,
                            "best_epoch": result.best_epoch, "best_val_accuracy": result.best_val_accuracy},
                           cp_dir / f"best_{experiment}.pt")
                criterion = torch.nn.CrossEntropyLoss()
                test = evaluate(model, loaders.test_loader, criterion, device=device)
                truth, predicted, _probabilities, mistakes = collect_predictions(
                    model, loaders.test_loader, device=device, max_misclassified=25)
                test_accuracy = float(accuracy_score(truth, predicted))
                macro_f1 = float(f1_score(truth, predicted, labels=list(range(10)), average="macro", zero_division=0))
                if not all(math.isfinite(value) for value in (test.loss, test_accuracy, macro_f1)):
                    raise FloatingPointError(f"Non-finite test metrics: {experiment}")
                description = experiment_description(experiment, kwargs)
                row = {"experiment": experiment, "attention": description["attention"],
                       "tokenization": description["tokenization"], "image_tokens": token_info["image_tokens"],
                       "total_tokens": token_info["total_tokens"], "parameters": result.parameter_count,
                       "best_epoch": result.best_epoch, "best_val_accuracy": result.best_val_accuracy,
                       "test_loss": test.loss, "test_accuracy": test_accuracy, "macro_f1": macro_f1,
                       "training_time": seconds_to_text(result.training_time_seconds),
                       "training_time_seconds": result.training_time_seconds}
                rows.append(row)
                histories[experiment] = result.history
                payload["experiments"][experiment] = {**row, "purpose": description["purpose"],
                    "tokenizer_type": description["tokenizer_type"], "tokenizer": token_info,
                    "initialization_sha256": initial_hash, "verification_passed": True,
                    "best_checkpoint": f"checkpoints/best_{experiment}.pt"}
                # Persist completed experiments before plotting or starting another long run.
                save_json(payload, out_dir / "metrics.json")
                save_summary_csv(rows, out_dir / "summary.csv")
                save_summary_csv([{"target": int(t), "prediction": int(p)} for t, p in zip(truth, predicted)],
                                 out_dir / f"predictions_{experiment}.csv")
                matrix = confusion_matrix(truth, predicted, labels=list(range(10)))
                save_summary_csv([{"true_label": i, **{f"pred_{j}": int(v) for j, v in enumerate(values)}}
                                  for i, values in enumerate(matrix)], out_dir / f"confusion_matrix_{experiment}.csv")
                plot_confusion_matrix(truth, predicted, CLASS_NAMES, out_dir / f"confusion_matrix_{experiment}.png",
                                      title=f"Confusion matrix - {experiment}")
                plot_misclassified(mistakes, CLASS_NAMES, out_dir / f"misclassified_{experiment}.png",
                                   title=f"Misclassified samples - {experiment}")
                write_comparisons(payload["experiments"], histories, out_dir)
                print(f"{experiment}: test acc {test_accuracy:.4f}, macro F1 {macro_f1:.4f}", flush=True)
                del model
            payload.update(status="completed", active_experiment=None)
            save_json(payload, out_dir / "metrics.json")
            return payload
        except BaseException as error:
            payload.update(status="failed", error=f"{type(error).__name__}: {error}")
            save_json(payload, out_dir / "metrics.json")
            raise


def main() -> None:
    args = parse_args()
    try:
        run_experiments(args)
        print(f"\nSaved results to: {args.output_dir.resolve()}")
    except (OSError, ValueError, RuntimeError, AssertionError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
