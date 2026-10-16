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
from E3.baselines import DEFAULT_BASELINES, load_e1_baselines
from E3.comparison import write_comparisons
from E3.models import SEQUENCE_EXPERIMENTS, build_image_sequence_model, sequence_metadata

EXPERIMENTS = list(SEQUENCE_EXPERIMENTS)
DEFAULT_OUTPUT = "E3/results/sequence_v2"
CLASS_NAMES = [str(i) for i in range(10)]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train eight LSTM/GRU classifiers; reuse fixed E1 MLP/CNN results.")
    parser.add_argument("--experiments", nargs="+", default=["all"], choices=["all", *EXPERIMENTS])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--baseline-results", type=Path, default=DEFAULT_BASELINES,
                        help="Existing E1 metrics/history directory; baselines are never retrained")
    for name, default in (("epochs",100),("batch-size",256),("seed",42),("num-workers",2),("hidden-dim",128),("num-layers",1),("patch-size",7),("overlap-stride",3),("stem-dim",64)):
        parser.add_argument(f"--{name}", type=int, default=default)
    for name, default in (("learning-rate",.001),("weight-decay",.0001),("val-ratio",.1),("dropout",.1)):
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--train-seed", type=int, help="Independent training seed; --seed fixes the split")
    parser.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--probe-order", action="store_true", help="Evaluate reversed/fixed shuffled tokens on validation only")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    args.train_seed = args.seed if args.train_seed is None else args.train_seed
    if min(args.epochs,args.batch_size,args.hidden_dim,args.num_layers,args.stem_dim) < 1 or args.num_workers < 0:
        parser.error("Epochs, batch/dimensions/layers must be positive; num-workers nonnegative.")
    if not 0 <= args.seed < 2**32 or not 0 <= args.train_seed < 2**32:
        parser.error("Seeds must be in [0, 2**32).")
    if not 0 < args.val_ratio < 1 or not 0 <= args.dropout < 1:
        parser.error("val-ratio must be in (0,1); dropout in [0,1).")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("Learning rate must be positive and finite; weight decay nonnegative and finite.")
    if len(set(args.experiments)) != len(args.experiments) or ("all" in args.experiments and len(args.experiments)!=1):
        parser.error("Use all alone or distinct explicit experiment names.")
    return args


def selected_experiments(values):
    return list(EXPERIMENTS) if values == ["all"] else list(values)


def model_kwargs(args):
    return {k:getattr(args,k) for k in ("hidden_dim","num_layers","patch_size","overlap_stride","stem_dim","dropout","bidirectional")}


def build_experiment_model(name, args):
    if name in SEQUENCE_EXPERIMENTS:
        return build_image_sequence_model(name, **model_kwargs(args))
    raise ValueError(f"Unknown experiment: {name}")


def seed_worker(_worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def reset_loaders(data, args):
    def loader(dataset, shuffle, seed):
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                          generator=torch.Generator().manual_seed(seed), worker_init_fn=seed_worker)
    return replace(data, train_loader=loader(data.train_loader.dataset,True,args.train_seed),
                   val_loader=loader(data.val_loader.dataset,False,args.train_seed+1),
                   test_loader=loader(data.test_loader.dataset,False,args.train_seed+2))


@contextlib.contextmanager
def output_guard(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = out_dir / ".run.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        if any(p.name != ".run.lock" for p in out_dir.iterdir()):
            raise FileExistsError(f"Output contains results: {out_dir}. Choose a NEW directory; no overwrite/resume.")
        yield
    finally:
        lock.unlink()


class OrderedSequenceProbe(torch.nn.Module):
    """Reorder tokens at inference; keep the image, learned tokenizer and weights fixed."""
    def __init__(self, model, permutation):
        super().__init__()
        self.model = model
        self.register_buffer("permutation", permutation)

    def forward(self, images):
        return self.model.classify_sequence(self.model.tokenizer(images).index_select(1,self.permutation))


def probe_order(model, val_loader, device, seed):
    count = model.tokenizer.sequence_length
    permutations = {"raster":torch.arange(count),"reverse":torch.arange(count-1,-1,-1),
                    "shuffle":torch.randperm(count,generator=torch.Generator().manual_seed(seed))}
    result = {"split":"validation","permutation_seed":seed,"trained_order":"raster",
              "interpretation":"Order sensitivity, not proof of long-range memory.","orders":{}}
    for name, permutation in permutations.items():
        wrapper = OrderedSequenceProbe(model,permutation).to(device)
        score = evaluate(wrapper,val_loader,torch.nn.CrossEntropyLoss(),device)
        result["orders"][name] = {"accuracy":score.accuracy,"loss":score.loss,"permutation":permutation.tolist()}
    return result


def run_experiments(args, data_bundle: DataBundle | None = None):
    # Fail before downloading/training if the referenced E1 run is missing or incompatible.
    baselines, baseline_histories = load_e1_baselines(args, "MNIST" if data_bundle is None else "synthetic_test")
    experiments = selected_experiments(args.experiments)
    for name in experiments:
        build_experiment_model(name,args)  # Validate before downloading or training.
    device = get_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    with output_guard(args.output_dir):
        set_seed(args.train_seed)
        data = data_bundle or build_mnist_loaders(args.data_dir,args.batch_size,args.val_ratio,args.seed,args.num_workers)
        for baseline in baselines.values():
            if any(baseline["split"][key] != value for key, value in
                   (("train", data.train_size), ("validation", data.val_size), ("test", data.test_size))):
                raise ValueError("E1 baseline sample counts differ from the E3 split.")
        def split_hash(loader):
            indices = getattr(loader.dataset,"indices",None)
            return hashlib.sha256(json.dumps(list(indices)).encode()).hexdigest() if indices is not None else None
        sources = ("E3/models.py","E3/train.py","E3/comparison.py","E3/baselines.py","common/data.py","common/engine.py","common/plots.py","common/utils.py")
        payload = {"schema_version":2,"status":"running","dataset":"MNIST" if data_bundle is None else "synthetic_test",
                   "split":{"train":data.train_size,"validation":data.val_size,"test":data.test_size,"seed":args.seed,
                            "val_ratio":args.val_ratio,"train_indices_sha256":split_hash(data.train_loader),"validation_indices_sha256":split_hash(data.val_loader)},
                   "training":{"epochs":args.epochs,"batch_size":args.batch_size,"learning_rate":args.learning_rate,
                               "weight_decay":args.weight_decay,"optimizer":"Adam","scheduler":None,"early_stopping":False,
                               "train_seed":args.train_seed,"num_workers":args.num_workers,"device":str(device)},
                   "rnn_config":model_kwargs(args),"baseline_source":"Existing E1 results, reused without training or test re-evaluation",
                   "baselines":baselines,
                   "environment":{"python":platform.python_version(),"torch":str(torch.__version__),"torchvision":str(torchvision.__version__),
                                  "cuda_version":torch.version.cuda,"gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else None,
                                  "num_threads":torch.get_num_threads(),"platform":platform.platform(),
                                  "tf32_matmul":torch.backends.cuda.matmul.allow_tf32,"tf32_cudnn":torch.backends.cudnn.allow_tf32},
                   "source_sha256":{f:hashlib.sha256((ROOT_DIR/f).read_bytes()).hexdigest() for f in sources},
                   "planned_experiments":experiments,"experiments":{}}
        save_json(payload,args.output_dir/"metrics.json")
        save_json({"baselines": baselines, "note": "Fixed E1 results; not trained in this E3 run."}, args.output_dir/"baselines_e1.json")
        save_json({k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},args.output_dir/"run_config.json")
        histories, rows = {}, []
        try:
            for name in experiments:
                payload["active_experiment"] = name
                save_json(payload,args.output_dir/"metrics.json")
                set_seed(args.train_seed)
                model = build_experiment_model(name,args)
                loaders = reset_loaders(data,args)
                set_seed(args.train_seed)  # Shuffle uses its own generator.
                print(f"\n=== {name.upper()} ===",flush=True)
                result = fit_classifier(model,loaders.train_loader,loaders.val_loader,args.epochs,args.learning_rate,args.weight_decay,device,args.progress)
                if not all(math.isfinite(v) for r in result.history for v in r.values()):
                    raise FloatingPointError(f"Non-finite history: {name}")
                save_history_csv(result.history,args.output_dir/f"history_{name}.csv")
                checkpoints = args.output_dir/"checkpoints"
                checkpoints.mkdir(exist_ok=True)
                torch.save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"experiment":name,
                            "model_kwargs":model_kwargs(args),"best_epoch":result.best_epoch,"best_val_accuracy":result.best_val_accuracy},checkpoints/f"best_{name}.pt")
                test = evaluate(model,loaders.test_loader,torch.nn.CrossEntropyLoss(),device)
                truth,predicted,_probabilities,mistakes = collect_predictions(model,loaders.test_loader,device,25)
                accuracy = float(accuracy_score(truth,predicted))
                f1 = float(f1_score(truth,predicted,labels=list(range(10)),average="macro",zero_division=0))
                if not all(math.isfinite(v) for v in (test.loss,accuracy,f1)):
                    raise FloatingPointError(f"Non-finite test metrics: {name}")
                info = sequence_metadata(model)
                row = {"experiment":name,"family":name.split("_")[0].upper(),"sequence":info["kind"],
                       "sequence_length":info.get("sequence_length"),"feature_dim":info.get("feature_dim"),"parameters":result.parameter_count,
                       "best_epoch":result.best_epoch,"best_val_accuracy":result.best_val_accuracy,"test_loss":test.loss,"test_accuracy":accuracy,
                       "macro_f1":f1,"training_time":seconds_to_text(result.training_time_seconds),"training_time_seconds":result.training_time_seconds,
                       "source":"E3_current","trained_in_this_run":True,"epochs":args.epochs,"batch_size":args.batch_size}
                payload["experiments"][name] = {**row,"tokenizer":info,"best_checkpoint":f"checkpoints/best_{name}.pt"}
                if args.probe_order and name in SEQUENCE_EXPERIMENTS:
                    payload["experiments"][name]["order_probe"] = probe_order(model,loaders.val_loader,device,args.train_seed)
                rows.append(row)
                histories[name] = result.history
                save_json(payload,args.output_dir/"metrics.json")
                save_summary_csv(rows,args.output_dir/"summary.csv")
                save_summary_csv([{"target":int(t),"prediction":int(p)} for t,p in zip(truth,predicted)],args.output_dir/f"predictions_{name}.csv")
                matrix = confusion_matrix(truth,predicted,labels=list(range(10)))
                save_summary_csv([{"true_label":i,**{f"pred_{j}":int(v) for j,v in enumerate(r)}} for i,r in enumerate(matrix)],args.output_dir/f"confusion_matrix_{name}.csv")
                plot_confusion_matrix(truth,predicted,CLASS_NAMES,args.output_dir/f"confusion_matrix_{name}.png",title=f"Confusion matrix - {name}")
                plot_misclassified(mistakes,CLASS_NAMES,args.output_dir/f"misclassified_{name}.png",title=f"Misclassified - {name}")
                write_comparisons(payload["experiments"],histories,args.output_dir,baselines,baseline_histories)
                print(f"{name}: test acc {accuracy:.4f}, macro F1 {f1:.4f}",flush=True)
                del model
            payload.update(status="completed",active_experiment=None)
            save_json(payload,args.output_dir/"metrics.json")
            return payload
        except BaseException as error:
            payload.update(status="failed",error=f"{type(error).__name__}: {error}")
            save_json(payload,args.output_dir/"metrics.json")
            raise


def main():
    args = parse_args()
    try:
        run_experiments(args)
        print(f"\nSaved results to: {args.output_dir.resolve()}")
    except (OSError,ValueError,RuntimeError,AssertionError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
