"""Offline CPU tests using synthetic MNIST-shaped images, not scientific results."""
from __future__ import annotations

import copy
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import transforms

from E1.analyze_softmax_experiments import analyze, mean_std
from E1.models import SoftmaxRegression
from E1.run_softmax_experiments import (
    DEFAULT_CONFIG, FiniteCrossEntropy, build_plan, file_hash, lock_selection,
    read_json, test_selected, train_run, validate_config,
)
from E1.softmax_data import build_softmax_loaders, fixed_split


class SyntheticDigits(Dataset):
    def __init__(self, size=40):
        generator = torch.Generator().manual_seed(123)
        self.data = torch.randint(0, 256, (size, 28, 28), dtype=torch.uint8, generator=generator)
        self.targets = torch.arange(size) % 10
        self.transform = None

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return Image.fromarray(self.data[index].numpy()), int(self.targets[index])


def tiny_config():
    config = read_json(DEFAULT_CONFIG)
    config.update(study_id="smoke", train_seeds=[11, 22], val_ratio=0.25)
    config["baseline"].update(epochs=1, batch_size=8)
    config["levels"] = {key: [value] for key, value in config["baseline"].items()}
    config["levels"]["augmentation"] = [False, True]
    config["levels"]["epochs"] = [1, 2]
    return config


class SoftmaxExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="softmax_smoke_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.study = self.root / "results"
        self.cp = self.root / "checkpoints"
        self.raw = SyntheticDigits()
        self.config = tiny_config()

    def run_one(self, run, config=None, study=None, cp=None, **kwargs):
        return train_run(config or self.config, run, study or self.study, cp or self.cp,
                         num_workers=0, raw_train=self.raw, **kwargs)

    def test_default_plan_and_invalid_config(self):
        config = read_json(DEFAULT_CONFIG)
        plan = build_plan(config)
        self.assertEqual(len(plan["experiments"]), 20)
        self.assertEqual(len(plan["runs"]), 45)
        self.assertEqual(sum(len(r["experiments"]) for r in plan["runs"]), 60)
        self.assertEqual(plan, build_plan(config))
        for item in plan["experiments"][1:]:
            self.assertEqual(sum(item[key] != config["baseline"][key] for key in config["baseline"]), 1)
        bad = copy.deepcopy(config)
        bad["study_id"] = "../escape"
        with self.assertRaises(ValueError):
            validate_config(bad)
        bad = copy.deepcopy(config)
        bad["levels"]["learning_rate"].append(float("nan"))
        with self.assertRaises(ValueError):
            validate_config(bad)
        self.assertEqual(sum(p.numel() for p in SoftmaxRegression().parameters()), 7850)

    def test_fixed_split_augmentation_and_worker(self):
        original = random_split(range(60000), [54000, 6000],
                                generator=torch.Generator().manual_seed(42))
        train, val = fixed_split(60000, 0.1, 42)
        self.assertEqual(train, original[0].indices)
        self.assertEqual(val, original[1].indices)
        self.assertFalse(set(train) & set(val))
        bundles = []
        for seed, batch, augmentation in [(11, 2, False), (22, 16, True)]:
            bundles.append(build_softmax_loaders("unused", batch, .25, 42, seed, augmentation,
                                                self.config["rotation"], 0, raw_train=self.raw))
        self.assertEqual(bundles[0].metadata["train_indices_sha256"],
                         bundles[1].metadata["train_indices_sha256"])
        self.assertEqual(bundles[0].metadata["validation_indices_sha256"],
                         bundles[1].metadata["validation_indices_sha256"])
        self.assertEqual(self.raw.transform, None)
        for bundle in bundles:
            self.assertFalse(any(isinstance(t, transforms.RandomRotation)
                                 for t in bundle.val_loader.dataset.transform.transforms))
        self.assertTrue(torch.equal(bundles[0].val_loader.dataset[0][0],
                                    bundles[1].val_loader.dataset[0][0]))
        self.assertTrue(any(isinstance(t, transforms.RandomRotation)
                            for t in bundles[1].train_loader.dataset.transform.transforms))
        # Windows spawn: a real worker process must be able to pickle the raw/view datasets.
        worker = build_softmax_loaders("unused", 8, .25, 42, 11, True,
                                      self.config["rotation"], 1, raw_train=self.raw)
        first = list(worker.train_loader)
        second = list(worker.train_loader)
        self.assertEqual(sum(len(y) for _, y in first), 30)
        self.assertEqual(sum(len(y) for _, y in second), 30)
        worker_repeat = build_softmax_loaders("unused", 8, .25, 42, 11, True,
                                             self.config["rotation"], 1, raw_train=self.raw)
        repeated = list(worker_repeat.train_loader)
        self.assertTrue(all(torch.equal(x, rx) and torch.equal(y, ry)
                            for (x, y), (rx, ry) in zip(first, repeated)))

    def test_prefix_matches_independent_run_and_no_test_access(self):
        for augmentation in (False, True):
            with self.subTest(augmentation=augmentation):
                config = tiny_config()
                config["baseline"]["augmentation"] = augmentation
                config["levels"]["augmentation"] = [augmentation]
                run = build_plan(config)["runs"][0]
                study, cp = self.root / f"prefix{augmentation}", self.root / f"cp{augmentation}"
                with patch("E1.run_softmax_experiments.build_softmax_test_loader",
                           side_effect=AssertionError("Training touched test")):
                    result = self.run_one(run, config, study, cp)
                standalone = copy.deepcopy(config)
                standalone["levels"]["epochs"] = [1]
                single = build_plan(standalone)["runs"][0]
                single_cp = self.root / f"single_cp{augmentation}"
                independent = self.run_one(single, standalone, self.root / f"single{augmentation}", single_cp)
                for kind in ("best_checkpoint", "last_checkpoint"):
                    a = torch.load(cp / run["run_id"] / result["budgets"]["1"][kind]["filename"],
                                   weights_only=True)["state_dict"]
                    b = torch.load(single_cp / single["run_id"] / independent["budgets"]["1"][kind]["filename"],
                                   weights_only=True)["state_dict"]
                    self.assertTrue(all(torch.equal(a[key], b[key]) for key in a))
                self.assertEqual(result["budgets"]["1"]["last_val_accuracy"],
                                 independent["budgets"]["1"]["last_val_accuracy"])
                with (study / run["run_id"] / "history.csv").open(encoding="utf-8", newline="") as stream:
                    history = list(csv.DictReader(stream))
                for cap in (1, 2):
                    best = max(float(row["val_accuracy"]) for row in history[:cap])
                    first_epoch = next(int(row["epoch"]) for row in history[:cap]
                                       if float(row["val_accuracy"]) == best)
                    self.assertEqual(result["budgets"][str(cap)]["best_epoch"], first_epoch)
                    self.assertEqual(result["budgets"][str(cap)]["best_val_accuracy"], best)
                with self.assertRaises(FileExistsError):
                    self.run_one(run, config, study, cp)
                self.assertEqual(self.run_one(run, config, study, cp, skip_completed=True)["status"], "completed")

    def test_end_to_end_analysis_selection_and_test(self):
        plan = build_plan(self.config)
        first = self.run_one(plan["runs"][0])
        partial = analyze(self.study, make_plot=False)
        self.assertEqual(partial["validation_ranking_complete_only"], [])
        self.assertEqual(partial["aggregates"][0]["missing_seeds"], [22])
        with self.assertRaises(FileNotFoundError):
            lock_selection(self.config, self.study, self.cp, ["S00"])
        with self.assertRaises(FileNotFoundError):
            test_selected(self.study, self.cp, num_workers=0, raw_test=SyntheticDigits(20))
        for run in plan["runs"][1:]:
            self.run_one(run)
        aggregate = analyze(self.study)
        self.assertEqual(len(aggregate["validation_ranking_complete_only"]), 3)
        self.assertTrue((self.study / "analysis/comparison.png").is_file())
        self.assertEqual(mean_std([1., 3.])["std"], 2**.5)
        self.assertIsNone(mean_std([1.])["std"])
        locked = lock_selection(self.config, self.study, self.cp, ["S00", "S01"])
        self.assertEqual(len(locked["entries"]), 4)
        with self.assertRaises(ValueError):
            self.run_one(plan["runs"][0], skip_completed=True)
        with self.assertRaises(FileExistsError):
            lock_selection(self.config, self.study, self.cp, ["S00"])
        reports = test_selected(self.study, self.cp, num_workers=0, raw_test=SyntheticDigits(20))
        self.assertEqual(len(reports), 4)
        self.assertTrue(all(0 <= r["macro_f1"] <= 1 for r in reports))
        final = read_json(self.study / "final_test" / locked["sha256"][:12] / "metrics.json")
        self.assertEqual(len(final["aggregates"]), 2)
        self.assertTrue(all(item["n"] == 2 for item in final["aggregates"]))
        self.assertEqual(aggregate["validation_ranking_complete_only"],
                         analyze(self.study, make_plot=False)["validation_ranking_complete_only"])
        with self.assertRaises(FileExistsError):
            test_selected(self.study, self.cp, num_workers=0, raw_test=SyntheticDigits(20))
        self.assertEqual(len(test_selected(self.study, self.cp, num_workers=0,
                                          raw_test=SyntheticDigits(20), skip_completed=True)), 4)
        entry = locked["entries"][0]
        path = self.cp / entry["run_id"] / entry["checkpoint"]["filename"]
        with path.open("ab") as stream:
            stream.write(b"tampered")
        with patch("E1.run_softmax_experiments.build_softmax_test_loader") as loader:
            with self.assertRaises(ValueError):
                test_selected(self.study, self.cp, num_workers=0, raw_test=SyntheticDigits(20))
            loader.assert_not_called()
        self.assertNotEqual(file_hash(path), first["budgets"]["1"]["best_checkpoint"]["sha256"])

    def test_failed_runs_and_source_drift(self):
        run = build_plan(self.config)["runs"][0]
        with patch("E1.run_softmax_experiments.train_one_epoch", side_effect=FloatingPointError("bad loss")):
            with self.assertRaises(FloatingPointError):
                self.run_one(run)
        self.assertEqual(read_json(self.study / run["run_id"] / "metrics.json")["status"], "failed")
        self.assertEqual(analyze(self.study, make_plot=False)["validation_ranking_complete_only"], [])
        with self.assertRaises(FloatingPointError):
            FiniteCrossEntropy()(torch.full((2, 10), float("nan")), torch.tensor([0, 1]))
        changed = copy.deepcopy(self.config)
        changed["rotation"]["degrees"] = 15
        with self.assertRaises(ValueError):
            self.run_one(build_plan(changed)["runs"][0], changed)

    def test_cli_plan_help_and_invalid_arguments(self):
        script = str(ROOT / "E1/run_softmax_experiments.py")
        result = subprocess.run([sys.executable, script, "--phase", "plan"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"physical_training_runs": 45', result.stdout)
        result = subprocess.run([sys.executable, script, "--phase", "train", "--seeds", "999"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("subset", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
