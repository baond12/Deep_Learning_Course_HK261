"""Offline E2 tests: numerical equivalence, paired initialization, and a tiny full run."""
from __future__ import annotations

import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".matplotlib-cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from common.data import DataBundle
from E2.models import (
    EXPERIMENT_CONFIGS, TOKENIZATIONS, CustomMultiHeadSelfAttention, PatchTokenizer,
    OverlapPatchTokenizer, build_model, copy_custom_to_pytorch, tokenization_metadata,
)
from E2.train import experiment_description, paired_model, parse_args, reset_loaders, run_experiments
from E2.verify_msa import VerificationError, verify_experiments


def fake_data() -> DataBundle:
    generator = torch.Generator().manual_seed(123)
    images = torch.randn(60, 1, 28, 28, generator=generator)
    images[:, 0, 0, 0] = torch.arange(60)
    dataset = TensorDataset(images, torch.arange(60) % 10)
    train, val = random_split(dataset, [40, 20], generator=torch.Generator().manual_seed(42))
    test = TensorDataset(torch.randn(20, 1, 28, 28, generator=generator), torch.arange(20) % 10)
    return DataBundle(DataLoader(train), DataLoader(val), DataLoader(test), 40, 20, 20, 10, (1, 28, 28))


class E2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".e2_smoke_", dir=ROOT_DIR)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def args(self, *more):
        return parse_args(["--device", "cpu", "--num-workers", "0",
                           "--output-dir", str(self.directory / "results"), *more])

    def test_four_tokenizers_eight_models_and_geometry(self):
        expected = {"patch": 16, "row": 28, "cnn_stem": 49, "overlap_patch": 64}
        self.assertEqual(len(EXPERIMENT_CONFIGS), 8)
        for name, config in EXPERIMENT_CONFIGS.items():
            with self.subTest(name=name):
                model = build_model(name)
                image = torch.randn(2, 1, 28, 28)
                tokens = model.tokenizer(image)
                self.assertEqual(tuple(tokens.shape), (2, expected[config["tokenizer_type"]], 64))
                self.assertEqual(tokenization_metadata(model)["total_tokens"], tokens.shape[1] + 1)
                loss = nn.functional.cross_entropy(model(image), torch.tensor([0, 9]))
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                if config["attention_type"] == "custom":
                    self.assertFalse(any(isinstance(m, (nn.MultiheadAttention, nn.TransformerEncoder,
                                                        nn.TransformerEncoderLayer)) for m in model.modules()))
        for tokenizer in (PatchTokenizer(), OverlapPatchTokenizer()):
            with torch.no_grad():
                tokenizer.projection.weight.fill_(1)
                tokenizer.projection.bias.zero_()
            image = torch.zeros(1, 1, 28, 28)
            image[0, 0, -1, -1] = 1
            self.assertEqual(float(tokenizer(image)[0, -1, 0].detach()), 1.)
        with self.assertRaises(ValueError):
            PatchTokenizer(patch_size=5)
        with self.assertRaises(ValueError):
            OverlapPatchTokenizer(stride=7)
        with self.assertRaises(ValueError):
            build_model("custom_patch")(torch.randn(2, 1, 32, 32))
        self.assertEqual(experiment_description("custom_patch", {"patch_size": 4})["tokenization"], "patch_4x4")

    def test_numerical_equivalence_all_tokenizers(self):
        for kwargs in ({}, {"embed_dim": 32, "num_heads": 8, "ffn_dim": 48,
                            "depth": 2, "patch_size": 4, "overlap_stride": 3, "dropout": .25}):
            report = verify_experiments(list(TOKENIZATIONS), kwargs, torch.device("cpu"))
            self.assertTrue(report["passed"])
            self.assertEqual(len(report["cases"]), 8)
            self.assertTrue(all(case["parameter_tensors_checked"] > 0 for case in report["cases"]))

    def test_pair_initialization_dropout_and_loader_order(self):
        for tokenizer in TOKENIZATIONS:
            a, hash_a = paired_model(f"custom_{tokenizer}", {}, 42)
            b, hash_b = paired_model(f"pytorch_{tokenizer}", {}, 42)
            self.assertEqual(hash_a, hash_b)
            self.assertEqual(sum(p.numel() for p in a.parameters()), sum(p.numel() for p in b.parameters()))
            self.assertFalse(hasattr(a.blocks[0].attention, "out_drop"))
            self.assertEqual(a.blocks[0].attention.attn_drop.p, b.blocks[0].attention.attention.dropout)
            self.assertEqual(a.blocks[0].drop1.p, b.blocks[0].drop1.p)
        with self.assertRaises(ValueError):
            copy_custom_to_pytorch(build_model("custom_row"), build_model("pytorch_patch"))
        data = fake_data()
        args = self.args("--batch-size", "8")
        first = reset_loaders(data, args)
        order_a = [[int(v) for x, _ in first.train_loader for v in x[:, 0, 0, 0]] for _ in range(2)]
        torch.rand(3000)  # Simulate a different amount of dropout RNG consumption.
        second = reset_loaders(data, args)
        order_b = [[int(v) for x, _ in second.train_loader for v in x[:, 0, 0, 0]] for _ in range(2)]
        self.assertEqual(order_a, order_b)
        self.assertEqual(set(order_a[0]), set(order_a[1]))
        self.assertEqual(first.train_loader.dataset.indices, data.train_loader.dataset.indices)
        self.assertEqual(first.val_loader.dataset.indices, data.val_loader.dataset.indices)
        third = reset_loaders(data, self.args("--batch-size", "8", "--train-seed", "43"))
        self.assertEqual(third.train_loader.dataset.indices, first.train_loader.dataset.indices)
        self.assertEqual(third.val_loader.dataset.indices, first.val_loader.dataset.indices)
        order_c = [int(v) for x, _ in third.train_loader for v in x[:, 0, 0, 0]]
        self.assertNotEqual(order_a[0], order_c)

    def test_actual_cli_verification_and_invalid_options(self):
        output = self.directory / "cli"
        task_environment = os.environ.copy()
        task_environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        result = subprocess.run([sys.executable, str(ROOT_DIR / "E2/train.py"), "--verify-only",
                                 "--device", "cpu", "--output-dir", str(output)],
                                capture_output=True, text=True, env=task_environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        verification = json.loads((output / "verification.json").read_text())
        self.assertTrue(verification["passed"])
        self.assertEqual(len(verification["cases"]), 8)
        self.assertFalse((output / "metrics.json").exists())
        result = subprocess.run([sys.executable, str(ROOT_DIR / "E2/train.py"), "--epochs", "0"],
                                capture_output=True, text=True, env=task_environment)
        self.assertEqual(result.returncode, 2)
        self.assertIn("positive", result.stderr)

    def test_failure_blocks_training_and_data_access(self):
        original = CustomMultiHeadSelfAttention.forward

        def broken_forward(module, x):
            return original(module, x) + .1

        with patch.object(CustomMultiHeadSelfAttention, "forward", broken_forward):
            with patch("E2.train.build_mnist_loaders") as data, patch("E2.train.fit_classifier") as fit:
                with self.assertRaises(VerificationError):
                    run_experiments(self.args("--experiments", "custom_row"))
                data.assert_not_called()
                fit.assert_not_called()
        report = json.loads((self.directory / "results/verification.json").read_text())
        self.assertFalse(report["passed"])
        self.assertFalse((self.directory / "results/.run.lock").exists())

    def test_verify_only_preserves_rng_and_never_loads_mnist(self):
        torch.manual_seed(12345)
        before = torch.get_rng_state().clone()
        with patch("E2.train.build_mnist_loaders") as data:
            result = run_experiments(self.args("--verify-only"))
            data.assert_not_called()
        self.assertTrue(result["verification"]["passed"])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertFalse((self.directory / "results/metrics.json").exists())

    def test_full_tiny_training_export_and_overwrite_guard(self):
        args = self.args("--epochs", "1", "--batch-size", "256", "--embed-dim", "16", "--ffn-dim", "32")
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_experiments(args, data_bundle=fake_data())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["dataset"], "synthetic_smoke")
        self.assertEqual(len(result["experiments"]), 8)
        out = self.directory / "results"
        for filename, expected_rows in (("summary.csv", 8), ("comparison_msa.csv", 4), ("comparison_tokenization.csv", 8)):
            with (out / filename).open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), expected_rows)
            if filename == "comparison_msa.csv":
                self.assertTrue(all(row["pair_complete"] == "True" for row in rows))
        self.assertTrue((out / "comparison.png").is_file())
        commentary = (out / "comparison.md").read_text(encoding="utf-8")
        self.assertIn("custom/cnn_stem:", commentary)
        self.assertIn("pytorch/overlap_patch:", commentary)
        self.assertIn("So với patch trong cùng backend:", commentary)
        for tokenizer in TOKENIZATIONS:
            self.assertTrue((out / f"learning_curves_{tokenizer}.png").is_file())
            a = result["experiments"][f"custom_{tokenizer}"]
            b = result["experiments"][f"pytorch_{tokenizer}"]
            self.assertEqual(a["initialization_sha256"], b["initialization_sha256"])
        for name in EXPERIMENT_CONFIGS:
            checkpoint = torch.load(out / f"checkpoints/best_{name}.pt", weights_only=True)
            self.assertEqual(checkpoint["experiment"], name)
            self.assertEqual(checkpoint["best_epoch"], 1)
            model = build_model(name, **checkpoint["model_kwargs"])
            model.load_state_dict(checkpoint["state_dict"])
            self.assertTrue((out / f"confusion_matrix_{name}.png").is_file())
        original_metrics = (out / "metrics.json").read_bytes()
        with self.assertRaises(FileExistsError):
            run_experiments(args, data_bundle=fake_data())
        self.assertEqual((out / "metrics.json").read_bytes(), original_metrics)
        self.assertFalse((out / ".run.lock").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
