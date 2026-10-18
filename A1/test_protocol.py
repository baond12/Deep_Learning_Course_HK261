"""Offline CPU integration tests; these results are never CIFAR-100 measurements."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from A1.data import A1DataBundle
from A1.run_protocol import PROTOCOL_FILE, parse_args, run, evaluate_once, export_dataset_summary
from A1.test_pilot import ForbiddenTestLoader, tiny_factory


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="a1_protocol_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
        self.protocol["epochs"] = 2
        images = torch.randn(4, 3, 32, 32, generator=torch.Generator().manual_seed(17))
        dataset = TensorDataset(images, torch.tensor([0, 1, 0, 1]))
        self.data = A1DataBundle(
            DataLoader(dataset, batch_size=2, shuffle=True, generator=torch.Generator()),
            DataLoader(dataset, batch_size=2, generator=torch.Generator()), ForbiddenTestLoader(),
            ["zero", "one"], 4, 4, 0, 32, {"train": [0, 1, 2, 3], "validation": [4, 5, 6, 7]})
        self.test_data = A1DataBundle(self.data.train_loader, self.data.val_loader,
            DataLoader(dataset, batch_size=2), ["zero", "one"], 4, 4, 4, 32,
            {"train": [0, 1, 2, 3], "validation": [4, 5, 6, 7], "test": [30, 10, 20, 40]})

    def case(self, phase="train", names=None, output="run", factory=tiny_factory, protocol=None, data=None):
        argv = ["--phase", phase, "--data-dir", "unused", "--device", "cpu", "--num-workers", "0", "--output-dir", str(self.root/output)]
        if names:
            argv += ["--experiments", *names]
        with contextlib.redirect_stdout(io.StringIO()):
            return run(parse_args(argv), data_bundle=data or (self.test_data if phase == "test" else self.data),
                       model_factory=factory, protocol_override=protocol or self.protocol)

    def test_all_eight_train_with_exact_lr_and_no_test_access(self):
        result = self.case()
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["test_evaluated"])
        self.assertTrue(result["synthetic_test"])
        self.assertEqual(len(result["experiments"]), 8)
        for name, record in result["experiments"].items():
            self.assertEqual(record["learning_rate"], 1e-5 if name == "vit_b16_pretrained_full" else 1e-4)
            checkpoint = torch.load(self.root/"run"/record["checkpoint"], weights_only=True)
            best = max(record["history"], key=lambda r: r["val_accuracy"])
            self.assertEqual(checkpoint["best_epoch"], best["epoch"])
            self.assertEqual(checkpoint["best_val_accuracy"], best["val_accuracy"])
            self.assertEqual(record["epochs_completed"], 2)
            self.assertTrue(record["frozen_state_unchanged"])
            self.assertTrue(record["trainable_state_changed"])

    def test_checkpoint_best_is_not_overwritten_by_worse_or_tied_epoch(self):
        from A1.engine import A1TrainResult
        from A1.models import set_a1_train_mode
        first_state = {}
        def controlled_fit(model, train_loader, val_loader, epochs, lr, wd, device, progress, on_epoch, on_phase):
            set_a1_train_mode(model)
            history, phases = [], []
            for epoch, accuracy in enumerate([0.75, 0.50, 0.75], 1):
                with torch.no_grad():
                    model.fc.weight.add_(1)
                row = {"epoch": epoch, "train_loss": 1.0, "train_accuracy": 0.5, "val_loss": 1.0, "val_accuracy": accuracy}
                phase = {"seconds": 1.0, "samples": 4, "batches": 2, "peak_allocated_mib": None, "peak_reserved_mib": None}
                pair = {"epoch": epoch, "train": copy.deepcopy(phase), "validation": copy.deepcopy(phase)}
                history.append(row)
                phases.append(pair)
                on_epoch(row, pair)
                if epoch == 1:
                    first_state.update({k: v.clone() for k, v in model.state_dict().items()})
            model.load_state_dict(first_state)
            return A1TrainResult(history, 1, .75, 1, 6., phases)
        protocol = {**self.protocol, "epochs": 3}
        with patch("A1.run_protocol.fit_classifier", side_effect=controlled_fit):
            result = self.case(names=["resnet18_scratch"], protocol=protocol)
        self.assertEqual(result["status"], "completed")
        checkpoint = torch.load(self.root/"run/checkpoints/resnet18_scratch.pt", weights_only=True)
        self.assertEqual(checkpoint["best_epoch"], 1)
        for k, value in checkpoint["state_dict"].items():
            self.assertTrue(torch.equal(value, first_state[k]))

    def test_final_evaluation_is_once_and_preserves_training_results(self):
        import csv
        trained = self.case(names=["resnet18_scratch", "vit_b16_pretrained_full"])
        before = (self.root/"run/metrics.json").read_bytes()
        result = self.case("test")
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["test_evaluated"])
        self.assertEqual(before, (self.root/"run/metrics.json").read_bytes())
        for name, r in result["experiments"].items():
            with (self.root/f"run/test/predictions_{name}.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(x["original_test_index"]) for x in rows], [30, 10, 20, 40])
            accuracy = sum(x["true_label"] == x["predicted_label"] for x in rows)/4
            self.assertEqual(accuracy, r["test_accuracy"])
            self.assertEqual(trained["experiments"][name]["checkpoint_sha256"], r["checkpoint_sha256"])
            self.assertTrue((self.root/f"run/test/confusion_matrix_{name}.csv").exists())
        with self.assertRaises(FileExistsError):
            self.case("test")

    def test_existing_training_output_is_preserved(self):
        self.case(names=["resnet18_scratch"])
        before = (self.root/"run/metrics.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.case(names=["resnet18_scratch"])
        self.assertEqual(before, (self.root/"run/metrics.json").read_bytes())

    def test_tampered_checkpoint_rejected_before_loading_test_data(self):
        self.case(names=["resnet18_scratch"])
        path = self.root/"run/checkpoints/resnet18_scratch.pt"
        with path.open("ab") as stream:
            stream.write(b"changed")
        with patch("A1.run_protocol.get_data", side_effect=AssertionError("Test data was accessed")):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                self.case("test")
        self.assertFalse((self.root/"run/test").exists())

    def test_changed_protocol_or_source_rejected_before_test(self):
        self.case(names=["resnet18_scratch"])
        changed = {**self.protocol, "epochs": 3}
        with self.assertRaisesRegex(ValueError, "protocol"):
            self.case("test", protocol=changed)
        path = self.root/"run/metrics.json"
        m = json.loads(path.read_text(encoding="utf-8"))
        m["environment"]["source_sha256_lf"]["A1/engine.py"] = "changed"
        path.write_text(json.dumps(m), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Source changed"):
            self.case("test")

    def test_failed_run_keeps_completed_results_and_blocks_final_test(self):
        def fail(**kwargs):
            if kwargs["mode"] == "scratch":
                raise torch.cuda.OutOfMemoryError("synthetic OOM")
            return tiny_factory(**kwargs)
        result = self.case(names=["resnet18_scratch", "resnet18_pretrained_head"], factory=fail)
        self.assertEqual(result["status"], "completed_with_failures")
        self.assertEqual(result["experiments"]["resnet18_pretrained_head"]["status"], "completed")
        with self.assertRaisesRegex(ValueError, "completed training"):
            self.case("test", factory=fail)

    def test_nonfinite_training_is_recorded(self):
        def fail(**kwargs):
            model = tiny_factory(**kwargs)
            model.fc.register_forward_hook(lambda _, inputs, output: output * float("nan"))
            return model
        result = self.case(names=["resnet18_scratch"], factory=fail)
        self.assertEqual(result["experiments"]["resnet18_scratch"]["status"], "nonfinite")
        self.assertFalse((self.root/"run/checkpoints/resnet18_scratch.pt").exists())

    def test_test_split_order_mismatch_is_rejected(self):
        self.case(names=["resnet18_scratch"])
        bad_data = A1DataBundle(self.test_data.train_loader, self.test_data.val_loader, self.test_data.test_loader,
            ["one", "zero"], 4, 4, 4, 32, self.test_data.split_indices)
        with self.assertRaisesRegex(ValueError, "class order"):
            self.case("test", data=bad_data)

    def test_nonfinite_test_logits_are_rejected(self):
        model = tiny_factory(architecture="resnet18", mode="scratch", num_classes=2, download_weights=False)
        model.fc.register_forward_hook(lambda _, inputs, output: output * float("nan"))
        with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
            evaluate_once(model, self.test_data.test_loader, torch.device("cpu"), [0, 1, 2, 3], ["zero", "one"], False)

    def test_eda_uses_only_training_source_and_checks_actual_counts(self):
        import numpy as np
        from PIL import Image
        class Raw:
            targets = [i // 2 for i in range(200)]
            data = np.zeros((200, 32, 32, 3), dtype=np.uint8)
            def __getitem__(self, index):
                return Image.fromarray(self.data[index]), self.targets[index]
        class Selection:
            dataset = Raw()
        class Loader:
            dataset = Selection()
        indices = {"train": list(range(0, 200, 2)), "validation": list(range(1, 200, 2))}
        data = A1DataBundle(Loader(), None, ForbiddenTestLoader(), [str(i) for i in range(100)], 100, 100, 0, 224, indices)
        protocol = {**self.protocol, "train_per_class": 1, "val_per_class": 1}
        export_dataset_summary(data, protocol, self.root)
        summary = json.loads((self.root/"dataset_summary.json").read_text())
        self.assertFalse(summary["test_loaded"])
        self.assertEqual(summary["original_training_label_counts"], [2] * 100)
        self.assertTrue(summary["train_validation_indices_disjoint"])
        self.assertTrue((self.root/"dataset_samples_train.png").exists())
        with self.assertRaisesRegex(ValueError, "per-class counts"):
            export_dataset_summary(data, {**protocol, "train_per_class": 2}, self.root)


if __name__ == "__main__":
    unittest.main()
