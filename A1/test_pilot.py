"""Offline CPU checks. Synthetic data/weights here are never reported as CIFAR results."""
import contextlib
import gc
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models.vision_transformer import VisionTransformer

from A1.data import A1DataBundle, build_cifar100_balanced_loaders
from A1.models import (build_a1_model, count_parameters, _configure_resnet_trainability,
                       _configure_vit_trainability)
from A1.pilot import parse_args, run_pilot


class TinyResNet(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(4, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU())
        self.fc = nn.Linear(8, classes)

    def forward(self, x):
        return self.fc(self.layer4(self.stem(x)).mean((2, 3)))


def tiny_factory(architecture, mode, num_classes, download_weights):
    if architecture == "resnet18":
        model = TinyResNet(num_classes)
        _configure_resnet_trainability(model, mode)
    else:
        model = VisionTransformer(image_size=32, patch_size=8, num_layers=2, num_heads=2,
                                  hidden_dim=16, mlp_dim=32, num_classes=num_classes)
        _configure_vit_trainability(model, mode)
    model._a1_architecture, model._a1_mode = architecture, mode
    model._a1_weights = "synthetic-test"
    return model


class ForbiddenTestLoader:
    generator = None

    def __iter__(self):
        raise AssertionError("Pilot accessed the test set")


class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="a1_pilot_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        generator = torch.Generator().manual_seed(99)
        images = torch.randn(4, 3, 32, 32, generator=generator)
        dataset = TensorDataset(images, torch.tensor([0, 1, 0, 1]))
        self.data = A1DataBundle(DataLoader(dataset, batch_size=2, shuffle=True, generator=torch.Generator()),
                                 DataLoader(dataset, batch_size=2), ForbiddenTestLoader(),
                                 ["zero", "one"], 4, 4, 0, 32,
                                 {"train": [0, 1, 2, 3], "validation": [4, 5, 6, 7]})

    def run_case(self, name="run", experiments=None, factory=tiny_factory):
        argv = ["--output-dir", str(self.root/name), "--device", "cpu", "--num-workers", "0", "--batch-size", "2"]
        if experiments:
            argv += ["--experiments", *experiments]
        with contextlib.redirect_stdout(io.StringIO()):
            return run_pilot(parse_args(argv), data_bundle=self.data, model_factory=factory)

    def test_all_eight_train_and_validate_without_test(self):
        payload = self.run_case()
        self.assertEqual(payload["status"], "completed")
        self.assertTrue(payload["synthetic_test"])
        self.assertFalse(payload["test_evaluated"])
        self.assertEqual(len(payload["experiments"]), 8)
        for record in payload["experiments"].values():
            self.assertEqual(record["epochs_completed"], 1)
            self.assertTrue(record["frozen_state_unchanged"])
            self.assertTrue(record["trainable_state_changed"])
            self.assertIsNone(record["peak_allocated_mib"])
            self.assertEqual(record["phase_measurements"][0]["train"]["samples"], 4)
        loaded = json.loads((self.root/'run/metrics.json').read_text(encoding='utf-8'))
        self.assertEqual(loaded["status"], "completed")
        self.assertEqual(len(list((self.root/'run').glob('run_*.json'))), 8)

    def test_nonfinite_loss_is_recorded_and_next_run_continues(self):
        def factory(**kwargs):
            model = tiny_factory(**kwargs)
            if kwargs["mode"] == "scratch":
                model.fc.register_forward_hook(lambda module, inputs, output: output * float('nan'))
            return model
        payload = self.run_case(experiments=['resnet18_scratch', 'resnet18_pretrained_head'], factory=factory)
        self.assertEqual(payload["status"], "completed_with_failures")
        self.assertEqual(payload["experiments"]['resnet18_scratch']["status"], "nonfinite")
        self.assertEqual(payload["experiments"]['resnet18_pretrained_head']["status"], "completed")
        self.assertNotIn('NaN', (self.root/'run/metrics.json').read_text(encoding='utf-8'))

    def test_nonfinite_gradient_is_recorded(self):
        def factory(**kwargs):
            model = tiny_factory(**kwargs)
            model.fc.weight.register_hook(lambda gradient: torch.full_like(gradient, float('nan')))
            return model
        payload = self.run_case(experiments=['resnet18_scratch'], factory=factory)
        self.assertEqual(payload["experiments"]['resnet18_scratch']["status"], "nonfinite")
        self.assertIn('gradient norm', payload["experiments"]['resnet18_scratch']["error"])

    def test_validation_failure_keeps_completed_training_measurement(self):
        def factory(**kwargs):
            model = tiny_factory(**kwargs)
            model.fc.register_forward_hook(lambda module, inputs, output: output if module.training else output * float('nan'))
            return model
        payload = self.run_case(experiments=['resnet18_scratch'], factory=factory)
        record = payload['experiments']['resnet18_scratch']
        self.assertEqual(record['status'], 'nonfinite')
        self.assertEqual(record['last_completed_phase']['phase'], 'train')
        self.assertEqual(record['last_completed_phase']['samples'], 4)
        self.assertEqual(record['epochs_completed'], 0)

    def test_missing_data_keeps_preparation_diagnostics(self):
        args = parse_args(['--output-dir', str(self.root/'missing'), '--device', 'cpu'])
        with patch('A1.pilot.build_cifar100_balanced_loaders', side_effect=RuntimeError('missing data')):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                run_pilot(args)
        payload = json.loads((self.root/'missing/metrics.json').read_text(encoding='utf-8'))
        self.assertEqual(payload['status'], 'preparation_failed')
        self.assertIn('missing data', payload['error'])

    def test_simulated_oom_is_recorded_and_next_run_continues(self):
        def factory(**kwargs):
            if kwargs["mode"] == "scratch":
                raise torch.cuda.OutOfMemoryError("controlled synthetic out of memory")
            return tiny_factory(**kwargs)
        payload = self.run_case(experiments=['resnet18_scratch', 'resnet18_pretrained_head'], factory=factory)
        self.assertEqual(payload["experiments"]['resnet18_scratch']["status"], "out_of_memory")
        self.assertEqual(payload["experiments"]['resnet18_pretrained_head']["status"], "completed")

    def test_interrupt_stops_remaining_runs_but_keeps_report(self):
        def factory(**kwargs):
            raise KeyboardInterrupt()
        payload = self.run_case(experiments=['resnet18_scratch', 'resnet18_pretrained_head'], factory=factory)
        self.assertEqual(payload["status"], "interrupted")
        self.assertEqual(payload["experiments"]['resnet18_scratch']["status"], "interrupted")
        self.assertEqual(payload["experiments"]['resnet18_pretrained_head']["status"], "not_run")

    def test_existing_output_is_not_overwritten(self):
        self.run_case(experiments=['resnet18_scratch'])
        original = (self.root/'run/metrics.json').read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_case(experiments=['resnet18_scratch'])
        self.assertEqual(original, (self.root/'run/metrics.json').read_bytes())

    def test_seed_is_independent_of_experiment_order(self):
        a = self.run_case('a', ['resnet18_scratch', 'resnet18_pretrained_head'])
        b = self.run_case('b', ['resnet18_pretrained_head', 'resnet18_scratch'])
        for name in a['experiments']:
            self.assertEqual(a['experiments'][name]['history'], b['experiments'][name]['history'])

    def test_changed_frozen_buffer_is_rejected(self):
        def factory(**kwargs):
            model = tiny_factory(**kwargs)
            def corrupt(module, inputs, output):
                module.running_mean.add_(0.1)
            model.stem[1].register_forward_hook(corrupt)
            return model
        payload = self.run_case(experiments=['resnet18_pretrained_head'], factory=factory)
        self.assertEqual(payload['experiments']['resnet18_pretrained_head']['status'], 'failed')
        self.assertFalse(payload['experiments']['resnet18_pretrained_head']['frozen_state_unchanged'])

    def test_data_split_fixed_and_test_not_loaded(self):
        class FakeCifar:
            def __init__(self, root, train, download, transform):
                if not train:
                    raise AssertionError('Pilot loaded test data')
                self.classes = [str(i) for i in range(100)]
                self.targets = [i for i in range(100) for _ in range(500)]

            def __len__(self):
                return len(self.targets)
        with patch('A1.data.datasets.CIFAR100', FakeCifar):
            a = build_cifar100_balanced_loaders('unused', 224, 100, 400, 100, 100, 16, 42, 0, include_test=False, train_seed=42)
            b = build_cifar100_balanced_loaders('unused', 224, 100, 400, 100, 100, 16, 42, 0, include_test=False, train_seed=43)
        self.assertEqual(a.split_indices, b.split_indices)
        self.assertEqual((a.train_size, a.val_size), (40000, 10000))
        self.assertFalse(set(a.split_indices['train']) & set(a.split_indices['validation']))
        self.assertIsNone(a.test_loader)

    def test_actual_model_parameter_counts_and_freeze_masks(self):
        expected = {'resnet18': (11227812, [11227812, 51300, 8445028, 11227812]),
                    'vit_b_16': (85875556, [85875556, 76900, 14252644, 85875556])}
        for architecture, (total, counts) in expected.items():
            for mode, count in zip(('scratch','pretrained_head','pretrained_partial','pretrained_full'), counts):
                model = build_a1_model(architecture, mode, 100, download_weights=False)
                self.assertEqual(count_parameters(model), (total, count))
                del model
                gc.collect()


if __name__ == '__main__':
    unittest.main()
