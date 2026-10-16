"""MNIST views with a fixed split and separate training randomness.

The training builder never constructs the official test dataset. Augmentation is
attached to a view of the training indices, never to the shared raw dataset.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from common.data import MNIST_MEAN, MNIST_STD


class DatasetView(Dataset):
    def __init__(self, raw: Dataset, indices: list[int], transform) -> None:
        self.raw = raw
        self.indices = tuple(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, target = self.raw[self.indices[index]]
        return self.transform(image), int(target)


def index_hash(indices) -> str:
    payload = json.dumps(list(indices), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def fixed_split(size: int, val_ratio: float, split_seed: int) -> tuple[list[int], list[int]]:
    val_size = int(size * val_ratio)
    if not 0 < val_size < size:
        raise ValueError("The split must contain nonempty training and validation sets.")
    # Same random_split algorithm and order as common/data.py.
    train, val = random_split(
        range(size), [size - val_size, val_size],
        generator=torch.Generator().manual_seed(split_seed),
    )
    return list(train.indices), list(val.indices)


def seed_worker(_worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def eval_transform():
    return transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])


def train_transform(augmentation: bool, rotation: dict):
    steps = []
    if augmentation:
        interpolation = {
            "bilinear": InterpolationMode.BILINEAR,
            "nearest": InterpolationMode.NEAREST,
        }[rotation["interpolation"]]
        steps.append(transforms.RandomRotation(
            rotation["degrees"], interpolation=interpolation,
            expand=False, fill=rotation["fill"],
        ))
    steps.extend([transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)])
    return transforms.Compose(steps)


def _loader(view: Dataset, batch_size: int, workers: int, seed: int, shuffle: bool):
    return DataLoader(
        view, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )


def _fingerprint(raw: Dataset) -> str:
    """Include image contents and labels, not only the number of examples."""
    digest = hashlib.sha256()
    if not hasattr(raw, "data") or not hasattr(raw, "targets"):
        raise ValueError("The raw dataset must expose MNIST-style data and targets.")
    for value in [raw.data, raw.targets]:
        array = torch.as_tensor(value).cpu().contiguous().numpy()
        digest.update(str((array.shape, array.dtype)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass
class SoftmaxTrainData:
    train_loader: DataLoader
    val_loader: DataLoader
    train_eval_loader: DataLoader
    metadata: dict


def build_softmax_loaders(
    data_dir: str | Path, batch_size: int, val_ratio: float,
    split_seed: int, train_seed: int, augmentation: bool, rotation: dict,
    num_workers: int = 2, download: bool = False, *, raw_train: Dataset | None = None,
) -> SoftmaxTrainData:
    injected = raw_train is not None
    if raw_train is None:
        raw_train = datasets.MNIST(data_dir, train=True, download=download, transform=None)
    if getattr(raw_train, "transform", None) is not None:
        raise ValueError("Pass an untransformed raw training dataset.")
    train_indices, val_indices = fixed_split(len(raw_train), val_ratio, split_seed)
    train = DatasetView(raw_train, train_indices, train_transform(augmentation, rotation))
    val = DatasetView(raw_train, val_indices, eval_transform())
    train_eval = DatasetView(raw_train, train_indices, eval_transform())
    metadata = {
        "dataset": "injected_smoke_dataset" if injected else "MNIST",
        "raw_training_size": len(raw_train), "train_size": len(train), "validation_size": len(val),
        "split_seed": split_seed, "train_seed": train_seed,
        "train_indices_sha256": index_hash(train_indices),
        "validation_indices_sha256": index_hash(val_indices),
        "dataset_sha256": _fingerprint(raw_train),
        "normalization_mean": list(MNIST_MEAN), "normalization_std": list(MNIST_STD),
        "augmentation_train_only": augmentation, "num_workers": num_workers,
    }
    return SoftmaxTrainData(
        _loader(train, batch_size, num_workers, train_seed, True),
        _loader(val, batch_size, num_workers, 0, False),
        _loader(train_eval, batch_size, num_workers, 1, False), metadata,
    )


def build_softmax_test_loader(
    data_dir: str | Path, batch_size: int, num_workers: int = 2,
    download: bool = False, *, raw_test: Dataset | None = None,
):
    injected = raw_test is not None
    if raw_test is None:
        raw_test = datasets.MNIST(data_dir, train=False, download=download, transform=None)
    if getattr(raw_test, "transform", None) is not None:
        raise ValueError("Pass an untransformed raw test dataset.")
    view = DatasetView(raw_test, list(range(len(raw_test))), eval_transform())
    return _loader(view, batch_size, num_workers, 0, False), {
        "dataset": "injected_smoke_dataset" if injected else "MNIST",
        "test_size": len(view), "test_dataset_sha256": _fingerprint(raw_test),
    }
