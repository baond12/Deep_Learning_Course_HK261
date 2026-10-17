from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class A1DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader | None
    class_names: list[str]
    train_size: int
    val_size: int
    test_size: int
    image_size: int
    split_indices: dict[str, list[int]] | None = None


def seed_worker(_worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def reset_loader_seeds(data: A1DataBundle, seed: int) -> None:
    for offset, loader in enumerate((data.train_loader, data.val_loader, data.test_loader)):
        if loader is not None and loader.generator is not None:
            loader.generator.manual_seed(seed + offset)


class RemappedSelection(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        indices: list[int],
        class_to_new: dict[int, int],
        transform=None,
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.class_to_new = class_to_new
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, target = self.dataset[self.indices[index]]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.class_to_new[int(target)]


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return train_transform, eval_transform


def _indices_by_class(targets: list[int], selected_classes: list[int]) -> dict[int, list[int]]:
    selected = set(selected_classes)
    grouped = {class_id: [] for class_id in selected_classes}
    for index, target in enumerate(targets):
        target = int(target)
        if target in selected:
            grouped[target].append(index)
    return grouped


def _take_balanced_split(
    targets: list[int],
    selected_classes: list[int],
    train_per_class: int,
    val_per_class: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    grouped = _indices_by_class(targets, selected_classes)
    train_indices: list[int] = []
    val_indices: list[int] = []
    required = train_per_class + val_per_class

    for class_id in selected_classes:
        class_indices = list(grouped[class_id])
        if len(class_indices) < required:
            raise ValueError(
                f"class {class_id} has {len(class_indices)} train samples, "
                f"but train_per_class + val_per_class = {required}"
            )
        rng = random.Random(seed + class_id)
        rng.shuffle(class_indices)
        train_indices.extend(class_indices[:train_per_class])
        val_indices.extend(class_indices[train_per_class:required])

    random.Random(seed).shuffle(train_indices)
    random.Random(seed + 10_000).shuffle(val_indices)
    return train_indices, val_indices


def _take_balanced_indices(
    targets: list[int],
    selected_classes: list[int],
    per_class: int,
    seed: int,
) -> list[int]:
    grouped = _indices_by_class(targets, selected_classes)
    selected_indices: list[int] = []
    for class_id in selected_classes:
        class_indices = list(grouped[class_id])
        if len(class_indices) < per_class:
            raise ValueError(
                f"class {class_id} has {len(class_indices)} test samples, but per_class = {per_class}"
            )
        rng = random.Random(seed + class_id)
        rng.shuffle(class_indices)
        selected_indices.extend(class_indices[:per_class])
    random.Random(seed + 20_000).shuffle(selected_indices)
    return selected_indices


def build_cifar100_balanced_loaders(
    data_dir: str | Path,
    image_size: int,
    num_classes: int,
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
    batch_size: int,
    seed: int,
    num_workers: int,
    include_test: bool = True,
    train_seed: int | None = None,
) -> A1DataBundle:
    root = Path(data_dir)
    try:
        raw_train = datasets.CIFAR100(root=root, train=True, download=False, transform=None)
        raw_test = datasets.CIFAR100(root=root, train=False, download=False, transform=None) if include_test else None
    except RuntimeError as exc:
        raise RuntimeError(
            "CIFAR-100 was not found under data_dir. Download it once with:\n"
            f"  python A1/prepare_data.py --data-dir {root}\n"
            "Then rerun train.py with the same --data-dir."
        ) from exc

    if not 1 <= num_classes <= len(raw_train.classes):
        raise ValueError(f"num_classes must be between 1 and {len(raw_train.classes)}, got {num_classes}")

    selected_class_ids = list(range(num_classes))
    class_to_new = {class_id: new_id for new_id, class_id in enumerate(selected_class_ids)}
    class_names = [raw_train.classes[class_id] for class_id in selected_class_ids]
    train_indices, val_indices = _take_balanced_split(
        targets=raw_train.targets,
        selected_classes=selected_class_ids,
        train_per_class=train_per_class,
        val_per_class=val_per_class,
        seed=seed,
    )
    test_indices = _take_balanced_indices(
        targets=raw_test.targets,
        selected_classes=selected_class_ids,
        per_class=test_per_class,
        seed=seed,
    ) if raw_test is not None else []

    train_transform, eval_transform = build_transforms(image_size)
    train_set = RemappedSelection(raw_train, train_indices, class_to_new, transform=train_transform)
    val_set = RemappedSelection(raw_train, val_indices, class_to_new, transform=eval_transform)
    test_set = RemappedSelection(raw_test, test_indices, class_to_new, transform=eval_transform) if raw_test is not None else None

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
    }
    loader_seed = seed if train_seed is None else train_seed
    train_loader = DataLoader(train_set, shuffle=True, generator=torch.Generator().manual_seed(loader_seed), **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, generator=torch.Generator().manual_seed(loader_seed + 1), **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, generator=torch.Generator().manual_seed(loader_seed + 2), **loader_kwargs) if test_set is not None else None

    return A1DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=class_names,
        train_size=len(train_set),
        val_size=len(val_set),
        test_size=len(test_set) if test_set is not None else 0,
        image_size=image_size,
        split_indices={"train": train_indices, "validation": val_indices, **({"test": test_indices} if include_test else {})},
    )

