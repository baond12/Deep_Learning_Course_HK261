from __future__ import annotations

import argparse
from pathlib import Path

from torchvision import datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CIFAR-100 once for A1 experiments.")
    parser.add_argument("--data-dir", type=str, default="data", help="Local/Colab path used by train.py.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)

    train_set = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=None)
    test_set = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=None)

    print(f"CIFAR-100 is ready under: {data_dir.resolve()}")
    print(f"classes: {len(train_set.classes)}")
    print(f"raw train samples: {len(train_set)}")
    print(f"raw test samples: {len(test_set)}")
    print("Use the same path with train.py, for example:")
    print(f"  python A1/train.py --data-dir {data_dir} --experiments resnet18_scratch")


if __name__ == "__main__":
    main()
