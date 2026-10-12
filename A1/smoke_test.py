from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from A1.models import build_a1_model, count_parameters, parse_experiment_name, smoke_forward


EXPERIMENTS = [
    "resnet18_scratch",
    "resnet18_pretrained_head",
    "resnet18_pretrained_partial",
    "resnet18_pretrained_full",
    "vit_b16_scratch",
    "vit_b16_pretrained_head",
    "vit_b16_pretrained_partial",
    "vit_b16_pretrained_full",
]


def main() -> None:
    for experiment in EXPERIMENTS:
        architecture, mode = parse_experiment_name(experiment)
        model = build_a1_model(
            architecture=architecture,
            mode=mode,
            num_classes=100,
            download_weights=False,
        )
        logits = smoke_forward(model, image_size=224, batch_size=2)
        total, trainable = count_parameters(model)
        if tuple(logits.shape) != (2, 100):
            raise AssertionError(f"{experiment}: expected logits (2, 100), got {tuple(logits.shape)}")
        if mode.startswith("pretrained") and "head" in mode and trainable >= total:
            raise AssertionError(f"{experiment}: head-only freeze did not reduce trainable params")
        print(f"OK {experiment}: logits={tuple(logits.shape)}, trainable={trainable}/{total}")
    print("All A1 smoke tests passed.")


if __name__ == "__main__":
    main()

