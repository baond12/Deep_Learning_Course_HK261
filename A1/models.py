from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights, resnet18, vit_b_16


def set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def set_a1_train_mode(model: nn.Module) -> None:
    """Freeze both parameters and running statistics outside the opened stages."""
    mode = model._a1_mode
    architecture = model._a1_architecture
    if mode in {"scratch", "pretrained_full"}:
        model.train()
        return
    model.eval()
    if architecture == "resnet18":
        if mode == "pretrained_partial":
            model.layer4.train()
        model.fc.train()
    else:
        if mode == "pretrained_partial":
            for layer in list(model.encoder.layers.children())[-2:]:
                layer.train()
        model.heads.train()


def _replace_resnet_head(model: nn.Module, num_classes: int) -> None:
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)


def _replace_vit_head(model: nn.Module, num_classes: int) -> None:
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)


def _configure_resnet_trainability(model: nn.Module, mode: str) -> None:
    if mode in {"scratch", "pretrained_full"}:
        set_trainable(model, True)
    elif mode == "pretrained_head":
        set_trainable(model, False)
        set_trainable(model.fc, True)
    elif mode == "pretrained_partial":
        set_trainable(model, False)
        set_trainable(model.layer4, True)
        set_trainable(model.fc, True)
    else:
        raise ValueError(f"Unknown ResNet mode: {mode}")


def _configure_vit_trainability(model: nn.Module, mode: str, partial_layers: int = 2) -> None:
    if mode in {"scratch", "pretrained_full"}:
        set_trainable(model, True)
    elif mode == "pretrained_head":
        set_trainable(model, False)
        set_trainable(model.heads, True)
    elif mode == "pretrained_partial":
        set_trainable(model, False)
        layers = list(model.encoder.layers.children())
        for layer in layers[-partial_layers:]:
            set_trainable(layer, True)
        set_trainable(model.heads, True)
    else:
        raise ValueError(f"Unknown ViT mode: {mode}")


def build_a1_model(
    architecture: str,
    mode: str,
    num_classes: int,
    download_weights: bool = True,
) -> nn.Module:
    pretrained = mode.startswith("pretrained")

    if architecture == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained and download_weights else None
        model = resnet18(weights=weights)
        _replace_resnet_head(model, num_classes)
        _configure_resnet_trainability(model, mode)
        model._a1_architecture = architecture
        model._a1_mode = mode
        model._a1_weights = weights.name if weights is not None else None
        return model

    if architecture == "vit_b_16":
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained and download_weights else None
        model = vit_b_16(weights=weights)
        _replace_vit_head(model, num_classes)
        _configure_vit_trainability(model, mode)
        model._a1_architecture = architecture
        model._a1_mode = mode
        model._a1_weights = weights.name if weights is not None else None
        return model

    raise ValueError(f"Unknown architecture: {architecture}")


def parse_experiment_name(experiment: str) -> tuple[str, str]:
    mapping = {
        "resnet18_scratch": ("resnet18", "scratch"),
        "resnet18_pretrained_head": ("resnet18", "pretrained_head"),
        "resnet18_pretrained_partial": ("resnet18", "pretrained_partial"),
        "resnet18_pretrained_full": ("resnet18", "pretrained_full"),
        "vit_b16_scratch": ("vit_b_16", "scratch"),
        "vit_b16_pretrained_head": ("vit_b_16", "pretrained_head"),
        "vit_b16_pretrained_partial": ("vit_b_16", "pretrained_partial"),
        "vit_b16_pretrained_full": ("vit_b_16", "pretrained_full"),
    }
    if experiment not in mapping:
        raise ValueError(f"Unknown experiment: {experiment}")
    return mapping[experiment]


@torch.no_grad()
def smoke_forward(model: nn.Module, image_size: int, batch_size: int = 2) -> torch.Tensor:
    model.eval()
    x = torch.randn(batch_size, 3, image_size, image_size)
    return model(x)

