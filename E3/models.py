from __future__ import annotations

import torch
from torch import nn

TOKENIZATIONS = ("row", "patch", "cnn_stem", "overlap_patch")
SEQUENCE_EXPERIMENTS = [f"{family}_{kind}" for kind in TOKENIZATIONS for family in ("lstm", "gru")]


def validate_images(x: torch.Tensor) -> None:
    if x.ndim != 4 or tuple(x.shape[1:]) != (1, 28, 28):
        raise ValueError("E3 expects MNIST images shaped (B, 1, 28, 28).")


class RowSequenceTokenizer(nn.Module):
    sequence_length = 28
    feature_dim = 28

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_images(x)
        return x.squeeze(1)


class PatchSequenceTokenizer(nn.Module):
    def __init__(self, patch_size: int = 7, stride: int | None = None) -> None:
        super().__init__()
        stride = patch_size if stride is None else stride
        if not (0 < stride <= patch_size <= 28) or (28 - patch_size) % stride:
            raise ValueError("Patch size/stride must cover the full 28x28 image without a dropped border.")
        self.patch_size = patch_size
        self.stride = stride
        self.sequence_length = ((28 - patch_size) // stride + 1) ** 2
        self.feature_dim = patch_size * patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_images(x)
        patches = x.unfold(2, self.patch_size, self.stride).unfold(
            3, self.patch_size, self.stride
        )
        patches = patches.contiguous().view(x.size(0), 1, -1, self.patch_size, self.patch_size)
        patches = patches.squeeze(1)
        return patches.flatten(2)


class CNNStemSequenceTokenizer(nn.Module):
    sequence_length = 49

    def __init__(self, feature_dim: int = 64) -> None:
        super().__init__()
        if feature_dim < 1:
            raise ValueError("stem_dim must be positive.")
        self.feature_dim = feature_dim
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, feature_dim, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_images(x)
        return self.stem(x).flatten(2).transpose(1, 2)


class ImageSequenceClassifier(nn.Module):
    def __init__(
        self,
        rnn_type: str = "lstm",
        sequence_type: str = "row",
        hidden_dim: int = 128,
        num_layers: int = 1,
        patch_size: int = 7,
        num_classes: int = 10,
        dropout: float = 0.1,
        bidirectional: bool = False,
        overlap_stride: int = 3,
        stem_dim: int = 64,
    ) -> None:
        super().__init__()
        if hidden_dim < 1 or num_layers < 1 or not 0 <= dropout < 1:
            raise ValueError("hidden_dim/num_layers must be positive; dropout must be in [0, 1).")
        if sequence_type == "row":
            self.tokenizer = RowSequenceTokenizer()
        elif sequence_type == "patch":
            self.tokenizer = PatchSequenceTokenizer(patch_size=patch_size)
        elif sequence_type == "overlap_patch":
            if overlap_stride >= patch_size:
                raise ValueError("Overlapping patches require stride < patch_size.")
            self.tokenizer = PatchSequenceTokenizer(patch_size, overlap_stride)
        elif sequence_type == "cnn_stem":
            self.tokenizer = CNNStemSequenceTokenizer(stem_dim)
        else:
            raise ValueError(f"Unknown sequence_type: {sequence_type}")

        rnn_cls: type[nn.LSTM] | type[nn.GRU]
        if rnn_type == "lstm":
            rnn_cls = nn.LSTM
        elif rnn_type == "gru":
            rnn_cls = nn.GRU
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")

        self.rnn_type = rnn_type
        self.sequence_type = sequence_type
        self.bidirectional = bidirectional
        self.rnn = rnn_cls(
            input_size=self.tokenizer.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        direction_factor = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * direction_factor, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = self.tokenizer(x)
        return self.classify_sequence(sequence)

    def classify_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        _output, state = self.rnn(sequence)
        hidden = state[0] if self.rnn_type == "lstm" else state
        # Last layer: concatenate the final states of BOTH directions, rather
        # than the backward output at the last spatial position.
        last = torch.cat((hidden[-2], hidden[-1]), dim=1) if self.bidirectional else hidden[-1]
        return self.head(last)


def build_image_sequence_model(
    experiment: str,
    hidden_dim: int = 128,
    num_layers: int = 1,
    patch_size: int = 7,
    dropout: float = 0.1,
    bidirectional: bool = False,
    overlap_stride: int = 3,
    stem_dim: int = 64,
) -> ImageSequenceClassifier:
    configs = {f"{family}_{kind}": {"rnn_type": family, "sequence_type": kind}
               for kind in TOKENIZATIONS for family in ("lstm", "gru")}
    if experiment not in configs:
        raise ValueError(f"Unknown image sequence experiment: {experiment}")

    return ImageSequenceClassifier(
        **configs[experiment],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        patch_size=patch_size,
        dropout=dropout,
        bidirectional=bidirectional,
        overlap_stride=overlap_stride,
        stem_dim=stem_dim,
    )


def sequence_metadata(model: ImageSequenceClassifier) -> dict:
    tokenizer = model.tokenizer
    info = {"kind": model.sequence_type, "sequence_length": tokenizer.sequence_length,
            "feature_dim": tokenizer.feature_dim, "uses_cls": False,
            "scan_order": "top-to-bottom rows; left-to-right then top-to-bottom spatial grid",
            "readout": "last-layer final hidden state(s)",
            "bidirectional": model.bidirectional}
    if isinstance(tokenizer, PatchSequenceTokenizer):
        info.update(patch_size=tokenizer.patch_size, stride=tokenizer.stride, padding=0)
    if isinstance(tokenizer, CNNStemSequenceTokenizer):
        info["stem"] = "Conv(1,32,3,pad1)-ReLU-Pool2-Conv(32,D,3,pad1)-ReLU-Pool2"
    return info

