from __future__ import annotations

import math

import torch
from torch import nn


TOKENIZATIONS = ("patch", "row", "cnn_stem", "overlap_patch")
EXPERIMENT_CONFIGS = {
    f"{attention}_{tokenizer}": {"tokenizer_type": tokenizer, "attention_type": attention}
    for tokenizer in TOKENIZATIONS for attention in ("custom", "pytorch")
}


class PatchTokenizer(nn.Module):
    def __init__(self, patch_size: int = 7, embed_dim: int = 64, stride: int | None = None) -> None:
        super().__init__()
        stride = patch_size if stride is None else stride
        if not (0 < stride <= patch_size <= 28) or (28 - patch_size) % stride:
            raise ValueError("Patch size/stride must cover the full 28x28 image without a dropped border.")
        self.patch_size = patch_size
        self.stride = stride
        self.grid_size = (28 - patch_size) // stride + 1
        self.num_tokens = self.grid_size**2
        self.projection = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.projection(x)
        return tokens.flatten(2).transpose(1, 2)


class RowTokenizer(nn.Module):
    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        self.num_tokens = 28
        self.projection = nn.Linear(28, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.squeeze(1)
        return self.projection(rows)


class OverlapPatchTokenizer(PatchTokenizer):
    def __init__(self, patch_size: int = 7, embed_dim: int = 64, stride: int = 3) -> None:
        if stride >= patch_size:
            raise ValueError("Overlapping patches require stride < patch_size.")
        super().__init__(patch_size, embed_dim, stride)


class CNNStemTokenizer(nn.Module):
    """Two conv/pooling blocks; each location in the 7x7 feature map is a token."""
    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        self.grid_size = 7
        self.num_tokens = self.grid_size**2
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, embed_dim, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x).flatten(2).transpose(1, 2)


class CustomMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int = 64, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, embed_dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, token_count, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = torch.softmax(scores, dim=-1)
        attention = self.attn_drop(attention)

        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().reshape(batch_size, token_count, embed_dim)
        # Residual/output dropout lives in TransformerEncoderBlock.drop1 for BOTH backends.
        return self.out_proj(context)


class TorchMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int = 64, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _weights = self.attention(x, x, x, need_weights=False)
        return output


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        ffn_dim: int = 128,
        attention_type: str = "custom",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if attention_type == "custom":
            attention = CustomMultiHeadSelfAttention(embed_dim, num_heads, dropout)
        elif attention_type == "pytorch":
            attention = TorchMultiHeadSelfAttention(embed_dim, num_heads, dropout)
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = attention
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.attention(self.norm1(x)))
        x = x + self.ffn(self.norm2(x))
        return x


class ImageTransformerClassifier(nn.Module):
    def __init__(
        self,
        tokenizer_type: str = "patch",
        attention_type: str = "custom",
        embed_dim: int = 64,
        num_heads: int = 4,
        ffn_dim: int = 128,
        depth: int = 1,
        patch_size: int = 7,
        num_classes: int = 10,
        dropout: float = 0.1,
        overlap_stride: int = 3,
    ) -> None:
        super().__init__()
        if depth < 1 or embed_dim < 1 or ffn_dim < 1 or num_heads < 1 or embed_dim % num_heads:
            raise ValueError("Positive dimensions/depth and embed_dim divisible by num_heads are required.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        self.attention_type = attention_type
        self.model_config = {"tokenizer_type": tokenizer_type, "embed_dim": embed_dim,
                             "num_heads": num_heads, "ffn_dim": ffn_dim, "depth": depth,
                             "patch_size": patch_size, "num_classes": num_classes,
                             "dropout": dropout, "overlap_stride": overlap_stride}
        if tokenizer_type == "patch":
            self.tokenizer = PatchTokenizer(patch_size=patch_size, embed_dim=embed_dim)
        elif tokenizer_type == "row":
            self.tokenizer = RowTokenizer(embed_dim=embed_dim)
        elif tokenizer_type == "cnn_stem":
            self.tokenizer = CNNStemTokenizer(embed_dim=embed_dim)
        elif tokenizer_type == "overlap_patch":
            self.tokenizer = OverlapPatchTokenizer(patch_size, embed_dim, overlap_stride)
        else:
            raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}")

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.tokenizer.num_tokens + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    attention_type=attention_type,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(module.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 28, 28):
            raise ValueError("E2 expects MNIST images shaped (B, 1, 28, 28).")
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos_drop(tokens + self.pos_embedding)

        for block in self.blocks:
            tokens = block(tokens)

        cls_representation = tokens[:, 0]
        return self.head(cls_representation)


def build_model(
    experiment: str,
    embed_dim: int = 64,
    num_heads: int = 4,
    ffn_dim: int = 128,
    depth: int = 1,
    patch_size: int = 7,
    dropout: float = 0.1,
    overlap_stride: int = 3,
) -> ImageTransformerClassifier:
    if experiment not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment: {experiment}")

    return ImageTransformerClassifier(
        **EXPERIMENT_CONFIGS[experiment],
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        depth=depth,
        patch_size=patch_size,
        dropout=dropout,
        overlap_stride=overlap_stride,
    )


def copy_custom_to_pytorch(source: ImageTransformerClassifier, target: ImageTransformerClassifier) -> None:
    """Copy the entire classifier, including tokenizer, CLS, FFN and attention weights."""
    if source.attention_type != "custom" or target.attention_type != "pytorch":
        raise ValueError("Expected custom source and PyTorch target.")
    if source.model_config != target.model_config:
        raise ValueError("Paired classifiers must have identical architecture and dropout settings.")
    mapped = dict(source.state_dict())
    for index in range(len(source.blocks)):
        prefix = f"blocks.{index}.attention."
        for original, replacement in (
            ("qkv.weight", "attention.in_proj_weight"), ("qkv.bias", "attention.in_proj_bias"),
            ("out_proj.weight", "attention.out_proj.weight"), ("out_proj.bias", "attention.out_proj.bias"),
        ):
            mapped[prefix + replacement] = mapped.pop(prefix + original)
    target.load_state_dict(mapped, strict=True)


def tokenization_metadata(model: ImageTransformerClassifier) -> dict:
    config, tokenizer = model.model_config, model.tokenizer
    kind = config["tokenizer_type"]
    metadata = {"kind": kind, "image_tokens": tokenizer.num_tokens,
                "total_tokens": tokenizer.num_tokens + 1, "embed_dim": config["embed_dim"],
                "uses_cls": True}
    if isinstance(tokenizer, PatchTokenizer):
        metadata.update(patch_size=tokenizer.patch_size, stride=tokenizer.stride,
                        grid_size=tokenizer.grid_size, padding=0)
    if kind == "cnn_stem":
        metadata.update(grid_size=7, stem="Conv(1,32,3,pad1)-ReLU-Pool2-Conv(32,D,3,pad1)-ReLU-Pool2")
    if kind == "row":
        metadata.update(row_length=28)
    return metadata

