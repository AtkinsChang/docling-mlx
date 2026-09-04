# SPDX-License-Identifier: Apache-2.0

"""Pinned PyTorch CPU helpers for TableFormer v1 token-decoder parity."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np

SOURCE_PREFIX = "_tag_transformer."
TOKEN_DECODER_PREFIXES = (
    "_embedding.",
    "_positional_encoding.",
    "_decoder.",
    "_fc.",
)


def _native_name(source_name: str) -> str:
    if not source_name.startswith(TOKEN_DECODER_PREFIXES):
        raise ValueError(f"Unknown TableFormer v1 decoder tensor: {source_name}")
    return source_name


def token_decoder_state_dict(checkpoint: Path) -> dict[str, np.ndarray]:
    """Extract the exact decoder-owned tensors from a source safetensors file."""

    from safetensors import safe_open

    tensors: dict[str, np.ndarray] = {}
    with safe_open(checkpoint, framework="np") as source:
        for source_name in source.keys():
            if not source_name.startswith(SOURCE_PREFIX):
                continue
            name = source_name.removeprefix(SOURCE_PREFIX)
            if name.startswith(TOKEN_DECODER_PREFIXES):
                tensors[name] = source.get_tensor(source_name).copy()
    return tensors


def build_reference_decoder(num_layers: int = 6) -> Any:
    """Build a pinned source decoder without its encoder or bbox head."""

    import torch.nn as nn
    from docling_ibm_models.tableformer.models.table04_rs.transformer_rs import (
        PositionalEncoding,
        TMTransformerDecoder,
        TMTransformerDecoderLayer,
    )

    class ReferenceDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._embedding = nn.Embedding(13, 512)
            self._positional_encoding = PositionalEncoding(512, max_len=1024)
            self._decoder = TMTransformerDecoder(
                TMTransformerDecoderLayer(
                    d_model=512,
                    nhead=8,
                    dim_feedforward=1024,
                ),
                num_layers=num_layers,
            )
            self._fc = nn.Linear(512, 13)

        def step(self, input_ids: Any, memory: Any, cache: Any = None) -> tuple[Any, Any, Any]:
            embedded = self._positional_encoding(self._embedding(input_ids))
            hidden, next_cache = self._decoder(embedded, memory, cache)
            return self._fc(hidden[-1]), hidden, next_cache

    return ReferenceDecoder().eval()


def load_reference_decoder(source: Mapping[str, np.ndarray], reference: Any) -> None:
    """Strict-load extracted source tensors into the pinned PyTorch decoder."""

    import torch

    reference.load_state_dict({name: torch.from_numpy(value) for name, value in source.items()})
    reference.eval()


def load_same_weight_decoder(
    source: Mapping[str, np.ndarray], native_decoder: Any, mx: Any
) -> None:
    """Strict-load source tensors without key or layout translation."""

    from mlx.utils import tree_flatten

    flattened = cast(list[tuple[str, Any]], tree_flatten(native_decoder.parameters()))
    translated = {_native_name(name): tensor for name, tensor in source.items()}
    target_shapes = {name: tuple(value.shape) for name, value in flattened}
    if set(translated) != set(target_shapes):
        raise ValueError(
            "TableFormer v1 decoder namespace mismatch: "
            f"missing={sorted(set(target_shapes) - set(translated))}, "
            f"extra={sorted(set(translated) - set(target_shapes))}"
        )
    for name, tensor in translated.items():
        if tensor.dtype != np.float32:
            raise ValueError(f"TableFormer v1 decoder tensor is not FP32: {name}")
        if tuple(tensor.shape) != target_shapes[name]:
            raise ValueError(
                f"TableFormer v1 decoder shape mismatch for {name}: "
                f"{tensor.shape} != {target_shapes[name]}"
            )
    native_decoder.load_weights(
        [(name, mx.array(tensor)) for name, tensor in translated.items()], strict=True
    )
    native_decoder.eval()


__all__ = [
    "SOURCE_PREFIX",
    "TOKEN_DECODER_PREFIXES",
    "build_reference_decoder",
    "load_reference_decoder",
    "load_same_weight_decoder",
    "token_decoder_state_dict",
]
