# SPDX-License-Identifier: Apache-2.0

"""Checkpoint sanitizing shared by EfficientNet runtime and converter."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np


def rename_key(source_key: str) -> str:
    """Map common wrapper prefixes to the native EfficientNet namespace."""

    return re.sub(r"^(?:module\.|model\.)", "", source_key)


def is_hf_checkpoint(state: Mapping[str, np.ndarray]) -> bool:
    """Recognize a PyTorch-style checkpoint without importing PyTorch."""

    return any(
        key.endswith("num_batches_tracked") or key.startswith(("module.", "model."))
        for key in state
    )


def sanitize_state_dict(
    source: Mapping[str, np.ndarray], target_shapes: Mapping[str, tuple[int, ...]]
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[str]]:
    """Rename, transpose convolution kernels, and drop BatchNorm counters."""

    renamed: dict[str, tuple[str, np.ndarray]] = {}
    ignored: list[str] = []
    for source_key, tensor in source.items():
        if source_key.endswith("num_batches_tracked"):
            ignored.append(source_key)
            continue
        target_key = rename_key(source_key)
        if target_key in renamed:
            previous, _ = renamed[target_key]
            raise ValueError(f"Source keys map to {target_key}: {previous}, {source_key}")
        renamed[target_key] = (source_key, tensor)

    missing = set(target_shapes) - set(renamed)
    extra = set(renamed) - set(target_shapes)
    if missing:
        raise ValueError(f"Source state missing target keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"Source state has unexpected target keys: {sorted(extra)}")

    converted: dict[str, np.ndarray] = {}
    mappings: list[dict[str, Any]] = []
    for target_key in sorted(target_shapes):
        source_key, tensor = renamed[target_key]
        if not np.issubdtype(tensor.dtype, np.floating) or not np.isfinite(tensor).all():
            raise ValueError(f"Inference tensor must be finite floating-point data: {source_key}")
        transformed = tensor.transpose(0, 2, 3, 1) if tensor.ndim == 4 else tensor
        if tuple(transformed.shape) != target_shapes[target_key]:
            raise ValueError(
                f"Target shape mismatch for {target_key}: {tuple(transformed.shape)} "
                f"!= {target_shapes[target_key]}"
            )
        converted[target_key] = np.ascontiguousarray(transformed)
        mappings.append(
            {
                "source_key": source_key,
                "target_key": target_key,
                "source_shape": list(tensor.shape),
                "target_shape": list(transformed.shape),
                "source_dtype": str(tensor.dtype),
                "target_dtype": str(transformed.dtype),
                "transform": "OIHW_to_OHWI" if tensor.ndim == 4 else "identity",
            }
        )
    return converted, mappings, sorted(ignored)


def load_state_dict(
    path: str,
    target_shapes: Mapping[str, tuple[int, ...]],
) -> tuple[dict[str, np.ndarray], bool]:
    """Load either an MLX-layout or upstream HF safetensors checkpoint."""

    from safetensors.numpy import load_file

    try:
        state = load_file(path)
    except Exception as error:
        raise ValueError(f"Unable to load EfficientNet safetensors: {path}") from error
    if is_hf_checkpoint(state):
        converted, _, _ = sanitize_state_dict(state, target_shapes)
        return converted, True
    missing = set(target_shapes) - set(state)
    extra = set(state) - set(target_shapes)
    if missing or extra:
        raise ValueError(
            "MLX state keys do not strictly match model "
            f"(missing={sorted(missing)}, extra={sorted(extra)})"
        )
    converted = {}
    for key, value in state.items():
        if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
            raise ValueError(f"Inference tensor must be finite floating-point data: {key}")
        if tuple(value.shape) != target_shapes[key]:
            raise ValueError(
                f"Target shape mismatch for {key}: {tuple(value.shape)} != {target_shapes[key]}"
            )
        converted[key] = np.ascontiguousarray(value)
    return converted, False
