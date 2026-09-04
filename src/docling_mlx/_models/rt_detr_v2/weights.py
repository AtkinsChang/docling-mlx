# SPDX-License-Identifier: Apache-2.0

"""Torch-safetensor to MLX-state conversion shared by runtime and converter."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np

_HF_PREFIX = "model."
_RENAMES = (
    (r"^backbone\.model\.", "vision.backbone."),
    (r"\.shortcut\.1\.", ".shortcut.proj."),
    (r"\.convolution\.", ".conv."),
    (r"\.normalization\.", ".bn."),
    (r"^encoder\.encoder\.", "vision.hybrid_encoder.aifi."),
    (r"^encoder_input_proj\.(\d+)\.0\.", r"vision.encoder_input_proj.\1.conv."),
    (r"^encoder_input_proj\.(\d+)\.1\.", r"vision.encoder_input_proj.\1.bn."),
    (r"^encoder\.", "vision.hybrid_encoder."),
    (r"\.norm\.", ".bn."),
    (r"^decoder_input_proj\.(\d+)\.0\.", r"decoder_input_proj.\1.conv."),
    (r"^decoder_input_proj\.(\d+)\.1\.", r"decoder_input_proj.\1.bn."),
    (r"^enc_output\.0\.", "enc_output.fc."),
    (r"^enc_output\.1\.", "enc_output.ln."),
)


def rename_key(source_key: str) -> str:
    """Translate a Hugging Face RT-DETR-v2 key into the MLX namespace."""

    target_key = source_key.removeprefix(_HF_PREFIX)
    for pattern, replacement in _RENAMES:
        target_key = re.sub(pattern, replacement, target_key)
    return target_key


def is_hf_checkpoint(state: Mapping[str, np.ndarray]) -> bool:
    """Recognize native Hugging Face state names without depending on Torch."""

    return any(
        key.startswith(
            ("model.backbone.", "model.decoder.", "backbone.model.", "encoder_input_proj.")
        )
        for key in state
    )


def sanitize_state_dict(
    source: Mapping[str, np.ndarray], target_shapes: Mapping[str, tuple[int, ...]]
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[str]]:
    """Rename, transpose convolutions, and validate one HF inference state dict."""

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
        is_convolution = tensor.ndim == 4 and target_key.endswith(".conv.weight")
        converted_tensor = tensor.transpose(0, 2, 3, 1) if is_convolution else tensor
        if tuple(converted_tensor.shape) != target_shapes[target_key]:
            raise ValueError(
                f"Target shape mismatch for {target_key}: {tuple(converted_tensor.shape)} "
                f"!= {target_shapes[target_key]}"
            )
        converted[target_key] = np.ascontiguousarray(converted_tensor)
        mappings.append(
            {
                "source_key": source_key,
                "target_key": target_key,
                "source_shape": list(tensor.shape),
                "target_shape": list(converted_tensor.shape),
                "source_dtype": str(tensor.dtype),
                "target_dtype": str(converted_tensor.dtype),
                "transform": "OIHW_to_OHWI" if is_convolution else "identity",
            }
        )
    return converted, mappings, sorted(ignored)


def load_state_dict(
    path: str,
    target_shapes: Mapping[str, tuple[int, ...]],
) -> tuple[dict[str, np.ndarray], bool]:
    """Load either an MLX-layout or HF PyTorch safetensor checkpoint."""

    from safetensors.numpy import load_file

    state = load_file(path)
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
    return {key: np.ascontiguousarray(value) for key, value in state.items()}, False
