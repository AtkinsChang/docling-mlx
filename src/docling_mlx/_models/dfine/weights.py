# SPDX-License-Identifier: Apache-2.0

"""Torch-safetensor to MLX-state conversion shared by runtime and converter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

_DENOISING_KEY = "model.denoising_class_embed.weight"


def rename_key(source_key: str) -> str:
    """Translate a Hugging Face D-FINE key into the MLX composition namespace."""

    if not source_key.startswith("model."):
        raise ValueError(f"Unexpected D-FINE source key: {source_key}")
    key = source_key.removeprefix("model.")
    if key.startswith("backbone.model."):
        return "vision.backbone." + key.removeprefix("backbone.model.")
    if key.startswith("encoder.encoder."):
        return "vision.encoder.aifi." + key.removeprefix("encoder.encoder.").replace(
            ".fc1.", ".mlp.layers.0."
        ).replace(".fc2.", ".mlp.layers.1.").replace(".self_attn.out_proj.", ".self_attn.o_proj.")
    if key.startswith("encoder."):
        return "vision." + key
    if key.startswith("encoder_input_proj."):
        return _rename_projection(key, "encoder_input_proj", "vision.encoder_input_proj")
    if key.startswith("decoder_input_proj."):
        return _rename_projection(key, "decoder_input_proj", "decoder_input_proj")
    if key.startswith("enc_output.0."):
        return "enc_output.fc." + key.removeprefix("enc_output.0.")
    if key.startswith("enc_output.1."):
        return "enc_output.norm." + key.removeprefix("enc_output.1.")
    if key.startswith("decoder.layers."):
        return (
            key.replace(".self_attn.o_proj.", ".self_attn.out_proj.")
            .replace(".mlp.layers.0.", ".fc1.")
            .replace(".mlp.layers.1.", ".fc2.")
        )
    return key


def _rename_projection(key: str, prefix: str, target_prefix: str) -> str:
    parts = key.split(".", 3)
    if len(parts) != 4 or parts[0] != prefix or parts[2] not in {"0", "1"}:
        raise ValueError(f"Unexpected D-FINE projection key: model.{key}")
    name = "conv" if parts[2] == "0" else "norm"
    return f"{target_prefix}.{parts[1]}.{name}.{parts[3]}"


def is_hf_checkpoint(state: Mapping[str, np.ndarray]) -> bool:
    """Recognize Hugging Face state names without importing Torch."""

    return any(key.startswith("model.") for key in state)


def _ignored_reason(
    source_key: str,
    tensor: np.ndarray,
    source: Mapping[str, np.ndarray],
    target_shapes: Mapping[str, tuple[int, ...]],
) -> str | None:
    if source_key == _DENOISING_KEY:
        if not np.issubdtype(tensor.dtype, np.floating) or not np.isfinite(tensor).all():
            raise ValueError(
                f"Ignored denoising embedding must be finite floating-point data: {source_key}"
            )
        return "eval_unreachable_denoising_embedding"
    if not source_key.endswith(".num_batches_tracked"):
        return None
    target_key = rename_key(source_key)
    source_stem = source_key.removesuffix("num_batches_tracked")
    target_stem = target_key.removesuffix("num_batches_tracked")
    if (
        source_stem + "running_mean" not in source
        or source_stem + "running_var" not in source
        or target_stem + "running_mean" not in target_shapes
        or target_stem + "running_var" not in target_shapes
    ):
        raise ValueError(f"Unpaired BatchNorm training counter is not ignorable: {source_key}")
    if tensor.dtype != np.int64 or tensor.shape != ():
        raise ValueError(f"Ignored BatchNorm counter must be an int64 scalar: {source_key}")
    return "batch_norm_training_counter"


def sanitize_state_dict(
    source: Mapping[str, np.ndarray], target_shapes: Mapping[str, tuple[int, ...]]
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, str]]:
    """Rename, transpose, and strictly validate one HF D-FINE inference state."""

    renamed: dict[str, tuple[str, np.ndarray]] = {}
    ignored: dict[str, str] = {}
    for source_key, tensor in source.items():
        reason = _ignored_reason(source_key, tensor, source, target_shapes)
        if reason is not None:
            ignored[source_key] = reason
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
        is_convolution = tensor.ndim == 4 and target_key.endswith(".weight")
        transformed = tensor.transpose(0, 2, 3, 1) if is_convolution else tensor
        if tuple(transformed.shape) != target_shapes[target_key]:
            raise ValueError(
                f"Target shape mismatch for {source_key} -> {target_key}: "
                f"{tuple(transformed.shape)} != {target_shapes[target_key]}"
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
                "transform": "OIHW_to_OHWI" if is_convolution else "identity",
            }
        )
    return converted, mappings, dict(sorted(ignored.items()))


def load_state_dict(
    path: str,
    target_shapes: Mapping[str, tuple[int, ...]],
) -> tuple[dict[str, np.ndarray], bool]:
    """Load either an MLX-layout or upstream HF PyTorch safetensor checkpoint."""

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
    if any(
        not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all()
        for value in state.values()
    ):
        raise ValueError("MLX state must contain finite floating-point tensors")
    return {key: np.ascontiguousarray(value) for key, value in state.items()}, False
