# SPDX-License-Identifier: Apache-2.0

"""Pure conversion of upstream TableFormerV2 tensors to MLX layouts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np

_UNUSED_CLASSIFIER_KEYS = frozenset(
    {
        "feature_extractor.classifier.1.bias",
        "feature_extractor.classifier.1.weight",
    }
)
_BBOX_KEY_RENAMES = (
    (re.compile(r"^bbox_head\.bbox_mlp\.(0|3|6)\."), r"bbox_head.bbox_mlp.layers.\1."),
    (re.compile(r"^bbox_head\.layers\.(0|1)\.ffn\.(0|3)\."), r"bbox_head.layers.\1.ffn.layers.\2."),
)


def rename_upstream_key(source_key: str) -> str:
    """Translate the two MLX Sequential namespaces in the bbox head."""

    for pattern, replacement in _BBOX_KEY_RENAMES:
        source_key = pattern.sub(replacement, source_key)
    return source_key


def _ignored_reason(key: str, tensor: np.ndarray, source: Mapping[str, np.ndarray]) -> str | None:
    if key in _UNUSED_CLASSIFIER_KEYS:
        if tensor.dtype != np.float32 or not np.isfinite(tensor).all():
            raise ValueError(f"Ignored classifier tensor must be finite float32: {key}")
        return "unused_torchvision_classifier"
    if not key.endswith(".num_batches_tracked"):
        return None
    stem = key.removesuffix("num_batches_tracked")
    if stem + "running_mean" not in source or stem + "running_var" not in source:
        raise ValueError(f"Unpaired BatchNorm training counter is not ignorable: {key}")
    if tensor.dtype != np.int64 or tensor.shape != ():
        raise ValueError(f"Ignored BatchNorm counter must be an int64 scalar: {key}")
    return "batch_norm_training_counter"


def convert_upstream_state_dict(
    source: Mapping[str, np.ndarray], target_shapes: Mapping[str, tuple[int, ...]]
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, str]]:
    """Convert every upstream inference tensor to its native MLX layout."""

    ignored: dict[str, str] = {}
    renamed: dict[str, tuple[str, np.ndarray]] = {}
    for source_key, tensor in source.items():
        reason = _ignored_reason(source_key, tensor, source)
        if reason is not None:
            ignored[source_key] = reason
            continue
        target_key = rename_upstream_key(source_key)
        if target_key in renamed:
            raise ValueError(
                f"Source keys map to the same target key {target_key}: "
                f"{renamed[target_key][0]}, {source_key}"
            )
        renamed[target_key] = source_key, tensor
    if missing := target_shapes.keys() - renamed.keys():
        raise ValueError(f"Source state missing target keys: {sorted(missing)}")
    if extra := renamed.keys() - target_shapes.keys():
        raise ValueError(
            "Source state has unexpected inference keys: "
            f"{sorted(renamed[key][0] for key in extra)}"
        )
    converted: dict[str, np.ndarray] = {}
    mappings: list[dict[str, Any]] = []
    for target_key, target_shape in sorted(target_shapes.items()):
        source_key, tensor = renamed[target_key]
        if tensor.dtype != np.float32 or not np.isfinite(tensor).all():
            raise ValueError(f"Inference tensor must be finite float32: {source_key}")
        value = tensor.transpose(0, 2, 3, 1) if tensor.ndim == 4 else tensor
        if value.shape != target_shape:
            raise ValueError(
                f"Target shape mismatch for {source_key} -> {target_key}: "
                f"{value.shape} != {target_shape}"
            )
        converted[target_key] = np.ascontiguousarray(value)
        mappings.append(
            {
                "source_key": source_key,
                "target_key": target_key,
                "source_shape": list(tensor.shape),
                "target_shape": list(target_shape),
                "source_dtype": "float32",
                "target_dtype": "float32",
                "transform": "OIHW_to_OHWI" if tensor.ndim == 4 else "identity",
            }
        )
    return converted, mappings, dict(sorted(ignored.items()))


__all__ = ["convert_upstream_state_dict", "rename_upstream_key"]
