# SPDX-License-Identifier: Apache-2.0

"""Same-weight Torch CPU helpers for TableFormerV2 vision-encoder parity."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np

ENCODER_SOURCE_PREFIXES = (
    "feature_extractor.features.",
    "se_module.",
    "conv_mixer.",
    "feature_to_embedding.",
)


def encoder_state_dict(model: Any) -> dict[str, np.ndarray]:
    """Extract exactly the state tensors consumed by the native vision encoder."""

    return {
        name: tensor.detach().cpu().numpy().copy()
        for name, tensor in model.state_dict().items()
        if name.startswith(ENCODER_SOURCE_PREFIXES) and not name.endswith(".num_batches_tracked")
    }


def load_same_weight_encoder(
    source: Mapping[str, np.ndarray], native_encoder: Any, mx: Any
) -> None:
    """Strict-load Torchvision OIHW source tensors into the native NHWC encoder."""

    from mlx.utils import tree_flatten

    flattened = cast(list[tuple[str, Any]], tree_flatten(native_encoder.parameters()))
    target_shapes = {name: tuple(value.shape) for name, value in flattened}
    if set(source) != set(target_shapes):
        raise ValueError(
            "TableFormerV2 encoder namespace mismatch: "
            f"missing={sorted(set(target_shapes) - set(source))}, "
            f"extra={sorted(set(source) - set(target_shapes))}"
        )
    converted: list[tuple[str, Any]] = []
    for name, tensor in source.items():
        if tensor.dtype != np.float32:
            raise ValueError(f"TableFormerV2 encoder tensor is not FP32: {name}")
        if tensor.ndim == 4 and name.endswith(".weight"):
            tensor = np.ascontiguousarray(tensor.transpose(0, 2, 3, 1))
        if tuple(tensor.shape) != target_shapes[name]:
            raise ValueError(
                f"TableFormerV2 encoder shape mismatch for {name}: "
                f"{tensor.shape} != {target_shapes[name]}"
            )
        converted.append((name, mx.array(tensor)))
    native_encoder.load_weights(converted, strict=True)
    native_encoder.eval()


__all__ = ["ENCODER_SOURCE_PREFIXES", "encoder_state_dict", "load_same_weight_encoder"]
