# SPDX-License-Identifier: Apache-2.0

"""Same-weight Torch CPU helpers for TableFormer v1 encoder parity."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np

ENCODER_SOURCE_PREFIXES = (
    "_encoder._resnet.",
    "_tag_transformer._input_filter.",
    "_tag_transformer._encoder.",
)


def encoder_state_dict(model: Any) -> dict[str, np.ndarray]:
    return {
        name: tensor.detach().cpu().numpy().copy()
        for name, tensor in model.state_dict().items()
        if name.startswith(ENCODER_SOURCE_PREFIXES) and not name.endswith(".num_batches_tracked")
    }


def convert_encoder_state_dict(
    source: Mapping[str, np.ndarray], target_shapes: Mapping[str, tuple[int, ...]]
) -> dict[str, np.ndarray]:
    if set(source) != set(target_shapes):
        raise ValueError(
            "TableFormer v1 encoder namespace mismatch: "
            f"missing={sorted(set(target_shapes) - set(source))}, "
            f"extra={sorted(set(source) - set(target_shapes))}"
        )
    converted = {}
    for name, tensor in source.items():
        if tensor.dtype != np.float32 or not np.isfinite(tensor).all():
            raise ValueError(f"TableFormer v1 encoder tensor must be finite FP32: {name}")
        value = (
            np.ascontiguousarray(tensor.transpose(0, 2, 3, 1))
            if tensor.ndim == 4 and name.endswith(".weight")
            else np.ascontiguousarray(tensor)
        )
        if value.shape != target_shapes[name]:
            raise ValueError(
                f"TableFormer v1 encoder shape mismatch for {name}: "
                f"{value.shape} != {target_shapes[name]}"
            )
        converted[name] = value
    return converted


def load_same_weight_encoder(
    source: Mapping[str, np.ndarray], native_encoder: Any, mx: Any
) -> None:
    from mlx.utils import tree_flatten

    flattened = cast(list[tuple[str, Any]], tree_flatten(native_encoder.parameters()))
    target_shapes = {name: tuple(value.shape) for name, value in flattened}
    converted = convert_encoder_state_dict(source, target_shapes)
    native_encoder.load_weights(
        [(name, mx.array(value)) for name, value in converted.items()], strict=True
    )
    native_encoder.eval()


__all__ = [
    "ENCODER_SOURCE_PREFIXES",
    "convert_encoder_state_dict",
    "encoder_state_dict",
    "load_same_weight_encoder",
]
