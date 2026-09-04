# SPDX-License-Identifier: Apache-2.0

"""Same-weight PyTorch CPU helpers for TableFormerV2 token-decoder parity."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np

TOKEN_DECODER_SOURCE_PREFIXES = (
    "input_embedding.",
    "positional_encoding",
    "transformer_decoder.",
    "output_projection.",
)


def token_decoder_state_dict(model: Any) -> dict[str, np.ndarray]:
    """Extract exactly the source state tensors owned by the native decoder."""

    return {
        name: tensor.detach().cpu().numpy().copy()
        for name, tensor in model.state_dict().items()
        if name == "positional_encoding" or name.startswith(TOKEN_DECODER_SOURCE_PREFIXES)
    }


def load_same_weight_decoder(
    source: Mapping[str, np.ndarray], native_decoder: Any, mx: Any
) -> None:
    """Strict-load source decoder tensors without key or layout translation."""

    from mlx.utils import tree_flatten

    flattened = cast(list[tuple[str, Any]], tree_flatten(native_decoder.parameters()))
    target_shapes = {name: tuple(value.shape) for name, value in flattened}
    if set(source) != set(target_shapes):
        raise ValueError(
            "TableFormerV2 decoder namespace mismatch: "
            f"missing={sorted(set(target_shapes) - set(source))}, "
            f"extra={sorted(set(source) - set(target_shapes))}"
        )
    for name, tensor in source.items():
        if tensor.dtype != np.float32:
            raise ValueError(f"TableFormerV2 decoder tensor is not FP32: {name}")
        if tuple(tensor.shape) != target_shapes[name]:
            raise ValueError(
                f"TableFormerV2 decoder shape mismatch for {name}: "
                f"{tensor.shape} != {target_shapes[name]}"
            )
    native_decoder.load_weights(
        [(name, mx.array(tensor)) for name, tensor in source.items()], strict=True
    )
    native_decoder.eval()


__all__ = ["TOKEN_DECODER_SOURCE_PREFIXES", "load_same_weight_decoder", "token_decoder_state_dict"]
