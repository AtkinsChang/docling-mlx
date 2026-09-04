# SPDX-License-Identifier: Apache-2.0

"""Differential contracts for TableFormerV2 fused MLX attention."""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

pytestmark = pytest.mark.mlx


def _required_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except (ImportError, ModuleNotFoundError) as error:
        pytest.fail(f"selected MLX qualification requires {name}: {error}")


def _manual_sdpa(
    mx: Any,
    query: Any,
    key: Any,
    value: Any,
    *,
    scale: float,
    mask: str | Any | None,
) -> Any:
    scores = (query @ key.transpose(0, 1, 3, 2)) * scale
    if isinstance(mask, str):
        assert mask == "causal"
        query_positions = mx.arange(query.shape[2])[:, None]
        key_positions = mx.arange(key.shape[2])[None, :]
        scores = mx.where(
            key_positions > query_positions,
            mx.array(-mx.inf, dtype=scores.dtype),
            scores,
        )
    elif mask is not None:
        scores = scores + mask
    return mx.softmax(scores, axis=-1) @ value


@pytest.mark.parametrize(
    ("query_length", "key_length", "mask"),
    [(1, 37, None), (19, 19, "causal"), (19, 23, None)],
)
def test_decoder_fast_sdpa_matches_manual_attention(
    query_length: int,
    key_length: int,
    mask: str | None,
) -> None:
    mx = _required_module("mlx.core")
    decoder = _required_module("docling_mlx._models.tableformer_v2.decoder")
    mx.random.seed(731)
    attention = decoder.FusedMultiHeadAttention(embed_dim=32, num_heads=4)
    query = mx.random.normal((2, 4, query_length, 8), dtype=mx.float32)
    key = mx.random.normal((2, 4, key_length, 8), dtype=mx.float32)
    value = mx.random.normal((2, 4, key_length, 8), dtype=mx.float32)
    expected = _manual_sdpa(
        mx,
        query,
        key,
        value,
        scale=attention.scale,
        mask=mask,
    )

    original = mx.fast.scaled_dot_product_attention
    with patch.object(mx.fast, "scaled_dot_product_attention", wraps=original) as fast:
        actual = attention._scaled_dot_product_attention(query, key, value, mask)
        mx.eval(actual, expected)

    assert fast.call_count == 1
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-5, rtol=0.0)
