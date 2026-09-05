# SPDX-License-Identifier: Apache-2.0

"""Projected attention-cache contracts for native TableFormerV2."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.mlx


def _required_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except (ImportError, ModuleNotFoundError) as error:
        pytest.fail(f"selected MLX qualification requires {name}: {error}")


def _config() -> Any:
    config_type = _required_module("docling_mlx._models.tableformer_v2.config").TableFormerV2Config
    return config_type.from_dict(
        {
            "architectures": ["TableFormerV2"],
            "model_type": "TableFormerV2",
            "embed_dim": 512,
            "num_heads": 8,
            "ff_dim": 2048,
            "num_decoder_layers": 4,
            "vocab_size": 13,
            "conv_mixer_expansion": 1.0,
            "data_cells": [4, 5, 10, 11, 12],
            "pad_token_id": 0,
            "eos_token_id": 3,
            "use_fpn": False,
            "dtype": "float32",
        }
    )


def test_projected_cache_matches_one_shot_projection_and_enforces_capacity() -> None:
    # Projecting one position at a time and all three at once take different matmul
    # paths, so the results may differ by a few float32 ulps; the tolerance allows that.
    mx = _required_module("mlx.core")
    decoder = _required_module("docling_mlx._models.tableformer_v2.decoder")
    attention = decoder.FusedMultiHeadAttention(embed_dim=16, num_heads=4)
    states = mx.arange(48, dtype=mx.float32).reshape(1, 3, 16) / 48.0
    _, expected_keys, expected_values = attention.project_self(states)
    cache = decoder.ProjectedKVCache(
        batch_size=1,
        num_heads=4,
        head_dim=4,
        capacity=3,
        dtype=mx.float32,
    )

    for position in range(3):
        _, current_keys, current_values = attention.project_self(states[:, position : position + 1])
        cached_keys, cached_values = cache.update_and_fetch(current_keys, current_values)
        mx.eval(cached_keys, cached_values, expected_keys, expected_values)
        assert cache.offset == position + 1
        assert cached_keys.shape == (1, 4, position + 1, 4)
        np.testing.assert_allclose(
            np.asarray(cached_keys),
            np.asarray(expected_keys[:, :, : position + 1]),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(cached_values),
            np.asarray(expected_values[:, :, : position + 1]),
            rtol=1e-6,
            atol=1e-6,
        )

    with pytest.raises(ValueError, match="capacity"):
        cache.update_and_fetch(expected_keys[:, :, :1], expected_values[:, :, :1])


def test_projected_cache_rejects_incompatible_updates() -> None:
    mx = _required_module("mlx.core")
    cache_type = _required_module("docling_mlx._models.tableformer_v2.decoder").ProjectedKVCache
    cache = cache_type(
        batch_size=1,
        num_heads=2,
        head_dim=4,
        capacity=4,
        dtype=mx.float32,
    )

    with pytest.raises(TypeError, match="dtype"):
        cache.update_and_fetch(
            mx.zeros((1, 2, 1, 4), dtype=mx.float16),
            mx.zeros((1, 2, 1, 4), dtype=mx.float16),
        )
    with pytest.raises(ValueError, match="shape"):
        cache.update_and_fetch(
            mx.zeros((1, 2, 1, 3), dtype=mx.float32),
            mx.zeros((1, 2, 1, 3), dtype=mx.float32),
        )


def test_decoder_preprojects_one_cross_attention_cache_per_layer() -> None:
    mx = _required_module("mlx.core")
    decoder = _required_module("docling_mlx._models.tableformer_v2.decoder")
    stack = decoder.CachedTransformerDecoder(_config())
    memory = mx.zeros((1, 7, 512), dtype=mx.float32)
    projected = stack.prepare_cross_attention_cache(memory)
    mx.eval(*(array for entry in projected for array in entry))

    assert len(projected) == 4
    for keys, values in projected:
        assert keys.shape == (1, 8, 7, 64)
        assert values.shape == (1, 8, 7, 64)
        assert keys.dtype == mx.float32
        assert values.dtype == mx.float32


def test_decoder_creates_fresh_fixed_capacity_cache_for_each_sequence() -> None:
    mx = _required_module("mlx.core")
    decoder = _required_module("docling_mlx._models.tableformer_v2.decoder")
    stack = decoder.TableFormerV2TokenDecoder(_config())
    memory = mx.zeros((1, 7, 512), dtype=mx.float32)
    cross_cache = stack.prepare_cross_attention_cache(memory)

    first = stack.cached_token_step(
        mx.array([[2]], dtype=mx.int32),
        memory,
        None,
        cross_attention_cache=cross_cache,
    )
    repeated = stack.cached_token_step(
        mx.array([[2]], dtype=mx.int32),
        memory,
        None,
        cross_attention_cache=cross_cache,
    )
    assert first.past_key_values is not None
    assert repeated.past_key_values is not None
    mx.eval(
        *(array for cache in first.past_key_values for array in cache.state),
        *(array for cache in repeated.past_key_values for array in cache.state),
    )

    for original, fresh in zip(first.past_key_values, repeated.past_key_values, strict=True):
        assert original is not fresh
        assert original.capacity == fresh.capacity == 512
        assert original.offset == fresh.offset == 1
        np.testing.assert_allclose(np.asarray(original.state[0]), np.asarray(fresh.state[0]))
        np.testing.assert_allclose(np.asarray(original.state[1]), np.asarray(fresh.state[1]))
    assert (
        sum(cache.keys.nbytes + cache.values.nbytes for cache in first.past_key_values)
        == 8 * 1024 * 1024
    )
