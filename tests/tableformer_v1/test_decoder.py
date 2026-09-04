# SPDX-License-Identifier: Apache-2.0

"""Same-weight differential gates for the TableFormer v1 decoder."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.tableformer_v1.conftest import accurate_config, fast_config

pytestmark = [pytest.mark.mlx, pytest.mark.parity]

mx: Any
torch: Any
reference: Any
native: Any


@pytest.fixture(scope="module", autouse=True)
def _same_weight_models() -> None:
    global mx, torch, reference, native

    try:
        import mlx.core as mlx_core
        import torch as torch_module

        from docling_mlx._models.tableformer_v1.config import TableFormerV1Config
        from docling_mlx._models.tableformer_v1.decoder import TableFormerV1TokenDecoder
        from tools.tableformer_v1.reference_decoder import (
            build_reference_decoder,
            load_reference_decoder,
            load_same_weight_decoder,
            token_decoder_state_dict,
        )
        from tools.tableformer_v1.source import verify_source
    except ImportError as error:
        pytest.fail(f"selected parity lane is missing a required dependency: {error}")

    source_value = os.environ.get("DOCLING_MLX_TABLEFORMER_V1_SOURCE")
    if source_value is None:
        pytest.fail(
            "selected parity lane requires the pinned TableFormer v1 source; "
            "set DOCLING_MLX_TABLEFORMER_V1_SOURCE"
        )
    source = Path(source_value).expanduser()
    try:
        verify_source(source)
    except (OSError, ValueError) as error:
        pytest.fail(f"selected parity lane requires the pinned TableFormer v1 source: {error}")
    checkpoint = source / "model_artifacts/tableformer/accurate/tableformer_accurate.safetensors"
    weights = token_decoder_state_dict(checkpoint)

    torch_module.set_num_threads(1)
    torch_module.use_deterministic_algorithms(True)
    reference = build_reference_decoder()
    load_reference_decoder(weights, reference)
    native = TableFormerV1TokenDecoder(TableFormerV1Config.from_dict(accurate_config()))
    load_same_weight_decoder(weights, native, mlx_core)
    mx = mlx_core
    torch = torch_module


def _assert_close(actual: Any, expected: np.ndarray) -> None:
    actual_array = np.array(actual, copy=True)
    error = np.abs(actual_array - expected)
    assert actual_array.shape == expected.shape
    assert float(error.mean()) <= 1e-4
    assert float(error.max()) <= 1e-3


@pytest.mark.parametrize("seed", [73, 1861])
def test_cached_decoder_matches_pinned_torch_cpu(seed: int) -> None:
    generator = np.random.default_rng(seed)
    memory = generator.normal(0.0, 0.1, (7, 1, 512)).astype(np.float32)
    token_ids = np.array([[2], [4], [9], [5]], dtype=np.int64)
    reference_cache = None
    native_cache = None

    for position in range(token_ids.shape[0]):
        current_ids = token_ids[: position + 1]
        with torch.inference_mode():
            expected_logits, expected_hidden, reference_cache = reference.step(
                torch.from_numpy(current_ids), torch.from_numpy(memory), reference_cache
            )
        actual = native.step(mx.array(current_ids), mx.array(memory), native_cache)
        mx.eval(actual.logits, actual.hidden_state, actual.cache)

        _assert_close(actual.logits, expected_logits.numpy())
        _assert_close(actual.hidden_state, expected_hidden[-1:].numpy())
        _assert_close(actual.cache, reference_cache.numpy())
        assert actual.cache.shape == (6, position + 1, 1, 512)
        if native_cache is not None:
            np.testing.assert_array_equal(
                np.asarray(actual.cache[:, :-1]), np.asarray(native_cache)
            )
        native_cache = actual.cache


def test_uncached_full_input_matches_pinned_torch_cpu() -> None:
    generator = np.random.default_rng(5107)
    memory = generator.normal(0.0, 0.1, (7, 2, 512)).astype(np.float32)
    token_ids = np.array([[2, 2], [4, 5], [9, 9], [5, 4]], dtype=np.int64)
    with torch.inference_mode():
        expected_logits, expected_hidden, expected_cache = reference.step(
            torch.from_numpy(token_ids), torch.from_numpy(memory)
        )
    actual = native.step(mx.array(token_ids), mx.array(memory))
    mx.eval(actual.logits, actual.hidden_state, actual.cache)

    _assert_close(actual.logits, expected_logits.numpy())
    _assert_close(actual.hidden_state, expected_hidden[-1:].numpy())
    _assert_close(actual.cache, expected_cache.numpy())
    assert actual.cache.shape == (6, 1, 2, 512)


def test_sinusoidal_positions_match_the_pinned_source() -> None:
    token_ids = np.array([[2], [4], [9], [5]], dtype=np.int64)
    with torch.inference_mode():
        expected = reference._positional_encoding(reference._embedding(torch.from_numpy(token_ids)))
    actual = native._positional_encoding(native._embedding(mx.array(token_ids)))
    mx.eval(actual)
    np.testing.assert_array_equal(np.asarray(actual), expected.numpy())


def test_decoder_rejects_cache_that_does_not_trail_input() -> None:
    with pytest.raises(ValueError, match="trail"):
        native.step(
            mx.array([[2], [4]], dtype=mx.int32),
            mx.zeros((7, 1, 512), dtype=mx.float32),
            mx.zeros((6, 2, 1, 512), dtype=mx.float32),
        )


def test_fast_decoder_matches_pinned_torch_cpu() -> None:
    from docling_mlx._models.tableformer_v1.config import TableFormerV1Config
    from docling_mlx._models.tableformer_v1.decoder import TableFormerV1TokenDecoder
    from tools.tableformer_v1.reference_decoder import (
        build_reference_decoder,
        load_reference_decoder,
        load_same_weight_decoder,
        token_decoder_state_dict,
    )

    source_value = os.environ.get("DOCLING_MLX_TABLEFORMER_V1_SOURCE")
    if source_value is None:
        pytest.fail(
            "selected parity lane requires the pinned TableFormer v1 source; "
            "set DOCLING_MLX_TABLEFORMER_V1_SOURCE"
        )
    source = Path(source_value).expanduser()
    checkpoint = source / "model_artifacts/tableformer/fast/tableformer_fast.safetensors"
    weights = token_decoder_state_dict(checkpoint)
    fast_reference = build_reference_decoder(num_layers=2)
    load_reference_decoder(weights, fast_reference)
    fast_native = TableFormerV1TokenDecoder(TableFormerV1Config.from_dict(fast_config()))
    load_same_weight_decoder(weights, fast_native, mx)

    generator = np.random.default_rng(1861)
    memory = generator.normal(0.0, 0.1, (7, 1, 512)).astype(np.float32)
    token_ids = np.array([[2], [4]], dtype=np.int64)
    with torch.inference_mode():
        expected_logits, expected_hidden, expected_cache = fast_reference.step(
            torch.from_numpy(token_ids), torch.from_numpy(memory)
        )
    actual = fast_native.step(mx.array(token_ids), mx.array(memory))
    mx.eval(actual.logits, actual.hidden_state, actual.cache)

    _assert_close(actual.logits, expected_logits.numpy())
    _assert_close(actual.hidden_state, expected_hidden[-1:].numpy())
    _assert_close(actual.cache, expected_cache.numpy())
    assert actual.cache.shape == (2, 1, 1, 512)
