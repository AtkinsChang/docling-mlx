# SPDX-License-Identifier: Apache-2.0

"""Same-weight differential gates for the TableFormerV2 token decoder."""

from __future__ import annotations

import subprocess
import sys
from importlib import import_module
from typing import Any

import numpy as np
import pytest

from docling_mlx._models.tableformer_v2.config import TableFormerV2Config

pytestmark = [pytest.mark.mlx, pytest.mark.parity]

mx: Any
torch: Any
TorchTableFormerV2: Any
TorchTableFormerV2Config: Any
TableFormerV2TokenDecoder: Any
load_same_weight_decoder: Any
token_decoder_state_dict: Any


@pytest.fixture(scope="module", autouse=True)
def _load_requirements() -> None:
    global mx, torch, TorchTableFormerV2, TorchTableFormerV2Config
    global TableFormerV2TokenDecoder, load_same_weight_decoder, token_decoder_state_dict

    probe = subprocess.run(
        [sys.executable, "-c", "import mlx.core"],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode:
        pytest.fail(f"selected parity lane requires Metal: {probe.stderr.strip()}")
    try:
        mx = import_module("mlx.core")
        torch = import_module("torch")
        source = import_module("docling_ibm_models.tableformer_v2.model")
        native = import_module("docling_mlx._models.tableformer_v2.decoder")
        helpers = import_module("tools.tableformer_v2.reference_decoder")
    except ImportError as error:
        pytest.fail(f"selected parity lane is missing a required dependency: {error}")
    TorchTableFormerV2 = source.TableFormerV2
    TorchTableFormerV2Config = source.TableFormerV2Config
    TableFormerV2TokenDecoder = native.TableFormerV2TokenDecoder
    load_same_weight_decoder = helpers.load_same_weight_decoder
    token_decoder_state_dict = helpers.token_decoder_state_dict


def _native_config() -> TableFormerV2Config:
    return TableFormerV2Config.from_dict(
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


def _same_weight_models() -> tuple[Any, Any]:
    torch.manual_seed(5107)
    reference = TorchTableFormerV2(
        TorchTableFormerV2Config(
            embed_dim=512,
            num_heads=8,
            ff_dim=2048,
            num_decoder_layers=4,
            vocab_size=13,
            conv_mixer_expansion=1.0,
            data_cells=[4, 5, 10, 11, 12],
            pad_token_id=0,
            eos_token_id=3,
            use_fpn=False,
        )
    ).eval()
    native = TableFormerV2TokenDecoder(_native_config())
    load_same_weight_decoder(token_decoder_state_dict(reference), native, mx)
    return reference, native


def _assert_close(actual: Any, expected: np.ndarray) -> None:
    actual_array = np.array(actual, copy=True)
    error = np.abs(actual_array - expected)
    assert actual_array.shape == expected.shape
    assert float(error.mean()) <= 1e-4
    assert float(error.max()) <= 1e-3


@pytest.mark.parametrize("seed", [73, 1861])
def test_full_sequence_decoder_matches_pinned_torch_cpu(seed: int) -> None:
    reference, native = _same_weight_models()
    generator = np.random.default_rng(seed)
    token_ids = np.array([[2, 9, 3, 2, 9], [2, 3, 9, 2, 3]], dtype=np.int64)
    memory = generator.normal(0.0, 0.1, (2, 7, 512)).astype(np.float32)
    with torch.inference_mode():
        output = reference(
            input_ids=torch.from_numpy(token_ids),
            encoder_outputs={"last_hidden_state": torch.from_numpy(memory)},
            use_cache=False,
        )
    native_output = native.full_sequence(mx.array(token_ids), mx.array(memory))
    mx.eval(native_output.logits, native_output.hidden_states)
    _assert_close(native_output.logits, output.logits.numpy())
    _assert_close(native_output.hidden_states, output.hidden_states.numpy())


def test_cached_steps_match_full_sequence_and_pinned_torch_cpu() -> None:
    reference, native = _same_weight_models()
    generator = np.random.default_rng(1861)
    token_ids = np.array([[2, 9, 3, 2, 9]], dtype=np.int64)
    memory = generator.normal(0.0, 0.1, (1, 7, 512)).astype(np.float32)
    past_reference = None
    past_native = None
    cross_native = native.prepare_cross_attention_cache(mx.array(memory))
    mx.eval(*(array for entry in cross_native for array in entry))
    cached_logits: list[np.ndarray] = []
    for position in range(token_ids.shape[1]):
        current = token_ids[:, position : position + 1]
        with torch.inference_mode():
            expected = reference(
                input_ids=torch.from_numpy(current),
                encoder_outputs={"last_hidden_state": torch.from_numpy(memory)},
                past_key_values=past_reference,
                use_cache=True,
            )
        next_token, actual = native.greedy_token_step(
            mx.array(current),
            mx.array(memory),
            past_native,
            cross_attention_cache=cross_native,
        )
        assert actual.past_key_values is not None
        mx.eval(
            next_token,
            actual.logits,
            actual.hidden_states,
            *(array for cache in actual.past_key_values for array in cache.state),
        )
        _assert_close(actual.logits, expected.logits.numpy())
        _assert_close(actual.hidden_states, expected.hidden_states.numpy())
        assert np.array(next_token, copy=True).tolist() == (
            expected.logits[:, -1, :].argmax(dim=-1, keepdim=True).numpy().tolist()
        )
        assert expected.past_key_values is not None
        for layer, actual_cache, expected_state in zip(
            native.transformer_decoder.layers,
            actual.past_key_values,
            expected.past_key_values,
            strict=True,
        ):
            expected_key, expected_value = layer.self_attn.project_key_values(
                mx.array(expected_state[0].numpy())
            )
            actual_key, actual_value = actual_cache.state
            mx.eval(expected_key, expected_value, actual_key, actual_value)
            _assert_close(actual_key, np.array(expected_key, copy=True))
            _assert_close(actual_value, np.array(expected_value, copy=True))
            assert actual_cache.offset == position + 1
        cached_logits.append(np.array(actual.logits[:, -1, :], copy=True))
        past_reference = expected.past_key_values
        past_native = actual.past_key_values
    full = native.full_sequence(mx.array(token_ids), mx.array(memory))
    mx.eval(full.logits)
    np.testing.assert_allclose(
        np.stack(cached_logits, axis=1), np.array(full.logits, copy=True), rtol=1e-4, atol=1e-3
    )


def test_positional_encoding_repeats_at_the_source_boundary() -> None:
    reference, native = _same_weight_models()
    with torch.inference_mode():
        expected = reference._positional_encoding(2, 3, offset=511).numpy()
    actual = native._positional_encoding(2, 3, offset=511)
    mx.eval(actual)
    np.testing.assert_array_equal(np.array(actual, copy=True), expected)


def test_causal_mask_matches_the_source_negative_infinity_contract() -> None:
    _, native = _same_weight_models()
    mask = native.causal_mask(3, dtype=mx.float32)
    mx.eval(mask)
    np.testing.assert_array_equal(
        np.asarray(mask),
        np.array(
            [[0.0, -np.inf, -np.inf], [0.0, 0.0, -np.inf], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
    )


def test_cached_step_rejects_multiple_tokens() -> None:
    _, native = _same_weight_models()
    with pytest.raises(ValueError, match="shape"):
        native.cached_token_step(
            mx.array(np.array([[2, 3]], dtype=np.int64)),
            mx.zeros((1, 7, 512), dtype=mx.float32),
            None,
        )
