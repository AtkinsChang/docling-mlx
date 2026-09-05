# SPDX-License-Identifier: Apache-2.0

"""Composition and generation contracts for native TableFormerV2."""

from __future__ import annotations

import importlib
from collections.abc import Callable
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
    config_type = importlib.import_module(
        "docling_mlx._models.tableformer_v2.config"
    ).TableFormerV2Config
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


def _model() -> Any:
    return importlib.import_module("docling_mlx._models.tableformer_v2.model").TableFormerV2(
        _config()
    )


def test_top_level_keeps_source_parameter_namespace_and_fp32_eval() -> None:
    mx = _required_module("mlx.core")
    tree_flatten = _required_module("mlx.utils").tree_flatten
    model = _model()
    parameters = dict(tree_flatten(model.parameters()))

    assert not model.training
    assert parameters["feature_extractor.features.0.0.weight"].dtype == mx.float32
    assert parameters["input_embedding.weight"].shape == (13, 512)
    assert parameters["positional_encoding"].shape == (1, 512, 512)
    assert parameters["transformer_decoder.layers.3.linear2.weight"].shape == (512, 2048)
    assert parameters["output_projection.weight"].shape == (13, 512)
    assert parameters["bbox_head.input_proj.weight"].shape == (512, 512)
    assert not any(key.startswith(("vision.", "token_decoder.")) for key in parameters)
    assert all(value.dtype == mx.float32 for value in parameters.values())


def test_top_level_exposes_full_and_cached_decoder_with_wrapped_position_512() -> None:
    mx = _required_module("mlx.core")
    model = _model()
    memory = mx.zeros((1, 2, 512), dtype=mx.float32)
    cross_cache = model.prepare_cross_attention_cache(memory)

    first = model.decode_tokens(
        mx.array([[2]], dtype=mx.int32),
        memory,
        cross_attention_cache=cross_cache,
        use_cache=True,
    )
    assert first.past_key_values is not None
    second = model.decode_tokens(
        mx.array([[9]], dtype=mx.int32),
        memory,
        past_key_values=first.past_key_values,
        cross_attention_cache=cross_cache,
        use_cache=True,
    )
    full = model.decode_tokens(
        mx.array([[2, 9]], dtype=mx.int32),
        memory,
        cross_attention_cache=cross_cache,
    )
    wrapped = model._positional_encoding(1, 1, offset=512)
    mx.eval(
        second.logits,
        full.logits,
        wrapped,
        model.positional_encoding,
        *(array for cache in second.past_key_values or () for array in cache.state),
    )

    assert second.logits.shape == (1, 1, 13)
    assert full.logits.shape == (1, 2, 13)
    assert second.past_key_values is not None
    assert all(cache.offset == 2 for cache in second.past_key_values)
    assert all(cache.state[0].shape == (1, 8, 2, 64) for cache in second.past_key_values)
    np.testing.assert_array_equal(
        np.asarray(wrapped[:, 0]), np.asarray(model.positional_encoding[:, 0])
    )


def test_compiled_image_backbone_is_idempotent_and_preserves_encoder_output() -> None:
    mx = _required_module("mlx.core")
    model = _model()
    pixels = mx.arange(1 * 64 * 64 * 3, dtype=mx.float32).reshape(1, 64, 64, 3) / 255
    expected = model.encode_images(pixels)
    mx.eval(expected.last_hidden_state)

    model.compile_image_backbone()
    compiled = model._feature_forward
    model.compile_image_backbone()
    actual = model.encode_images(pixels)
    mx.eval(actual.last_hidden_state)

    assert model._feature_forward is compiled
    assert actual.spatial_size == expected.spatial_size
    np.testing.assert_allclose(
        np.asarray(actual.last_hidden_state),
        np.asarray(expected.last_hidden_state),
        rtol=0,
        atol=3e-6,
    )


def _install_generation_fakes(
    monkeypatch: pytest.MonkeyPatch,
    model: Any,
    *,
    next_tokens: Callable[[int, int], np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any]]:
    mx = _required_module("mlx.core")
    model_module = importlib.import_module("docling_mlx._models.tableformer_v2.model")
    decoder_module = importlib.import_module("docling_mlx._models.tableformer_v2.decoder")
    calls: dict[str, Any] = {"encode": 0, "decode": [], "bbox": 0}
    captures: dict[str, Any] = {}

    def encode_images(pixels: Any) -> Any:
        calls["encode"] += 1
        return model_module.TableFormerV2EncoderOutput(
            mx.zeros((pixels.shape[0], 2, 512), dtype=mx.float32),
            (1, 2),
        )

    def decode_tokens(
        input_ids: Any,
        encoder_hidden_states: Any,
        *,
        past_key_values: Any = None,
        cross_attention_cache: Any = None,
        use_cache: bool = False,
    ) -> Any:
        del encoder_hidden_states, past_key_values, cross_attention_cache
        calls["decode"].append((tuple(input_ids.shape), use_cache))
        batch_size, sequence_length = input_ids.shape
        logits = np.zeros((batch_size, sequence_length, 13), dtype=np.float32)
        if use_cache:
            step = sum(cached for _, cached in calls["decode"]) - 1
            tokens = next_tokens(step, batch_size)
            logits[:, -1, :] = -1.0
            logits[np.arange(batch_size), -1, tokens] = 1.0
        positions = np.broadcast_to(
            np.arange(sequence_length, dtype=np.float32)[None, :, None],
            (batch_size, sequence_length, 512),
        ).copy()
        return decoder_module.TokenDecoderOutput(
            logits=mx.array(logits),
            hidden_states=mx.array(positions),
            past_key_values=() if use_cache else None,
        )

    def bbox_head(cell_embeddings: Any, encoder_hidden: Any, cell_batch_indices: Any) -> Any:
        del encoder_hidden
        calls["bbox"] += 1
        captures["cell_embeddings"] = np.asarray(cell_embeddings).copy()
        captures["cell_batch_indices"] = np.asarray(cell_batch_indices).copy()
        return mx.zeros((cell_embeddings.shape[0], 4), dtype=mx.float32)

    monkeypatch.setattr(model, "encode_images", encode_images)
    monkeypatch.setattr(model, "decode_tokens", decode_tokens)
    monkeypatch.setattr(model, "bbox_head", bbox_head)
    return calls, captures


def test_generation_encodes_and_boxes_once_and_caps_at_513_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = _required_module("mlx.core")
    model = _model()
    calls, _ = _install_generation_fakes(
        monkeypatch,
        model,
        next_tokens=lambda _step, batch_size: np.full(batch_size, 9, dtype=np.int32),
    )

    output = model.generate(mx.zeros((1, 8, 8, 3), dtype=mx.float32))
    mx.eval(output.generated_ids, output.hidden_states, output.predicted_bboxes)

    assert calls["encode"] == 1
    assert calls["bbox"] == 1
    assert calls["decode"] == [((1, 1), True)] * 512 + [((1, 513), False)]
    assert output.generated_ids.shape == (1, 513)
    assert np.asarray(output.generated_ids)[0, 0] == 2
    assert np.all(np.asarray(output.generated_ids)[0, 1:] == 9)
    assert output.hidden_states.shape == (1, 513, 512)
    assert output.predicted_bboxes.shape == (0, 4)
    assert output.cell_positions.shape == (0, 2)


def test_generation_stops_on_eos_and_vectorizes_exact_cell_correspondence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = _required_module("mlx.core")
    model = _model()
    schedule = (
        np.array([4, 9], dtype=np.int32),
        np.array([3, 5], dtype=np.int32),
        np.array([3, 3], dtype=np.int32),
        # Double buffering builds one step past the EOS it then discards.
        np.array([3, 3], dtype=np.int32),
    )
    calls, captures = _install_generation_fakes(
        monkeypatch,
        model,
        next_tokens=lambda step, _batch_size: schedule[step],
    )

    output = model.generate(mx.zeros((2, 8, 8, 3), dtype=mx.float32))
    mx.eval(output.generated_ids, output.cell_positions, output.predicted_bboxes)

    np.testing.assert_array_equal(
        np.asarray(output.generated_ids),
        np.array([[2, 4, 3, 3], [2, 9, 5, 3]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(output.cell_positions), np.array([[0, 1], [1, 2]], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        np.asarray(output.cell_batch_indices), np.array([0, 1], dtype=np.int32)
    )
    np.testing.assert_array_equal(captures["cell_batch_indices"], np.array([0, 1]))
    np.testing.assert_array_equal(captures["cell_embeddings"][:, 0], np.array([1.0, 2.0]))
    assert output.predicted_bboxes.shape == (2, 4)
    assert calls["encode"] == 1
    assert calls["bbox"] == 1
    assert calls["decode"][-1] == ((2, 4), False)


def test_generation_rejects_wrong_bbox_cardinality(monkeypatch: pytest.MonkeyPatch) -> None:
    mx = _required_module("mlx.core")
    model = _model()
    _install_generation_fakes(
        monkeypatch,
        model,
        next_tokens=lambda step, batch_size: np.full(
            batch_size, 4 if step == 0 else 3, dtype=np.int32
        ),
    )
    monkeypatch.setattr(
        model,
        "bbox_head",
        lambda cell_embeddings, _encoder_hidden, _indices: mx.zeros(
            (max(0, cell_embeddings.shape[0] - 1), 4), dtype=mx.float32
        ),
    )

    with pytest.raises(RuntimeError, match="exactly one box"):
        model.generate(mx.zeros((1, 8, 8, 3), dtype=mx.float32))


def test_model_rejects_training_and_non_fp32_generation() -> None:
    mx = _required_module("mlx.core")
    model = _model()
    with pytest.raises(ValueError, match="float32"):
        model.generate(mx.zeros((1, 8, 8, 3), dtype=mx.float16), max_generation_steps=1)
    model.train()
    with pytest.raises(ValueError, match="eval"):
        model.generate(mx.zeros((1, 8, 8, 3), dtype=mx.float32), max_generation_steps=1)
