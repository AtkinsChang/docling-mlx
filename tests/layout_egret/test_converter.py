# SPDX-License-Identifier: Apache-2.0

"""Strict D-FINE conversion contracts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from tools.layout_egret.convert_weights import (
    _atomic_output,
    convert_state_dict,
    rename_key,
)


def _source_state() -> dict[str, np.ndarray]:
    return {
        "model.backbone.model.embedder.stem1.convolution.weight": np.arange(
            24, dtype=np.float32
        ).reshape(2, 3, 2, 2),
        "model.backbone.model.embedder.stem1.normalization.running_mean": np.zeros(
            2, dtype=np.float32
        ),
        "model.backbone.model.embedder.stem1.normalization.running_var": np.ones(
            2, dtype=np.float32
        ),
        "model.backbone.model.embedder.stem1.normalization.num_batches_tracked": np.array(
            7, dtype=np.int64
        ),
        "model.encoder_input_proj.0.0.weight": np.arange(6, dtype=np.float32).reshape(2, 3, 1, 1),
        "model.encoder_input_proj.0.1.weight": np.ones(2, dtype=np.float32),
        "model.enc_output.0.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "model.enc_output.1.bias": np.zeros(2, dtype=np.float32),
        "model.decoder.layers.0.self_attn.q_proj.weight": np.arange(6, dtype=np.float32).reshape(
            2, 3
        ),
        "model.denoising_class_embed.weight": np.ones((2, 3), dtype=np.float32),
    }


def _target_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "vision.backbone.embedder.stem1.convolution.weight": (2, 2, 2, 3),
        "vision.backbone.embedder.stem1.normalization.running_mean": (2,),
        "vision.backbone.embedder.stem1.normalization.running_var": (2,),
        "vision.encoder_input_proj.0.conv.weight": (2, 1, 1, 3),
        "vision.encoder_input_proj.0.norm.weight": (2,),
        "enc_output.fc.weight": (2, 3),
        "enc_output.norm.bias": (2,),
        "decoder.layers.0.self_attn.q_proj.weight": (2, 3),
    }


def test_conversion_maps_the_complete_medium_state_once() -> None:
    source = _source_state()
    converted, mappings, ignored = convert_state_dict(source, _target_shapes())

    np.testing.assert_array_equal(
        converted["vision.backbone.embedder.stem1.convolution.weight"],
        source["model.backbone.model.embedder.stem1.convolution.weight"].transpose(0, 2, 3, 1),
    )
    assert set(converted) == set(_target_shapes())
    assert {item["target_key"] for item in mappings} == set(_target_shapes())
    assert ignored == {
        "model.backbone.model.embedder.stem1.normalization.num_batches_tracked": (
            "batch_norm_training_counter"
        ),
        "model.denoising_class_embed.weight": "eval_unreachable_denoising_embedding",
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda state: state.pop("model.enc_output.0.weight"),
            "missing target keys",
        ),
        (
            lambda state: state.__setitem__("model.unexpected", np.zeros(1, dtype=np.float32)),
            "unexpected",
        ),
        (
            lambda state: state.__setitem__(
                "model.enc_output.0.weight", np.zeros((3, 2), dtype=np.float32)
            ),
            "shape mismatch",
        ),
    ],
)
def test_conversion_rejects_incomplete_extra_or_wrong_shape_state(
    mutation: Callable[[dict[str, np.ndarray]], object], message: str
) -> None:
    source = _source_state()
    mutation(source)
    with pytest.raises(ValueError, match=message):
        convert_state_dict(source, _target_shapes())


def test_only_paired_batch_norm_counters_and_the_named_denoising_embedding_are_ignored() -> None:
    source = _source_state()
    source["model.backbone.model.embedder.stem1.normalization.num_batches_tracked"] = np.array(
        0, dtype=np.int32
    )
    with pytest.raises(ValueError, match="int64 scalar"):
        convert_state_dict(source, _target_shapes())

    source = _source_state()
    source["model.other.num_batches_tracked"] = np.array(0, dtype=np.int64)
    with pytest.raises(ValueError, match="Unpaired"):
        convert_state_dict(source, _target_shapes())


@pytest.mark.parametrize(
    ("source_key", "target_key"),
    [
        (
            "model.backbone.model.embedder.stem1.convolution.weight",
            "vision.backbone.embedder.stem1.convolution.weight",
        ),
        ("model.encoder_input_proj.1.0.weight", "vision.encoder_input_proj.1.conv.weight"),
        ("model.encoder_input_proj.1.1.bias", "vision.encoder_input_proj.1.norm.bias"),
        ("model.enc_output.0.weight", "enc_output.fc.weight"),
        ("model.enc_output.1.weight", "enc_output.norm.weight"),
        (
            "model.decoder.layers.0.self_attn.q_proj.weight",
            "decoder.layers.0.self_attn.q_proj.weight",
        ),
        ("model.decoder.layers.0.mlp.layers.0.weight", "decoder.layers.0.fc1.weight"),
        (
            "model.encoder.encoder.0.layers.0.fc1.weight",
            "vision.encoder.aifi.0.layers.0.mlp.layers.0.weight",
        ),
        (
            "model.encoder.encoder.0.layers.0.fc2.bias",
            "vision.encoder.aifi.0.layers.0.mlp.layers.1.bias",
        ),
        (
            "model.encoder.encoder.0.layers.0.self_attn.out_proj.weight",
            "vision.encoder.aifi.0.layers.0.self_attn.o_proj.weight",
        ),
        (
            "model.decoder_input_proj.0.0.weight",
            "decoder_input_proj.0.conv.weight",
        ),
        (
            "model.decoder_input_proj.2.1.running_var",
            "decoder_input_proj.2.norm.running_var",
        ),
    ],
)
def test_rename_key_matches_the_composed_dfine_namespace(source_key: str, target_key: str) -> None:
    assert rename_key(source_key) == target_key


def test_atomic_output_publishes_complete_directory_and_cleans_up_failure(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    with _atomic_output(output) as staging:
        (staging / "complete").write_text("yes")
        assert not output.exists()
    assert (output / "complete").read_text() == "yes"
    assert not list(tmp_path.glob(".artifact.convert-*"))

    failed = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="fail"):
        with _atomic_output(failed) as staging:
            (staging / "partial").write_text("no")
            raise RuntimeError("fail")
    assert not failed.exists()
    assert not list(tmp_path.glob(".failed.convert-*"))


def test_atomic_output_refuses_existing_directory(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        with _atomic_output(output):
            pytest.fail("existing output must not stage")


def test_xlarge_decoder_input_projections_map_and_transpose_to_target_shapes() -> None:
    source = {
        "model.decoder_input_proj.0.0.weight": np.arange(256 * 384, dtype=np.float32).reshape(
            256, 384, 1, 1
        ),
        "model.decoder_input_proj.0.1.weight": np.ones(256, dtype=np.float32),
        "model.decoder_input_proj.0.1.bias": np.zeros(256, dtype=np.float32),
        "model.decoder_input_proj.0.1.running_mean": np.zeros(256, dtype=np.float32),
        "model.decoder_input_proj.0.1.running_var": np.ones(256, dtype=np.float32),
        "model.decoder_input_proj.0.1.num_batches_tracked": np.array(0, dtype=np.int64),
    }
    target_shapes = {
        "decoder_input_proj.0.conv.weight": (256, 1, 1, 384),
        "decoder_input_proj.0.norm.weight": (256,),
        "decoder_input_proj.0.norm.bias": (256,),
        "decoder_input_proj.0.norm.running_mean": (256,),
        "decoder_input_proj.0.norm.running_var": (256,),
    }

    converted, mappings, ignored = convert_state_dict(source, target_shapes)

    assert converted["decoder_input_proj.0.conv.weight"].shape == (256, 1, 1, 384)
    assert {item["target_key"] for item in mappings} == set(target_shapes)
    assert ignored == {
        "model.decoder_input_proj.0.1.num_batches_tracked": "batch_norm_training_counter"
    }
