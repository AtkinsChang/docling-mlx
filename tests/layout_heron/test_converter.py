# SPDX-License-Identifier: Apache-2.0

"""Pure conversion checks for the shared RT-DETR-v2 sanitizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.layout_heron.convert_weights import _verify_source, convert_state_dict, rename_key


def test_conversion_transposes_only_oihw_convolutions_and_accounts_for_bn() -> None:
    source = {
        "model.backbone.model.conv.convolution.weight": np.arange(24, dtype=np.float32).reshape(
            2, 3, 2, 2
        ),
        "model.class_embed.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "model.backbone.model.bn.normalization.running_mean": np.zeros(2, dtype=np.float32),
        "model.backbone.model.bn.normalization.num_batches_tracked": np.array(7, dtype=np.int64),
    }
    target = {
        "vision.backbone.conv.conv.weight": (2, 2, 2, 3),
        "class_embed.weight": (2, 3),
        "vision.backbone.bn.bn.running_mean": (2,),
    }
    converted, mappings, ignored = convert_state_dict(source, target)
    np.testing.assert_array_equal(
        converted["vision.backbone.conv.conv.weight"],
        source["model.backbone.model.conv.convolution.weight"].transpose(0, 2, 3, 1),
    )
    np.testing.assert_array_equal(
        converted["class_embed.weight"], source["model.class_embed.weight"]
    )
    assert ignored == ["model.backbone.model.bn.normalization.num_batches_tracked"]
    assert {entry["transform"] for entry in mappings} == {"identity", "OIHW_to_OHWI"}
    assert mappings[0]["source_key"] != mappings[0]["target_key"]


@pytest.mark.parametrize(
    ("source_key", "target_key"),
    [
        (
            "model.backbone.model.encoder.stages.0.layers.0.shortcut.1.convolution.weight",
            "vision.backbone.encoder.stages.0.layers.0.shortcut.proj.conv.weight",
        ),
        (
            "model.encoder.encoder.layers.0.self_attn.k_proj.weight",
            "vision.hybrid_encoder.aifi.layers.0.self_attn.k_proj.weight",
        ),
        ("model.encoder_input_proj.2.0.weight", "vision.encoder_input_proj.2.conv.weight"),
        ("model.encoder_input_proj.2.1.running_var", "vision.encoder_input_proj.2.bn.running_var"),
        (
            "model.encoder.lateral_convs.1.norm.weight",
            "vision.hybrid_encoder.lateral_convs.1.bn.weight",
        ),
        ("model.decoder_input_proj.1.0.weight", "decoder_input_proj.1.conv.weight"),
        ("model.decoder_input_proj.1.1.bias", "decoder_input_proj.1.bn.bias"),
        ("model.enc_output.0.weight", "enc_output.fc.weight"),
        ("model.enc_output.1.bias", "enc_output.ln.bias"),
    ],
)
def test_rename_key_matches_the_reviewed_mlx_vlm_pipeline(source_key: str, target_key: str) -> None:
    assert rename_key(source_key) == target_key


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ({}, "missing"),
        ({"weight": np.zeros(3, dtype=np.float32)}, "shape"),
        ({"weight": np.array([np.nan, 0], dtype=np.float32)}, "finite"),
        ({"weight": np.zeros(2, dtype=np.float32), "extra": np.zeros(1)}, "unexpected"),
    ],
)
def test_conversion_rejects_missing_extra_or_invalid_tensors(source: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        convert_state_dict(source, {"weight": (2,)})


def test_conversion_drops_any_batchnorm_counter() -> None:
    converted, _, ignored = convert_state_dict(
        {"bn.running_mean": np.zeros(2, dtype=np.float32)}, {"bn.running_mean": (2,)}
    )
    assert set(converted) == {"bn.running_mean"}
    assert ignored == []
    _, _, ignored = convert_state_dict(
        {
            "bn.running_mean": np.zeros(2, dtype=np.float32),
            "other.num_batches_tracked": np.array(0, dtype=np.int32),
        },
        {"bn.running_mean": (2,)},
    )
    assert ignored == ["other.num_batches_tracked"]


def test_source_verification_requires_every_converter_input(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="Missing source file"):
        _verify_source(tmp_path)
