# SPDX-License-Identifier: Apache-2.0

"""Closed TableFormer v1 configuration tests."""

from __future__ import annotations

from copy import deepcopy

import pytest

from docling_mlx._models.tableformer_v1.config import (
    TABLEFORMER_V1_TOKENS,
    TableFormerV1Config,
)
from tests.tableformer_v1.conftest import accurate_config, fast_config


def test_frozen_accurate_config_is_accepted() -> None:
    config = TableFormerV1Config.from_dict(accurate_config())

    assert config.image_size == 448
    assert config.encoded_image_size == 28
    assert config.hidden_dim == 512
    assert config.encoder_layers == config.decoder_layers == 6
    assert config.num_heads == 8
    assert config.ff_dim == 1024
    assert config.vocab_size == len(TABLEFORMER_V1_TOKENS) == 13
    assert config.dtype == "float32"


def test_frozen_fast_config_is_accepted() -> None:
    config = TableFormerV1Config.from_dict(fast_config())

    assert (config.encoder_layers, config.decoder_layers) == (4, 2)


@pytest.mark.parametrize(("encoder_layers", "decoder_layers"), [(4, 6), (6, 2), (5, 3)])
def test_unshipped_depth_pairs_are_rejected(encoder_layers: int, decoder_layers: int) -> None:
    raw = deepcopy(accurate_config())
    raw["model"]["enc_layers"] = encoder_layers
    raw["model"]["dec_layers"] = decoder_layers

    with pytest.raises(ValueError, match="layer depths"):
        TableFormerV1Config.from_dict(raw)


@pytest.mark.parametrize(
    ("section", "name", "value", "message"),
    [
        ("dataset", "resized_image", True, "resized_image must be an integer"),
        ("model", "hidden_dim", 256, "embed_dim"),
        ("model", "nheads", 4, "num_heads"),
    ],
)
def test_architecture_drift_is_rejected(
    section: str, name: str, value: object, message: str
) -> None:
    raw = deepcopy(accurate_config())
    raw[section][name] = value

    error = TypeError if value is True else ValueError
    with pytest.raises(error, match=message):
        TableFormerV1Config.from_dict(raw)


def test_tag_vocabulary_is_exact() -> None:
    raw = accurate_config()
    raw["dataset_wordmap"]["word_map_tag"]["ecel"] = 99

    with pytest.raises(ValueError, match="vocabulary"):
        TableFormerV1Config.from_dict(raw)


def test_flat_hf_configuration_is_accepted() -> None:
    from tools.tableformer_v1.convert_weights import _flat_config

    config = TableFormerV1Config.from_dict(_flat_config(accurate_config()))
    assert config.model_type == "tableformer_v1"
    assert config.architectures == ("TableFormerV1",)
    assert config.embed_dim == 512
    assert config.num_encoder_layers == config.num_decoder_layers == 6
    assert config.data_cell_token_ids == frozenset({4, 5, 10, 11, 12})


def test_unknown_source_fields_are_accepted() -> None:
    raw = accurate_config()
    raw["future_metadata"] = {"producer_version": "next"}

    TableFormerV1Config.from_dict(raw)
