# SPDX-License-Identifier: Apache-2.0

"""Config-only RT-DETR-v2 contracts."""

import copy

import pytest

from docling_mlx._models.rt_detr_v2 import RtDetrV2Config


def _config() -> dict[str, object]:
    return {
        "model_type": "rt_detr_v2",
        "num_labels": 3,
        "id2label": {"0": "caption", "1": "table", "2": "text"},
        "label2id": {"caption": 0, "table": 1, "text": 2},
        "backbone_config": {
            "model_type": "rt_detr_resnet",
            "depths": [3, 4, 6, 3],
            "hidden_sizes": [256, 512, 1024, 2048],
            "out_features": ["stage2", "stage3", "stage4"],
        },
    }


def test_hf_defaults_fill_absent_runtime_fields() -> None:
    config = RtDetrV2Config.from_dict(_config())

    assert config.decoder_n_points == 4
    assert config.transformer_config.points_per_level == (4, 4, 4)
    assert config.id2label == {0: "caption", 1: "table", 2: "text"}
    assert config.backbone_config.depths == (3, 4, 6, 3)


def test_config_accepts_non_heron_labels_and_topology() -> None:
    raw = _config()
    backbone = raw["backbone_config"]
    assert isinstance(backbone, dict)
    backbone["depths"] = [2, 2, 2, 2]
    raw["id2label"] = {"0": "object", "1": "other", "2": "third"}
    raw["label2id"] = {"object": 0, "other": 1, "third": 2}
    raw["unknown_hf_metadata"] = True

    config = RtDetrV2Config.from_dict(raw)

    assert config.backbone_config.depths == (2, 2, 2, 2)
    assert config.id2label[0] == "object"


def test_parser_accepts_per_level_decoder_points() -> None:
    raw = _config()
    raw["decoder_n_points"] = [3, 6, 3]

    config = RtDetrV2Config.from_dict(raw)

    assert config.transformer_config.points_per_level == (3, 6, 3)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("model_type",), "other", "model_type"),
        (("decoder_method",), "other", "decoder_method"),
        (("decoder_n_points",), [4, 4], "decoder_n_points"),
        (("backbone_config", "layer_type"), "other", "layer_type"),
    ],
)
def test_rejects_invalid_runtime_structure(
    path: tuple[str, ...], value: object, message: str
) -> None:
    raw = _config()
    target = raw
    for key in path[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = value
    with pytest.raises((TypeError, ValueError), match=message):
        RtDetrV2Config.from_dict(raw)


def test_parser_copies_mutable_source_values() -> None:
    raw = _config()
    config = RtDetrV2Config.from_dict(raw)
    original = copy.deepcopy(config)
    raw["id2label"] = {"0": "changed", "1": "changed-1", "2": "changed-2"}
    backbone = raw["backbone_config"]
    assert isinstance(backbone, dict)
    backbone["depths"] = [1, 1, 1, 1]

    assert config == original
