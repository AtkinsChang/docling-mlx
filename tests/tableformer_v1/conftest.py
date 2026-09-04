# SPDX-License-Identifier: Apache-2.0

"""TableFormer v1 artifact builders."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from tools.tableformer_v1.convert_weights import _flat_config, _preprocessor_config

_TAG_MAP = {
    "<pad>": 0,
    "<unk>": 1,
    "<start>": 2,
    "<end>": 3,
    "ecel": 4,
    "fcel": 5,
    "lcel": 6,
    "ucel": 7,
    "xcel": 8,
    "nl": 9,
    "ched": 10,
    "rhed": 11,
    "srow": 12,
}
_CELL_SPECIALS = {0: "<pad>", 1: "<b>", 278: "<unk>", 279: "<start>", 280: "<end>"}


def accurate_config() -> dict:
    return {
        "dataset": {
            "type": "PTN_prepared",
            "name": "PubTabNet_300_100_512",
            "raw_data_dir": "./tests/test_data/ccs_api/model/",
            "load_cells": True,
            "bbox_format": "5plet",
            "resized_image": 448,
            "keep_AR": False,
            "up_scaling_enabled": True,
            "down_scaling_enabled": True,
            "padding_mode": "null",
            "padding_color": [0, 0, 0],
            "image_normalization": {
                "state": True,
                "mean": [0.94247851, 0.94254675, 0.94292611],
                "std": [0.17910956, 0.17940403, 0.17931663],
            },
            "color_jitter": True,
            "rand_crop": True,
            "rand_pad": True,
            "image_grayscale": False,
        },
        "model": {
            "type": "TableModel04_rs",
            "name": "14_128_256_4_true",
            "backbone": "resnet18",
            "enc_image_size": 28,
            "tag_embed_dim": 16,
            "hidden_dim": 512,
            "tag_decoder_dim": 512,
            "bbox_embed_dim": 256,
            "tag_attention_dim": 256,
            "bbox_attention_dim": 512,
            "enc_layers": 6,
            "dec_layers": 6,
            "nheads": 8,
            "dropout": 0.1,
            "bbox_classes": 2,
        },
        "train": {"bbox": True},
        "predict": {
            "max_steps": 1024,
            "beam_size": 5,
            "bbox": True,
            "pdf_cell_iou_thres": 0.05,
            "padding": False,
            "padding_size": 50,
            "disable_post_process": False,
            "profiling": False,
        },
        "debug": {"save_debug_images": False},
        "dataset_wordmap": {
            "word_map_tag": dict(_TAG_MAP),
            "word_map_cell": {
                _CELL_SPECIALS.get(index, f"token-{index}"): index for index in range(281)
            },
        },
    }


def fast_config() -> dict:
    config = deepcopy(accurate_config())
    config["model"]["enc_layers"] = 4
    config["model"]["dec_layers"] = 2
    return config


def build_artifact(
    directory: Path,
    *,
    profiles: tuple[str, ...] = ("accurate",),
) -> Path:
    configs = {"accurate": accurate_config, "fast": fast_config}
    for name in profiles:
        profile = directory / name
        profile.mkdir()
        (profile / "model.safetensors").write_bytes(b"semantic validation does not load weights")
        source_config = configs[name]()
        (profile / "config.json").write_text(json.dumps(_flat_config(source_config)))
        (profile / "generation_config.json").write_text(
            json.dumps({"max_generation_steps": source_config["predict"]["max_steps"]})
        )
        (profile / "preprocessor_config.json").write_text(
            json.dumps(_preprocessor_config(source_config))
        )
    return directory


@pytest.fixture
def artifact_root(tmp_path: Path) -> Path:
    return build_artifact(tmp_path)
