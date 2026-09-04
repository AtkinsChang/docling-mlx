# SPDX-License-Identifier: Apache-2.0

"""Portable tests for the closed TableFormerV2 artifact contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docling_mlx._models.tableformer_v2 import TABLEFORMER_V2_TOKENS
from docling_mlx.engines.table_structure.tableformer_v2.artifact import (
    validate_tableformer_v2_artifact,
)


def _config() -> dict:
    return {
        "architectures": ["TableFormerV2"],
        "model_type": "TableFormerV2",
        "embed_dim": 512,
        "num_heads": 8,
        "ff_dim": 2048,
        "num_decoder_layers": 4,
        "vocab_size": 13,
        "conv_mixer_expansion": 1.0,
        "data_cells": [5, 4, 10, 11, 12],
        "pad_token_id": 0,
        "eos_token_id": 3,
        "use_fpn": False,
        "dtype": "float32",
    }


def _artifact(directory: Path) -> Path:
    (directory / "model.safetensors").write_bytes(b"strict loading is tested separately")
    (directory / "config.json").write_text(json.dumps(_config()))
    (directory / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "do_convert_rgb": True,
                "do_resize": True,
                "size": {"height": 448, "width": 448},
                "interpolation": "bilinear",
                "antialias": True,
                "do_rescale": True,
                "rescale_factor": 1 / 255,
                "do_normalize": True,
                "image_mean": [0.485, 0.456, 0.406],
                "image_std": [0.229, 0.224, 0.225],
                "output_dtype": "float32",
                "output_layout": "NHWC",
            }
        )
    )
    (directory / "generation_config.json").write_text(json.dumps({"max_generation_steps": 512}))
    (directory / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [
                    {"id": index, "content": token}
                    for index, token in enumerate(TABLEFORMER_V2_TOKENS)
                ]
            }
        )
    )
    (directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "pad_token": "<pad>",
                "unk_token": "[UNK]",
                "bos_token": "<start>",
                "eos_token": "<end>",
            }
        )
    )
    (directory / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "pad_token": {"content": "<pad>"},
                "unk_token": {"content": "[UNK]"},
                "bos_token": {"content": "<start>"},
                "eos_token": {"content": "<end>"},
            }
        )
    )
    return directory


def test_closed_artifact_contract_accepts_current_checkpoint(tmp_path: Path) -> None:
    artifact = validate_tableformer_v2_artifact(_artifact(tmp_path))
    assert artifact.config.num_decoder_layers == 4
    assert artifact.preprocessing.size == (448, 448)
    assert artifact.generation.max_generation_steps == 512
    assert artifact.token_map.tokens == TABLEFORMER_V2_TOKENS


def test_upstream_checkpoint_uses_its_native_metadata_layout(tmp_path: Path) -> None:
    directory = _artifact(tmp_path)
    (directory / "preprocessor_config.json").unlink()
    (directory / "generation_config.json").write_text(json.dumps({"max_length": 512}))

    artifact = validate_tableformer_v2_artifact(directory)

    assert artifact.upstream_weights is True
    assert artifact.preprocessing.size == (448, 448)
    assert artifact.generation.max_generation_steps == 512


@pytest.mark.parametrize(
    ("filename", "mutation", "message"),
    [
        ("config.json", {"num_decoder_layers": 2}, "num_decoder_layers"),
        ("generation_config.json", {"max_generation_steps": 511}, "512"),
        ("tokenizer_config.json", {"bos_token": "<end>"}, "tokenizer metadata"),
    ],
)
def test_runtime_rejects_incompatible_contract(
    tmp_path: Path, filename: str, mutation: dict, message: str
) -> None:
    artifact = _artifact(tmp_path)
    path = artifact / filename
    value = json.loads(path.read_text())
    value.update(mutation)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match=message):
        validate_tableformer_v2_artifact(artifact)


def test_data_cell_order_is_not_semantic(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    path = artifact / "config.json"
    config = json.loads(path.read_text())
    config["data_cells"] = [12, 11, 10, 5, 4]
    path.write_text(json.dumps(config))
    validate_tableformer_v2_artifact(artifact)


def test_duplicate_data_cell_ids_are_rejected(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    path = artifact / "config.json"
    config = json.loads(path.read_text())
    config["data_cells"] = [4, 4, 5, 10, 11, 12]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="duplicates"):
        validate_tableformer_v2_artifact(artifact)


def test_generation_steps_must_be_an_integer(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    path = artifact / "generation_config.json"
    path.write_text(json.dumps({"max_generation_steps": 512.0}))
    with pytest.raises(ValueError, match="512 generation steps"):
        validate_tableformer_v2_artifact(artifact)


def test_tokenizer_accepts_unknown_fields(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    path = artifact / "tokenizer.json"
    tokenizer = json.loads(path.read_text())
    tokenizer["future_default"] = True
    path.write_text(json.dumps(tokenizer))
    validate_tableformer_v2_artifact(artifact)


def test_duplicate_token_ids_are_rejected(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    path = artifact / "tokenizer.json"
    tokenizer = json.loads(path.read_text())
    tokenizer["added_tokens"].append({"id": 12, "content": "<duplicate>"})
    path.write_text(json.dumps(tokenizer))
    with pytest.raises(ValueError, match="duplicate token IDs"):
        validate_tableformer_v2_artifact(artifact)
