# SPDX-License-Identifier: Apache-2.0

"""Closed TableFormer v1 artifact tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docling_mlx.engines.table_structure.tableformer_v1.artifact import (
    TABLEFORMER_V1_PREPROCESSING_SPEC,
    TABLEFORMER_V1_TAG_MAP,
    validate_tableformer_v1_artifact,
)
from tests.tableformer_v1.conftest import accurate_config, build_artifact


def _mutate(root: Path, section: str, field: str, value: object) -> None:
    filename = {
        "config": "config.json",
        "generation": "generation_config.json",
        "preprocessor": "preprocessor_config.json",
    }[section]
    path = root / f"accurate/{filename}"
    config = json.loads(path.read_text())
    config[field] = value
    path.write_text(json.dumps(config))


def test_accurate_profile_selects_its_repository_subdirectory(artifact_root: Path) -> None:
    artifact = validate_tableformer_v1_artifact(artifact_root / "accurate")
    assert artifact.directory == artifact_root / "accurate"
    assert artifact.preprocessing == TABLEFORMER_V1_PREPROCESSING_SPEC
    assert artifact.preprocessing.page_height == 1024
    assert artifact.preprocessing.image_size == 448
    assert artifact.config.encoder_layers == 6


def test_fast_profile_selects_its_repository_subdirectory(tmp_path: Path) -> None:
    root = build_artifact(tmp_path, profiles=("accurate", "fast"))
    artifact = validate_tableformer_v1_artifact(root / "fast")

    assert artifact.directory == root / "fast"
    assert (artifact.config.encoder_layers, artifact.config.decoder_layers) == (4, 2)


def test_plain_checkpoint_validation_reports_its_own_missing_files(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"wrong level")
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="preprocessor_config.json"):
        validate_tableformer_v1_artifact(tmp_path)


def test_upstream_profile_is_a_plain_engine_checkpoint(artifact_root: Path) -> None:
    profile = artifact_root / "accurate"
    (profile / "tm_config.json").write_text(json.dumps(accurate_config()))
    (profile / "tableformer_accurate.safetensors").write_bytes(b"loaded by the engine")

    artifact = validate_tableformer_v1_artifact(profile)

    assert artifact.upstream_weights is True
    assert artifact.weights_path.name == "tableformer_accurate.safetensors"
    assert artifact.config.encoder_layers == 6


def test_selected_profile_rejects_other_profiles_config(artifact_root: Path) -> None:
    _mutate(artifact_root, "config", "num_encoder_layers", 5)
    with pytest.raises(ValueError, match="layer depths"):
        validate_tableformer_v1_artifact(artifact_root / "accurate")


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("config", "image_size", 512),
        ("config", "num_heads", 4),
        ("generation", "max_generation_steps", 512),
    ],
)
def test_semantic_drift_is_rejected(
    artifact_root: Path, section: str, field: str, value: object
) -> None:
    _mutate(artifact_root, section, field, value)
    with pytest.raises(ValueError, match="TableFormer v1"):
        validate_tableformer_v1_artifact(artifact_root / "accurate")


def test_tag_map_is_canonical_and_immutable() -> None:
    assert tuple(TABLEFORMER_V1_TAG_MAP) == (
        "<pad>",
        "<unk>",
        "<start>",
        "<end>",
        "ecel",
        "fcel",
        "lcel",
        "ucel",
        "xcel",
        "nl",
        "ched",
        "rhed",
        "srow",
    )


def test_unknown_config_metadata_does_not_break_runtime(artifact_root: Path) -> None:
    path = artifact_root / "accurate/config.json"
    config = json.loads(path.read_text())
    config["future_metadata"] = {"producer_version": "next"}
    path.write_text(json.dumps(config))
    validate_tableformer_v1_artifact(artifact_root / "accurate")


@pytest.mark.parametrize(
    "name", ["config.json", "generation_config.json", "preprocessor_config.json"]
)
def test_required_metadata_files_are_required(artifact_root: Path, name: str) -> None:
    (artifact_root / "accurate" / name).unlink()
    with pytest.raises(FileNotFoundError, match=name):
        validate_tableformer_v1_artifact(artifact_root / "accurate")
