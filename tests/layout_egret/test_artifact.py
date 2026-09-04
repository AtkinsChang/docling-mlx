# SPDX-License-Identifier: Apache-2.0

"""Runtime artifact contracts for generic D-FINE checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docling_mlx.engines.object_detection.dfine.artifact import _validate_artifact
from tests.layout_egret.test_config import egret_config, preprocessor_config


def _artifact(directory: Path, *, profile: str = "medium") -> Path:
    (directory / "model.safetensors").write_bytes(b"weights are loaded only by the engine")
    (directory / "config.json").write_text(json.dumps(egret_config(profile)))
    (directory / "preprocessor_config.json").write_text(json.dumps(preprocessor_config()))
    return directory


@pytest.mark.parametrize("profile", ["medium", "large", "xlarge"])
def test_runtime_accepts_hf_shaped_artifact(tmp_path: Path, profile: str) -> None:
    config, preprocessing = _validate_artifact(_artifact(tmp_path, profile=profile))

    assert config.num_labels == 17
    assert preprocessing.size == (640, 640)


def test_runtime_requires_all_model_files(tmp_path: Path) -> None:
    _artifact(tmp_path)
    (tmp_path / "model.safetensors").unlink()

    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        _validate_artifact(tmp_path)
