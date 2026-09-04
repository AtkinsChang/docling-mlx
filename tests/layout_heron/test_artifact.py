# SPDX-License-Identifier: Apache-2.0

"""RT-DETR-v2 checkpoint-directory contracts independent of MLX."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from docling_mlx.engines.object_detection.rt_detr_v2.artifact import (
    _validate_artifact,
)
from docling_mlx.engines.object_detection.rt_detr_v2.preprocessing import (
    RtDetrPreprocessingSpec,
    parse_preprocessing_config,
    pil_to_rgb_uint8,
)


def _config() -> dict[str, object]:
    return {
        "model_type": "rt_detr_v2",
        "id2label": {"0": "caption", "1": "table"},
        "label2id": {"caption": 0, "table": 1},
        "backbone_config": {"out_features": ["stage2", "stage3", "stage4"]},
    }


def _processor_config() -> dict[str, object]:
    return {
        "image_processor_type": "RTDetrImageProcessor",
        "do_resize": True,
        "do_rescale": True,
        "do_normalize": False,
        "resample": 2,
        "rescale_factor": 1 / 255,
        "size": {"height": 640, "width": 640},
    }


def _artifact(directory: Path) -> Path:
    (directory / "model.safetensors").write_bytes(b"not read by semantic validation")
    (directory / "config.json").write_text(json.dumps(_config()))
    (directory / "preprocessor_config.json").write_text(json.dumps(_processor_config()))
    return directory


def test_runtime_accepts_a_generic_checkpoint_contract(tmp_path: Path) -> None:
    config, preprocessing = _validate_artifact(_artifact(tmp_path))

    assert config.id2label == {0: "caption", 1: "table"}
    assert preprocessing == RtDetrPreprocessingSpec()


def test_runtime_accepts_non_docling_labels(tmp_path: Path) -> None:
    _artifact(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    config["id2label"] = {"0": "person", "1": "car"}
    config["label2id"] = {"person": 0, "car": 1}
    (tmp_path / "config.json").write_text(json.dumps(config))

    parsed, _ = _validate_artifact(tmp_path)

    assert parsed.id2label == {0: "person", 1: "car"}


def test_runtime_rejects_noncanonical_label_identifiers(tmp_path: Path) -> None:
    _artifact(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    config["id2label"] = {"00": "caption", "1": "table"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(TypeError, match="id2label"):
        _validate_artifact(tmp_path)


def test_preprocessing_is_rgb_contract() -> None:
    spec = parse_preprocessing_config(_processor_config())
    assert spec.size == (640, 640)
    converted = pil_to_rgb_uint8(Image.new("L", (2, 3), 12))
    assert converted.shape == (3, 2, 3)


def test_preprocessing_honors_normalization() -> None:
    config = _processor_config() | {"do_normalize": True, "image_std": [0.0, 1.0, 1.0]}
    with pytest.raises(ValueError, match="image_std"):
        parse_preprocessing_config(config)
