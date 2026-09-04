# SPDX-License-Identifier: Apache-2.0

"""Runtime validation for EfficientNet checkpoint directories."""

from __future__ import annotations

from pathlib import Path

from docling_mlx._models.efficientnet.config import EfficientNetConfig
from docling_mlx.engines._shared import read_json_object, require_checkpoint_files

from .preprocessing import EfficientNetPreprocessingSpec, parse_preprocessing_config

CHECKPOINT_FILES = ("model.safetensors", "config.json", "preprocessor_config.json")


def _validate_artifact(
    directory: Path,
) -> tuple[EfficientNetConfig, EfficientNetPreprocessingSpec]:
    """Validate a generic MLX or upstream HF checkpoint directory."""

    require_checkpoint_files(directory, CHECKPOINT_FILES)
    config = EfficientNetConfig.from_dict(read_json_object(directory / "config.json"))
    preprocessing = parse_preprocessing_config(
        read_json_object(directory / "preprocessor_config.json")
    )
    return config, preprocessing


__all__ = ["CHECKPOINT_FILES", "_validate_artifact"]
