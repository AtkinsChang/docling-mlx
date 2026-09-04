# SPDX-License-Identifier: Apache-2.0

"""Runtime contracts for RT-DETR-v2 checkpoint directories."""

from __future__ import annotations

from pathlib import Path

from docling_mlx._models.rt_detr_v2.config import RtDetrV2Config
from docling_mlx.engines._shared import read_json_object, require_checkpoint_files
from docling_mlx.engines.object_detection.rt_detr_v2.preprocessing import (
    RtDetrPreprocessingSpec,
    parse_preprocessing_config,
)

CHECKPOINT_FILES = (
    "model.safetensors",
    "config.json",
    "preprocessor_config.json",
)


def _validate_artifact(
    directory: Path,
) -> tuple[RtDetrV2Config, RtDetrPreprocessingSpec]:
    """Validate a converted or upstream checkpoint directory."""

    require_checkpoint_files(directory, CHECKPOINT_FILES)
    config = RtDetrV2Config.from_dict(read_json_object(directory / "config.json"))
    preprocessing = parse_preprocessing_config(
        read_json_object(directory / "preprocessor_config.json")
    )
    return config, preprocessing


__all__ = ["CHECKPOINT_FILES", "_validate_artifact"]
