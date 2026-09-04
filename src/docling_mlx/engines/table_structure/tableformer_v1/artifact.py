# SPDX-License-Identifier: Apache-2.0

"""Runtime contract for the native TableFormer v1 profiles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from docling_mlx._models.tableformer_v1.config import (
    TABLEFORMER_V1_DATA_CELL_TOKEN_IDS,
    TABLEFORMER_V1_TOKENS,
    TableFormerV1Config,
)
from docling_mlx.engines._shared import read_json_object, require_checkpoint_files

from .conversion import upstream_preprocessor_config

CHECKPOINT_FILES = (
    "model.safetensors",
    "config.json",
    "preprocessor_config.json",
    "generation_config.json",
)


@dataclass(frozen=True, slots=True)
class TableFormerV1PreprocessingSpec:
    page_height: int
    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    rescale_factor: float = 1 / 255
    interpolation: str = "bilinear"
    antialias: bool = False
    output_layout: str = "CWH"


TABLEFORMER_V1_PREPROCESSING_SPEC = TableFormerV1PreprocessingSpec(
    page_height=1024,
    image_size=448,
    mean=(0.94247851, 0.94254675, 0.94292611),
    std=(0.17910956, 0.17940403, 0.17931663),
)


@dataclass(frozen=True, slots=True)
class TableFormerV1GenerationSpec:
    max_generation_steps: int


TABLEFORMER_V1_TAG_MAP = MappingProxyType(
    {token: index for index, token in enumerate(TABLEFORMER_V1_TOKENS)}
)


@dataclass(frozen=True, slots=True)
class TableFormerV1Artifact:
    directory: Path
    weights_path: Path
    upstream_weights: bool
    preprocessing: TableFormerV1PreprocessingSpec
    generation: TableFormerV1GenerationSpec
    config: TableFormerV1Config


def _float_triplet(raw: Mapping[str, Any], name: str) -> tuple[float, float, float]:
    value = raw.get(name)
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"TableFormer v1 {name} must contain three values")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise TypeError(f"TableFormer v1 {name} must contain real numbers")
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _parse_preprocessing(raw: Mapping[str, Any]) -> TableFormerV1PreprocessingSpec:
    size = raw.get("size")
    if not isinstance(size, Mapping) or size.get("height") != 448 or size.get("width") != 448:
        raise ValueError("TableFormer v1 preprocessing must resize to 448x448")
    for name in ("do_convert_rgb", "do_resize", "do_rescale", "do_normalize"):
        if raw.get(name) is not True:
            raise ValueError(f"TableFormer v1 preprocessing requires {name}")
    if raw.get("interpolation") != "bilinear" or raw.get("resample") != "bilinear":
        raise ValueError("TableFormer v1 preprocessing requires bilinear resampling")
    if raw.get("antialias") is not False:
        raise ValueError("TableFormer v1 preprocessing must disable antialiasing")
    if raw.get("normalize_before_resize") is not True:
        raise ValueError("TableFormer v1 preprocessing must normalize before resize")
    if raw.get("output_dtype") != "float32" or raw.get("output_layout") != "CWH":
        raise ValueError("TableFormer v1 preprocessing must output float32 CWH")
    factor = raw.get("rescale_factor")
    if isinstance(factor, bool) or not isinstance(factor, (int, float)) or factor != 1 / 255:
        raise ValueError("TableFormer v1 preprocessing requires rescale_factor 1/255")
    if raw.get("page_height") != 1024 or raw.get("page_interpolation") != "area":
        raise ValueError("TableFormer v1 page preprocessing must use 1024px area resize")
    mean = _float_triplet(raw, "image_mean")
    std = _float_triplet(raw, "image_std")
    if mean != TABLEFORMER_V1_PREPROCESSING_SPEC.mean:
        raise ValueError("TableFormer v1 preprocessing mean does not match the checkpoint")
    if std != TABLEFORMER_V1_PREPROCESSING_SPEC.std:
        raise ValueError("TableFormer v1 preprocessing std does not match the checkpoint")
    return TableFormerV1PreprocessingSpec(
        page_height=1024,
        image_size=448,
        mean=mean,
        std=std,
        rescale_factor=float(factor),
        interpolation="bilinear",
        antialias=False,
        output_layout="CWH",
    )


def _parse_generation(raw: Mapping[str, Any]) -> TableFormerV1GenerationSpec:
    steps = raw.get("max_generation_steps")
    if type(steps) is not int or steps != 1024:
        raise ValueError("TableFormer v1 requires exactly 1024 generation steps")
    return TableFormerV1GenerationSpec(max_generation_steps=steps)


def _validate_upstream_checkpoint(directory: Path) -> TableFormerV1Artifact:
    require_checkpoint_files(directory, ("tm_config.json",))
    weights = tuple(directory.glob("tableformer_*.safetensors"))
    if len(weights) != 1:
        raise FileNotFoundError(
            f"Expected exactly one upstream TableFormer v1 checkpoint in {directory}"
        )
    raw = read_json_object(directory / "tm_config.json")
    config = TableFormerV1Config.from_dict(raw)
    preprocessing = _parse_preprocessing(upstream_preprocessor_config(raw))
    predict = raw.get("predict")
    if not isinstance(predict, Mapping):
        raise TypeError("TableFormer v1 tm_config.json must contain predict metadata")
    generation = _parse_generation({"max_generation_steps": predict.get("max_steps")})
    if config.data_cell_token_ids != TABLEFORMER_V1_DATA_CELL_TOKEN_IDS:
        raise ValueError("TableFormer v1 data-cell token map does not match the closed profile")
    return TableFormerV1Artifact(directory, weights[0], True, preprocessing, generation, config)


def validate_tableformer_v1_artifact(directory: Path) -> TableFormerV1Artifact:
    """Validate one converted MLX or upstream TableFormerV1 checkpoint directory."""

    if (directory / "tm_config.json").is_file():
        return _validate_upstream_checkpoint(directory)

    require_checkpoint_files(directory, CHECKPOINT_FILES)
    config = TableFormerV1Config.from_dict(read_json_object(directory / "config.json"))
    preprocessing = _parse_preprocessing(read_json_object(directory / "preprocessor_config.json"))
    generation = _parse_generation(read_json_object(directory / "generation_config.json"))
    if config.data_cell_token_ids != TABLEFORMER_V1_DATA_CELL_TOKEN_IDS:
        raise ValueError("TableFormer v1 data-cell token map does not match the closed profile")
    return TableFormerV1Artifact(
        directory,
        directory / "model.safetensors",
        False,
        preprocessing,
        generation,
        config,
    )


__all__ = [
    "CHECKPOINT_FILES",
    "TABLEFORMER_V1_PREPROCESSING_SPEC",
    "TABLEFORMER_V1_TAG_MAP",
    "TableFormerV1Artifact",
    "TableFormerV1GenerationSpec",
    "TableFormerV1PreprocessingSpec",
    "validate_tableformer_v1_artifact",
]
