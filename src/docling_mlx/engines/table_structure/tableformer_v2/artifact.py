# SPDX-License-Identifier: Apache-2.0

"""Runtime contract for the native FP32 TableFormerV2 artifact."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docling_mlx._models.tableformer_v2.config import (
    TABLEFORMER_V2_TOKENS,
    TableFormerV2Config,
)
from docling_mlx.engines._shared import read_json_object, require_checkpoint_files

CHECKPOINT_FILES = (
    "model.safetensors",
    "config.json",
    "preprocessor_config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
UPSTREAM_CHECKPOINT_FILES = tuple(
    name for name in CHECKPOINT_FILES if name != "preprocessor_config.json"
)


@dataclass(frozen=True, slots=True)
class TableFormerV2PreprocessingSpec:
    size: tuple[int, int]
    interpolation: str
    antialias: bool
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class TableFormerV2GenerationSpec:
    max_generation_steps: int


@dataclass(frozen=True, slots=True)
class TableFormerV2TokenMap:
    tokens: tuple[str, ...]
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    data_cell_token_ids: frozenset[int]


@dataclass(frozen=True, slots=True)
class TableFormerV2Artifact:
    directory: Path
    weights_path: Path
    upstream_weights: bool
    config: TableFormerV2Config
    preprocessing: TableFormerV2PreprocessingSpec
    generation: TableFormerV2GenerationSpec
    token_map: TableFormerV2TokenMap


def _float_triplet(raw: Mapping[str, Any], name: str) -> tuple[float, float, float]:
    value = raw.get(name)
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"TableFormerV2 {name} must contain three values")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise TypeError(f"TableFormerV2 {name} must contain real numbers")
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _parse_preprocessing(raw: Mapping[str, Any]) -> TableFormerV2PreprocessingSpec:
    size = raw.get("size")
    if not isinstance(size, Mapping) or not {"height", "width"} <= set(size):
        raise ValueError("TableFormerV2 preprocessing must specify height and width")
    if size.get("height") != 448 or size.get("width") != 448:
        raise ValueError("TableFormerV2 preprocessing must resize to 448x448")
    spec = TableFormerV2PreprocessingSpec(
        size=(448, 448),
        interpolation="bilinear",
        antialias=True,
        mean=_float_triplet(raw, "image_mean"),
        std=_float_triplet(raw, "image_std"),
    )
    if spec.mean != (0.485, 0.456, 0.406) or spec.std != (0.229, 0.224, 0.225):
        raise ValueError("TableFormerV2 preprocessing requires ImageNet normalization")
    return spec


def _parse_generation(
    raw: Mapping[str, Any], *, upstream: bool = False
) -> TableFormerV2GenerationSpec:
    steps = raw.get("max_length") if upstream else raw.get("max_generation_steps")
    if type(steps) is not int or steps != 512:
        raise ValueError("TableFormerV2 requires exactly 512 generation steps")
    return TableFormerV2GenerationSpec(max_generation_steps=512)


def _special_token_content(raw: Mapping[str, Any], name: str) -> str:
    value = raw.get(name)
    if not isinstance(value, Mapping) or not isinstance(value.get("content"), str):
        raise TypeError(f"TableFormerV2 special_tokens_map {name} must contain content")
    return value["content"]


def _parse_token_map(
    tokenizer: Mapping[str, Any],
    tokenizer_config: Mapping[str, Any],
    special_tokens: Mapping[str, Any],
    data_cell_token_ids: frozenset[int],
) -> TableFormerV2TokenMap:
    raw_tokens = tokenizer.get("added_tokens")
    if not isinstance(raw_tokens, list):
        raise TypeError("TableFormerV2 tokenizer added_tokens must be a list")
    token_ids: dict[int, str] = {}
    for token in raw_tokens:
        if not isinstance(token, Mapping) or type(token.get("id")) is not int:
            raise TypeError("TableFormerV2 tokenizer entries must have integer IDs")
        if not isinstance(token.get("content"), str):
            raise TypeError("TableFormerV2 tokenizer entries must have string content")
        if token["id"] in token_ids:
            raise ValueError("TableFormerV2 tokenizer contains duplicate token IDs")
        token_ids[token["id"]] = token["content"]
    tokens = tuple(token_ids.get(index) for index in range(len(TABLEFORMER_V2_TOKENS)))
    if tokens != TABLEFORMER_V2_TOKENS or len(token_ids) != len(tokens):
        raise ValueError("TableFormerV2 token IDs must match the exact 13-token contract")
    special_ids = {
        name: tokens.index(_special_token_content(special_tokens, name))
        for name in ("pad_token", "unk_token", "bos_token", "eos_token")
    }
    if any(
        tokenizer_config.get(name) != tokens[token_id] for name, token_id in special_ids.items()
    ):
        raise ValueError("TableFormerV2 tokenizer metadata disagrees with special token map")
    result = TableFormerV2TokenMap(
        tokens=tokens,
        pad_token_id=special_ids["pad_token"],
        bos_token_id=special_ids["bos_token"],
        eos_token_id=special_ids["eos_token"],
        data_cell_token_ids=data_cell_token_ids,
    )
    if (
        result.pad_token_id != 0
        or special_ids["unk_token"] != 1
        or result.bos_token_id != 2
        or result.eos_token_id != 3
    ):
        raise ValueError("TableFormerV2 token map does not match the closed profile")
    return result


def validate_tableformer_v2_artifact(directory: Path) -> TableFormerV2Artifact:
    """Validate a converted MLX or upstream TableFormerV2 checkpoint directory."""

    upstream = not (directory / "preprocessor_config.json").is_file()
    require_checkpoint_files(directory, UPSTREAM_CHECKPOINT_FILES if upstream else CHECKPOINT_FILES)
    raw_config = read_json_object(directory / "config.json")
    raw_generation = read_json_object(directory / "generation_config.json")
    raw_special_tokens = read_json_object(directory / "special_tokens_map.json")
    raw_tokenizer = read_json_object(directory / "tokenizer.json")
    raw_tokenizer_config = read_json_object(directory / "tokenizer_config.json")
    config = TableFormerV2Config.from_dict(raw_config)
    preprocessing = (
        TableFormerV2PreprocessingSpec(
            size=(448, 448),
            interpolation="bilinear",
            antialias=True,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        if upstream
        else _parse_preprocessing(read_json_object(directory / "preprocessor_config.json"))
    )
    generation = _parse_generation(raw_generation, upstream=upstream)
    token_map = _parse_token_map(
        raw_tokenizer,
        raw_tokenizer_config,
        raw_special_tokens,
        config.data_cell_token_ids,
    )
    if (
        config.pad_token_id != token_map.pad_token_id
        or config.eos_token_id != token_map.eos_token_id
    ):
        raise ValueError("TableFormerV2 config and token map disagree")
    if config.data_cell_token_ids != token_map.data_cell_token_ids:
        raise ValueError("TableFormerV2 config and data-cell token map disagree")

    return TableFormerV2Artifact(
        directory,
        directory / "model.safetensors",
        upstream,
        config,
        preprocessing,
        generation,
        token_map,
    )


__all__ = [
    "CHECKPOINT_FILES",
    "UPSTREAM_CHECKPOINT_FILES",
    "TableFormerV2Artifact",
    "validate_tableformer_v2_artifact",
]
