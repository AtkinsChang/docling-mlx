# SPDX-License-Identifier: Apache-2.0

"""Immutable model identities selected by Docling-facing adaptors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class ModelPreset:
    """Data needed to construct one native engine from a published checkpoint."""

    engine_kind: str
    repo_id: str
    revision: str
    engine_options: Mapping[str, object] = MappingProxyType({})


PRESETS: Mapping[str, ModelPreset] = MappingProxyType(
    {
        "layout_heron_default": ModelPreset(
            "object_detection/rt_detr_v2",
            "atkinschang/docling-layout-heron-mlx",
            "0e868578f271f8f6a6c907fb4f3aa723143d85f4",
        ),
        "layout_heron_101": ModelPreset(
            "object_detection/rt_detr_v2",
            "atkinschang/docling-layout-heron-101-mlx",
            "42b86329d788e310562a608190a1a5e54ef79bda",
        ),
        "layout_egret_medium": ModelPreset(
            "object_detection/dfine",
            "atkinschang/docling-layout-egret-medium-mlx",
            "a500e62df89b586b716111709ec8626aa28072c0",
        ),
        "layout_egret_large": ModelPreset(
            "object_detection/dfine",
            "atkinschang/docling-layout-egret-large-mlx",
            "3f75c0befdd32a4ba4c1f42720cfcf95e3be04db",
        ),
        "layout_egret_xlarge": ModelPreset(
            "object_detection/dfine",
            "atkinschang/docling-layout-egret-xlarge-mlx",
            "0df67dc77ff5794e5ebc75adc2bff8b75c08e2b2",
        ),
        "document_figure_classifier_v2": ModelPreset(
            "image_classification/efficientnet",
            "atkinschang/DocumentFigureClassifier-v2.5-MLX",
            "673c86192056a6f0e6c6c295647ac3232fde5f34",
        ),
        "tableformer_v1_accurate": ModelPreset(
            "table_structure/tableformer_v1",
            "atkinschang/TableFormer-MLX",
            "28bb5171682eed2a7d3c0a2f29f80f32dcccc18e",
            MappingProxyType({"checkpoint_subdirectory": "accurate"}),
        ),
        "tableformer_v1_fast": ModelPreset(
            "table_structure/tableformer_v1",
            "atkinschang/TableFormer-MLX",
            "28bb5171682eed2a7d3c0a2f29f80f32dcccc18e",
            MappingProxyType({"checkpoint_subdirectory": "fast"}),
        ),
        "tableformer_v2": ModelPreset(
            "table_structure/tableformer_v2",
            "atkinschang/TableFormerV2-MLX",
            "79a9ab108f1bf6882c64226b5794886ffd972c18",
        ),
    }
)


def resolve_preset(name: str) -> ModelPreset:
    """Return one known immutable model identity."""

    try:
        return PRESETS[name]
    except KeyError as error:
        raise ValueError(f"Unsupported MLX preset: {name!r}") from error


__all__ = ["ModelPreset", "PRESETS", "resolve_preset"]
