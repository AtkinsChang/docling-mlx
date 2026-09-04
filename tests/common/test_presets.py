# SPDX-License-Identifier: Apache-2.0

"""Pure preset catalog contracts."""

from __future__ import annotations

import pytest
from docling.datamodel.picture_classification_options import DocumentPictureClassifierOptions
from docling.datamodel.pipeline_options import LayoutObjectDetectionOptions

from docling_mlx.presets import PRESETS, resolve_preset


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_has_one_immutable_engine_identity(name: str) -> None:
    preset = resolve_preset(name)
    assert preset.repo_id and preset.revision
    assert preset.engine_kind.count("/") == 1


def test_project_presets_use_docling_ids() -> None:
    assert {
        "layout_heron_default",
        "layout_heron_101",
        "layout_egret_medium",
        "layout_egret_large",
        "layout_egret_xlarge",
    } <= set(LayoutObjectDetectionOptions.list_preset_ids())
    assert "document_figure_classifier_v2" in DocumentPictureClassifierOptions.list_preset_ids()


def test_legacy_layout_preset_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported MLX preset"):
        resolve_preset("heron-r50")


def test_unknown_preset_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported MLX preset"):
        resolve_preset("missing")
