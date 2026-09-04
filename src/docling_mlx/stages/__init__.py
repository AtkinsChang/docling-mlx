# SPDX-License-Identifier: Apache-2.0

"""Docling stage adaptors. Applications own pipeline integration."""

from docling_mlx.stages._granite_vision import (
    GRANITE_VISION_4_1_REPO_ID,
    GRANITE_VISION_4_1_REVISION,
)
from docling_mlx.stages.chart_extraction import (
    MlxChartExtractionModelOptions,
    MlxGraniteVisionChartExtractionModel,
)
from docling_mlx.stages.layout import (
    MlxLayoutObjectDetectionModel,
    MlxLayoutObjectDetectionOptions,
    MlxObjectDetectionEngineOptions,
)
from docling_mlx.stages.picture_classification import (
    MlxDocumentPictureClassifier,
    MlxDocumentPictureClassifierOptions,
    MlxImageClassificationEngineOptions,
)
from docling_mlx.stages.table_structure import (
    MlxGraniteVisionTableStructureModel,
    MlxGraniteVisionTableStructureOptions,
)
from docling_mlx.stages.table_structure_v1 import (
    MlxTableFormerV1Model,
    MlxTableStructureEngineOptions,
    MlxTableStructureOptions,
)
from docling_mlx.stages.table_structure_v2 import (
    MlxTableFormerV2Model,
    MlxTableStructureV2EngineOptions,
    MlxTableStructureV2Options,
)

__all__ = [
    "MlxDocumentPictureClassifier",
    "MlxDocumentPictureClassifierOptions",
    "MlxImageClassificationEngineOptions",
    "MlxLayoutObjectDetectionModel",
    "MlxLayoutObjectDetectionOptions",
    "MlxObjectDetectionEngineOptions",
    "MlxGraniteVisionChartExtractionModel",
    "MlxChartExtractionModelOptions",
    "GRANITE_VISION_4_1_REPO_ID",
    "GRANITE_VISION_4_1_REVISION",
    "MlxGraniteVisionTableStructureModel",
    "MlxGraniteVisionTableStructureOptions",
    "MlxTableFormerV1Model",
    "MlxTableStructureEngineOptions",
    "MlxTableStructureOptions",
    "MlxTableFormerV2Model",
    "MlxTableStructureV2EngineOptions",
    "MlxTableStructureV2Options",
]
