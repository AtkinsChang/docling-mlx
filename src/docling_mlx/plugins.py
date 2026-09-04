# SPDX-License-Identifier: Apache-2.0

"""Docling plugin entry points.

Concrete model adapters are intentionally imported by callbacks, rather than
at module import time, so probing installed plugins never initializes MLX or
pulls reference frameworks into the runtime process.
"""

from __future__ import annotations


def layout_engines() -> dict[str, list[type[object]]]:
    """Expose the native MLX layout stage to Docling."""

    from docling_mlx.stages.layout import MlxLayoutObjectDetectionModel

    return {"layout_engines": [MlxLayoutObjectDetectionModel]}


def table_structure_engines() -> dict[str, list[type[object]]]:
    """Expose project-owned table-structure adapters."""

    from docling_mlx.stages.table_structure import MlxGraniteVisionTableStructureModel
    from docling_mlx.stages.table_structure_v1 import MlxTableFormerV1Model
    from docling_mlx.stages.table_structure_v2 import MlxTableFormerV2Model

    return {
        "table_structure_engines": [
            MlxGraniteVisionTableStructureModel,
            MlxTableFormerV2Model,
            MlxTableFormerV1Model,
        ]
    }


__all__ = ["layout_engines", "table_structure_engines"]
