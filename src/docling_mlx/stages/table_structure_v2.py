# SPDX-License-Identifier: Apache-2.0

"""Native MLX TableFormerV2 table-structure stage for Docling."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import groupby
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Cluster, Page, Table, TableStructurePrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import TableStructureV2Options
from docling.datamodel.spatial import (
    BoundingBoxSpatialIndex,
    has_positive_area,
    ordered_bounding_box,
)
from docling.models.base_table_model import BaseTableStructureModel
from docling.utils.profiling import TimeRecorder
from docling_core.types.doc import BoundingBox, DocItemLabel, TableCell
from docling_core.types.doc.page import TextCell
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from docling_mlx._compat.docling import require_page_backend
from docling_mlx.engines._shared import resolve_artifact_checkpoint
from docling_mlx.engines.table_structure.tableformer_v2.artifact import CHECKPOINT_FILES
from docling_mlx.engines.table_structure.tableformer_v2.engine import (
    TableFormerV2Engine,
    TableFormerV2EngineOptions,
    TableFormerV2Prediction,
)
from docling_mlx.engines.table_structure.tableformer_v2.model_spec import TableFormerV2ModelSpec
from docling_mlx.presets import resolve_preset
from docling_mlx.runtime.guards import validate_mlx_accelerator

_CELL_TOKENS = frozenset({"fcel", "ecel", "ched", "rhed", "srow"})
_TEXTCELL_OVERLAP = 0.3


class MlxTableStructureV2EngineOptions(BaseModel):
    """MLX runtime settings for TableFormerV2."""

    model_config = ConfigDict(extra="forbid")

    warmup: bool = False


class MlxTableStructureV2Options(TableStructureV2Options):
    """Official TableFormerV2 options with MLX model and runtime settings."""

    model_config = ConfigDict(extra="forbid")

    kind: ClassVar[str] = "mlx_tableformer_v2"
    model_spec: TableFormerV2ModelSpec | None = None
    engine_options: MlxTableStructureV2EngineOptions = Field(
        default_factory=MlxTableStructureV2EngineOptions
    )


class MlxTableFormerV2Model(BaseTableStructureModel):
    """Recognize table structure with the project-owned native MLX engine."""

    scale = 2.0

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Path | None,
        options: MlxTableStructureV2Options,
        accelerator_options: AcceleratorOptions,
        enable_remote_services: Literal[False] = False,
    ) -> None:
        del enable_remote_services
        self.enabled = enabled
        self.options = options.model_copy(deep=True)
        self.accelerator_options = accelerator_options
        self.do_cell_matching = self.options.do_cell_matching
        self.engine: TableFormerV2Engine | None = None
        if not enabled:
            return
        validate_mlx_accelerator(accelerator_options)
        preset_name = "tableformer_v2"
        preset = resolve_preset(preset_name)
        if preset.engine_kind != "table_structure/tableformer_v2":
            raise ValueError(f"Preset {preset_name!r} is not a TableFormerV2 preset")
        model_spec = self.options.model_spec or TableFormerV2ModelSpec(
            repo_id=preset.repo_id, revision=preset.revision
        )
        if artifacts_path is not None and model_spec.path is None:
            if model_spec.repo_id is None or model_spec.revision is None:
                raise ValueError("TableFormerV2 model_spec requires repo_id and revision")
            model_spec = TableFormerV2ModelSpec(
                path=resolve_artifact_checkpoint(
                    model_spec.repo_id,
                    model_spec.revision,
                    artifacts_path,
                    files=CHECKPOINT_FILES,
                )
            )
        self.engine = TableFormerV2Engine(model_spec, TableFormerV2EngineOptions())
        self.engine.initialize(warmup=self.options.engine_options.warmup)

    @classmethod
    def get_options_type(cls) -> type[MlxTableStructureV2Options]:
        """Return the exact custom type used by Docling's table factory."""

        return MlxTableStructureV2Options

    @staticmethod
    def _match_texts(
        bboxes: list[BoundingBox],
        text_cells: list[TextCell],
        textcell_overlap: float = _TEXTCELL_OVERLAP,
    ) -> list[str]:
        """Match predicted cells against cluster text in stable source order."""

        if not bboxes:
            return []
        if not text_cells:
            return [""] * len(bboxes)

        spatial_index = BoundingBoxSpatialIndex()
        cell_bboxes: list[BoundingBox] = []
        for cell_idx, text_cell in enumerate(text_cells):
            cell_bbox = text_cell.rect.to_bounding_box()
            cell_bboxes.append(cell_bbox)
            if has_positive_area(cell_bbox):
                spatial_index.insert(cell_idx, cell_bbox)

        matched: list[str] = []
        for bbox in bboxes:
            ordered_bbox = ordered_bounding_box(bbox)
            fragments: list[str] = []
            for cell_idx in sorted(spatial_index.intersection(ordered_bbox)):
                cell_bbox = cell_bboxes[cell_idx]
                if (
                    cell_bbox.get_intersection_bbox(ordered_bbox) is not None
                    and cell_bbox.intersection_over_self(ordered_bbox) > textcell_overlap
                ):
                    fragments.append(text_cells[cell_idx].text.strip())
            matched.append(" ".join(fragments))
        return matched

    @staticmethod
    def _crop_table_image(
        page_image: np.ndarray, bbox: BoundingBox
    ) -> tuple[Image.Image, list[float]]:
        """Apply Docling V2's round-then-scale table crop semantics exactly."""

        table_box = [
            round(bbox.l) * MlxTableFormerV2Model.scale,
            round(bbox.t) * MlxTableFormerV2Model.scale,
            round(bbox.r) * MlxTableFormerV2Model.scale,
            round(bbox.b) * MlxTableFormerV2Model.scale,
        ]
        x1, y1, x2, y2 = (int(value) for value in table_box)
        return Image.fromarray(page_image[y1:y2, x1:x2]).convert("RGB"), table_box

    @staticmethod
    def _build_table_cells(
        otsl_seq: list[str],
        cell_bboxes: Sequence[tuple[float, float, float, float]],
        table_bbox: list[float],
    ) -> tuple[list[dict[str, object]], int, int]:
        """Build Docling table-cell mappings from tags and normalized cell boxes."""

        rows = [
            list(group)
            for is_newline, group in groupby(otsl_seq, lambda token: token == "nl")
            if not is_newline
        ]
        if not rows:
            if cell_bboxes:
                raise RuntimeError("TableFormerV2 emitted bboxes without data-cell tokens")
            return [], 0, 0

        num_rows = len(rows)
        num_cols = max(len(row) for row in rows)
        grid = [row + [""] * (num_cols - len(row)) for row in rows]
        expected_bboxes = sum(token in _CELL_TOKENS for row in grid for token in row)
        if len(cell_bboxes) != expected_bboxes:
            raise RuntimeError(
                "TableFormerV2 emitted a different number of cell bboxes than data-cell tokens"
            )

        left, top, right, bottom = table_bbox
        width = right - left
        height = bottom - top
        cells: list[dict[str, object]] = []
        bbox_idx = 0
        for row_idx, row in enumerate(grid):
            for col_idx, token in enumerate(row):
                if token not in _CELL_TOKENS:
                    continue
                normalized_bbox = cell_bboxes[bbox_idx]
                bbox_idx += 1
                if len(normalized_bbox) != 4:
                    raise RuntimeError("TableFormerV2 emitted an invalid cell bbox")
                cell_box = ordered_bounding_box(
                    BoundingBox(
                        l=left + normalized_bbox[0] * width,
                        t=top + normalized_bbox[1] * height,
                        r=left + normalized_bbox[2] * width,
                        b=top + normalized_bbox[3] * height,
                    )
                ).model_dump(exclude={"coord_origin"})
                col_span = 1
                for next_col in range(col_idx + 1, num_cols):
                    if grid[row_idx][next_col] != "lcel":
                        break
                    col_span += 1
                row_span = 1
                for next_row in range(row_idx + 1, num_rows):
                    if grid[next_row][col_idx] != "ucel":
                        break
                    row_span += 1
                cells.append(
                    {
                        "bbox": cell_box,
                        "row_span": row_span,
                        "col_span": col_span,
                        "start_row_offset_idx": row_idx,
                        "end_row_offset_idx": row_idx + row_span,
                        "start_col_offset_idx": col_idx,
                        "end_col_offset_idx": col_idx + col_span,
                        "column_header": token == "ched",
                        "row_header": token == "rhed",
                        "row_section": token == "srow",
                    }
                )
        return cells, num_rows, num_cols

    def _prediction_to_table(
        self,
        prediction: TableFormerV2Prediction,
        *,
        table_cluster: Cluster,
        page: Page,
        table_box: list[float],
        crop_size: tuple[int, int],
    ) -> Table:
        otsl_seq = list(prediction.otsl_tokens)
        table_bbox = [value / self.scale for value in table_box]
        width, height = crop_size
        normalized_boxes = tuple(
            (left / width, top / height, right / width, bottom / height)
            for left, top, right, bottom in prediction.cell_bboxes
        )
        cell_data, num_rows, num_cols = self._build_table_cells(
            otsl_seq,
            normalized_boxes,
            table_bbox,
        )
        cell_matches = [
            (element, BoundingBox.model_validate(element["bbox"])) for element in cell_data
        ]
        cluster_cells = table_cluster.cells
        matched_texts = (
            self._match_texts([bbox for _, bbox in cell_matches], cluster_cells)
            if self.do_cell_matching
            else [""] * len(cell_matches)
        )
        backend = require_page_backend(page, "MLX table structure stage")
        for (element, bbox), matched_text in zip(cell_matches, matched_texts, strict=True):
            text = matched_text if self.do_cell_matching else ""
            if not text.strip():
                text = backend.get_text_in_rect(bbox)
            # Docling V2 writes this under bbox["token"], which is not part of the
            # TableCell schema.  The native adapter places actual text on the cell.
            element["text"] = text

        table_cells = [TableCell.model_validate(element) for element in cell_data]
        return Table(
            otsl_seq=otsl_seq,
            table_cells=table_cells,
            num_rows=num_rows,
            num_cols=num_cols,
            id=table_cluster.id,
            page_no=page.page_no,
            cluster=table_cluster,
            label=table_cluster.label,
        )

    def predict_tables(
        self,
        conv_res: ConversionResult,
        pages: Sequence[Page],
    ) -> Sequence[TableStructurePrediction]:
        """Predict valid table clusters while preserving Docling V2 page semantics."""

        predictions: list[TableStructurePrediction] = []
        for page in pages:
            backend = require_page_backend(page, "MLX table structure stage")
            if not backend.is_valid():
                existing = page.predictions.tablestructure or TableStructurePrediction()
                page.predictions.tablestructure = existing
                predictions.append(existing)
                continue

            with TimeRecorder(conv_res, "table_structure"):
                if page.predictions.layout is None:
                    raise RuntimeError("MLX table structure stage requires layout predictions")
                if page.size is None:
                    raise RuntimeError("MLX table structure stage requires page size")
                table_prediction = TableStructurePrediction()
                page.predictions.tablestructure = table_prediction
                clusters = [
                    cluster
                    for cluster in page.predictions.layout.clusters
                    if cluster.label in {DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX}
                ]
                if not clusters or not self.enabled:
                    predictions.append(table_prediction)
                    continue
                if self.engine is None:
                    raise RuntimeError("TableFormerV2 engine is not initialized")

                page_image = np.asarray(page.get_image(scale=self.scale))
                crops_and_boxes = [
                    (cluster, *self._crop_table_image(page_image, cluster.bbox))
                    for cluster in clusters
                ]
                outputs = self.engine.predict([crop for _, crop, _ in crops_and_boxes])
                if len(outputs) != len(crops_and_boxes):
                    raise RuntimeError(
                        "TableFormerV2 engine returned a different number of outputs than inputs"
                    )
                for (cluster, crop, table_box), output in zip(
                    crops_and_boxes, outputs, strict=True
                ):
                    table_prediction.table_map[cluster.id] = self._prediction_to_table(
                        output,
                        table_cluster=cluster,
                        page=page,
                        table_box=table_box,
                        crop_size=crop.size,
                    )
                predictions.append(table_prediction)
        return predictions


__all__ = [
    "MlxTableFormerV2Model",
    "MlxTableStructureV2EngineOptions",
    "MlxTableStructureV2Options",
]
