# SPDX-License-Identifier: Apache-2.0

"""Native MLX TableFormerV1 table-structure stage for Docling."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Literal, cast

import numpy as np
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Cluster, Page, Table, TableStructurePrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import TableFormerMode, TableStructureOptions
from docling.models.base_table_model import BaseTableStructureModel
from docling.utils.profiling import TimeRecorder
from docling_core.types.doc import BoundingBox, DocItemLabel, TableCell
from docling_core.types.doc.page import BoundingRectangle, TextCellUnit
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from docling_mlx._compat.docling import require_page_backend
from docling_mlx.engines._shared import resolve_artifact_checkpoint
from docling_mlx.engines.table_structure.tableformer_v1.artifact import CHECKPOINT_FILES
from docling_mlx.engines.table_structure.tableformer_v1.engine import (
    TableFormerV1Engine,
    TableFormerV1EngineOptions,
    TableFormerV1Prediction,
)
from docling_mlx.engines.table_structure.tableformer_v1.model_spec import TableFormerV1ModelSpec
from docling_mlx.presets import resolve_preset
from docling_mlx.runtime.guards import validate_mlx_accelerator


class MlxTableStructureEngineOptions(BaseModel):
    """MLX runtime settings for TableFormerV1."""

    model_config = ConfigDict(extra="forbid")

    warmup: bool = False


class MlxTableStructureOptions(TableStructureOptions):
    """Official TableFormerV1 options with MLX model and runtime settings."""

    model_config = ConfigDict(extra="forbid")

    kind: ClassVar[str] = "mlx_tableformer"
    model_spec: TableFormerV1ModelSpec | None = None
    engine_options: MlxTableStructureEngineOptions = Field(
        default_factory=MlxTableStructureEngineOptions
    )


class MlxTableFormerV1Model(BaseTableStructureModel):
    """Recognize table structure with native TableFormerV1."""

    scale = 2.0

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Path | None,
        options: MlxTableStructureOptions,
        accelerator_options: AcceleratorOptions,
        enable_remote_services: Literal[False] = False,
    ) -> None:
        del enable_remote_services
        self.enabled = enabled
        self.options = options.model_copy(deep=True)
        self.accelerator_options = accelerator_options
        self.do_cell_matching = self.options.do_cell_matching
        self.engine: TableFormerV1Engine | None = None
        if not enabled:
            return
        validate_mlx_accelerator(accelerator_options)
        self._validate_options()
        preset_name = f"tableformer_v1_{self.options.mode.value}"
        preset = resolve_preset(preset_name)
        if preset.engine_kind != "table_structure/tableformer_v1":
            raise ValueError(f"Preset {preset_name!r} is not a TableFormerV1 preset")
        model_spec = self.options.model_spec or TableFormerV1ModelSpec(
            repo_id=preset.repo_id, revision=preset.revision
        )
        engine_options = TableFormerV1EngineOptions(
            checkpoint_subdirectory=cast(
                str | None, preset.engine_options.get("checkpoint_subdirectory")
            )
        )
        if artifacts_path is not None and model_spec.path is None:
            if model_spec.repo_id is None or model_spec.revision is None:
                raise ValueError("TableFormerV1 model_spec requires repo_id and revision")
            patterns = tuple(
                f"{engine_options.checkpoint_subdirectory}/{name}" for name in CHECKPOINT_FILES
            )
            model_spec = TableFormerV1ModelSpec(
                path=resolve_artifact_checkpoint(
                    model_spec.repo_id,
                    model_spec.revision,
                    artifacts_path,
                    files=patterns,
                )
            )
        self.engine = TableFormerV1Engine(model_spec, engine_options)
        self.engine.initialize(warmup=self.options.engine_options.warmup)

    @classmethod
    def get_options_type(cls) -> type[MlxTableStructureOptions]:
        return MlxTableStructureOptions

    def _validate_options(self) -> None:
        if self.options.mode not in {TableFormerMode.ACCURATE, TableFormerMode.FAST}:
            raise ValueError(f"Unsupported TableFormerV1 mode: {self.options.mode!r}")

    @classmethod
    def _table_box(cls, cluster: Cluster) -> list[float]:
        return [
            round(cluster.bbox.l) * cls.scale,
            round(cluster.bbox.t) * cls.scale,
            round(cluster.bbox.r) * cls.scale,
            round(cluster.bbox.b) * cls.scale,
        ]

    @classmethod
    def _tokens_for_cluster(cls, page: Page, cluster: Cluster) -> list[dict[str, object]]:
        backend = require_page_backend(page, "MLX table structure stage")
        segmented_page = backend.get_segmented_page()
        if segmented_page is not None:
            cells = segmented_page.get_cells_in_bbox(
                cell_unit=TextCellUnit.WORD,
                bbox=cluster.bbox,
            )
            if not cells:
                cells = cluster.cells
        else:
            cells = cluster.cells

        tokens: list[dict[str, object]] = []
        for cell in cells:
            if not cell.text.strip():
                continue
            copied = copy.deepcopy(cell)
            copied.rect = BoundingRectangle.from_bounding_box(
                copied.rect.to_bounding_box().scaled(scale=cls.scale)
            )
            tokens.append(
                {
                    "id": copied.index,
                    "text": copied.text,
                    "bbox": copied.rect.to_bounding_box().model_dump(),
                }
            )
        return tokens

    def _prediction_to_table(
        self,
        prediction: TableFormerV1Prediction,
        *,
        table_cluster: Cluster,
        page: Page,
        table_box: list[float],
        tokens: list[dict[str, object]],
        crop_size: tuple[int, int],
    ) -> Table:
        from docling_mlx.engines.table_structure.tableformer_v1.postprocessing import (
            postprocess_prediction,
        )

        width, height = crop_size
        normalized_boxes = tuple(
            (left / width, top / height, right / width, bottom / height)
            for left, top, right, bottom in prediction.cell_bboxes
        )
        otsl_seq = list(prediction.otsl_tokens)
        cell_data, num_rows, num_cols = postprocess_prediction(
            otsl_seq,
            normalized_boxes,
            prediction.bbox_classes,
            table_bbox=table_box,
            pdf_cells=tokens,
            do_cell_matching=self.do_cell_matching,
        )
        backend = require_page_backend(page, "MLX table structure stage")
        table_cells: list[TableCell] = []
        for element in cell_data:
            bbox = BoundingBox.model_validate(element["bbox"])
            if self.do_cell_matching:
                text_bboxes = element.get("text_cell_bboxes")
                if not isinstance(text_bboxes, list):
                    raise TypeError("TableFormerV1 matching returned invalid text boxes")
                text = " ".join(
                    str(text_bbox.get("token", "")).strip()
                    for text_bbox in text_bboxes
                    if isinstance(text_bbox, dict)
                    if str(text_bbox.get("token", "")).strip()
                )
            else:
                text = backend.get_text_in_rect(bbox.scaled(1 / self.scale))
            element["text"] = text
            element["bbox"] = bbox.scaled(1 / self.scale)
            table_cells.append(TableCell.model_validate(element))

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
                    raise RuntimeError("TableFormerV1 engine is not initialized")

                from docling_mlx.engines.table_structure.tableformer_v1.preprocessing import (
                    resize_page_image,
                )

                page_image, page_scale = resize_page_image(
                    np.asarray(page.get_image(scale=self.scale)),
                    self.engine.artifact.preprocessing,
                )
                table_boxes = [self._table_box(cluster) for cluster in clusters]
                crops = [
                    Image.fromarray(
                        page_image[
                            round(table_box[1] * page_scale) : round(table_box[3] * page_scale),
                            round(table_box[0] * page_scale) : round(table_box[2] * page_scale),
                        ]
                    ).convert("RGB")
                    for table_box in table_boxes
                ]
                outputs = self.engine.predict(crops)
                if len(outputs) != len(clusters):
                    raise RuntimeError(
                        "TableFormerV1 engine returned a different number of outputs than inputs"
                    )
                for cluster, table_box, crop, output in zip(
                    clusters, table_boxes, crops, outputs, strict=True
                ):
                    table_prediction.table_map[cluster.id] = self._prediction_to_table(
                        output,
                        table_cluster=cluster,
                        page=page,
                        table_box=table_box,
                        tokens=self._tokens_for_cluster(page, cluster),
                        crop_size=crop.size,
                    )
                predictions.append(table_prediction)
        return predictions


__all__ = [
    "MlxTableFormerV1Model",
    "MlxTableStructureEngineOptions",
    "MlxTableStructureOptions",
]
