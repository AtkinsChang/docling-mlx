# SPDX-License-Identifier: Apache-2.0

"""Granite Vision 4.1 table structure adapter for Docling's MLX VLM engine."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Literal

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page, Table, TableStructurePrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import GraniteVisionTableStructureOptions
from docling.datamodel.pipeline_options_vlm_model import ResponseFormat
from docling.datamodel.stage_model_specs import VlmModelSpec
from docling.datamodel.vlm_engine_options import MlxVlmEngineOptions
from docling.models.base_table_model import BaseTableStructureModel
from docling.models.inference_engines.vlm.base import (
    BaseVlmEngine,
    VlmEngineOptionsMixin,
    VlmEngineType,
)
from docling.utils.profiling import TimeRecorder
from docling_core.types.doc import DocItemLabel
from pydantic import ConfigDict, Field

from docling_mlx._compat.docling import require_page_backend
from docling_mlx.runtime.guards import validate_mlx_accelerator
from docling_mlx.stages._granite_vision import (
    build_granite_vision_input,
    create_granite_vision_engine,
    granite_vision_model_spec,
)
from docling_mlx.stages._otsl import parse_otsl_output

_log = logging.getLogger(__name__)

GRANITE_TABLE_PROMPT = "<tables_otsl>"


def _default_model_spec() -> VlmModelSpec:
    return granite_vision_model_spec(
        name="Granite Vision 4.1 Table",
        prompt=GRANITE_TABLE_PROMPT,
        response_format=ResponseFormat.OTSL,
    )


class MlxGraniteVisionTableStructureOptions(
    VlmEngineOptionsMixin, GraniteVisionTableStructureOptions
):
    """Serializable model and MLX runtime settings for Granite table extraction."""

    model_config = ConfigDict(extra="forbid")

    kind: ClassVar[str] = "mlx_granite_vision_table"
    model_spec: VlmModelSpec = Field(default_factory=_default_model_spec)
    engine_options: MlxVlmEngineOptions = Field(default_factory=MlxVlmEngineOptions)


class MlxGraniteVisionTableStructureModel(BaseTableStructureModel):
    """Populate Docling table predictions through the official MLX VLM engine."""

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Path | None,
        options: MlxGraniteVisionTableStructureOptions,
        accelerator_options: AcceleratorOptions,
        enable_remote_services: Literal[False] = False,
        *,
        engine: BaseVlmEngine | None = None,
    ) -> None:
        self.enabled = enabled
        self.options = options.model_copy(deep=True)
        self.accelerator_options = accelerator_options
        self.engine: BaseVlmEngine | None = None
        self._owns_engine = False

        if not enabled:
            return
        validate_mlx_accelerator(accelerator_options)
        self._validate_options()

        if engine is not None:
            from docling_mlx.stages.granite_vision_engine import MlxGraniteVision41Engine

            if not isinstance(engine, MlxGraniteVision41Engine):
                raise ValueError(
                    "injected Granite Vision table engine must use the corrected "
                    "Granite Vision 4.1 engine"
                )
            self.engine = engine
        else:
            self.engine = create_granite_vision_engine(
                engine_options=self.options.engine_options,
                model_spec=self.options.model_spec,
                artifacts_path=artifacts_path,
                accelerator_options=accelerator_options,
            )
            self._owns_engine = True
        self.engine.initialize()

    @classmethod
    def get_options_type(cls) -> type[MlxGraniteVisionTableStructureOptions]:
        """Return the exact options type used by Docling's table factory."""

        return MlxGraniteVisionTableStructureOptions

    def _validate_options(self) -> None:
        if self.options.engine_options.engine_type != VlmEngineType.MLX:
            raise ValueError("Granite Vision table extraction requires the MLX VLM engine")
        if not self.options.model_spec.is_engine_supported(VlmEngineType.MLX):
            raise ValueError("model_spec does not support the MLX VLM engine")
        if self.options.model_spec.response_format != ResponseFormat.OTSL:
            raise ValueError("Granite Vision table extraction requires an OTSL response format")

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

                valid_pairs = []
                for cluster in clusters:
                    crop = page.get_image(scale=1.0, cropbox=cluster.bbox)
                    if crop is not None:
                        valid_pairs.append((cluster, crop))
                if not valid_pairs:
                    predictions.append(table_prediction)
                    continue

                if self.engine is None:
                    raise RuntimeError("Granite Vision table engine is not initialized")
                inputs = [
                    build_granite_vision_input(self.options.model_spec, image)
                    for _, image in valid_pairs
                ]
                outputs = self.engine.predict_batch(inputs)
                if len(outputs) != len(valid_pairs):
                    raise RuntimeError(
                        "Granite Vision table engine returned a different number of outputs "
                        "than inputs"
                    )

                for (cluster, _), output in zip(valid_pairs, outputs, strict=True):
                    try:
                        otsl_seq, table_cells, num_rows, num_cols = parse_otsl_output(output.text)
                    except Exception as exc:
                        _log.warning(
                            "Failed to parse OTSL output for page %s table cluster %s: %s",
                            page.page_no,
                            cluster.id,
                            exc,
                        )
                        otsl_seq, table_cells, num_rows, num_cols = [], [], 0, 0

                    table_prediction.table_map[cluster.id] = Table(
                        otsl_seq=otsl_seq,
                        table_cells=table_cells,
                        num_rows=num_rows,
                        num_cols=num_cols,
                        id=cluster.id,
                        page_no=page.page_no,
                        cluster=cluster,
                        label=cluster.label,
                    )

                predictions.append(table_prediction)

        return predictions

    def cleanup(self) -> None:
        """Release only an engine created and owned by this stage."""

        if self._owns_engine and self.engine is not None:
            self.engine.cleanup()
        self.engine = None
        self._owns_engine = False


__all__ = [
    "GRANITE_TABLE_PROMPT",
    "MlxGraniteVisionTableStructureModel",
    "MlxGraniteVisionTableStructureOptions",
]
