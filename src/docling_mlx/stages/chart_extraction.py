# SPDX-License-Identifier: Apache-2.0

"""Granite Vision 4.1 chart extraction through Docling's MLX VLM engine."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.datamodel.chart_extraction_options import (
    ChartExtractionModelKind,
    ChartExtractionModelOptions,
)
from docling.datamodel.pipeline_options_vlm_model import ResponseFormat
from docling.datamodel.stage_model_specs import VlmModelSpec
from docling.datamodel.vlm_engine_options import MlxVlmEngineOptions
from docling.models.base_model import BaseItemAndImageEnrichmentModel
from docling.models.inference_engines.vlm import BaseVlmEngine
from docling_core.types.doc import (
    CodeLanguageLabel,
    DescriptionMetaField,
    DoclingDocument,
    NodeItem,
    PictureClassificationMetaField,
    PictureItem,
    PictureMeta,
    TabularChartMetaField,
)
from docling_core.types.doc.document import CodeMetaField
from pydantic import ConfigDict, Field

from docling_mlx.runtime.guards import validate_mlx_accelerator
from docling_mlx.stages._chart_granite import extract_csv_table, extract_python_code
from docling_mlx.stages._granite_vision import (
    build_granite_vision_input,
    create_granite_vision_engine,
    granite_vision_model_spec,
)

_LOG = logging.getLogger(__name__)

_CHART2CSV = "<chart2csv>"
_CHART2SUMMARY = "<chart2summary>"
_CHART2CODE = "<chart2code>"
_SUPPORTED_CHART_TYPES = frozenset({"bar_chart", "pie_chart", "line_chart"})


def _default_granite_vision_chart_spec() -> VlmModelSpec:
    return granite_vision_model_spec(
        name="Granite Vision 4.1 Chart",
        prompt=_CHART2CSV,
        response_format=ResponseFormat.PLAINTEXT,
    )


class MlxChartExtractionModelOptions(ChartExtractionModelOptions):
    """Chart tasks, official model identity, and Docling MLX runtime options."""

    model_config = ConfigDict(extra="forbid")

    model: Literal[ChartExtractionModelKind.GRANITE_VISION_V4] = (
        ChartExtractionModelKind.GRANITE_VISION_V4
    )
    model_spec: VlmModelSpec = Field(default_factory=_default_granite_vision_chart_spec)
    engine_options: MlxVlmEngineOptions = Field(default_factory=MlxVlmEngineOptions)


class MlxGraniteVisionChartExtractionModel(BaseItemAndImageEnrichmentModel):
    """Enrich classified chart pictures with Granite Vision 4.1 outputs.

    This is a directly constructible component because Docling (as of 2.124.0) has no
    chart-extraction factory.  Applications own its placement in an enrichment
    pipeline.
    """

    images_scale = 2.0

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Path | None,
        options: MlxChartExtractionModelOptions,
        accelerator_options: AcceleratorOptions,
        *,
        engine: BaseVlmEngine | None = None,
    ) -> None:
        self.enabled = enabled
        self.options = options.model_copy(deep=True)
        self.engine: BaseVlmEngine | None = None
        self._owns_engine = False

        if not enabled:
            return
        validate_mlx_accelerator(accelerator_options)

        if engine is not None:
            from docling_mlx.stages.granite_vision_engine import MlxGraniteVision41Engine

            if not isinstance(engine, MlxGraniteVision41Engine):
                raise ValueError(
                    "injected Granite Vision chart engine must use the corrected "
                    "Granite Vision 4.1 engine"
                )
            self.engine = engine
            self.engine.initialize()
            return

        if not self._active_prompts():
            return

        self.engine = create_granite_vision_engine(
            engine_options=self.options.engine_options,
            model_spec=self.options.model_spec,
            artifacts_path=artifacts_path,
            accelerator_options=accelerator_options,
        )
        self._owns_engine = True
        self.engine.initialize()

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        del doc
        if not self.enabled or not self._active_prompts():
            return False
        if not isinstance(element, PictureItem) or not isinstance(element.meta, PictureMeta):
            return False
        classification = element.meta.classification
        if not isinstance(classification, PictureClassificationMetaField):
            return False
        return classification.get_main_prediction().class_name in _SUPPORTED_CHART_TYPES

    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
    ) -> Iterable[NodeItem]:
        del doc
        elements = list(element_batch)
        if not self.enabled:
            yield from (element.item for element in elements)
            return

        active_prompts = self._active_prompts()
        if not active_prompts:
            yield from (element.item for element in elements)
            return
        if not elements:
            return
        if self.engine is None:
            raise RuntimeError("Granite Vision chart engine is not initialized")

        inputs = [
            build_granite_vision_input(
                self.options.model_spec,
                element.image,
                prompt=prompt,
            )
            for element in elements
            for prompt in active_prompts
        ]
        outputs = self.engine.predict_batch(inputs)
        if len(outputs) != len(inputs):
            raise RuntimeError(
                "Granite Vision chart engine returned "
                f"{len(outputs)} outputs for {len(inputs)} inputs"
            )

        prompt_count = len(active_prompts)
        for image_index, element in enumerate(elements):
            item = element.item
            if not isinstance(item, PictureItem):
                yield item
                continue
            if not isinstance(item.meta, PictureMeta):
                item.meta = PictureMeta()

            for prompt_index, prompt in enumerate(active_prompts):
                result = outputs[image_index * prompt_count + prompt_index].text
                try:
                    self._apply_result(item.meta, prompt, result)
                except Exception as exc:
                    _LOG.error(
                        "Failed to process Granite Vision chart output [%s] for image %d (%s): %s",
                        prompt,
                        image_index,
                        item.self_ref,
                        exc,
                    )
            yield item

    def cleanup(self) -> None:
        """Release an internally-created engine without touching injected engines."""

        if self._owns_engine and self.engine is not None:
            self.engine.cleanup()
        self.engine = None
        self._owns_engine = False

    def _active_prompts(self) -> list[str]:
        prompts: list[str] = []
        if self.options.chart2csv:
            prompts.append(_CHART2CSV)
        if self.options.chart2summary:
            prompts.append(_CHART2SUMMARY)
        if self.options.chart2code:
            prompts.append(_CHART2CODE)
        return prompts

    def _apply_result(self, meta: PictureMeta, prompt: str, result: str) -> None:
        if prompt == _CHART2CSV:
            meta.tabular_chart = TabularChartMetaField(chart_data=extract_csv_table(result))
        elif prompt == _CHART2SUMMARY:
            meta.description = DescriptionMetaField(text=result)
        elif prompt == _CHART2CODE:
            code = extract_python_code(result)
            if code is not None:
                meta.code = CodeMetaField(text=code, language=CodeLanguageLabel.PYTHON)


__all__ = [
    "MlxGraniteVisionChartExtractionModel",
    "MlxChartExtractionModelOptions",
]
