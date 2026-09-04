# SPDX-License-Identifier: Apache-2.0

"""Application-owned Granite engine lifecycle across table and chart stages."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import (
    Cluster,
    ItemAndImageEnrichmentElement,
    LayoutPrediction,
    Page,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.stage_model_specs import EngineModelConfig
from docling.datamodel.vlm_engine_options import MlxVlmEngineOptions
from docling.models.inference_engines.vlm import (
    BaseVlmEngine,
    VlmEngineInput,
    VlmEngineOutput,
)
from docling_core.types.doc import (
    BoundingBox,
    DocItemLabel,
    DoclingDocument,
    PictureClassificationMetaField,
    PictureClassificationPrediction,
    PictureMeta,
    Size,
)
from PIL import Image

from docling_mlx.stages._granite_vision import (
    GRANITE_VISION_4_1_REPO_ID,
    GRANITE_VISION_4_1_REVISION,
)
from docling_mlx.stages.chart_extraction import (
    MlxChartExtractionModelOptions,
    MlxGraniteVisionChartExtractionModel,
)
from docling_mlx.stages.granite_vision_engine import MlxGraniteVision41Engine
from docling_mlx.stages.table_structure import (
    MlxGraniteVisionTableStructureModel,
    MlxGraniteVisionTableStructureOptions,
)


class SharedFakeEngine(MlxGraniteVision41Engine):
    def __init__(self, batches: Sequence[Sequence[str]]) -> None:
        BaseVlmEngine.__init__(
            self,
            MlxVlmEngineOptions(),
            model_config=EngineModelConfig(
                repo_id=GRANITE_VISION_4_1_REPO_ID,
                revision=GRANITE_VISION_4_1_REVISION,
            ),
        )
        self.pending_batches = [list(batch) for batch in batches]
        self.inputs: list[list[VlmEngineInput]] = []
        self.initialize_calls = 0
        self.cleanup_calls = 0

    def initialize(self) -> None:
        self.initialize_calls += 1
        self._initialized = True

    def predict_batch(self, input_batch: list[VlmEngineInput]) -> list[VlmEngineOutput]:
        self.inputs.append(input_batch)
        return [VlmEngineOutput(text=text) for text in self.pending_batches.pop(0)]

    def cleanup(self) -> None:
        self.cleanup_calls += 1


class PageBackend:
    def is_valid(self) -> bool:
        return True

    def get_page_image(
        self,
        scale: float,
        cropbox: BoundingBox | None = None,
    ) -> Image.Image:
        assert scale == 1.0
        assert cropbox is not None
        return Image.new("RGB", (8, 6), "white")


def test_application_can_share_one_engine_across_table_and_chart_stages() -> None:
    engine = SharedFakeEngine(
        [
            ["<ched>Name</ched><nl/><fcel>A</fcel><nl/>"],
            ["Category,Value\nA,3"],
        ]
    )
    accelerator = AcceleratorOptions(device="auto")
    table_stage = MlxGraniteVisionTableStructureModel(
        enabled=True,
        artifacts_path=Path("/unused"),
        options=MlxGraniteVisionTableStructureOptions(),
        accelerator_options=accelerator,
        engine=engine,
    )
    chart_stage = MlxGraniteVisionChartExtractionModel(
        enabled=True,
        artifacts_path=Path("/unused"),
        options=MlxChartExtractionModelOptions(),
        accelerator_options=accelerator,
        engine=engine,
    )

    page = Page(page_no=0, size=Size(width=100, height=80))
    cluster = Cluster(
        id=7,
        label=DocItemLabel.TABLE,
        bbox=BoundingBox(l=1, t=1, r=9, b=7),
    )
    page.predictions.layout = LayoutPrediction(clusters=[cluster])
    page._backend = cast(Any, PageBackend())
    conversion = ConversionResult.model_construct(timings={})
    table_stage.predict_tables(conversion, [page])

    document = DoclingDocument(name="shared-granite")
    picture = document.add_picture()
    picture.meta = PictureMeta(
        classification=PictureClassificationMetaField(
            predictions=[
                PictureClassificationPrediction(
                    class_name="bar_chart",
                    confidence=1.0,
                    created_by="test",
                )
            ]
        )
    )
    element = ItemAndImageEnrichmentElement(
        item=picture,
        image=Image.new("RGB", (8, 6), "white"),
    )
    list(chart_stage(document, [element]))

    assert engine.initialize_calls == 2
    assert [entry.prompt for batch in engine.inputs for entry in batch] == [
        "<tables_otsl>",
        "<chart2csv>",
    ]
    assert page.predictions.tablestructure is not None
    assert page.predictions.tablestructure.table_map[7].table_cells[0].text == "Name"
    assert picture.meta is not None
    assert picture.meta.tabular_chart is not None
    assert picture.meta.tabular_chart.chart_data.table_cells[-1].text == "3"

    table_stage.cleanup()
    chart_stage.cleanup()
    assert engine.cleanup_calls == 0
    engine.cleanup()
    assert engine.cleanup_calls == 1
