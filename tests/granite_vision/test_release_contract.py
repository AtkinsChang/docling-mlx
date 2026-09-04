# SPDX-License-Identifier: Apache-2.0

"""Official Granite Vision 4.1 release qualification through Docling's MLX engine."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import (
    Cluster,
    ItemAndImageEnrichmentElement,
    LayoutPrediction,
    Page,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.vlm_engine_options import MlxVlmEngineOptions
from docling.models.inference_engines.vlm import BaseVlmEngine, VlmEngineType
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

pytestmark = [pytest.mark.mlx, pytest.mark.release]

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/granite_vision"
DEFAULT_ARTIFACTS_ROOT = ROOT / ".artifacts/granite-vlm"
REPO_FOLDER = GRANITE_VISION_4_1_REPO_ID.replace("/", "--")


def _artifacts_root() -> Path:
    if override := os.environ.get("DOCLING_MLX_GRANITE_VISION_ARTIFACTS_ROOT"):
        return Path(override).expanduser()
    return DEFAULT_ARTIFACTS_ROOT


@pytest.fixture(scope="module")
def granite_engine() -> Iterator[BaseVlmEngine]:
    root = _artifacts_root()
    artifact = root / REPO_FOLDER
    required = ["config.json", "preprocessor_config.json", "model.safetensors.index.json"]
    if not artifact.is_dir() or any(not (artifact / name).is_file() for name in required):
        pytest.fail(
            f"provision the official Granite Vision 4.1 artifact at {artifact} or set "
            "DOCLING_MLX_GRANITE_VISION_ARTIFACTS_ROOT; release runs require zero skips"
        )

    options = MlxGraniteVisionTableStructureOptions()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    with patch.object(
        MlxGraniteVision41Engine,
        "download_models",
        side_effect=AssertionError("local Granite qualification must not download"),
    ):
        engine = MlxGraniteVision41Engine(
            MlxVlmEngineOptions(),
            artifacts_path=root,
            model_config=options.model_spec.get_engine_config(VlmEngineType.MLX),
        )
    assert engine.model_config is not None
    assert engine.model_config.repo_id == GRANITE_VISION_4_1_REPO_ID
    assert engine.model_config.revision == GRANITE_VISION_4_1_REVISION
    yield engine
    engine.cleanup()


class FixturePageBackend:
    def __init__(self, image: Image.Image) -> None:
        self.image = image

    def is_valid(self) -> bool:
        return True

    def get_page_image(
        self,
        scale: float,
        cropbox: BoundingBox | None = None,
    ) -> Image.Image:
        assert scale == 1.0
        return self.image if cropbox is None else self.image.crop(cropbox.as_tuple())


def _page(image_name: str, boxes: list[BoundingBox]) -> Page:
    image = Image.open(FIXTURES / image_name).convert("RGB")
    page = Page(page_no=0, size=Size(width=image.width, height=image.height))
    page.predictions.layout = LayoutPrediction(
        clusters=[
            Cluster(id=index, label=DocItemLabel.TABLE, bbox=box) for index, box in enumerate(boxes)
        ]
    )
    page._backend = cast(Any, FixturePageBackend(image))
    return page


def _table_stage(engine: BaseVlmEngine) -> MlxGraniteVisionTableStructureModel:
    options = MlxGraniteVisionTableStructureOptions()
    options.model_spec.max_new_tokens = 384
    return MlxGraniteVisionTableStructureModel(
        enabled=True,
        artifacts_path=_artifacts_root(),
        options=options,
        accelerator_options=AcceleratorOptions(device="auto"),
        engine=engine,
    )


def test_official_model_extracts_simple_merged_and_multiple_tables(
    granite_engine: BaseVlmEngine,
) -> None:
    stage = _table_stage(granite_engine)
    simple = _page("table-simple.png", [BoundingBox(l=0, t=0, r=720, b=420)])
    merged = _page("table-merged.png", [BoundingBox(l=0, t=0, r=720, b=470)])
    multi = _page(
        "table-multi.png",
        [
            BoundingBox(l=20, t=0, r=490, b=500),
            BoundingBox(l=510, t=0, r=980, b=500),
        ],
    )

    predictions = stage.predict_tables(
        ConversionResult.model_construct(timings={}),
        [simple, merged, multi],
    )

    simple_table = predictions[0].table_map[0]
    assert simple_table.num_rows >= 3
    assert {"Region", "Revenue", "North", "South"} <= {
        cell.text for cell in simple_table.table_cells
    }

    merged_table = predictions[1].table_map[0]
    assert {"Quarterly Results", "Quarter", "Revenue", "Q1", "Q2"} <= {
        cell.text for cell in merged_table.table_cells
    }
    assert any(cell.col_span == 2 for cell in merged_table.table_cells)

    assert set(predictions[2].table_map) == {0, 1}
    multi_text = [
        {cell.text for cell in predictions[2].table_map[index].table_cells} for index in (0, 1)
    ]
    assert {"Item", "Value", "A", "B", "10", "20"} <= multi_text[0]
    assert {"Item", "Value", "A", "B", "6", "8"} <= multi_text[1]


def _chart_element(
    document: DoclingDocument,
    image_name: str,
    class_name: str,
) -> ItemAndImageEnrichmentElement:
    picture = document.add_picture()
    picture.meta = PictureMeta(
        classification=PictureClassificationMetaField(
            predictions=[
                PictureClassificationPrediction(
                    class_name=class_name,
                    confidence=1.0,
                    created_by="release-fixture",
                )
            ]
        )
    )
    return ItemAndImageEnrichmentElement(
        item=picture,
        image=Image.open(FIXTURES / image_name).convert("RGB"),
    )


def _chart_stage(
    engine: BaseVlmEngine,
    *,
    summary: bool = False,
    code: bool = False,
) -> MlxGraniteVisionChartExtractionModel:
    options = MlxChartExtractionModelOptions(
        chart2csv=True,
        chart2summary=summary,
        chart2code=code,
    )
    return MlxGraniteVisionChartExtractionModel(
        enabled=True,
        artifacts_path=_artifacts_root(),
        options=options,
        accelerator_options=AcceleratorOptions(device="auto"),
        engine=engine,
    )


def test_official_model_extracts_bar_line_and_pie_chart_data(
    granite_engine: BaseVlmEngine,
) -> None:
    document = DoclingDocument(name="granite-chart-types")
    elements = [
        _chart_element(document, "chart-bar.png", "bar_chart"),
        _chart_element(document, "chart-line.png", "line_chart"),
        _chart_element(document, "chart-pie.png", "pie_chart"),
    ]

    output = list(_chart_stage(granite_engine)(document, elements))

    assert len(output) == 3
    for item in output:
        assert item.meta is not None
        assert item.meta.tabular_chart is not None
        table = item.meta.tabular_chart.chart_data
        assert table.num_rows >= 2
        assert table.num_cols >= 2
        assert table.table_cells


def test_official_model_generates_all_chart_metadata_tasks(
    granite_engine: BaseVlmEngine,
) -> None:
    document = DoclingDocument(name="granite-chart-tasks")
    element = _chart_element(document, "chart-bar.png", "bar_chart")

    [item] = list(_chart_stage(granite_engine, summary=True, code=True)(document, [element]))

    assert item.meta is not None
    assert item.meta.tabular_chart is not None
    assert item.meta.tabular_chart.chart_data.table_cells
    assert item.meta.description is not None
    assert item.meta.description.text.strip()
    assert item.meta.code is not None
    assert item.meta.code.text.strip()
