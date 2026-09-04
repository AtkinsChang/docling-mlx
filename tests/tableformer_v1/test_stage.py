# SPDX-License-Identifier: Apache-2.0

"""Docling adaptor contracts for native TableFormerV1."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Cluster, LayoutPrediction, Page, TableStructurePrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import TableFormerMode
from docling_core.types.doc import BoundingBox, DocItemLabel, Size
from docling_core.types.doc.base import CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell, TextCellUnit
from PIL import Image

from docling_mlx.engines.table_structure.tableformer_v1 import (
    TableFormerV1ModelSpec,
    TableFormerV1Prediction,
)
from docling_mlx.stages.table_structure_v1 import (
    MlxTableFormerV1Model,
    MlxTableStructureEngineOptions,
    MlxTableStructureOptions,
)


class SegmentedPage:
    def __init__(self, cells: list[TextCell]) -> None:
        self.cells = cells

    def get_cells_in_bbox(self, *, cell_unit: TextCellUnit, bbox: BoundingBox) -> list[TextCell]:
        del cell_unit, bbox
        return self.cells


class PageBackend:
    def __init__(
        self,
        *,
        valid: bool = True,
        segmented: SegmentedPage | None = None,
        text: str = "backend text",
    ) -> None:
        self.valid = valid
        self.segmented = segmented
        self.text = text
        self.images: list[float] = []
        self.text_boxes: list[BoundingBox] = []
        self.image = Image.fromarray(np.arange(20 * 20 * 3, dtype=np.uint8).reshape(20, 20, 3))

    def is_valid(self) -> bool:
        return self.valid

    def get_page_image(self, scale: float, cropbox: BoundingBox | None = None) -> Image.Image:
        assert cropbox is None
        self.images.append(scale)
        return self.image

    def get_segmented_page(self) -> SegmentedPage | None:
        return self.segmented

    def get_text_in_rect(self, bbox: BoundingBox) -> str:
        self.text_boxes.append(bbox)
        return self.text


class FakeEngine:
    next_outputs: list[TableFormerV1Prediction] = []
    instances: list[FakeEngine] = []

    def __init__(self, model_spec: object, options: object) -> None:
        self.model_spec = model_spec
        self.options = options
        self.batches: list[list[Image.Image]] = []
        self.outputs = list(self.next_outputs)
        self.initialize_calls: list[bool] = []
        self.artifact = SimpleNamespace(preprocessing=SimpleNamespace(page_height=1024))
        self.instances.append(self)

    def initialize(self, *, warmup: bool = False) -> None:
        self.initialize_calls.append(warmup)

    def predict(self, images: list[Image.Image]) -> list[TableFormerV1Prediction]:
        self.batches.append(images)
        return list(self.outputs)


@pytest.fixture(autouse=True)
def _runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeEngine.next_outputs = []
    FakeEngine.instances = []
    monkeypatch.setattr("docling_mlx.stages.table_structure_v1.TableFormerV1Engine", FakeEngine)
    monkeypatch.setattr(
        "docling_mlx.engines.table_structure.tableformer_v1.preprocessing.resize_page_image",
        lambda image, spec: (image, 1.0),
    )


def _options(
    *,
    mode: TableFormerMode = TableFormerMode.ACCURATE,
    do_cell_matching: bool = True,
    warmup: bool = False,
) -> MlxTableStructureOptions:
    return MlxTableStructureOptions(
        model_spec=TableFormerV1ModelSpec(path="/local"),
        mode=mode,
        do_cell_matching=do_cell_matching,
        engine_options=MlxTableStructureEngineOptions(warmup=warmup),
    )


def _cell(text: str, *, index: int = 7) -> TextCell:
    return TextCell(
        index=index,
        text=text,
        orig=text,
        from_ocr=True,
        rect=BoundingRectangle(
            r_x0=2,
            r_y0=4,
            r_x1=12,
            r_y1=4,
            r_x2=12,
            r_y2=16,
            r_x3=2,
            r_y3=16,
            coord_origin=CoordOrigin.TOPLEFT,
        ),
    )


def _cluster(
    cluster_id: int,
    label: DocItemLabel = DocItemLabel.TABLE,
    *,
    cells: list[TextCell] | None = None,
) -> Cluster:
    return Cluster(
        id=cluster_id,
        label=label,
        bbox=BoundingBox(l=1.6, t=2.2, r=6.6, b=8.2),
        cells=[] if cells is None else cells,
    )


def _page(
    *clusters: Cluster,
    valid: bool = True,
    segmented: SegmentedPage | None = None,
    text: str = "backend text",
) -> Page:
    page = Page(page_no=3, size=Size(width=10, height=10))
    page.predictions.layout = LayoutPrediction(clusters=list(clusters))
    page._backend = cast(Any, PageBackend(valid=valid, segmented=segmented, text=text))
    return page


def _stage(*, enabled: bool = True, do_cell_matching: bool = True) -> MlxTableFormerV1Model:
    return MlxTableFormerV1Model(
        enabled,
        None,
        _options(do_cell_matching=do_cell_matching),
        AcceleratorOptions(device="auto"),
    )


def _prediction() -> TableFormerV1Prediction:
    return TableFormerV1Prediction(
        token_ids=(2, 5, 9, 3),
        otsl_tokens=("fcel", "nl"),
        cell_bboxes=((0.0, 0.0, 10.0, 12.0),),
        bbox_classes=(2,),
    )


def _conversion_result() -> ConversionResult:
    return ConversionResult.model_construct(timings={})


@pytest.mark.parametrize("device", ["cpu", "cuda", "xpu"])
def test_disabled_stage_accepts_non_mlx_accelerators_and_empty_batches(device: str) -> None:
    stage = MlxTableFormerV1Model(False, None, _options(), AcceleratorOptions(device=device))
    assert list(stage.predict_tables(_conversion_result(), [])) == []
    with pytest.raises(RuntimeError, match="not supported by this model"):
        MlxTableFormerV1Model(True, None, _options(), AcceleratorOptions(device=device))


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ("backend", "initialized page backend"),
        ("layout", "layout predictions"),
        ("size", "page size"),
    ],
)
def test_missing_page_state_raises_runtime_error(state: str, message: str) -> None:
    page = _page(_cluster(1))
    if state == "backend":
        page._backend = None
    elif state == "layout":
        page.predictions.layout = None
    else:
        page.size = None
    with pytest.raises(RuntimeError, match=message):
        _stage().predict_tables(_conversion_result(), [page])


def test_official_mode_selects_its_checkpoint() -> None:
    options = MlxTableStructureOptions()
    assert options.mode is TableFormerMode.ACCURATE
    assert options.engine_options.warmup is False
    assert "preset" not in MlxTableStructureOptions.model_fields
    fast = MlxTableFormerV1Model(
        True,
        None,
        MlxTableStructureOptions(mode=TableFormerMode.FAST),
        AcceleratorOptions(device="auto"),
    )
    assert isinstance(fast.engine, FakeEngine)
    assert fast.engine.options.checkpoint_subdirectory == "fast"


def test_warmup_option_initializes_at_construction() -> None:
    stage = MlxTableFormerV1Model(
        True,
        None,
        _options(warmup=True),
        AcceleratorOptions(device="auto"),
    )
    assert stage.engine.initialize_calls == [True]  # type: ignore[union-attr]


def test_invalid_page_and_no_table_page_avoid_inference() -> None:
    stage = _stage()
    invalid = _page(_cluster(1), valid=False)
    existing = TableStructurePrediction()
    invalid.predictions.tablestructure = existing
    plain = _page(_cluster(2, DocItemLabel.TEXT))
    predictions = stage.predict_tables(_conversion_result(), [invalid, plain])
    assert predictions[0] is existing
    assert predictions[1].table_map == {}
    assert isinstance(stage.engine, FakeEngine)
    assert stage.engine.batches == []


def test_table_and_index_use_one_page_render_and_crop_pixel_boxes() -> None:
    FakeEngine.next_outputs = [_prediction(), _prediction()]
    stage = _stage(do_cell_matching=False)
    page = _page(
        _cluster(7),
        _cluster(9, DocItemLabel.TEXT),
        _cluster(8, DocItemLabel.DOCUMENT_INDEX),
    )
    prediction = stage.predict_tables(_conversion_result(), [page])[0]
    assert list(prediction.table_map) == [7, 8]
    assert isinstance(stage.engine, FakeEngine)
    assert [crop.size for crop in stage.engine.batches[0]] == [(10, 12), (10, 12)]
    assert cast(PageBackend, page._backend).images == [2.0]
    assert prediction.table_map[7].table_cells[0].text == "backend text"


def test_word_cells_are_preferred_with_cluster_fallback() -> None:
    word = _cell("word text", index=11)
    cluster_cell = _cell("cluster text", index=12)
    FakeEngine.next_outputs = [_prediction()]
    stage = _stage()
    with_words = _page(_cluster(1, cells=[cluster_cell]), segmented=SegmentedPage([word]))
    without_words = _page(_cluster(2, cells=[cluster_cell]), segmented=SegmentedPage([]))
    first = stage.predict_tables(_conversion_result(), [with_words])[0].table_map[1]
    second = stage.predict_tables(_conversion_result(), [without_words])[0].table_map[2]
    assert first.table_cells[0].text == "word text"
    assert second.table_cells[0].text == "cluster text"
    assert first.table_cells[0].bbox is not None
    assert first.table_cells[0].bbox.as_tuple() == pytest.approx((2.0, 4.0, 12.0, 16.0))
