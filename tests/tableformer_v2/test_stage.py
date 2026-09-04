# SPDX-License-Identifier: Apache-2.0

"""Docling adaptor contracts for native TableFormerV2."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Cluster, LayoutPrediction, Page, TableStructurePrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import TableStructureV2Options
from docling_core.types.doc import BoundingBox, DocItemLabel, Size
from docling_core.types.doc.base import CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell
from PIL import Image

from docling_mlx.engines.table_structure.tableformer_v2 import (
    TableFormerV2ModelSpec,
    TableFormerV2Prediction,
)
from docling_mlx.stages.table_structure_v2 import (
    MlxTableFormerV2Model,
    MlxTableStructureV2EngineOptions,
    MlxTableStructureV2Options,
)


class PageBackend:
    def __init__(self, *, valid: bool = True, text: str = "backend text") -> None:
        self.valid = valid
        self.text = text
        self.images: list[float] = []
        self.text_boxes: list[BoundingBox] = []
        self.image = Image.fromarray(np.arange(20 * 20 * 3, dtype=np.uint8).reshape(20, 20, 3))

    def is_valid(self) -> bool:
        return self.valid

    def get_page_image(self, scale: float, cropbox: BoundingBox | None = None) -> Image.Image:
        assert scale == 2.0 and cropbox is None
        self.images.append(scale)
        return self.image

    def get_text_in_rect(self, bbox: BoundingBox) -> str:
        self.text_boxes.append(bbox)
        return self.text


class FakeEngine:
    next_outputs: list[TableFormerV2Prediction] = []
    instances: list[FakeEngine] = []

    def __init__(self, model_spec: object, options: object) -> None:
        self.model_spec = model_spec
        self.options = options
        self.batches: list[list[Image.Image]] = []
        self.outputs = list(self.next_outputs)
        self.initialize_calls: list[bool] = []
        self.instances.append(self)

    def initialize(self, *, warmup: bool = False) -> None:
        self.initialize_calls.append(warmup)

    def predict(self, images: list[Image.Image]) -> list[TableFormerV2Prediction]:
        self.batches.append(images)
        return list(self.outputs)


@pytest.fixture(autouse=True)
def _install_fake_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeEngine.next_outputs = []
    FakeEngine.instances = []
    monkeypatch.setattr("docling_mlx.stages.table_structure_v2.TableFormerV2Engine", FakeEngine)


def _options(*, do_cell_matching: bool = True, warmup: bool = False) -> MlxTableStructureV2Options:
    return MlxTableStructureV2Options(
        model_spec=TableFormerV2ModelSpec(path="/local"),
        do_cell_matching=do_cell_matching,
        engine_options=MlxTableStructureV2EngineOptions(warmup=warmup),
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


def _page(*clusters: Cluster, valid: bool = True, text: str = "backend text") -> Page:
    page = Page(page_no=3, size=Size(width=10, height=10))
    page.predictions.layout = LayoutPrediction(clusters=list(clusters))
    page._backend = cast(Any, PageBackend(valid=valid, text=text))
    return page


def _conversion_result() -> ConversionResult:
    return ConversionResult.model_construct(timings={})


def _stage(*, enabled: bool = True, do_cell_matching: bool = True) -> MlxTableFormerV2Model:
    return MlxTableFormerV2Model(
        enabled,
        None,
        _options(do_cell_matching=do_cell_matching),
        AcceleratorOptions(device="auto"),
    )


def _prediction(
    otsl_tokens: tuple[str, ...] = ("fcel", "nl"),
    cell_bboxes: tuple[tuple[float, float, float, float], ...] = ((2.0, 3.6, 7.0, 9.6),),
) -> TableFormerV2Prediction:
    return TableFormerV2Prediction(
        token_ids=(2, 5, 9, 3), otsl_tokens=otsl_tokens, cell_bboxes=cell_bboxes
    )


def test_options_match_the_official_v2_shape() -> None:
    assert MlxTableStructureV2Options.kind == "mlx_tableformer_v2"
    assert issubclass(MlxTableStructureV2Options, TableStructureV2Options)
    options = MlxTableStructureV2Options()
    assert options.engine_options.warmup is False
    assert "preset" not in MlxTableStructureV2Options.model_fields
    assert options.model_spec is None


def test_warmup_option_initializes_at_construction() -> None:
    stage = MlxTableFormerV2Model(
        True,
        None,
        _options(warmup=True),
        AcceleratorOptions(device="auto"),
    )
    assert stage.engine.initialize_calls == [True]  # type: ignore[union-attr]


def test_constructor_keeps_explicit_model_spec_override() -> None:
    stage = _stage()
    assert isinstance(stage.engine, FakeEngine)
    assert stage.engine.model_spec == TableFormerV2ModelSpec(path="/local")


def test_invalid_page_preserves_existing_prediction_without_inference() -> None:
    stage = _stage()
    page = _page(_cluster(1), valid=False)
    existing = TableStructurePrediction()
    page.predictions.tablestructure = existing
    assert stage.predict_tables(_conversion_result(), [page]) == [existing]
    assert isinstance(stage.engine, FakeEngine)
    assert stage.engine.batches == []


def test_no_table_clusters_create_an_empty_prediction_without_page_rendering() -> None:
    stage = _stage()
    page = _page(_cluster(1, DocItemLabel.TEXT))
    assert stage.predict_tables(_conversion_result(), [page])[0].table_map == {}
    assert cast(PageBackend, page._backend).images == []


def test_predict_tables_reconstructs_crop_pixel_boxes() -> None:
    FakeEngine.next_outputs = [_prediction(), _prediction(("ched", "nl"))]
    stage = _stage()
    page = _page(
        _cluster(7),
        _cluster(9, DocItemLabel.TEXT),
        _cluster(8, DocItemLabel.DOCUMENT_INDEX),
    )
    predictions = stage.predict_tables(_conversion_result(), [page])
    assert list(predictions[0].table_map) == [7, 8]
    assert isinstance(stage.engine, FakeEngine)
    assert [crop.size for crop in stage.engine.batches[0]] == [(10, 12), (10, 12)]
    backend = cast(PageBackend, page._backend)
    assert backend.images == [2.0]
    first_cell = predictions[0].table_map[7].table_cells[0]
    assert first_cell.text == "backend text"
    assert first_cell.bbox is not None
    assert first_cell.bbox.as_tuple() == pytest.approx((3.0, 3.8, 5.5, 6.8))
    assert predictions[0].table_map[8].table_cells[0].column_header is True


def test_otsl_grid_preserves_spans_and_all_header_flags() -> None:
    cells, rows, columns = MlxTableFormerV2Model._build_table_cells(
        ["ched", "lcel", "nl", "ucel", "ecel", "nl", "rhed", "srow"],
        ((0.0, 0.0, 0.5, 0.5), (0.5, 0.5, 1.0, 1.0), (0.0, 0.75, 0.5, 1.0), (0.5, 0.75, 1.0, 1.0)),
        [0.0, 0.0, 100.0, 80.0],
    )
    assert (rows, columns) == (3, 2)
    assert len(cells) == 4
    assert cells[0]["column_header"] is True
    assert cells[0]["col_span"] == 2
    assert cells[0]["row_span"] == 2
    assert cells[2]["row_header"] is True
    assert cells[3]["row_section"] is True


def test_matching_prefers_cluster_text_then_falls_back_to_backend() -> None:
    text_cell = TextCell(
        text="cluster text",
        orig="cluster text",
        from_ocr=True,
        rect=BoundingRectangle(
            r_x0=2,
            r_y0=2,
            r_x1=7,
            r_y1=2,
            r_x2=7,
            r_y2=8,
            r_x3=2,
            r_y3=8,
            coord_origin=CoordOrigin.TOPLEFT,
        ),
    )
    FakeEngine.next_outputs = [_prediction(cell_bboxes=((0.0, 0.0, 10.0, 12.0),))]
    page = _page(_cluster(1, cells=[text_cell]))
    table = _stage().predict_tables(_conversion_result(), [page])[0].table_map[1]
    assert table.table_cells[0].text == "cluster text"
    assert cast(PageBackend, page._backend).text_boxes == []


def test_bbox_cardinality_and_engine_output_count_fail_fast() -> None:
    FakeEngine.next_outputs = [_prediction(cell_bboxes=())]
    with pytest.raises(RuntimeError, match="different number of cell bboxes"):
        _stage().predict_tables(_conversion_result(), [_page(_cluster(1))])
    FakeEngine.next_outputs = []
    with pytest.raises(RuntimeError, match="different number of outputs"):
        _stage().predict_tables(_conversion_result(), [_page(_cluster(1))])
