# SPDX-License-Identifier: Apache-2.0

"""Portable disabled-stage contracts for TableFormerV2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Cluster, LayoutPrediction, Page
from docling.datamodel.document import ConversionResult
from docling_core.types.doc import BoundingBox, DocItemLabel, Size

from docling_mlx.engines.table_structure.tableformer_v2 import TableFormerV2Prediction
from docling_mlx.stages.table_structure_v2 import (
    MlxTableFormerV2Model,
    MlxTableStructureV2Options,
)


class _Engine:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def initialize(self, *, warmup: bool = False) -> None:
        del warmup


@pytest.fixture(autouse=True)
def _fake_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("docling_mlx.stages.table_structure_v2.TableFormerV2Engine", _Engine)


def _conversion_result() -> ConversionResult:
    return ConversionResult.model_construct(timings={})


@pytest.mark.parametrize("device", ["cpu", "cuda", "xpu"])
def test_disabled_stage_accepts_non_mlx_accelerators_and_passes_empty_batches(
    device: str,
) -> None:
    stage = MlxTableFormerV2Model(
        enabled=False,
        artifacts_path=Path("/not-used"),
        options=MlxTableStructureV2Options(),
        accelerator_options=AcceleratorOptions(device=device),
    )

    assert list(stage.predict_tables(_conversion_result(), [])) == []
    with pytest.raises(RuntimeError, match="not supported by this model"):
        MlxTableFormerV2Model(
            enabled=True,
            artifacts_path=Path("/not-used"),
            options=MlxTableStructureV2Options(),
            accelerator_options=AcceleratorOptions(device=device),
        )


class _ValidBackend:
    def is_valid(self) -> bool:
        return True


def _page() -> Page:
    page = Page(page_no=1, size=Size(width=10, height=10))
    page.predictions.layout = LayoutPrediction(clusters=[])
    page._backend = cast(Any, _ValidBackend())
    return page


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ("backend", "initialized page backend"),
        ("layout", "layout predictions"),
        ("size", "page size"),
    ],
)
def test_missing_page_state_raises_runtime_error(state: str, message: str) -> None:
    page = _page()
    if state == "backend":
        page._backend = None
    elif state == "layout":
        page.predictions.layout = None
    else:
        page.size = None

    stage = MlxTableFormerV2Model(
        enabled=True,
        artifacts_path=None,
        options=MlxTableStructureV2Options(),
        accelerator_options=AcceleratorOptions(device="auto"),
    )
    with pytest.raises(RuntimeError, match=message):
        stage.predict_tables(_conversion_result(), [page])


def test_table_conversion_requires_an_initialized_page_backend() -> None:
    stage = MlxTableFormerV2Model(
        enabled=True,
        artifacts_path=None,
        options=MlxTableStructureV2Options(),
        accelerator_options=AcceleratorOptions(device="auto"),
    )
    page = Page(page_no=1, size=Size(width=10, height=10))
    cluster = Cluster(
        id=1,
        label=DocItemLabel.TABLE,
        bbox=BoundingBox(l=0, t=0, r=10, b=10),
        cells=[],
    )

    with pytest.raises(RuntimeError, match="initialized page backend"):
        stage._prediction_to_table(
            TableFormerV2Prediction(
                token_ids=(5, 9),
                cell_bboxes=((0.0, 0.0, 1.0, 1.0),),
                otsl_tokens=("fcel", "nl"),
            ),
            table_cluster=cluster,
            page=page,
            table_box=[0.0, 0.0, 10.0, 10.0],
            crop_size=(1, 1),
        )
