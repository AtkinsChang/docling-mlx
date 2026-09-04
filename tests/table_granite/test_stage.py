# SPDX-License-Identifier: Apache-2.0

"""Granite Vision table stage contracts without loading the real 4B model."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import (
    Cluster,
    LayoutPrediction,
    Page,
    TableStructurePrediction,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options_vlm_model import ResponseFormat
from docling.datamodel.stage_model_specs import EngineModelConfig, VlmModelSpec
from docling.datamodel.vlm_engine_options import MlxVlmEngineOptions
from docling.models.inference_engines.vlm.base import (
    BaseVlmEngine,
    VlmEngineInput,
    VlmEngineOutput,
    VlmEngineType,
)
from docling_core.types.doc import BoundingBox, DocItemLabel, Size
from PIL import Image

from docling_mlx.stages._granite_vision import (
    GRANITE_VISION_4_1_REPO_ID,
    GRANITE_VISION_4_1_REVISION,
)
from docling_mlx.stages.granite_vision_engine import MlxGraniteVision41Engine
from docling_mlx.stages.table_structure import (
    GRANITE_TABLE_PROMPT,
    MlxGraniteVisionTableStructureModel,
    MlxGraniteVisionTableStructureOptions,
)


class FakeMlxEngine(MlxGraniteVision41Engine):
    def __init__(
        self,
        outputs: Sequence[str] = (),
        *,
        repo_id: str = GRANITE_VISION_4_1_REPO_ID,
        revision: str = GRANITE_VISION_4_1_REVISION,
        options: MlxVlmEngineOptions | None = None,
    ) -> None:
        BaseVlmEngine.__init__(
            self,
            options or MlxVlmEngineOptions(),
            model_config=EngineModelConfig(repo_id=repo_id, revision=revision),
        )
        self.outputs = list(outputs)
        self.batches: list[list[VlmEngineInput]] = []
        self.cleanup_calls = 0
        self.initialize_calls = 0
        self._initialized = True

    def initialize(self) -> None:
        self.initialize_calls += 1
        self._initialized = True

    def predict_batch(self, input_batch: list[VlmEngineInput]) -> list[VlmEngineOutput]:
        self.batches.append(input_batch)
        return [VlmEngineOutput(text=text) for text in self.outputs]

    def cleanup(self) -> None:
        self.cleanup_calls += 1


class PageBackend:
    def __init__(self, *, valid: bool = True, missing_cluster_ids: set[int] | None = None):
        self.valid = valid
        self.missing_cluster_ids = missing_cluster_ids or set()
        self.crops: list[tuple[float, BoundingBox]] = []

    def is_valid(self) -> bool:
        return self.valid

    def get_page_image(
        self,
        scale: float,
        cropbox: BoundingBox | None = None,
    ) -> Image.Image | None:
        assert cropbox is not None
        self.crops.append((scale, cropbox))
        if int(cropbox.l) in self.missing_cluster_ids:
            return None
        return Image.new("RGB", (8, 6), color=(int(cropbox.l), 0, 0))


def _cluster(cluster_id: int, label: DocItemLabel) -> Cluster:
    return Cluster(
        id=cluster_id,
        label=label,
        bbox=BoundingBox(l=float(cluster_id), t=1, r=float(cluster_id + 4), b=5),
    )


def _page(
    *clusters: Cluster,
    valid: bool = True,
    missing_cluster_ids: set[int] | None = None,
) -> Page:
    page = Page(page_no=3, size=Size(width=100, height=80))
    page.predictions.layout = LayoutPrediction(clusters=list(clusters))
    page._backend = cast(
        Any,
        PageBackend(valid=valid, missing_cluster_ids=missing_cluster_ids),
    )
    return page


def _conversion_result() -> ConversionResult:
    return ConversionResult.model_construct(timings={})


def _stage(
    engine: BaseVlmEngine,
    *,
    options: MlxGraniteVisionTableStructureOptions | None = None,
) -> MlxGraniteVisionTableStructureModel:
    return MlxGraniteVisionTableStructureModel(
        enabled=True,
        artifacts_path=Path("/not-used-by-injected-engine"),
        options=options or MlxGraniteVisionTableStructureOptions(),
        accelerator_options=AcceleratorOptions(device="auto"),
        engine=engine,
    )


def test_default_options_declare_the_otsl_mlx_table_task() -> None:
    options = MlxGraniteVisionTableStructureOptions()

    assert options.kind == "mlx_granite_vision_table"
    assert options.model_spec.response_format == ResponseFormat.OTSL
    assert options.model_spec.supported_engines == {VlmEngineType.MLX}
    assert options.engine_options.engine_type == VlmEngineType.MLX
    assert MlxGraniteVisionTableStructureModel.get_options_type() is type(options)


def test_disabled_stage_is_lazy_and_does_not_call_the_engine_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import docling_mlx.stages.table_structure as stage_module

    def unexpected_factory(**kwargs: Any) -> BaseVlmEngine:
        raise AssertionError(f"factory must remain lazy: {kwargs}")

    monkeypatch.setattr(stage_module, "create_granite_vision_engine", unexpected_factory)

    stage = MlxGraniteVisionTableStructureModel(
        enabled=False,
        artifacts_path=None,
        options=MlxGraniteVisionTableStructureOptions(),
        accelerator_options=AcceleratorOptions(device="auto"),
    )

    assert stage.engine is None


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:1", "xpu"])
def test_disabled_stage_accepts_non_mlx_accelerators_and_empty_batches(device: str) -> None:
    stage = MlxGraniteVisionTableStructureModel(
        enabled=False,
        artifacts_path=None,
        options=MlxGraniteVisionTableStructureOptions(),
        accelerator_options=AcceleratorOptions(device=device),
    )

    assert list(stage.predict_tables(_conversion_result(), [])) == []


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ("backend", "initialized page backend"),
        ("layout", "layout predictions"),
        ("size", "page size"),
    ],
)
def test_missing_page_state_raises_runtime_error(state: str, message: str) -> None:
    page = _page(_cluster(1, DocItemLabel.TABLE))
    if state == "backend":
        page._backend = None
    elif state == "layout":
        page.predictions.layout = None
    else:
        page.size = None

    with pytest.raises(RuntimeError, match=message):
        _stage(FakeMlxEngine()).predict_tables(_conversion_result(), [page])


def test_owned_engine_uses_copied_options_and_corrected_engine_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import docling_mlx.stages.table_structure as stage_module

    created: list[dict[str, Any]] = []
    fake = FakeMlxEngine()

    def create_fake(**kwargs: Any) -> BaseVlmEngine:
        created.append(kwargs)
        return fake

    monkeypatch.setattr(stage_module, "create_granite_vision_engine", create_fake)
    options = MlxGraniteVisionTableStructureOptions()
    artifacts_path = Path("/models")

    stage = MlxGraniteVisionTableStructureModel(
        enabled=True,
        artifacts_path=artifacts_path,
        options=options,
        accelerator_options=AcceleratorOptions(device="mps"),
    )
    options.model_spec.prompt = "mutated"

    assert stage.options.model_spec.prompt == GRANITE_TABLE_PROMPT
    assert created[0]["model_spec"] is stage.options.model_spec
    assert created[0]["artifacts_path"] == artifacts_path
    assert created[0]["engine_options"] is stage.options.engine_options
    assert set(created[0]) == {
        "engine_options",
        "model_spec",
        "artifacts_path",
        "accelerator_options",
    }
    stage.cleanup()
    assert fake.cleanup_calls == 1


def test_injected_engine_must_use_corrected_engine() -> None:
    class StockMlxEngine(BaseVlmEngine):
        def __init__(self) -> None:
            super().__init__(
                MlxVlmEngineOptions(),
                model_config=EngineModelConfig(
                    repo_id=GRANITE_VISION_4_1_REPO_ID,
                    revision=GRANITE_VISION_4_1_REVISION,
                ),
            )

        def initialize(self) -> None:
            self._initialized = True

        def predict_batch(self, input_batch: list[VlmEngineInput]) -> list[VlmEngineOutput]:
            return []

        def cleanup(self) -> None:
            return None

    with pytest.raises(ValueError, match="corrected Granite Vision 4.1 engine"):
        _stage(StockMlxEngine())

    _stage(FakeMlxEngine(repo_id="example/wrong"))
    wrong_type = FakeMlxEngine()
    wrong_type.options.engine_type = VlmEngineType.TRANSFORMERS  # type: ignore[assignment]
    _stage(wrong_type)


def test_external_engine_is_not_cleaned_up_by_the_stage() -> None:
    engine = FakeMlxEngine()
    stage = _stage(engine)

    stage.cleanup()

    assert engine.cleanup_calls == 0


def test_predict_tables_crops_only_table_clusters_and_preserves_order() -> None:
    engine = FakeMlxEngine(
        [
            "<ched>Name</ched><nl/><fcel>A</fcel><nl/>",
            "<ched>Section</ched><ched>Page</ched><nl/><fcel>Intro</fcel><fcel>1</fcel><nl/>",
        ]
    )
    options = MlxGraniteVisionTableStructureOptions(
        model_spec=VlmModelSpec(
            name="Configured Granite",
            default_repo_id=GRANITE_VISION_4_1_REPO_ID,
            revision=GRANITE_VISION_4_1_REVISION,
            prompt=GRANITE_TABLE_PROMPT,
            response_format=ResponseFormat.OTSL,
            supported_engines={VlmEngineType.MLX},
            temperature=0.25,
            max_new_tokens=321,
            stop_strings=["</otsl>"],
            extra_generation_config={"seed": 7},
        )
    )
    stage = _stage(engine, options=options)
    table = _cluster(1, DocItemLabel.TABLE)
    text = _cluster(2, DocItemLabel.TEXT)
    index = _cluster(3, DocItemLabel.DOCUMENT_INDEX)
    page = _page(table, text, index)

    predictions = stage.predict_tables(_conversion_result(), [page])

    assert predictions == [page.predictions.tablestructure]
    assert list(predictions[0].table_map) == [1, 3]
    assert predictions[0].table_map[1].page_no == 3
    assert predictions[0].table_map[1].cluster is table
    assert predictions[0].table_map[1].num_rows == 2
    assert predictions[0].table_map[3].label == DocItemLabel.DOCUMENT_INDEX
    assert [cell.text for cell in predictions[0].table_map[3].table_cells] == [
        "Section",
        "Page",
        "Intro",
        "1",
    ]
    backend = page._backend
    assert isinstance(backend, PageBackend)
    assert [(scale, int(box.l)) for scale, box in backend.crops] == [(1.0, 1), (1.0, 3)]
    assert len(engine.batches) == 1
    assert [item.image.getpixel((0, 0))[0] for item in engine.batches[0]] == [1, 3]
    for item in engine.batches[0]:
        assert item.prompt == GRANITE_TABLE_PROMPT
        assert item.temperature == 0.25
        assert item.max_new_tokens == 321
        assert item.stop_strings == ["</otsl>"]
        assert item.extra_generation_config == {"seed": 7}


def test_inherited_call_assigns_the_prediction_and_yields_the_page() -> None:
    engine = FakeMlxEngine(["<fcel>one</fcel><nl/>"])
    stage = _stage(engine)
    page = _page(_cluster(8, DocItemLabel.TABLE))

    yielded = list(stage(_conversion_result(), [page]))

    assert yielded == [page]
    assert page.predictions.tablestructure is not None
    assert page.predictions.tablestructure.table_map[8].table_cells[0].text == "one"


def test_invalid_page_preserves_existing_prediction_without_inference() -> None:
    engine = FakeMlxEngine(["must not be consumed"])
    stage = _stage(engine)
    page = _page(_cluster(1, DocItemLabel.TABLE), valid=False)
    existing = TableStructurePrediction()
    page.predictions.tablestructure = existing

    predictions = stage.predict_tables(_conversion_result(), [page])

    assert predictions == [existing]
    assert engine.batches == []


def test_missing_crops_are_skipped_and_output_count_must_match() -> None:
    skipped = _page(
        _cluster(1, DocItemLabel.TABLE),
        _cluster(3, DocItemLabel.TABLE),
        missing_cluster_ids={1, 3},
    )
    engine = FakeMlxEngine()
    stage = _stage(engine)

    predictions = stage.predict_tables(_conversion_result(), [skipped])

    assert predictions[0].table_map == {}
    assert engine.batches == []

    mismatch_page = _page(_cluster(4, DocItemLabel.TABLE))
    with pytest.raises(RuntimeError, match="different number of outputs"):
        stage.predict_tables(_conversion_result(), [mismatch_page])


def test_malformed_otsl_is_an_empty_table() -> None:
    page = _page(_cluster(5, DocItemLabel.TABLE), _cluster(6, DocItemLabel.TABLE))
    stage = _stage(FakeMlxEngine(["not an OTSL table", "<fcel>ok</fcel>"]))

    predictions = stage.predict_tables(_conversion_result(), [page])

    table = predictions[0].table_map[5]
    assert (table.num_rows, table.num_cols, table.table_cells) == (0, 0, [])
    assert predictions[0].table_map[6].table_cells[0].text == "ok"


def test_otsl_parser_error_is_an_empty_table_with_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import docling_mlx.stages.table_structure as stage_module

    def broken_parser(_text: str) -> tuple[list[str], list[object], int, int]:
        raise AttributeError("parser implementation bug")

    monkeypatch.setattr(stage_module, "parse_otsl_output", broken_parser)
    page = _page(_cluster(5, DocItemLabel.TABLE))

    caplog.set_level(logging.WARNING)
    prediction = _stage(FakeMlxEngine(["<fcel>ok</fcel>"])).predict_tables(
        _conversion_result(), [page]
    )[0]

    assert prediction.table_map[5].table_cells == []
    assert "parser implementation bug" in caplog.text
