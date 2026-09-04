# SPDX-License-Identifier: Apache-2.0

"""Portable contract tests for the Granite Vision chart component."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.datamodel.stage_model_specs import EngineModelConfig
from docling.datamodel.vlm_engine_options import MlxVlmEngineOptions
from docling.exceptions import AcceleratorDeviceNotAvailableError
from docling.models.inference_engines.vlm import (
    BaseVlmEngine,
    VlmEngineInput,
    VlmEngineOutput,
)
from docling_core.types.doc import (
    CodeLanguageLabel,
    DescriptionMetaField,
    DoclingDocument,
    PictureClassificationMetaField,
    PictureClassificationPrediction,
    PictureMeta,
    TableData,
    TabularChartMetaField,
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


class FakeMlxVlmEngine(MlxGraniteVision41Engine):
    def __init__(
        self,
        outputs: list[str] | None = None,
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
        self.outputs = outputs or []
        self.inputs: list[VlmEngineInput] = []
        self.initialize_calls = 0
        self.cleanup_calls = 0

    def initialize(self) -> None:
        self.initialize_calls += 1
        self._initialized = True

    def predict_batch(self, input_batch: list[VlmEngineInput]) -> list[VlmEngineOutput]:
        self.inputs.extend(input_batch)
        return [VlmEngineOutput(text=text) for text in self.outputs]

    def cleanup(self) -> None:
        self.cleanup_calls += 1


def _options(**overrides: object) -> MlxChartExtractionModelOptions:
    return MlxChartExtractionModelOptions(**overrides)


def _stage(
    engine: BaseVlmEngine,
    options: MlxChartExtractionModelOptions | None = None,
) -> MlxGraniteVisionChartExtractionModel:
    return MlxGraniteVisionChartExtractionModel(
        enabled=True,
        artifacts_path=None,
        options=options or _options(),
        accelerator_options=AcceleratorOptions(device="auto"),
        engine=engine,
    )


def _classified_picture(
    document: DoclingDocument,
    class_name: str,
    *,
    image_color: str = "white",
) -> ItemAndImageEnrichmentElement:
    picture = document.add_picture()
    picture.meta = PictureMeta(
        classification=PictureClassificationMetaField(
            predictions=[
                PictureClassificationPrediction(
                    class_name=class_name,
                    confidence=0.9,
                    created_by="test-classifier",
                )
            ]
        )
    )
    return ItemAndImageEnrichmentElement(
        item=picture,
        image=Image.new("RGB", (4, 3), color=image_color),
    )


def test_options_default_to_the_mlx_engine_and_csv_only_tasks() -> None:
    options = _options()

    assert options.engine_options.engine_type.value == "mlx"
    assert options.chart2csv is True
    assert options.chart2summary is False
    assert options.chart2code is False


def test_constructor_uses_corrected_engine_helper_and_copies_options(monkeypatch) -> None:
    created: list[dict[str, object]] = []
    fake = FakeMlxVlmEngine()

    def fake_create_vlm_engine(**kwargs: object) -> BaseVlmEngine:
        created.append(kwargs)
        return fake

    monkeypatch.setattr(
        "docling_mlx.stages.chart_extraction.create_granite_vision_engine",
        fake_create_vlm_engine,
    )
    source_options = _options(chart2summary=True)
    stage = MlxGraniteVisionChartExtractionModel(
        enabled=True,
        artifacts_path=Path("/models"),
        options=source_options,
        accelerator_options=AcceleratorOptions(device="mps"),
    )
    source_options.chart2csv = False
    source_options.model_spec.revision = "mutated"

    assert stage.options.chart2csv is True
    assert stage.options.model_spec.revision == GRANITE_VISION_4_1_REVISION
    assert fake.initialize_calls == 1
    assert created == [
        {
            "engine_options": stage.options.engine_options,
            "model_spec": stage.options.model_spec,
            "artifacts_path": Path("/models"),
            "accelerator_options": AcceleratorOptions(device="mps"),
        }
    ]
    assert stage._owns_engine is True


def test_disabled_and_no_task_stages_do_not_create_an_engine(monkeypatch) -> None:
    def fail_create(**_kwargs: object) -> BaseVlmEngine:
        raise AssertionError("engine factory must not be called")

    monkeypatch.setattr(
        "docling_mlx.stages.chart_extraction.create_granite_vision_engine",
        fail_create,
    )
    disabled = MlxGraniteVisionChartExtractionModel(
        enabled=False,
        artifacts_path=None,
        options=_options(),
        accelerator_options=AcceleratorOptions(device="cpu"),
    )
    no_tasks = MlxGraniteVisionChartExtractionModel(
        enabled=True,
        artifacts_path=None,
        options=_options(chart2csv=False, chart2summary=False, chart2code=False),
        accelerator_options=AcceleratorOptions(device="auto"),
    )

    assert disabled.engine is None
    assert no_tasks.engine is None
    disabled_doc = DoclingDocument(name="disabled")
    disabled_element = _classified_picture(disabled_doc, "bar_chart")
    assert list(disabled(disabled_doc, [disabled_element])) == [disabled_element.item]
    assert list(no_tasks(DoclingDocument(name="no-tasks"), [])) == []


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:1", "xpu"])
def test_stage_rejects_non_mlx_accelerators_at_construction(device: str) -> None:
    with pytest.raises(AcceleratorDeviceNotAvailableError, match="not supported by this model"):
        MlxGraniteVisionChartExtractionModel(
            enabled=True,
            artifacts_path=None,
            options=_options(),
            accelerator_options=AcceleratorOptions(device=device),
            engine=FakeMlxVlmEngine(),
        )


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

    _stage(FakeMlxVlmEngine(repo_id="other/model"))


def test_external_engine_is_detached_but_not_cleaned_up() -> None:
    engine = FakeMlxVlmEngine()
    stage = _stage(engine)

    stage.cleanup()

    assert stage.engine is None
    assert engine.initialize_calls == 1
    assert engine.cleanup_calls == 0


def test_classification_prerequisite_and_scale_match_docling_chart_contract() -> None:
    document = DoclingDocument(name="processable")
    stage = _stage(FakeMlxVlmEngine())
    chart = _classified_picture(document, "bar_chart")
    figure = _classified_picture(document, "photograph")
    missing_meta = document.add_picture()

    assert stage.images_scale == 2.0
    assert stage.is_processable(document, chart.item) is True
    assert stage.is_processable(document, figure.item) is False
    assert stage.is_processable(document, missing_meta) is False


def test_image_major_prompt_batch_and_metadata_updates_are_stable() -> None:
    document = DoclingDocument(name="chart-batch")
    first = _classified_picture(document, "bar_chart", image_color="red")
    second = _classified_picture(document, "line_chart", image_color="blue")
    assert first.item.meta is not None
    assert second.item.meta is not None
    second.item.meta.tabular_chart = TabularChartMetaField(chart_data=TableData())
    existing_table = second.item.meta.tabular_chart

    engine = FakeMlxVlmEngine(
        outputs=[
            "```csv\nRegion,Revenue\nNorth,12.5\n```",
            "Revenue rises in the north.",
            "```python\nprint('first')\n```",
            'a,b\n"unterminated',
            "Second summary survives the CSV parser failure.",
            "not a fenced Python block",
        ]
    )
    stage = _stage(
        engine,
        _options(chart2csv=True, chart2summary=True, chart2code=True),
    )

    output = list(stage(document, [first, second]))

    assert output == [first.item, second.item]
    assert [entry.prompt for entry in engine.inputs] == [
        "<chart2csv>",
        "<chart2summary>",
        "<chart2code>",
        "<chart2csv>",
        "<chart2summary>",
        "<chart2code>",
    ]
    assert [entry.image.getpixel((0, 0)) for entry in engine.inputs] == [
        (255, 0, 0),
        (255, 0, 0),
        (255, 0, 0),
        (0, 0, 255),
        (0, 0, 255),
        (0, 0, 255),
    ]
    assert all(
        entry.max_new_tokens == stage.options.model_spec.max_new_tokens for entry in engine.inputs
    )

    assert first.item.meta is not None
    assert first.item.meta.tabular_chart is not None
    table = first.item.meta.tabular_chart.chart_data
    assert (table.num_rows, table.num_cols) == (2, 2)
    assert [cell.text for cell in table.table_cells] == [
        "Region",
        "Revenue",
        "North",
        "12.5",
    ]
    assert [cell.column_header for cell in table.table_cells] == [True, True, False, False]
    assert [cell.row_header for cell in table.table_cells] == [False, False, True, False]
    assert first.item.meta.description == DescriptionMetaField(text="Revenue rises in the north.")
    assert first.item.meta.code is not None
    assert first.item.meta.code.text == "print('first')"
    assert first.item.meta.code.language == CodeLanguageLabel.PYTHON

    assert second.item.meta is not None
    assert second.item.meta.tabular_chart is existing_table
    assert second.item.meta.description == DescriptionMetaField(
        text="Second summary survives the CSV parser failure."
    )
    assert second.item.meta.code is None


def test_no_active_prompts_do_not_call_an_injected_engine() -> None:
    document = DoclingDocument(name="no-prompts")
    element = _classified_picture(document, "pie_chart")
    engine = FakeMlxVlmEngine(outputs=["must not be consumed"])
    stage = _stage(
        engine,
        _options(chart2csv=False, chart2summary=False, chart2code=False),
    )

    assert list(stage(document, [element])) == [element.item]
    assert engine.inputs == []
    assert stage.is_processable(document, element.item) is False


def test_engine_output_count_mismatch_fails_before_misaligned_metadata() -> None:
    document = DoclingDocument(name="short-output")
    element = _classified_picture(document, "pie_chart")
    stage = _stage(
        FakeMlxVlmEngine(outputs=[]),
        _options(chart2csv=True, chart2summary=True),
    )

    with pytest.raises(RuntimeError, match="0 outputs for 2 inputs"):
        list(stage(document, [element]))


def test_repeated_cleanup_releases_an_owned_engine_once(monkeypatch) -> None:
    owned = FakeMlxVlmEngine()
    monkeypatch.setattr(
        "docling_mlx.stages.chart_extraction.create_granite_vision_engine",
        lambda **_kwargs: owned,
    )
    owner = MlxGraniteVisionChartExtractionModel(
        enabled=True,
        artifacts_path=None,
        options=_options(),
        accelerator_options=AcceleratorOptions(device="auto"),
    )

    owner.cleanup()
    owner.cleanup()

    assert owned.cleanup_calls == 1


def test_malformed_csv_is_skipped_with_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR)
    document = DoclingDocument(name="malformed-chart")
    element = _classified_picture(document, "bar_chart")
    stage = _stage(
        FakeMlxVlmEngine(outputs=['a,b\n"unterminated', "Chart summary survives"]),
        _options(chart2csv=True, chart2summary=True),
    )

    output = list(stage(document, [element]))

    assert output == [element.item]
    assert element.item.meta is not None
    assert element.item.meta.tabular_chart is None
    assert element.item.meta.description == DescriptionMetaField(text="Chart summary survives")
    assert "for image 0" in caplog.text
    assert element.item.self_ref in caplog.text
    assert "EOF inside string" in caplog.text


def test_chart_parser_error_is_logged_and_skipped(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import docling_mlx.stages.chart_extraction as stage_module

    def broken_parser(_text: str) -> TableData:
        raise AttributeError("parser implementation bug")

    monkeypatch.setattr(stage_module, "extract_csv_table", broken_parser)
    document = DoclingDocument(name="parser-bug")
    element = _classified_picture(document, "bar_chart")

    caplog.set_level(logging.ERROR)
    output = list(_stage(FakeMlxVlmEngine(outputs=["a,b\n1,2"]))(document, [element]))

    assert output == [element.item]
    assert element.item.meta is not None
    assert element.item.meta.tabular_chart is None
    assert "parser implementation bug" in caplog.text
