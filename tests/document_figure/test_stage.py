# SPDX-License-Identifier: Apache-2.0

"""Docling-facing behaviour of the explicit MLX picture-classifier adapter."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import ItemAndImageEnrichmentElement, Page
from docling.datamodel.image_classification_engine_options import (
    TransformersImageClassificationEngineOptions,
)
from docling.datamodel.picture_classification_options import DocumentPictureClassifierOptions
from docling.datamodel.stage_model_specs import EngineModelConfig, ImageClassificationModelSpec
from docling_core.types.doc import (
    BoundingBox,
    DoclingDocument,
    PictureClassificationMetaField,
    PictureClassificationPrediction,
    PictureMeta,
    ProvenanceItem,
    Size,
)
from docling_core.types.doc.common.meta import SummaryMetaField
from PIL import Image

from docling_mlx.stages.picture_classification import (
    MlxDocumentPictureClassifierOptions,
    MlxImageClassificationEngineOptions,
)

ARTIFACT = Path(__file__).parents[2] / ".artifacts/document-figure-classifier"
REFERENCE_IMAGES = Path(__file__).parents[1] / "fixtures/document_figure/reference_images"


def _stage_type():
    from docling_mlx.stages.picture_classification import MlxDocumentPictureClassifier

    return MlxDocumentPictureClassifier


def _options(*, keep_deprecated_annotations: bool = True) -> MlxDocumentPictureClassifierOptions:
    options = MlxDocumentPictureClassifierOptions(
        model_spec=ImageClassificationModelSpec(
            name="Document Figure Classifier",
            repo_id=ARTIFACT.name,
            revision="local-test-revision",
        ),
        engine_options=MlxImageClassificationEngineOptions(top_k=None),
    )
    options._keep_deprecated_annotations = keep_deprecated_annotations
    return options


def _stage(
    options: MlxDocumentPictureClassifierOptions | None = None,
    *,
    enabled: bool = True,
):
    return _stage_type()(
        enabled=enabled,
        artifacts_path=ARTIFACT.parent,
        options=options or _options(),
        accelerator_options=AcceleratorOptions(device="auto"),
    )


def _processable_stage():
    """Use the real inherited preparation methods without initializing MLX."""
    stage = object.__new__(_stage_type())
    stage.enabled = True
    return stage


def _picture(doc: DoclingDocument):
    return doc.add_picture()


@pytest.mark.mlx
def test_real_stage_classifies_and_preserves_picture_meta() -> None:
    if not ARTIFACT.is_dir():
        pytest.fail("pinned MLX artifact is required for the selected MLX lane")
    stage = _stage()
    document = DoclingDocument(name="real-stage")
    picture = _picture(document)
    picture.meta = PictureMeta(summary=SummaryMetaField(text="kept summary"))
    image = Image.open(REFERENCE_IMAGES / "bar_chart.png")
    try:
        output = list(
            stage(
                document,
                [ItemAndImageEnrichmentElement(item=picture, image=image)],
            )
        )
    finally:
        image.close()

    assert output == [picture]
    assert picture.meta is not None
    assert picture.meta.summary == SummaryMetaField(text="kept summary")
    assert picture.meta.classification is not None
    predictions = picture.meta.classification.predictions
    assert len(predictions) == 26
    assert predictions[0].class_name == "bar_chart"
    assert {prediction.class_name for prediction in predictions} == set(stage._classes.values())
    assert [prediction.confidence for prediction in predictions] == sorted(
        (prediction.confidence for prediction in predictions), reverse=True
    )
    assert {prediction.created_by for prediction in predictions} == {"DocumentPictureClassifier"}
    assert len(picture.annotations) == 1
    assert picture.annotations[0].provenance == "DocumentPictureClassifier"
    assert [entry.class_name for entry in picture.annotations[0].predicted_classes] == [
        prediction.class_name for prediction in predictions
    ]


def test_stage_can_suppress_deprecated_annotations_without_loading_a_model(monkeypatch) -> None:
    import docling_mlx.stages.picture_classification as engine_module

    class FakeEngine:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def initialize(self, *, warmup: bool = False) -> None:
            pass

        def get_label_mapping(self):
            return {0: "bar_chart"}

        def predict_batch(self, inputs):
            return [SimpleNamespace(label_ids=[0], scores=[1.0], metadata={}) for _ in inputs]

    monkeypatch.setattr(engine_module, "_MlxDocumentFigureClassificationEngine", FakeEngine)
    stage = _stage(_options(keep_deprecated_annotations=False))
    document = DoclingDocument(name="annotation-toggle")
    picture = _picture(document)
    assert list(
        stage(
            document,
            [ItemAndImageEnrichmentElement(item=picture, image=Image.new("RGB", (3, 2)))],
        )
    ) == [picture]

    assert picture.annotations == []
    assert picture.meta is not None
    assert picture.meta.classification is not None
    assert [prediction.class_name for prediction in picture.meta.classification.predictions] == [
        "bar_chart"
    ]


@pytest.mark.release
def test_inherited_prepare_element_crops_a_real_docling_page() -> None:
    document = DoclingDocument(name="page-crop")
    document.add_page(page_no=1, size=Size(width=10, height=10))
    picture = document.add_picture(
        prov=ProvenanceItem(
            page_no=1,
            bbox=BoundingBox(l=1, t=2, r=5, b=6),
            charspan=(0, 0),
        )
    )
    page = Page(page_no=1, size=Size(width=10, height=10))
    page_image = Image.fromarray(np.arange(20 * 20 * 3, dtype=np.uint8).reshape(20, 20, 3))

    class PageBackend:
        def get_page_image(self, scale: float, cropbox: BoundingBox | None = None) -> Image.Image:
            assert scale == 2
            if cropbox is None:
                return page_image
            return page_image.crop(
                cropbox.to_top_left_origin(page_height=10).scaled(scale=scale).as_tuple()
            )

    page._backend = PageBackend()

    prepared = _processable_stage().prepare_element(
        SimpleNamespace(document=document, pages=[page]), picture
    )

    assert prepared is not None
    assert prepared.item is picture
    np.testing.assert_array_equal(
        np.asarray(prepared.image), np.asarray(page_image.crop((2, 4, 10, 12)))
    )


def test_stage_uses_model_spec_and_official_options_identity(monkeypatch) -> None:
    import docling_mlx.stages.picture_classification as engine_module

    created = []

    class FakeEngine:
        def __init__(self, options, *, model_config, accelerator_options, artifacts_path) -> None:
            created.append((options, model_config, accelerator_options, artifacts_path))

        def initialize(self, *, warmup: bool = False) -> None:
            pass

        def get_label_mapping(self):
            return {0: "logo"}

    monkeypatch.setattr(engine_module, "_MlxDocumentFigureClassificationEngine", FakeEngine)
    options = MlxDocumentPictureClassifierOptions(
        model_spec=ImageClassificationModelSpec(
            name="Custom Figure",
            repo_id="example/custom-figure",
            revision="refs/pr/123",
        ),
        engine_options=MlxImageClassificationEngineOptions(top_k=5),
    )
    artifacts_path = Path("/not-resolved-for-stage-contract")
    stage = _stage_type()(
        True,
        artifacts_path,
        options,
        AcceleratorOptions(device="mps"),
    )

    assert stage.options is options
    assert stage.options.model_spec.repo_id == "example/custom-figure"
    assert stage.options.engine_options.top_k == 5
    assert created[0][0].top_k == 5
    assert created[0][1] == EngineModelConfig(
        repo_id="example/custom-figure", revision="refs/pr/123"
    )
    assert created[0][2].device == "mps"
    assert created[0][3] == artifacts_path


def test_warmup_option_is_forwarded_to_the_stage_engine(monkeypatch) -> None:
    calls: list[bool] = []

    class FakeEngine:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def initialize(self, *, warmup: bool = False) -> None:
            calls.append(warmup)

        def get_label_mapping(self) -> dict[int, str]:
            return {0: "logo"}

    import docling_mlx.stages.picture_classification as engine_module

    monkeypatch.setattr(engine_module, "_MlxDocumentFigureClassificationEngine", FakeEngine)
    stage = _stage_type()(
        True,
        None,
        MlxDocumentPictureClassifierOptions(
            engine_options=MlxImageClassificationEngineOptions(warmup=True)
        ),
        AcceleratorOptions(device="auto"),
    )
    assert stage.options.engine_options.warmup is True
    assert calls == [True]


def test_stage_ignores_non_mlx_engine_overrides(monkeypatch) -> None:
    from docling.models.inference_engines.image_classification.base import (
        ImageClassificationEngineType,
    )

    created: list[EngineModelConfig] = []

    class FakeEngine:
        def __init__(self, _options, *, model_config, **_kwargs) -> None:
            created.append(model_config)

        def initialize(self, *, warmup: bool = False) -> None:
            pass

        def get_label_mapping(self) -> dict[int, str]:
            return {0: "logo"}

    import docling_mlx.stages.picture_classification as engine_module

    monkeypatch.setattr(engine_module, "_MlxDocumentFigureClassificationEngine", FakeEngine)
    options = _options()
    options.model_spec.engine_overrides = {
        ImageClassificationEngineType.TRANSFORMERS: EngineModelConfig(repo_id="example/override")
    }

    _stage(options)

    assert created == [EngineModelConfig(repo_id=ARTIFACT.name, revision="local-test-revision")]


def test_stage_options_default_to_published_model() -> None:
    assert MlxDocumentPictureClassifierOptions.kind == "mlx_document_picture_classifier"
    assert issubclass(MlxDocumentPictureClassifierOptions, DocumentPictureClassifierOptions)
    assert MlxDocumentPictureClassifierOptions.list_preset_ids() == [
        "document_figure_classifier_v2"
    ]
    options = MlxDocumentPictureClassifierOptions.from_preset("document_figure_classifier_v2")
    assert type(options).repo_cache_folder is DocumentPictureClassifierOptions.repo_cache_folder
    assert options.engine_options == MlxImageClassificationEngineOptions()


def test_picture_preset_rejects_non_mlx_engine_options() -> None:
    with pytest.raises(TypeError, match="MlxImageClassificationEngineOptions"):
        MlxDocumentPictureClassifierOptions.from_preset(
            "document_figure_classifier_v2",
            engine_options=TransformersImageClassificationEngineOptions(),
        )


def test_disabled_stage_accepts_cpu_and_passes_batches_through() -> None:
    stage = _stage_type()(
        enabled=False,
        artifacts_path=None,
        options=MlxDocumentPictureClassifierOptions(
            engine_options=MlxImageClassificationEngineOptions(warmup=True)
        ),
        accelerator_options=AcceleratorOptions(device="cpu"),
    )
    document = DoclingDocument(name="disabled-cpu")
    first = _picture(document)
    second = _picture(document)
    batch = [
        ItemAndImageEnrichmentElement(item=first, image=Image.new("RGB", (2, 2))),
        ItemAndImageEnrichmentElement(item=second, image=Image.new("RGB", (2, 2))),
    ]

    assert list(stage(document, batch)) == [first, second]


@pytest.mark.release
def test_docling_has_no_picture_classification_plugin_factory() -> None:
    import docling.models.factories as factories

    assert not hasattr(factories, "get_picture_classification_factory")
    assert (
        importlib.util.find_spec("docling.models.factories.picture_classification_factory") is None
    )


@pytest.mark.release
def test_upstream_chart_routing_uses_main_picture_classification() -> None:
    from docling.models.stages.chart_extraction.granite_vision import (
        ChartExtractionModelGraniteVision,
    )

    model = object.__new__(ChartExtractionModelGraniteVision)
    model.enabled = True
    document = DoclingDocument(name="chart-routing")
    picture = document.add_picture()
    picture.meta = PictureMeta(
        classification=PictureClassificationMetaField(
            predictions=[
                PictureClassificationPrediction(
                    class_name="bar_chart", confidence=0.9, created_by="test"
                ),
                PictureClassificationPrediction(
                    class_name="photograph", confidence=0.1, created_by="test"
                ),
            ]
        )
    )
    assert model.is_processable(document, picture)

    picture.meta.classification = PictureClassificationMetaField(
        predictions=[
            PictureClassificationPrediction(
                class_name="geographical_map", confidence=0.9, created_by="test"
            )
        ]
    )
    assert not model.is_processable(document, picture)


def test_disabled_stage_never_constructs_an_engine_or_imports_mlx() -> None:
    script = """
import sys
from pathlib import Path
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.stage_model_specs import ImageClassificationModelSpec
import docling_mlx.stages.picture_classification as stage_module
from docling_mlx.stages.picture_classification import (
    MlxImageClassificationEngineOptions,
)
from docling_mlx.stages.picture_classification import MlxDocumentPictureClassifierOptions

class FailEngine:
    def __init__(self, *_args, **_kwargs):
        raise AssertionError("disabled stage constructed an engine")

def fail_resolve(*_args, **_kwargs):
    raise AssertionError("disabled stage resolved an artifact")

stage_module._MlxDocumentFigureClassificationEngine = FailEngine
stage_module.resolve_artifact_checkpoint = fail_resolve
stage = stage_module.MlxDocumentPictureClassifier(
    False,
    Path("/no-artifact"),
    MlxDocumentPictureClassifierOptions(
        model_spec=ImageClassificationModelSpec(
            name="disabled", repo_id="example/disabled", revision="testing"
        ),
        engine_options=MlxImageClassificationEngineOptions(),
    ),
    AcceleratorOptions(device="auto"),
)
assert stage.engine is None
assert not any(name == "mlx" or name.startswith("mlx.") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
