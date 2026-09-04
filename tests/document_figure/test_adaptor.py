# SPDX-License-Identifier: Apache-2.0

"""Portable DocumentFigure preset and adaptor contracts."""

from __future__ import annotations

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.stage_model_specs import ImageClassificationModelSpec

from docling_mlx.engines.image_classification.efficientnet.preprocessing import (
    parse_preprocessing_config,
)
from docling_mlx.stages.picture_classification import (
    MlxDocumentPictureClassifier,
    MlxDocumentPictureClassifierOptions,
    MlxImageClassificationEngineOptions,
)


def test_document_figure_uses_the_generic_vit_preprocessor_path() -> None:
    spec = parse_preprocessing_config(
        {
            "image_processor_type": "ViTImageProcessor",
            "size": {"height": 224, "width": 224},
            "resample": 2,
            "rescale_factor": 1 / 255,
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.47853944, 0.4732864, 0.47434163],
        }
    )
    assert spec.include_top is False


def test_options_default_to_the_document_figure_preset() -> None:
    options = MlxDocumentPictureClassifierOptions()
    assert (
        options.model_spec
        == MlxDocumentPictureClassifierOptions.get_preset(
            "document_figure_classifier_v2"
        ).model_spec
    )
    assert options.engine_options == MlxImageClassificationEngineOptions()


def test_disabled_adaptor_does_not_construct_a_generic_engine() -> None:
    stage = MlxDocumentPictureClassifier(
        enabled=False,
        artifacts_path=None,
        options=MlxDocumentPictureClassifierOptions(
            model_spec=ImageClassificationModelSpec(
                name="test", repo_id="example/figure", revision="test"
            )
        ),
        accelerator_options=AcceleratorOptions(device="cpu"),
    )
    assert stage.engine is None
