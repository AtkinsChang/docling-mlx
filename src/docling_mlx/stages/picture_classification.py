# SPDX-License-Identifier: Apache-2.0

"""Docling picture enrichment with an explicitly constructed MLX engine."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import numpy as np
from docling.datamodel import stage_model_specs
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.picture_classification_options import DocumentPictureClassifierOptions
from docling.datamodel.stage_model_specs import EngineModelConfig, ImageClassificationModelSpec
from docling.models.inference_engines.image_classification.base import (
    BaseImageClassificationEngine,
    BaseImageClassificationEngineOptions,
    ImageClassificationEngineInput,
    ImageClassificationEngineOutput,
)
from docling.models.stages.picture_classifier.document_picture_classifier import (
    DocumentPictureClassifier,
)
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from docling_mlx.engines._shared import resolve_artifact_checkpoint
from docling_mlx.engines.image_classification.efficientnet.artifact import CHECKPOINT_FILES
from docling_mlx.engines.image_classification.efficientnet.engine import (
    EfficientNetEngine,
    EfficientNetEngineOptions,
    EfficientNetModelSpec,
)
from docling_mlx.presets import resolve_preset
from docling_mlx.runtime.guards import validate_mlx_accelerator


class MlxImageClassificationEngineOptions(BaseModel):
    """Serializable native EfficientNet settings for the Docling adaptor."""

    model_config = ConfigDict(extra="forbid")

    top_k: int | None = Field(default=None, ge=1)
    dtype: Literal["float16", "float32", "bfloat16"] | None = None
    warmup: bool = False


def _picture_model_spec() -> ImageClassificationModelSpec:
    preset = resolve_preset("document_figure_classifier_v2")
    if preset.engine_kind != "image_classification/efficientnet":
        raise ValueError("Preset 'document_figure_classifier_v2' is not a DocumentFigure preset")
    return ImageClassificationModelSpec(
        name="DocumentFigureClassifier v2.5 MLX",
        repo_id=preset.repo_id,
        revision=preset.revision,
    )


class _MlxDocumentFigureClassificationEngine(BaseImageClassificationEngine):
    """Adapt generic EfficientNet classification results to Docling objects."""

    def __init__(
        self,
        options: MlxImageClassificationEngineOptions,
        *,
        model_config: EngineModelConfig,
        accelerator_options: AcceleratorOptions,
        artifacts_path: Path | str | None,
    ) -> None:
        copied_options = options.model_copy(deep=True)
        copied_model_config = model_config.model_copy(deep=True)
        super().__init__(
            cast(BaseImageClassificationEngineOptions, copied_options),
            model_config=copied_model_config,
        )
        validate_mlx_accelerator(accelerator_options)
        self._mlx_options = copied_options
        self._model_config = copied_model_config
        self._artifacts_path = artifacts_path
        repo_id = self._model_config.repo_id
        if not repo_id:
            raise ValueError("_MlxDocumentFigureClassificationEngine requires model_config.repo_id")
        if self._artifacts_path is None:
            spec = EfficientNetModelSpec(
                repo_id=repo_id,
                revision=self._model_config.revision,
            )
        else:
            revision = self._model_config.revision
            if not revision:
                raise ValueError(
                    "_MlxDocumentFigureClassificationEngine requires model_config.revision"
                )
            spec = EfficientNetModelSpec(
                path=resolve_artifact_checkpoint(
                    repo_id,
                    revision,
                    self._artifacts_path,
                    files=CHECKPOINT_FILES,
                )
            )
        dtype = None
        if self._mlx_options.dtype is not None:
            import mlx.core as mx

            dtype = getattr(mx, self._mlx_options.dtype)
        self._engine = EfficientNetEngine(
            spec,
            EfficientNetEngineOptions(top_k=self._mlx_options.top_k, dtype=dtype),
        )
        self._num_labels = 26
        self.artifact_path: Path | None = None

    def initialize(self, *, warmup: bool = False) -> None:
        self._engine.initialize(warmup=warmup)
        self.artifact_path = self._engine.directory
        self._num_labels = len(self._engine.get_label_mapping())
        self._initialized = True

    def get_label_mapping(self) -> dict[int, str]:
        return self._engine.get_label_mapping()

    def predict_logits(self, images: Sequence[Image.Image]) -> NDArray:
        if not images:
            return np.empty((0, self._num_labels), dtype=np.float32)
        return self._engine.predict_logits(images)

    def predict_batch(
        self, input_batch: list[ImageClassificationEngineInput]
    ) -> list[ImageClassificationEngineOutput]:
        if not input_batch:
            return []
        logits = self.predict_logits([item.image for item in input_batch])
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        count = (
            self._num_labels
            if self._mlx_options.top_k is None
            else min(self._mlx_options.top_k, self._num_labels)
        )
        return [
            ImageClassificationEngineOutput(
                label_ids=(indices := np.argsort(-scores, kind="stable")[:count]).tolist(),
                scores=scores[indices].tolist(),
                metadata=dict(item.metadata),
            )
            for item, scores in zip(input_batch, probabilities, strict=True)
        ]

    def predict(self, item: ImageClassificationEngineInput) -> ImageClassificationEngineOutput:
        return self.predict_batch([item])[0]

    def __call__(
        self, item: ImageClassificationEngineInput | list[ImageClassificationEngineInput]
    ) -> ImageClassificationEngineOutput | list[ImageClassificationEngineOutput]:
        return self.predict_batch(item) if isinstance(item, list) else self.predict(item)


class MlxDocumentPictureClassifierOptions(DocumentPictureClassifierOptions):
    """Official picture-classification options backed by the MLX engine."""

    model_config = ConfigDict(extra="forbid")

    kind: ClassVar[str] = "mlx_document_picture_classifier"
    model_spec: ImageClassificationModelSpec = Field(default_factory=_picture_model_spec)
    engine_options: MlxImageClassificationEngineOptions = Field(  # type: ignore[assignment]
        default_factory=MlxImageClassificationEngineOptions
    )

    @property
    def repo_id(self) -> str:
        return self.model_spec.repo_id

    @property
    def revision(self) -> str:
        return self.model_spec.revision

    @classmethod
    def from_preset(
        cls,
        preset_id: str,
        engine_options: BaseImageClassificationEngineOptions | None = None,
        **overrides: Any,
    ) -> MlxDocumentPictureClassifierOptions:
        preset = cls.get_preset(preset_id)
        if engine_options is not None and not isinstance(
            engine_options, MlxImageClassificationEngineOptions
        ):
            raise TypeError("MLX picture presets require MlxImageClassificationEngineOptions")
        mlx_engine_options = (
            MlxImageClassificationEngineOptions()
            if engine_options is None
            else cast(MlxImageClassificationEngineOptions, engine_options)
        )
        instance = cls(
            model_spec=preset.model_spec.model_copy(deep=True),
            engine_options=mlx_engine_options,
        )
        for key, value in {**preset.stage_options, **overrides}.items():
            setattr(instance, key, value)
        return instance


MlxDocumentPictureClassifierOptions.register_preset(
    stage_model_specs.IMAGE_CLASSIFICATION_DOCUMENT_FIGURE.model_copy(
        deep=True,
        update={"model_spec": _picture_model_spec()},
    )
)


class MlxDocumentPictureClassifier(DocumentPictureClassifier):
    """Reuse upstream image preparation and output writing, without pipeline patching."""

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Path | None,
        options: MlxDocumentPictureClassifierOptions,
        accelerator_options: AcceleratorOptions,
        enable_remote_services: bool = False,
    ) -> None:
        del enable_remote_services
        self.enabled = enabled
        self.options = options
        self.engine = None
        self._classes = {}
        if not enabled:
            return
        engine = _MlxDocumentFigureClassificationEngine(
            options.engine_options,
            model_config=EngineModelConfig(
                repo_id=options.repo_id,
                revision=options.revision,
            ),
            accelerator_options=accelerator_options,
            artifacts_path=artifacts_path,
        )
        engine.initialize(warmup=options.engine_options.warmup)
        self.engine = engine
        self._classes = engine.get_label_mapping()


__all__ = [
    "MlxDocumentPictureClassifier",
    "MlxDocumentPictureClassifierOptions",
    "MlxImageClassificationEngineOptions",
]
