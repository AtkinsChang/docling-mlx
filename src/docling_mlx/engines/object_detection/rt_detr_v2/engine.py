# SPDX-License-Identifier: Apache-2.0

"""Framework-free MLX RT-DETR-v2 inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from ...._models.rt_detr_v2.config import RtDetrV2Config
from ...._models.rt_detr_v2.weights import load_state_dict
from ....runtime.guards import require_apple_silicon
from ..._shared import resolve_checkpoint
from .._focal_postprocessing import _postprocess_focal_detections
from .._types import Detections
from .artifact import CHECKPOINT_FILES, _validate_artifact
from .preprocessing import (
    RtDetrPreprocessingSpec,
    preprocess_images,
)

if TYPE_CHECKING:
    import mlx.core as mx

    from ...._models.rt_detr_v2.model import RtDetrV2


@dataclass(frozen=True, slots=True)
class RtDetrV2ModelSpec:
    """An immutable Hub model identity or a local checkpoint directory."""

    repo_id: str | None = None
    revision: str | None = None
    path: Path | str | None = None

    def __post_init__(self) -> None:
        if (self.path is None) == (self.repo_id is None):
            raise ValueError("Specify exactly one of model path or repo_id")


@dataclass(frozen=True, slots=True)
class RtDetrV2EngineOptions:
    score_threshold: float = 0.3
    dtype: mx.Dtype | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be in [0, 1]")


class RtDetrV2Engine:
    """Load and run an MLX-layout or upstream HF RT-DETR-v2 checkpoint."""

    def __init__(
        self,
        model_spec: RtDetrV2ModelSpec,
        options: RtDetrV2EngineOptions | None = None,
    ) -> None:
        self.model_spec = model_spec
        self.options = options or RtDetrV2EngineOptions()
        self._model: RtDetrV2 | None = None
        self._config: RtDetrV2Config | None = None
        self._preprocessing: RtDetrPreprocessingSpec | None = None
        self._dtype: mx.Dtype | None = None
        self._init_lock = Lock()
        self._warmup_done = False
        self.directory: Path | None = None

    def _resolve_directory(self) -> Path:
        return resolve_checkpoint(self.model_spec, files=CHECKPOINT_FILES, component="RT-DETR-v2")

    def _warmup(self) -> None:
        if self._model is None or self._preprocessing is None or self._dtype is None:
            raise RuntimeError("RT-DETR-v2 warmup requires an initialized model")
        import mlx.core as mx

        height, width = self._preprocessing.size
        pixels = mx.array(
            preprocess_images([Image.new("RGB", (width, height))], self._preprocessing),
            dtype=self._dtype,
        )
        output = self._model(pixels)
        mx.eval(output["pred_logits"], output["pred_boxes"])

    def initialize(self, warmup: bool = False) -> None:
        with self._init_lock:
            if self._model is None:
                require_apple_silicon()
                directory = self._resolve_directory()
                config, preprocessing = _validate_artifact(directory)
                import mlx.core as mx
                from mlx.utils import tree_flatten

                from ...._models.rt_detr_v2 import RtDetrV2

                model = RtDetrV2(config)
                target_shapes = {}
                for entry in tree_flatten(model.parameters()):
                    if not isinstance(entry, tuple):
                        continue
                    key, value = entry
                    target_shapes[key] = tuple(value.shape)
                state, _ = load_state_dict(str(directory / "model.safetensors"), target_shapes)
                dtype = self.options.dtype or mx.array(next(iter(state.values()))).dtype
                model.load_weights(
                    [(key, mx.array(value, dtype=dtype)) for key, value in state.items()],
                    strict=True,
                )
                model.eval()
                mx.eval(model.parameters())
                model.compile_backbone()
                self._model = model
                self._config = config
                self._preprocessing = preprocessing
                self._dtype = dtype
                self.directory = directory
            if warmup and not self._warmup_done:
                self._warmup()
                self._warmup_done = True

    def get_label_mapping(self) -> dict[int, str]:
        self.initialize()
        if self._config is None:
            raise RuntimeError("RT-DETR-v2 initialization did not produce a configuration")
        return dict(self._config.id2label)

    def predict(self, images: list[Image.Image]) -> list[Detections]:
        """Return thresholded pixel ``xyxy`` detections in input order."""

        if not images:
            return []
        self.initialize()
        if (
            self._model is None
            or self._config is None
            or self._preprocessing is None
            or self._dtype is None
        ):
            raise RuntimeError("RT-DETR-v2 initialization did not produce a model")
        import mlx.core as mx

        pixels = mx.array(preprocess_images(images, self._preprocessing), dtype=self._dtype)
        output = self._model(pixels)
        raw_logits, raw_boxes = output["pred_logits"], output["pred_boxes"]
        mx.eval(raw_logits, raw_boxes)
        logits = np.array(raw_logits, copy=True)
        boxes = np.array(raw_boxes, copy=True)
        expected = (len(images), self._config.num_queries)
        if logits.shape != (*expected, self._config.num_labels) or boxes.shape != (*expected, 4):
            raise RuntimeError(
                f"RT-DETR-v2 model returned invalid shapes {logits.shape} and {boxes.shape}"
            )
        outputs = _postprocess_focal_detections(
            logits,
            boxes,
            [(image.width, image.height) for image in images],
            score_threshold=self.options.score_threshold,
        )
        return [
            Detections(
                boxes=pixel_boxes,
                scores=scores,
                label_ids=label_ids,
                id2label=dict(self._config.id2label),
            )
            for label_ids, scores, pixel_boxes in outputs
        ]


__all__ = ["Detections", "RtDetrV2Engine", "RtDetrV2EngineOptions", "RtDetrV2ModelSpec"]
