# SPDX-License-Identifier: Apache-2.0

"""Framework-free MLX EfficientNet image classification."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from ...._models.efficientnet.weights import load_state_dict
from ....runtime.guards import require_apple_silicon
from ..._shared import resolve_checkpoint
from .artifact import CHECKPOINT_FILES, _validate_artifact
from .preprocessing import EfficientNetPreprocessingSpec, preprocess_images

if TYPE_CHECKING:
    import mlx.core as mx

    from ...._models.efficientnet.config import EfficientNetConfig
    from ...._models.efficientnet.model import EfficientNet


@dataclass(frozen=True, slots=True)
class EfficientNetModelSpec:
    """An immutable Hub model identity or a local checkpoint directory."""

    repo_id: str | None = None
    revision: str | None = None
    path: Path | str | None = None

    def __post_init__(self) -> None:
        if (self.path is None) == (self.repo_id is None):
            raise ValueError("Specify exactly one of model path or repo_id")


@dataclass(frozen=True, slots=True)
class EfficientNetEngineOptions:
    """Classification output and model dtype settings."""

    top_k: int | None = 5
    dtype: mx.Dtype | None = None

    def __post_init__(self) -> None:
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be positive or None")


@dataclass(frozen=True, slots=True)
class Classification:
    """Top-k classification results for one input image."""

    label_ids: list[int]
    probabilities: list[float]
    id2label: dict[int, str]

    @property
    def scores(self) -> list[float]:
        """Compatibility alias for Docling's score terminology."""

        return self.probabilities


class EfficientNetEngine:
    """Load and run an MLX-layout or upstream HF EfficientNet checkpoint."""

    def __init__(
        self,
        model_spec: EfficientNetModelSpec,
        options: EfficientNetEngineOptions | None = None,
    ) -> None:
        self.model_spec = model_spec
        self.options = options or EfficientNetEngineOptions()
        self._model: EfficientNet | None = None
        self._config: EfficientNetConfig | None = None
        self._preprocessing: EfficientNetPreprocessingSpec | None = None
        self._dtype: mx.Dtype | None = None
        self._init_lock = Lock()
        self._warmup_done = False
        self.directory: Path | None = None

    def _resolve_directory(self) -> Path:
        return resolve_checkpoint(self.model_spec, files=CHECKPOINT_FILES, component="EfficientNet")

    def _warmup(self) -> None:
        if (
            self._model is None
            or self._config is None
            or self._preprocessing is None
            or self._dtype is None
        ):
            raise RuntimeError("EfficientNet warmup requires an initialized model")
        import mlx.core as mx

        height, width = self._config.image_size
        pixels = mx.array(
            preprocess_images([Image.new("RGB", (width, height))], self._preprocessing),
            dtype=self._dtype,
        )
        mx.eval(self._model(pixels))

    def initialize(self, warmup: bool = False) -> None:
        with self._init_lock:
            if self._model is None:
                require_apple_silicon()
                directory = self._resolve_directory()
                config, preprocessing = _validate_artifact(directory)
                import mlx.core as mx
                from mlx.utils import tree_flatten

                from ...._models.efficientnet.model import EfficientNet

                model = EfficientNet(config)
                target_shapes = {}
                for entry in tree_flatten(model.parameters()):
                    if isinstance(entry, tuple):
                        key, value = entry
                        target_shapes[key] = tuple(value.shape)
                state, _ = load_state_dict(str(directory / "model.safetensors"), target_shapes)
                if not state:
                    raise ValueError("EfficientNet checkpoint is empty")
                dtype = self.options.dtype or mx.array(next(iter(state.values()))).dtype
                # MLX's half-precision BatchNorm path can overflow on EfficientNet's
                # small running variances; keep the model in FP32 while allowing
                # callers to choose the input dtype.
                model_dtype = mx.float32 if dtype in (mx.float16, mx.bfloat16) else dtype
                model.load_weights(
                    [(key, mx.array(value, dtype=model_dtype)) for key, value in state.items()],
                    strict=True,
                )
                model.eval()
                mx.eval(model.parameters())
                model.compile_forward()
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
            raise RuntimeError("EfficientNet initialization did not produce a configuration")
        return dict(self._config.id2label)

    def predict_logits(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Return finite CPU logits in input order."""

        if not images:
            return np.empty((0, 0), dtype=np.float32)
        self.initialize()
        if self._model is None or self._config is None or self._preprocessing is None:
            raise RuntimeError("EfficientNet initialization did not produce a model")
        import mlx.core as mx

        pixels = mx.array(preprocess_images(images, self._preprocessing), dtype=self._dtype)
        logits = self._model(pixels)
        mx.eval(logits)
        result = np.array(logits, dtype=np.float32, copy=True)
        expected = (len(images), self._config.num_labels)
        if result.shape != expected or not np.isfinite(result).all():
            raise RuntimeError(f"EfficientNet returned invalid logits with shape {result.shape}")
        return result

    def predict(self, images: Sequence[Image.Image]) -> list[Classification]:
        """Return top-k labels and probabilities for each image."""

        if not images:
            return []
        logits = self.predict_logits(images)
        if self._config is None:
            raise RuntimeError("EfficientNet initialization did not produce a configuration")
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        count = (
            self._config.num_labels
            if self.options.top_k is None
            else min(self.options.top_k, self._config.num_labels)
        )
        mapping = dict(self._config.id2label)
        return [
            Classification(
                label_ids=(indices := np.argsort(-scores, kind="stable")[:count]).tolist(),
                probabilities=scores[indices].tolist(),
                id2label=mapping,
            )
            for scores in probabilities
        ]


__all__ = [
    "Classification",
    "EfficientNetEngine",
    "EfficientNetEngineOptions",
    "EfficientNetModelSpec",
]
