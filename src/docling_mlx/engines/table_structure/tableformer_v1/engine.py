# SPDX-License-Identifier: Apache-2.0

"""Framework-free MLX TableFormerV1 inference."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from PIL import Image

from ....runtime.guards import require_apple_silicon
from ..._shared import resolve_checkpoint
from .artifact import (
    CHECKPOINT_FILES,
    TABLEFORMER_V1_TAG_MAP,
    TableFormerV1Artifact,
    validate_tableformer_v1_artifact,
)
from .conversion import convert_upstream_state_dict
from .model_spec import TableFormerV1ModelSpec
from .preprocessing import preprocess_table_image

if TYPE_CHECKING:
    from ...._models.tableformer_v1.model import TableFormerV1


@dataclass(frozen=True, slots=True)
class TableFormerV1EngineOptions:
    """Generic TableFormerV1 loading options."""

    checkpoint_subdirectory: str | None = None

    def __post_init__(self) -> None:
        if self.checkpoint_subdirectory is not None and (
            not self.checkpoint_subdirectory
            or Path(self.checkpoint_subdirectory).is_absolute()
            or ".." in Path(self.checkpoint_subdirectory).parts
        ):
            raise ValueError("checkpoint_subdirectory must be a relative directory name")


@dataclass(frozen=True, slots=True)
class TableFormerV1Prediction:
    """Tag IDs, OTSL tags, and crop-pixel cell boxes for one table image."""

    token_ids: tuple[int, ...]
    otsl_tokens: tuple[str, ...]
    cell_bboxes: tuple[tuple[float, float, float, float], ...]
    bbox_classes: tuple[int, ...]


class TableFormerV1Engine:
    """Load and run an MLX TableFormerV1 checkpoint over Pillow crops."""

    def __init__(
        self,
        model_spec: TableFormerV1ModelSpec,
        options: TableFormerV1EngineOptions | None = None,
    ) -> None:
        self.model_spec = model_spec
        self.options = options or TableFormerV1EngineOptions()
        self._model: TableFormerV1 | None = None
        self._artifact: TableFormerV1Artifact | None = None
        self._init_lock = Lock()
        self._warmup_done = False
        self.directory: Path | None = None

    @property
    def artifact(self) -> TableFormerV1Artifact:
        self.initialize()
        if self._artifact is None:
            raise RuntimeError("TableFormerV1 initialization did not produce an artifact")
        return self._artifact

    def _resolve_directory(self) -> Path:
        subdirectory = self.options.checkpoint_subdirectory
        if subdirectory is None:
            return resolve_checkpoint(
                self.model_spec, files=CHECKPOINT_FILES, component="TableFormerV1"
            )
        files = (
            *(f"{subdirectory}/{name}" for name in CHECKPOINT_FILES),
            f"model_artifacts/tableformer/{subdirectory}/tm_config.json",
            f"model_artifacts/tableformer/{subdirectory}/tableformer_{subdirectory}.safetensors",
        )
        root = resolve_checkpoint(self.model_spec, files=files, component="TableFormerV1")
        converted = root / subdirectory
        upstream = root / "model_artifacts" / "tableformer" / subdirectory
        if (upstream / "tm_config.json").is_file():
            return upstream
        return converted

    def _warmup(self) -> None:
        if self._model is None or self._artifact is None:
            raise RuntimeError("TableFormerV1 warmup requires an initialized model")
        import mlx.core as mx

        size = self._artifact.preprocessing.image_size
        pixels = mx.array(
            preprocess_table_image(
                np.zeros((size, size, 3), dtype=np.uint8), self._artifact.preprocessing
            )
        )
        outputs = self._model.generate(pixels, max_generation_steps=1)
        mx.eval(*outputs)

    def initialize(self, warmup: bool = False) -> None:
        with self._init_lock:
            if self._model is None:
                require_apple_silicon()
                directory = self._resolve_directory()
                artifact = validate_tableformer_v1_artifact(directory)
                import mlx.core as mx

                from ...._models.tableformer_v1.model import TableFormerV1

                model = TableFormerV1(artifact.config)
                if artifact.upstream_weights:
                    from mlx.utils import tree_flatten
                    from safetensors.numpy import load_file

                    flattened = cast(list[tuple[str, Any]], tree_flatten(model.parameters()))
                    target_shapes = {key: tuple(value.shape) for key, value in flattened}
                    converted_weights, _, _ = convert_upstream_state_dict(
                        load_file(str(artifact.weights_path)), target_shapes
                    )
                    model.load_weights(
                        [(key, mx.array(value)) for key, value in converted_weights.items()],
                        strict=True,
                    )
                else:
                    loaded_weights: Any = mx.load(str(artifact.weights_path))
                    if not isinstance(loaded_weights, dict) or any(
                        tensor.dtype != mx.float32 for tensor in loaded_weights.values()
                    ):
                        raise ValueError("All TableFormerV1 model state tensors must be float32")
                    model.load_weights(cast(Any, list(loaded_weights.items())), strict=True)
                model.eval()
                mx.eval(model.parameters())
                model.compile_image_backbone()
                self._model = model
                self._artifact = artifact
                self.directory = directory
            if warmup and not self._warmup_done:
                self._warmup()
                self._warmup_done = True

    @staticmethod
    def _decode_token_ids(token_ids: Sequence[int]) -> tuple[str, ...]:
        tokens_by_id = tuple(TABLEFORMER_V1_TAG_MAP)
        for token_id in token_ids:
            if type(token_id) is not int or not 0 <= token_id < len(tokens_by_id):
                raise ValueError(f"TableFormerV1 emitted unsupported token ID {token_id!r}")
        return tuple(tokens_by_id[token_id] for token_id in token_ids[1:-1])

    @classmethod
    def _materialize_prediction(
        cls,
        generated_ids: object,
        predicted_bboxes: object,
        bbox_classes: object,
        image: Image.Image,
    ) -> TableFormerV1Prediction:
        tokens = np.array(generated_ids, copy=True)
        boxes = np.array(predicted_bboxes, copy=True)
        classes = np.array(bbox_classes, copy=True)
        if tokens.ndim == 2 and tokens.shape[0] == 1:
            tokens = tokens[0]
        if tokens.ndim != 1 or not tokens.size or not np.issubdtype(tokens.dtype, np.integer):
            raise RuntimeError("TableFormerV1 generated token IDs have an invalid shape or dtype")
        token_ids = tuple(int(token_id) for token_id in tokens)
        if token_ids[0] != TABLEFORMER_V1_TAG_MAP["<start>"]:
            raise RuntimeError("TableFormerV1 generated sequence does not start with BOS")
        otsl_tokens = cls._decode_token_ids(token_ids)
        if boxes.ndim != 2 or boxes.shape[1:] != (4,):
            raise RuntimeError("TableFormerV1 generated cell boxes have an invalid shape")
        if classes.ndim == 2:
            if classes.shape != (boxes.shape[0], 3):
                raise RuntimeError("TableFormerV1 generated cell classes have an invalid shape")
            classes = classes.argmax(axis=1)
        if classes.shape != (boxes.shape[0],):
            raise RuntimeError("TableFormerV1 generated cell classes have an invalid shape")
        scale = np.asarray((image.width, image.height, image.width, image.height), dtype=np.float32)
        cell_bboxes = cast(
            tuple[tuple[float, float, float, float], ...],
            tuple(tuple(float(coordinate) for coordinate in bbox * scale) for bbox in boxes),
        )
        return TableFormerV1Prediction(
            token_ids=token_ids,
            otsl_tokens=otsl_tokens,
            cell_bboxes=cell_bboxes,
            bbox_classes=tuple(int(value) for value in classes),
        )

    def predict(self, images: Sequence[Image.Image]) -> list[TableFormerV1Prediction]:
        """Return one pure crop-local structure prediction per input image."""

        if not images:
            return []
        self.initialize()
        if self._model is None or self._artifact is None:
            raise RuntimeError("TableFormerV1 initialization did not produce a model")
        import mlx.core as mx

        results: list[TableFormerV1Prediction] = []
        for image in images:
            pixels = mx.array(
                preprocess_table_image(
                    np.asarray(image.convert("RGB")), self._artifact.preprocessing
                )
            )
            generated_ids, bbox_classes, predicted_bboxes = self._model.generate(
                pixels,
                max_generation_steps=self._artifact.generation.max_generation_steps,
            )
            mx.eval(generated_ids, bbox_classes, predicted_bboxes)
            max_sequence_length = self._artifact.generation.max_generation_steps + 1
            if np.asarray(generated_ids).shape[-1] > max_sequence_length:
                raise RuntimeError("TableFormerV1 generated sequence exceeds its configured limit")
            results.append(
                self._materialize_prediction(generated_ids, predicted_bboxes, bbox_classes, image)
            )
        return results


__all__ = [
    "TableFormerV1Engine",
    "TableFormerV1EngineOptions",
    "TableFormerV1ModelSpec",
    "TableFormerV1Prediction",
]
