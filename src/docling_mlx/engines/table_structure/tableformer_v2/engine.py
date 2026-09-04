# SPDX-License-Identifier: Apache-2.0

"""Framework-free MLX TableFormerV2 inference."""

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
from .artifact import CHECKPOINT_FILES, TableFormerV2Artifact, validate_tableformer_v2_artifact
from .conversion import convert_upstream_state_dict
from .model_spec import TableFormerV2ModelSpec
from .preprocessing import preprocess_images

if TYPE_CHECKING:
    from ...._models.tableformer_v2.model import TableFormerV2


@dataclass(frozen=True, slots=True)
class TableFormerV2EngineOptions:
    """Reserved generic TableFormerV2 runtime options."""


@dataclass(frozen=True, slots=True)
class TableFormerV2Prediction:
    """Tag IDs, OTSL tags, and crop-pixel cell boxes for one table image."""

    token_ids: tuple[int, ...]
    otsl_tokens: tuple[str, ...]
    cell_bboxes: tuple[tuple[float, float, float, float], ...]


class TableFormerV2Engine:
    """Load and run an MLX TableFormerV2 checkpoint over Pillow crops."""

    def __init__(
        self,
        model_spec: TableFormerV2ModelSpec,
        options: TableFormerV2EngineOptions | None = None,
    ) -> None:
        self.model_spec = model_spec
        self.options = options or TableFormerV2EngineOptions()
        self._model: TableFormerV2 | None = None
        self._artifact: TableFormerV2Artifact | None = None
        self._init_lock = Lock()
        self._warmup_done = False
        self.directory: Path | None = None

    @property
    def artifact(self) -> TableFormerV2Artifact:
        self.initialize()
        if self._artifact is None:
            raise RuntimeError("TableFormerV2 initialization did not produce an artifact")
        return self._artifact

    def _warmup(self) -> None:
        if self._model is None or self._artifact is None:
            raise RuntimeError("TableFormerV2 warmup requires an initialized model")
        import mlx.core as mx

        height, width = self._artifact.preprocessing.size
        generated = self._model.generate(
            preprocess_images([Image.new("RGB", (width, height))], self._artifact.preprocessing),
            max_generation_steps=1,
        )
        mx.eval(generated.generated_ids, generated.predicted_bboxes)

    def initialize(self, warmup: bool = False) -> None:
        with self._init_lock:
            if self._model is None:
                require_apple_silicon()
                directory = resolve_checkpoint(
                    self.model_spec, files=CHECKPOINT_FILES, component="TableFormerV2"
                )
                artifact = validate_tableformer_v2_artifact(directory)
                import mlx.core as mx

                from ...._models.tableformer_v2.model import TableFormerV2

                model = TableFormerV2(artifact.config)
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
                        raise ValueError("All TableFormerV2 model state tensors must be float32")
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
    def _otsl_tokens(token_ids: Sequence[int], artifact: TableFormerV2Artifact) -> tuple[str, ...]:
        tags: list[str] = []
        for token_id in token_ids:
            if type(token_id) is not int or not 0 <= token_id < len(artifact.token_map.tokens):
                raise ValueError(f"TableFormerV2 emitted unsupported token ID {token_id!r}")
            token = artifact.token_map.tokens[token_id]
            if token_id in {
                artifact.token_map.pad_token_id,
                1,
                artifact.token_map.bos_token_id,
                artifact.token_map.eos_token_id,
            }:
                continue
            if not token.startswith("<") or not token.endswith(">"):
                raise ValueError(f"TableFormerV2 emitted unsupported token string {token!r}")
            tags.append(token[1:-1])
        return tuple(tags)

    @classmethod
    def _materialize_prediction(
        cls,
        generated_ids: object,
        predicted_bboxes: object,
        artifact: TableFormerV2Artifact,
        image: Image.Image,
    ) -> TableFormerV2Prediction:
        token_array = np.array(generated_ids, copy=True)
        bbox_array = np.array(predicted_bboxes, copy=True)
        max_sequence_length = artifact.generation.max_generation_steps + 1
        if (
            token_array.ndim != 2
            or token_array.shape[0] != 1
            or not 1 <= token_array.shape[1] <= max_sequence_length
            or not np.issubdtype(token_array.dtype, np.integer)
        ):
            raise RuntimeError("TableFormerV2 generated token IDs have an invalid shape or dtype")
        token_ids = tuple(int(token_id) for token_id in token_array[0])
        if token_ids[0] != artifact.token_map.bos_token_id:
            raise RuntimeError("TableFormerV2 generated sequence does not start with BOS")
        expected_bbox_count = sum(
            token_id in artifact.token_map.data_cell_token_ids for token_id in token_ids
        )
        if bbox_array.shape != (expected_bbox_count, 4):
            raise RuntimeError("TableFormerV2 generated an invalid number of cell boxes")
        scale = np.asarray((image.width, image.height, image.width, image.height), dtype=np.float32)
        cell_bboxes = cast(
            tuple[tuple[float, float, float, float], ...],
            tuple(tuple(float(coordinate) for coordinate in bbox * scale) for bbox in bbox_array),
        )
        return TableFormerV2Prediction(
            token_ids=token_ids,
            otsl_tokens=cls._otsl_tokens(token_ids, artifact),
            cell_bboxes=cell_bboxes,
        )

    def predict(self, images: Sequence[Image.Image]) -> list[TableFormerV2Prediction]:
        """Return one pure crop-local structure prediction per input image."""

        if not images:
            return []
        self.initialize()
        if self._model is None or self._artifact is None:
            raise RuntimeError("TableFormerV2 initialization did not produce a model")
        import mlx.core as mx

        results: list[TableFormerV2Prediction] = []
        for image in images:
            generated = self._model.generate(
                preprocess_images([image], self._artifact.preprocessing),
                max_generation_steps=self._artifact.generation.max_generation_steps,
            )
            mx.eval(generated.generated_ids, generated.predicted_bboxes)
            results.append(
                self._materialize_prediction(
                    generated.generated_ids,
                    generated.predicted_bboxes,
                    self._artifact,
                    image,
                )
            )
        return results


__all__ = [
    "TableFormerV2Engine",
    "TableFormerV2EngineOptions",
    "TableFormerV2ModelSpec",
    "TableFormerV2Prediction",
]
