# SPDX-License-Identifier: Apache-2.0

"""Docling layout detection backed by native MLX engines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal, cast

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import (
    BaseLayoutOptions,
    LayoutObjectDetectionOptions,
)
from docling.datamodel.stage_model_specs import (
    EngineModelConfig,
    ObjectDetectionModelSpec,
    ObjectDetectionStagePreset,
    ObjectDetectionStagePresetMixin,
)
from docling.models.inference_engines.object_detection.base import (
    BaseObjectDetectionEngine,
    BaseObjectDetectionEngineOptions,
    ObjectDetectionEngineInput,
    ObjectDetectionEngineOutput,
    ObjectDetectionEngineType,
)
from docling.models.stages.layout.layout_object_detection_model import (
    LayoutObjectDetectionModel,
)
from pydantic import BaseModel, ConfigDict, Field

from docling_mlx.engines._shared import read_json_object, resolve_artifact_checkpoint
from docling_mlx.engines.object_detection.dfine.artifact import CHECKPOINT_FILES as DFINE_FILES
from docling_mlx.engines.object_detection.dfine.engine import (
    DFineEngine,
    DFineEngineOptions,
    DFineModelSpec,
)
from docling_mlx.engines.object_detection.rt_detr_v2.artifact import (
    CHECKPOINT_FILES as RT_DETR_V2_FILES,
)
from docling_mlx.engines.object_detection.rt_detr_v2.engine import (
    RtDetrV2Engine,
    RtDetrV2EngineOptions,
    RtDetrV2ModelSpec,
)
from docling_mlx.presets import resolve_preset
from docling_mlx.runtime.guards import validate_mlx_accelerator


class MlxObjectDetectionEngineOptions(BaseModel):
    """Serializable additions used by the native MLX detector engines."""

    model_config = ConfigDict(extra="forbid")

    score_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    dtype: Literal["float16", "float32"] | None = None
    warmup: bool = False


def _layout_model_spec(preset_id: str) -> ObjectDetectionModelSpec:
    mirror = resolve_preset(preset_id)
    upstream = LayoutObjectDetectionOptions.get_preset(preset_id)
    return upstream.model_spec.model_copy(
        deep=True,
        update={
            "repo_id": mirror.repo_id,
            "revision": mirror.revision,
            "engine_overrides": {},
        },
    )


def _layout_preset(preset_id: str) -> ObjectDetectionStagePreset:
    upstream = LayoutObjectDetectionOptions.get_preset(preset_id)
    return upstream.model_copy(
        deep=True,
        update={
            "model_spec": _layout_model_spec(preset_id),
            # Docling requires every preset to name a member of its closed enum.
            "default_engine_type": ObjectDetectionEngineType.TRANSFORMERS,
        },
    )


class MlxLayoutObjectDetectionOptions(
    ObjectDetectionStagePresetMixin,
    BaseLayoutOptions,
):
    """Official Docling layout shape with an MLX-specific engine addition."""

    model_config = ConfigDict(extra="forbid")

    kind: ClassVar[str] = "mlx_layout_object_detection"
    model_spec: ObjectDetectionModelSpec = Field(
        default_factory=lambda: _layout_model_spec("layout_heron_default")
    )
    engine_options: MlxObjectDetectionEngineOptions = Field(
        default_factory=MlxObjectDetectionEngineOptions
    )

    @classmethod
    def from_preset(
        cls,
        preset_id: str,
        engine_options: BaseObjectDetectionEngineOptions
        | MlxObjectDetectionEngineOptions
        | None = None,
        **overrides: Any,
    ) -> MlxLayoutObjectDetectionOptions:
        """Build an MLX preset without constructing Docling's closed engine options."""

        if engine_options is not None and not isinstance(
            engine_options, MlxObjectDetectionEngineOptions
        ):
            raise TypeError("MLX layout presets require MlxObjectDetectionEngineOptions")
        mlx_engine_options = (
            MlxObjectDetectionEngineOptions()
            if engine_options is None
            else cast(MlxObjectDetectionEngineOptions, engine_options)
        )
        preset = cls.get_preset(preset_id)
        instance = cls(
            model_spec=preset.model_spec.model_copy(deep=True),
            engine_options=mlx_engine_options,
        )
        for key, value in {**preset.stage_options, **overrides}.items():
            setattr(instance, key, value)
        return instance


for _preset_id in (
    "layout_heron_default",
    "layout_heron_101",
    "layout_egret_medium",
    "layout_egret_large",
    "layout_egret_xlarge",
):
    MlxLayoutObjectDetectionOptions.register_preset(_layout_preset(_preset_id))


class _MlxObjectDetectionEngine(BaseObjectDetectionEngine):
    """Select the native detector family from checkpoint metadata."""

    def __init__(
        self,
        options: MlxObjectDetectionEngineOptions,
        *,
        model_config: EngineModelConfig,
        accelerator_options: AcceleratorOptions,
        artifacts_path: Path | str | None,
    ) -> None:
        copied_options = options.model_copy(deep=True)
        copied_model_config = model_config.model_copy(deep=True)
        # MLX cannot be represented by Docling's closed ObjectDetectionEngineType.
        super().__init__(
            cast(BaseObjectDetectionEngineOptions, copied_options),
            model_config=copied_model_config,
        )
        validate_mlx_accelerator(accelerator_options)
        self._mlx_options = copied_options
        self._model_config = copied_model_config
        self._artifacts_path = artifacts_path
        repo_id = self._model_config.repo_id
        revision = self._model_config.revision
        if not repo_id:
            raise ValueError("Mlx layout detection requires model_spec.repo_id")
        if not revision:
            raise ValueError("Mlx layout detection requires model_spec.revision")
        directory = resolve_artifact_checkpoint(
            repo_id,
            revision,
            self._artifacts_path,
            files=tuple(dict.fromkeys((*RT_DETR_V2_FILES, *DFINE_FILES))),
        )
        self._engine = self._build_engine(directory)
        self.artifact_path: Path | None = None

    def _build_engine(self, directory: Path) -> RtDetrV2Engine | DFineEngine:
        model_type = read_json_object(directory / "config.json").get("model_type")
        dtype = None
        if self._mlx_options.dtype is not None:
            import mlx.core as mx

            dtype = getattr(mx, self._mlx_options.dtype)
        if model_type == "rt_detr_v2":
            return RtDetrV2Engine(
                RtDetrV2ModelSpec(path=directory),
                RtDetrV2EngineOptions(
                    score_threshold=self._mlx_options.score_threshold,
                    dtype=dtype,
                ),
            )
        if model_type == "d_fine":
            return DFineEngine(
                DFineModelSpec(path=directory),
                DFineEngineOptions(
                    score_threshold=self._mlx_options.score_threshold,
                    dtype=dtype,
                ),
            )
        raise ValueError(
            f"Unsupported MLX layout checkpoint model_type in {directory / 'config.json'}: "
            f"{model_type!r}"
        )

    def initialize(self, *, warmup: bool = False) -> None:
        self._engine.initialize(warmup=warmup)
        self.artifact_path = self._engine.directory
        self._initialized = True

    def get_label_mapping(self) -> dict[int, str]:
        return self._engine.get_label_mapping()

    def predict_batch(
        self, input_batch: list[ObjectDetectionEngineInput]
    ) -> list[ObjectDetectionEngineOutput]:
        if not input_batch:
            return []
        detections = self._engine.predict([item.image for item in input_batch])
        return [
            ObjectDetectionEngineOutput(
                label_ids=result.label_ids,
                scores=result.scores,
                bboxes=result.boxes,
                metadata=dict(item.metadata),
            )
            for result, item in zip(detections, input_batch, strict=True)
        ]


class MlxLayoutObjectDetectionModel(LayoutObjectDetectionModel):
    """Use Docling's official layout stage around a native MLX detector."""

    def __init__(
        self,
        artifacts_path: Path | None,
        accelerator_options: AcceleratorOptions,
        options: MlxLayoutObjectDetectionOptions,
        enable_remote_services: bool = False,
    ) -> None:
        del enable_remote_services
        copied_options = options.model_copy(deep=True)
        self.options = cast(LayoutObjectDetectionOptions, copied_options)
        self.engine = _MlxObjectDetectionEngine(
            copied_options.engine_options,
            model_config=EngineModelConfig(
                repo_id=copied_options.model_spec.repo_id,
                revision=copied_options.model_spec.revision,
            ),
            accelerator_options=accelerator_options,
            artifacts_path=artifacts_path,
        )
        self.engine.initialize(warmup=copied_options.engine_options.warmup)
        self._label_map = self._build_label_map()
        self._unmapped_label_ids: set[int] = set()

    @classmethod
    def get_options_type(cls) -> type[LayoutObjectDetectionOptions]:
        """Return the exact options type used by Docling's layout factory."""

        return cast(type[LayoutObjectDetectionOptions], MlxLayoutObjectDetectionOptions)


__all__ = [
    "MlxLayoutObjectDetectionModel",
    "MlxLayoutObjectDetectionOptions",
    "MlxObjectDetectionEngineOptions",
]
