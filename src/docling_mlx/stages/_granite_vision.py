# SPDX-License-Identifier: Apache-2.0

"""Shared Docling VLM boundary for official Granite Vision 4.1 tasks."""

from __future__ import annotations

from pathlib import Path

import PIL.Image
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options_vlm_model import ResponseFormat
from docling.datamodel.stage_model_specs import VlmModelSpec
from docling.datamodel.vlm_engine_options import MlxVlmEngineOptions
from docling.models.inference_engines.vlm import (
    BaseVlmEngine,
    VlmEngineInput,
    VlmEngineType,
)

GRANITE_VISION_4_1_REPO_ID = "ibm-granite/granite-vision-4.1-4b"
GRANITE_VISION_4_1_REVISION = "dd48e97503de471803850df70843cf9eb5da8712"


def granite_vision_model_spec(
    *,
    name: str,
    prompt: str,
    response_format: ResponseFormat,
) -> VlmModelSpec:
    """Build one task-specific spec for the pinned official model."""

    return VlmModelSpec(
        name=name,
        default_repo_id=GRANITE_VISION_4_1_REPO_ID,
        revision=GRANITE_VISION_4_1_REVISION,
        prompt=prompt,
        response_format=response_format,
        supported_engines={VlmEngineType.MLX},
    )


def build_granite_vision_input(
    model_spec: VlmModelSpec,
    image: PIL.Image.Image,
    *,
    prompt: str | None = None,
) -> VlmEngineInput:
    """Translate a task crop and model spec into Docling's engine input."""

    return VlmEngineInput(
        image=image,
        prompt=model_spec.prompt if prompt is None else prompt,
        temperature=model_spec.temperature,
        max_new_tokens=model_spec.max_new_tokens,
        stop_strings=list(model_spec.stop_strings),
        extra_generation_config=model_spec.get_runtime_input_extra_config(VlmEngineType.MLX),
    )


def create_granite_vision_engine(
    *,
    engine_options: MlxVlmEngineOptions,
    model_spec: VlmModelSpec,
    artifacts_path: Path | str | None,
    accelerator_options: AcceleratorOptions,
) -> BaseVlmEngine:
    """Create the internal corrected engine without changing Docling's registry."""

    from docling_mlx.stages.granite_vision_engine import MlxGraniteVision41Engine

    return MlxGraniteVision41Engine(
        engine_options,
        artifacts_path=artifacts_path,
        model_config=model_spec.get_engine_config(VlmEngineType.MLX),
        accelerator_options=accelerator_options,
    )


__all__ = [
    "GRANITE_VISION_4_1_REPO_ID",
    "GRANITE_VISION_4_1_REVISION",
    "build_granite_vision_input",
    "create_granite_vision_engine",
    "granite_vision_model_spec",
]
