# SPDX-License-Identifier: Apache-2.0

"""Internal corrected Granite Vision 4.1 engine for Docling's MLX runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.stage_model_specs import EngineModelConfig
from docling.datamodel.vlm_engine_options import MlxVlmEngineOptions
from docling.models.inference_engines.vlm.mlx_engine import MlxVlmEngine

from docling_mlx._compat.docling import resolve_model_artifacts_path
from docling_mlx._compat.mlx_vlm import (
    apply_granite_vision_chat_template,
    correct_loaded_granite_vision_activations,
    load_granite_vision_config,
    load_granite_vision_model,
    replace_granite_vision_image_processor,
)
from docling_mlx.runtime.guards import validate_mlx_accelerator


def _apply_granite_vision_chat_template(
    processor: object,
    config: object,
    prompt: object,
    add_generation_prompt: bool = True,
    return_messages: bool = False,
    num_images: int = 0,
    num_audios: int = 0,
    **kwargs: object,
) -> Any:
    if return_messages:
        raise ValueError("Granite Vision chat template does not support return_messages")
    if num_images < 1:
        raise ValueError("Granite Vision chat template requires at least one image")
    if num_audios:
        raise ValueError("Granite Vision chat template does not support audio")

    content: list[dict[str, object]] = [{"type": "image"} for _ in range(num_images)]
    content.append({"type": "text", "text": prompt})
    messages: list[dict[str, object]] = [{"role": "user", "content": content}]
    return apply_granite_vision_chat_template(
        processor,
        messages,
        add_generation_prompt,
        **kwargs,
    )


class MlxGraniteVision41Engine(MlxVlmEngine):
    """Correct the Granite Vision activation boundary in mlx-vlm."""

    def __init__(
        self,
        options: MlxVlmEngineOptions,
        artifacts_path: Path | str | None,
        model_config: EngineModelConfig | None = None,
        *,
        accelerator_options: AcceleratorOptions | None = None,
    ) -> None:
        if accelerator_options is not None:
            validate_mlx_accelerator(accelerator_options)
        super().__init__(options, artifacts_path, model_config)

    def _load_model_for_repo(self, repo_id: str, revision: str = "main") -> None:
        from mlx import nn
        from transformers import AutoImageProcessor

        artifact_path = resolve_model_artifacts_path(
            repo_id,
            revision,
            self.artifacts_path,
            lambda requested_repo, requested_revision: self.download_models(
                requested_repo,
                revision=requested_revision,
            ),
        )
        config = load_granite_vision_config(artifact_path)
        model, processor = load_granite_vision_model(artifact_path)
        correct_loaded_granite_vision_activations(model, gelu_type=nn.GELU)
        replace_granite_vision_image_processor(
            processor,
            artifact_path,
            auto_image_processor=AutoImageProcessor,
        )

        self.vlm_model = model
        self.processor = processor
        self.config = config
        self.apply_chat_template = _apply_granite_vision_chat_template
