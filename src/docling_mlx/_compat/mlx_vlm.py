# SPDX-License-Identifier: Apache-2.0

"""Isolate upstream mlx-vlm private APIs; the supported lower bound lives in pyproject.toml."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast


def apply_granite_vision_chat_template(
    processor: object,
    messages: list[dict[str, object]],
    add_generation_prompt: bool,
    **kwargs: object,
) -> Any:
    """Apply mlx-vlm's template without importing it until Granite is loaded."""

    from mlx_vlm.prompt_utils import get_chat_template

    return get_chat_template(processor, messages, add_generation_prompt, **cast(Any, kwargs))


def load_granite_vision_config(artifact_path: Path) -> object:
    """Load the Granite config through mlx-vlm lazily."""

    from mlx_vlm.utils import load_config

    return load_config(artifact_path)


def load_granite_vision_model(artifact_path: Path) -> tuple[object, object]:
    """Strictly load Granite through mlx-vlm lazily."""

    from mlx_vlm import load

    return load(str(artifact_path), strict=True)


def correct_loaded_granite_vision_activations(model: object, *, gelu_type: type[Any]) -> None:
    """Apply the Granite Vision activation correction."""

    vision_model = model.vision_tower.vision_model  # type: ignore[attr-defined]
    layers = vision_model.encoder.layers
    if len(layers) != 27:
        raise ValueError("Granite Vision model must contain exactly 27 encoder layers")
    for layer in layers:
        layer.mlp.activation_fn = gelu_type(approx="tanh")
    vision_model.head.mlp.activation_fn = gelu_type(approx="tanh")


def replace_granite_vision_image_processor(
    processor: object,
    artifact_path: Path,
    *,
    auto_image_processor: type[Any],
    processor_type: type[Any] | None = None,
) -> None:
    """Require mlx-vlm's processor before installing Docling's torchvision backend."""

    if processor_type is None:
        from mlx_vlm.models.granite4_vision.processing_granite4_vision import (
            Granite4VisionProcessor,
        )

        processor_type = Granite4VisionProcessor
    if not isinstance(processor, processor_type):
        raise TypeError("unexpected Granite Vision processor type")

    replacement = auto_image_processor.from_pretrained(artifact_path)
    if getattr(replacement, "backend", None) != "torchvision":
        raise RuntimeError("Granite Vision 4.1 requires the torchvision image processor")
    processor.image_processor = replacement


__all__ = [
    "apply_granite_vision_chat_template",
    "correct_loaded_granite_vision_activations",
    "load_granite_vision_config",
    "load_granite_vision_model",
    "replace_granite_vision_image_processor",
]
