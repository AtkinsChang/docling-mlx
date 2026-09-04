# SPDX-License-Identifier: Apache-2.0

"""Pillow implementation of ``EfficientNetImageProcessor`` semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps


@dataclass(frozen=True, slots=True)
class EfficientNetPreprocessingSpec:
    do_resize: bool = True
    size: tuple[int, int] = (600, 600)
    resample: Image.Resampling = Image.Resampling.BICUBIC
    do_center_crop: bool = False
    crop_size: tuple[int, int] = (289, 289)
    do_rescale: bool = True
    rescale_factor: float = 1 / 255
    rescale_offset: bool = False
    do_normalize: bool = True
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.47853944, 0.4732864, 0.47434163)
    include_top: bool = True


def _size(value: object, name: str, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    try:
        result = (int(value["height"]), int(value["width"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must specify height and width") from error
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} must specify positive height and width")
    return result


def _triplet(
    value: object, name: str, default: tuple[float, float, float]
) -> tuple[float, float, float]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    if any(isinstance(item, bool) or not isinstance(item, Real) for item in value):
        raise TypeError(f"{name} must contain real values")
    result = tuple(float(item) for item in value)
    if name == "image_std" and any(item <= 0 for item in result):
        raise ValueError("image_std values must be positive")
    return result  # type: ignore[return-value]


def parse_preprocessing_config(
    config: Mapping[str, Any], *, include_top: bool | None = None
) -> EfficientNetPreprocessingSpec:
    """Parse the supported runtime portion of an HF processor config."""

    if not isinstance(config, Mapping):
        raise TypeError("preprocessor configuration must be a mapping")
    defaults = EfficientNetPreprocessingSpec()
    raw_resample = config.get("resample", int(defaults.resample))
    try:
        resample = Image.Resampling(int(raw_resample))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Unsupported PIL resample mode: {raw_resample!r}") from error
    factor = config.get("rescale_factor", defaults.rescale_factor)
    if isinstance(factor, bool) or not isinstance(factor, Real) or float(factor) <= 0:
        raise ValueError("rescale_factor must be a positive real number")
    processor_name = config.get("image_processor_type")
    if include_top is None:
        include_top = bool(
            config.get("include_top", processor_name in {None, "EfficientNetImageProcessor"})
        )
    return EfficientNetPreprocessingSpec(
        do_resize=bool(config.get("do_resize", defaults.do_resize)),
        size=_size(config.get("size"), "size", defaults.size),
        resample=resample,
        do_center_crop=bool(config.get("do_center_crop", defaults.do_center_crop)),
        crop_size=_size(config.get("crop_size"), "crop_size", defaults.crop_size),
        do_rescale=bool(config.get("do_rescale", defaults.do_rescale)),
        rescale_factor=float(factor),
        rescale_offset=bool(config.get("rescale_offset", defaults.rescale_offset)),
        do_normalize=bool(config.get("do_normalize", defaults.do_normalize)),
        image_mean=_triplet(config.get("image_mean"), "image_mean", defaults.image_mean),
        image_std=_triplet(config.get("image_std"), "image_std", defaults.image_std),
        include_top=include_top,
    )


def _rgb(image: Image.Image) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError("EfficientNet preprocessing requires PIL images")
    return image.convert("RGB")


def _center_crop(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    height, width = size
    image_width, image_height = image.size
    if width > image_width or height > image_height:
        left = max((width - image_width) // 2, 0)
        top = max((height - image_height) // 2, 0)
        right = max(width - image_width - left, 0)
        bottom = max(height - image_height - top, 0)
        image = ImageOps.expand(image, border=(left, top, right, bottom), fill=0)
        image_width, image_height = image.size
    left = int((image_width - width) / 2.0)
    top = int((image_height - height) / 2.0)
    return image.crop((left, top, left + width, top + height))


def preprocess_images(
    images: Sequence[Image.Image], spec: EfficientNetPreprocessingSpec
) -> NDArray[np.float32]:
    """Apply resize, crop, rescale, normalize, and optional top normalization."""

    prepared: list[NDArray[np.uint8]] = []
    for image in images:
        rgb = _rgb(image)
        if spec.do_resize:
            rgb = rgb.resize((spec.size[1], spec.size[0]), resample=spec.resample)
        if spec.do_center_crop:
            rgb = _center_crop(rgb, spec.crop_size)
        prepared.append(np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)))
    if not prepared:
        return np.empty((0, *spec.size, 3), dtype=np.float32)

    pixels = np.stack(prepared).astype(np.float32)
    mean = np.asarray(spec.image_mean, dtype=np.float32)
    std = np.asarray(spec.image_std, dtype=np.float32)
    if spec.do_rescale and spec.do_normalize and not spec.rescale_offset:
        # Match the HF backend's fused float32 mean/std path.
        pixels = (pixels - mean / np.float32(spec.rescale_factor)) / (
            std / np.float32(spec.rescale_factor)
        )
    else:
        if spec.do_rescale:
            pixels *= np.float32(spec.rescale_factor)
            if spec.rescale_offset:
                pixels -= np.float32(1)
        if spec.do_normalize:
            pixels = (pixels - mean) / std
    if spec.include_top:
        pixels /= std
    return np.ascontiguousarray(pixels)


__all__ = [
    "EfficientNetPreprocessingSpec",
    "parse_preprocessing_config",
    "preprocess_images",
]
