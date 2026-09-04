# SPDX-License-Identifier: Apache-2.0

"""Pillow implementation of ``RTDetrImageProcessor``'s image path."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image


@dataclass(frozen=True, slots=True)
class RtDetrPreprocessingSpec:
    do_resize: bool = True
    size: tuple[int, int] = (640, 640)
    resample: Image.Resampling = Image.Resampling.BILINEAR
    do_rescale: bool = True
    rescale_factor: float = 1 / 255
    do_normalize: bool = False
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)


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
    if name == "image_std" and any(item == 0.0 for item in result):
        raise ValueError("image_std values must be nonzero")
    return result  # type: ignore[return-value]


def parse_preprocessing_config(config: Mapping[str, Any]) -> RtDetrPreprocessingSpec:
    """Parse the runtime portion of an HF ``preprocessor_config.json``."""

    if not isinstance(config, Mapping):
        raise TypeError("preprocessor configuration must be a mapping")
    defaults = RtDetrPreprocessingSpec()
    raw_size = config.get("size", {"height": defaults.size[0], "width": defaults.size[1]})
    if not isinstance(raw_size, Mapping):
        raise TypeError("preprocessor size must be a mapping")
    try:
        height, width = int(raw_size["height"]), int(raw_size["width"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("preprocessor size must specify positive height and width") from error
    if height <= 0 or width <= 0:
        raise ValueError("preprocessor size must specify positive height and width")
    raw_resample = config.get("resample", int(defaults.resample))
    try:
        resample = Image.Resampling(int(raw_resample))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Unsupported PIL resample mode: {raw_resample!r}") from error
    factor = config.get("rescale_factor", defaults.rescale_factor)
    if isinstance(factor, bool) or not isinstance(factor, Real):
        raise TypeError("rescale_factor must be a real number")
    return RtDetrPreprocessingSpec(
        do_resize=bool(config.get("do_resize", defaults.do_resize)),
        size=(height, width),
        resample=resample,
        do_rescale=bool(config.get("do_rescale", defaults.do_rescale)),
        rescale_factor=float(factor),
        do_normalize=bool(config.get("do_normalize", defaults.do_normalize)),
        image_mean=_triplet(config.get("image_mean"), "image_mean", defaults.image_mean),
        image_std=_triplet(config.get("image_std"), "image_std", defaults.image_std),
    )


def pil_to_rgb_uint8(image: Image.Image) -> NDArray[np.uint8]:
    """Return a contiguous RGB uint8 image without importing MLX."""

    if not isinstance(image, Image.Image):
        raise TypeError("RT-DETR preprocessing requires PIL images")
    return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))


def resize_pil_rgb_uint8(
    image: Image.Image,
    size: Sequence[int],
    resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> NDArray[np.uint8]:
    """Pillow resize to ``(height, width)`` followed by RGB conversion."""

    if len(size) != 2:
        raise ValueError("size must contain height and width")
    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        raise ValueError("size must contain positive height and width")
    return pil_to_rgb_uint8(image.resize((width, height), resample=resample))


def preprocess_images(
    images: Sequence[Image.Image], spec: RtDetrPreprocessingSpec
) -> NDArray[np.float32]:
    """Apply the configured HF resize, rescale, and normalize operations."""

    prepared = [
        resize_pil_rgb_uint8(image, spec.size, spec.resample)
        if spec.do_resize
        else pil_to_rgb_uint8(image)
        for image in images
    ]
    if not prepared:
        return np.empty((0, *spec.size, 3), dtype=np.float32)
    pixels = np.stack(prepared).astype(np.float32)
    if spec.do_rescale:
        pixels *= spec.rescale_factor
    if spec.do_normalize:
        pixels -= np.asarray(spec.image_mean, dtype=np.float32)
        pixels /= np.asarray(spec.image_std, dtype=np.float32)
    return pixels
