# SPDX-License-Identifier: Apache-2.0

"""Pillow-exact TableFormerV2 image preparation with lazy MLX normalization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image

from docling_mlx.engines.table_structure.tableformer_v2.artifact import (
    TableFormerV2PreprocessingSpec,
)


def resize_pil_rgb_uint8(image: Image.Image, spec: TableFormerV2PreprocessingSpec) -> np.ndarray:
    """Match Torchvision's pinned PIL bilinear resize before ``ToTensor``."""

    rgb = image.convert("RGB")
    target = (spec.size[1], spec.size[0])
    if rgb.size != target:
        rgb = rgb.resize(target, resample=Image.Resampling.BILINEAR)
    return np.array(rgb, dtype=np.uint8, copy=True, order="C")


def preprocess_images(images: Sequence[Image.Image], spec: TableFormerV2PreprocessingSpec) -> Any:
    """Return one FP32 NHWC MLX batch resized to ``spec.size`` and normalized by
    ``spec.mean`` and ``spec.std``; the MLX import is deferred.
    """

    if not images:
        raise ValueError("TableFormerV2 preprocessing requires at least one image")
    import mlx.core as mx

    pixels = np.stack([resize_pil_rgb_uint8(image, spec) for image in images])
    values = mx.array(pixels).astype(mx.float32)
    mean = mx.array(spec.mean, dtype=mx.float32).reshape(1, 1, 1, 3)
    std = mx.array(spec.std, dtype=mx.float32).reshape(1, 1, 1, 3)
    return (values / mx.array(255.0, dtype=mx.float32) - mean) / std


__all__ = ["preprocess_images", "resize_pil_rgb_uint8"]
