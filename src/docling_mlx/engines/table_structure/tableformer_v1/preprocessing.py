# SPDX-License-Identifier: Apache-2.0

"""OpenCV-exact image preparation for upstream TableFormer v1."""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np

from docling_mlx.engines.table_structure.tableformer_v1.artifact import (
    TableFormerV1PreprocessingSpec,
)


@cache
def _opencv() -> Any:
    import cv2

    return cv2


def resize_page_image(
    image: np.ndarray, spec: TableFormerV1PreprocessingSpec
) -> tuple[np.ndarray, float]:
    """Resize a page to height 1024 with upstream ``INTER_AREA`` semantics."""

    if image.ndim not in {2, 3} or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("TableFormer v1 page image must be a non-empty OpenCV image")
    scale = spec.page_height / float(image.shape[0])
    size = (int(image.shape[1] * scale), spec.page_height)
    return _opencv().resize(image, size, interpolation=_opencv().INTER_AREA), scale


def preprocess_table_image(image: np.ndarray, spec: TableFormerV1PreprocessingSpec) -> np.ndarray:
    """Return the upstream batch layout and FP32 values, without importing Torch."""

    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("TableFormer v1 table image must be a non-empty three-channel array")
    normalized = (image.astype(np.float32) - np.array(spec.mean) / spec.rescale_factor) / np.array(
        spec.std
    )
    resized = _opencv().resize(
        normalized,
        (spec.image_size, spec.image_size),
        interpolation=_opencv().INTER_LINEAR,
    )
    cwh = resized.transpose(2, 1, 0) * spec.rescale_factor
    return np.asarray(cwh, dtype=np.float32)[None]


__all__ = ["preprocess_table_image", "resize_page_image"]
