# SPDX-License-Identifier: Apache-2.0

"""Pinned Torchvision CPU oracle for TableFormerV2 preprocessing."""

from __future__ import annotations

import numpy as np
from PIL import Image


def preprocess_with_torchvision(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as tvf

    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError("TableFormerV2 reference requires the float32 Torch default dtype")
    rgb = image.convert("RGB")
    resized = tvf.resize(
        rgb,
        [448, 448],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    resized_uint8 = np.array(resized, dtype=np.uint8, copy=True, order="C")
    tensor = tvf.to_tensor(resized)
    normalized = tvf.normalize(
        tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return resized_uint8, normalized.permute(1, 2, 0).contiguous().numpy()
