# SPDX-License-Identifier: Apache-2.0

"""MLX-lane preprocessing shape and normalization check."""

import numpy as np
import pytest
from PIL import Image

from docling_mlx.engines.image_classification.efficientnet.preprocessing import (
    parse_preprocessing_config,
    preprocess_images,
)

pytestmark = pytest.mark.mlx

CONFIG = {
    "image_processor_type": "ViTImageProcessor",
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.47853944, 0.4732864, 0.47434163],
    "resample": 2,
    "rescale_factor": 1 / 255,
    "size": {"height": 3, "width": 4},
}


def test_pil_preprocessor_preserves_shape_dtype_and_normalization() -> None:
    images = [
        Image.fromarray(np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)),
        Image.new("L", (7, 5), 12),
        Image.new("RGBA", (7, 5), (13, 17, 19, 23)),
    ]
    spec = parse_preprocessing_config(CONFIG)
    actual = preprocess_images(images, spec)
    expected_pixels = np.stack(
        [
            np.asarray(
                Image.fromarray(np.asarray(image.convert("RGB"), dtype=np.uint8)).resize(
                    (4, 3), resample=Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            )
            for image in images
        ]
    )
    factor = np.float32(spec.rescale_factor)
    expected = (expected_pixels - np.asarray(spec.image_mean, dtype=np.float32) / factor) / (
        np.asarray(spec.image_std, dtype=np.float32) / factor
    )
    assert actual.shape == (3, 3, 4, 3)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
