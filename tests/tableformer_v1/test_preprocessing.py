# SPDX-License-Identifier: Apache-2.0

"""Exact upstream OpenCV preprocessing tests for TableFormer v1."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from docling_mlx.engines.table_structure.tableformer_v1.artifact import (
    validate_tableformer_v1_artifact,
)
from docling_mlx.engines.table_structure.tableformer_v1.preprocessing import (
    preprocess_table_image,
    resize_page_image,
)


def _pattern(width: int, height: int) -> np.ndarray:
    y, x = np.indices((height, width))
    return np.stack(
        ((x * 17 + y) % 256, (x + y * 13) % 256, (x * 7 + y * 5) % 256), axis=-1
    ).astype(np.uint8)


def test_public_contract_import_does_not_load_opencv() -> None:
    code = (
        "import sys; import docling_mlx.engines.table_structure.tableformer_v1; "
        "assert 'cv2' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.release
def test_preprocessing_does_not_change_process_global_opencv_settings(
    artifact_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cv2

    module: Any = sys.modules["docling_mlx.engines.table_structure.tableformer_v1.preprocessing"]
    module._opencv.cache_clear()
    monkeypatch.setattr(
        cv2,
        "setNumThreads",
        lambda _threads: pytest.fail("preprocessing changed global OpenCV threads"),
    )
    monkeypatch.setattr(
        cv2.ocl,
        "setUseOpenCL",
        lambda _enabled: pytest.fail("preprocessing changed global OpenCL state"),
    )
    spec = validate_tableformer_v1_artifact(artifact_root / "accurate").preprocessing
    preprocess_table_image(_pattern(5, 3), spec)


@pytest.mark.release
def test_page_resize_matches_upstream_inter_area(artifact_root: Path) -> None:
    import cv2

    spec = validate_tableformer_v1_artifact(artifact_root / "accurate").preprocessing
    image = _pattern(91, 37)
    actual, scale = resize_page_image(image, spec)
    expected_scale = 1024 / 37.0
    expected = cv2.resize(
        image,
        (int(91 * expected_scale), 1024),
        interpolation=cv2.INTER_AREA,
    )
    assert scale == expected_scale
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.release
def test_table_preprocessing_matches_upstream_operation_order(artifact_root: Path) -> None:
    import cv2

    spec = validate_tableformer_v1_artifact(artifact_root / "accurate").preprocessing
    image = _pattern(91, 37)
    expected = (image.astype(np.float32) - 255.0 * np.array(spec.mean)) / np.array(spec.std)
    expected = cv2.resize(expected, (448, 448), interpolation=cv2.INTER_LINEAR)
    expected = np.asarray(expected.transpose(2, 1, 0) / 255.0, dtype=np.float32)[None]

    actual = preprocess_table_image(image, spec)

    assert actual.dtype == np.float32
    assert actual.shape == (1, 3, 448, 448)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.release
def test_width_height_transpose_is_not_standard_chw(artifact_root: Path) -> None:
    spec = validate_tableformer_v1_artifact(artifact_root / "accurate").preprocessing
    image = np.zeros((3, 5, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(5)
    actual = preprocess_table_image(image, spec)[0]
    assert actual.shape == (3, 448, 448)
    assert not np.array_equal(actual, actual.transpose(0, 2, 1))
