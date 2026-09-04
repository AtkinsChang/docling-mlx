# SPDX-License-Identifier: Apache-2.0

"""Portable contracts for shared focal-loss detector postprocessing."""

import numpy as np
import pytest

from docling_mlx.engines.object_detection._focal_postprocessing import (
    _postprocess_focal_detections,
    postprocess_detections,
)


def test_postprocess_orders_flattened_scores_and_scales_repeated_query_boxes() -> None:
    logits = np.full((1, 4, 3), -100.0, dtype=np.float32)
    logits[0, 0, 0] = 3.0
    logits[0, 0, 1] = 2.0
    logits[0, 1, 0] = 2.0
    logits[0, 2, 2] = 0.0
    boxes = np.array(
        [[[1.2, -0.2, 0.4, 0.4], [0.5, 0.5, 1.5, 1.0], [0.5, 0.5, 0.2, 0.2], [0, 0, 0, 0]]],
        dtype=np.float32,
    )
    labels, scores, pixel_boxes = _postprocess_focal_detections(
        logits, boxes, [(101, 81)], score_threshold=0.5
    )[0]

    assert labels == [0, 1, 0]
    assert scores == pytest.approx(
        [1 / (1 + np.exp(-3)), 1 / (1 + np.exp(-2)), 1 / (1 + np.exp(-2))]
    )
    np.testing.assert_allclose(
        pixel_boxes,
        [
            [101.0, -32.4, 141.4, 0.0],
            [101.0, -32.4, 141.4, 0.0],
            [-25.25, 0.0, 126.25, 81.0],
        ],
        rtol=0,
        atol=2e-5,
    )


def test_postprocess_softmax_excludes_no_object_class() -> None:
    logits = np.array([[[0.0, 2.0, -2.0], [0.0, -1.0, 3.0]]], dtype=np.float32)
    boxes = np.array([[[0.5, 0.5, 0.4, 0.2], [0.5, 0.5, 0.4, 0.2]]], dtype=np.float32)

    labels, scores, pixel_boxes = postprocess_detections(
        logits, boxes, [(100, 50)], score_threshold=0.5, use_focal_loss=False
    )[0]

    assert labels == [1]
    assert scores == pytest.approx([0.8668133])
    np.testing.assert_allclose(pixel_boxes, [[30.0, 20.0, 70.0, 30.0]])


@pytest.mark.parametrize(
    ("logits", "boxes", "image_sizes", "message"),
    [
        (
            np.zeros((1, 2, 3), dtype=np.float32),
            np.zeros((1, 3, 4), dtype=np.float32),
            [(1, 1)],
            "incompatible",
        ),
        (
            np.zeros((1, 2, 3), dtype=np.float32),
            np.zeros((1, 2, 5), dtype=np.float32),
            [(1, 1)],
            "box shape",
        ),
        (
            np.zeros((1, 2, 3), dtype=np.float32),
            np.zeros((1, 2, 4), dtype=np.float32),
            [],
            "batch",
        ),
        (
            np.full((1, 2, 3), np.inf, dtype=np.float32),
            np.zeros((1, 2, 4), dtype=np.float32),
            [(1, 1)],
            "non-finite",
        ),
        (
            np.zeros((1, 2, 3), dtype=np.float32),
            np.full((1, 2, 4), np.nan, dtype=np.float32),
            [(1, 1)],
            "non-finite",
        ),
    ],
)
def test_postprocess_rejects_invalid_outputs(
    logits: np.ndarray,
    boxes: np.ndarray,
    image_sizes: list[tuple[int, int]],
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _postprocess_focal_detections(logits, boxes, image_sizes, score_threshold=0.5)
