# SPDX-License-Identifier: Apache-2.0

"""Shared CPU postprocessing for object detectors."""

from collections.abc import Sequence

import numpy as np


def _postprocess_focal_detections(
    logits: np.ndarray,
    boxes: np.ndarray,
    image_sizes: Sequence[tuple[int, int]],
    *,
    score_threshold: float,
) -> list[tuple[list[int], list[float], list[list[float]]]]:
    """Select flattened query-class scores and scale their boxes to image pixels."""

    if logits.ndim != 3 or boxes.ndim != 3 or logits.shape[:2] != boxes.shape[:2]:
        raise RuntimeError("Focal object detector returned incompatible logits and boxes")
    if logits.shape[0] != len(image_sizes) or boxes.shape[2] != 4:
        raise RuntimeError("Focal object detector returned an invalid batch or box shape")
    if logits.shape[1] == 0 or logits.shape[2] == 0:
        raise RuntimeError("Focal object detector returned empty queries or labels")
    if not np.isfinite(logits).all() or not np.isfinite(boxes).all():
        raise RuntimeError("Focal object detector returned non-finite predictions")

    positive = logits >= 0
    probabilities = np.empty_like(logits)
    probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponent = np.exp(logits[~positive])
    probabilities[~positive] = exponent / (1.0 + exponent)

    query_count = logits.shape[1]
    label_count = logits.shape[2]
    outputs: list[tuple[list[int], list[float], list[list[float]]]] = []
    for scores_by_query, boxes_by_query, (image_width, image_height) in zip(
        probabilities, boxes, image_sizes, strict=True
    ):
        flat_scores = scores_by_query.reshape(-1)
        flattened_ids = np.arange(flat_scores.size, dtype=np.int64)
        selected = np.lexsort((flattened_ids, -flat_scores))[:query_count]
        selected = selected[flat_scores[selected] > score_threshold]
        query_ids = selected // label_count
        label_ids = selected % label_count
        selected_boxes = boxes_by_query[query_ids]
        center_x, center_y, width, height = selected_boxes.T
        pixel_boxes = np.stack(
            (
                (center_x - width / 2.0) * image_width,
                (center_y - height / 2.0) * image_height,
                (center_x + width / 2.0) * image_width,
                (center_y + height / 2.0) * image_height,
            ),
            axis=1,
        )
        outputs.append(
            (
                label_ids.tolist(),
                flat_scores[selected].astype(float).tolist(),
                pixel_boxes.astype(float).tolist(),
            )
        )
    return outputs


def postprocess_detections(
    logits: np.ndarray,
    boxes: np.ndarray,
    image_sizes: Sequence[tuple[int, int]],
    *,
    score_threshold: float,
    use_focal_loss: bool,
) -> list[tuple[list[int], list[float], list[list[float]]]]:
    """Postprocess focal or softmax classifier outputs into pixel ``xyxy`` boxes."""

    if use_focal_loss:
        return _postprocess_focal_detections(
            logits, boxes, image_sizes, score_threshold=score_threshold
        )
    if logits.ndim != 3 or boxes.ndim != 3 or logits.shape[:2] != boxes.shape[:2]:
        raise RuntimeError("Object detector returned incompatible logits and boxes")
    if logits.shape[0] != len(image_sizes) or boxes.shape[2] != 4:
        raise RuntimeError("Object detector returned an invalid batch or box shape")
    if logits.shape[1] == 0 or logits.shape[2] < 2:
        raise RuntimeError("Object detector returned empty queries or labels")
    if not np.isfinite(logits).all() or not np.isfinite(boxes).all():
        raise RuntimeError("Object detector returned non-finite predictions")

    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    outputs: list[tuple[list[int], list[float], list[list[float]]]] = []
    for scores_by_query, boxes_by_query, (image_width, image_height) in zip(
        probabilities[..., :-1], boxes, image_sizes, strict=True
    ):
        label_ids = scores_by_query.argmax(axis=-1)
        scores = scores_by_query[np.arange(scores_by_query.shape[0]), label_ids]
        selected = scores > score_threshold
        center_x, center_y, width, height = boxes_by_query[selected].T
        pixel_boxes = np.stack(
            (
                (center_x - width / 2.0) * image_width,
                (center_y - height / 2.0) * image_height,
                (center_x + width / 2.0) * image_width,
                (center_y + height / 2.0) * image_height,
            ),
            axis=1,
        )
        outputs.append(
            (
                label_ids[selected].astype(int).tolist(),
                scores[selected].astype(float).tolist(),
                pixel_boxes.astype(float).tolist(),
            )
        )
    return outputs
