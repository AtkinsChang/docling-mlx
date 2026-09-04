# SPDX-License-Identifier: Apache-2.0

"""Pure object-detection result types shared by native detector engines."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Detections:
    """Thresholded pixel ``xyxy`` detections for one source image."""

    boxes: list[list[float]]
    scores: list[float]
    label_ids: list[int]
    id2label: dict[int, str]


__all__ = ["Detections"]
