# SPDX-License-Identifier: Apache-2.0

"""End-to-end official D-FINE parity against Transformers on CPU."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from docling_mlx.engines.object_detection.dfine.engine import (
    Detections,
    DFineEngine,
    DFineModelSpec,
)

ROOT = Path(__file__).resolve().parents[2]
HF_CACHE = Path.home() / ".cache/huggingface/hub"
_REQUIRED_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")


def _snapshot(name: str) -> Path:
    snapshots = HF_CACHE / f"models--ustc-community--dfine-{name}-coco" / "snapshots"
    candidates = list(snapshots.glob("*")) if snapshots.is_dir() else []
    for candidate in candidates:
        if all((candidate / filename).is_file() for filename in _REQUIRED_FILES):
            return candidate
    pytest.fail(
        "D-FINE parity requires cached dfine-"
        f"{name}-coco; run: hf download ustc-community/dfine-{name}-coco config.json README.md"
    )


def _images() -> list[Image.Image]:
    inputs_value = os.environ.get("DOCLING_MLX_OBJECT_DETECTION_PARITY_INPUTS")
    if inputs_value is None:
        pytest.fail(
            "D-FINE parity requires DOCLING_MLX_OBJECT_DETECTION_PARITY_INPUTS "
            "to point to the pinned DPBench PDF directory"
        )
    inputs = Path(inputs_value).expanduser()
    fixture = ROOT / "tests/fixtures/layout_heron/benchmark_gradient.png"
    pdfs = [inputs / f"{index:04}.pdf" for index in range(3)]
    missing = [str(path) for path in [fixture, *pdfs] if not path.is_file()]
    if missing:
        pytest.fail(f"D-FINE parity requires fixture images: {missing}")
    import pypdfium2 as pdfium

    with Image.open(fixture) as image:
        images = [image.convert("RGB").copy()]
    for path in pdfs:
        document = pdfium.PdfDocument(path)
        images.append(document[0].render(scale=2).to_pil().convert("RGB"))
    return images


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    left, top = np.maximum(first[:2], second[:2])
    right, bottom = np.minimum(first[2:], second[2:])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def _assert_detections_match(native: Detections, reference: dict[str, object]) -> None:
    reference_labels = np.asarray(reference["labels"])
    reference_scores = np.asarray(reference["scores"])
    reference_boxes = np.asarray(reference["boxes"])
    assert native.label_ids == reference_labels.tolist()
    assert len(native.scores) == len(reference_scores) == len(reference_boxes)

    available = set(range(len(reference_scores)))
    for label, score, box in zip(native.label_ids, native.scores, native.boxes, strict=True):
        matching = [index for index in available if reference_labels[index] == label]
        assert matching, f"unmatched native label {label}"
        index = max(matching, key=lambda item: _iou(np.asarray(box), reference_boxes[item]))
        available.remove(index)
        assert abs(score - float(reference_scores[index])) <= 1e-2
        assert _iou(np.asarray(box), reference_boxes[index]) >= 0.99
    assert not available, f"unmatched Transformers detections: {sorted(available)}"


@pytest.mark.mlx
@pytest.mark.parity
@pytest.mark.parametrize("name", ("nano", "small", "medium", "large", "xlarge"))
def test_official_checkpoints_match_transformers(name: str) -> None:
    import torch
    from transformers import AutoImageProcessor, DFineForObjectDetection

    checkpoint = _snapshot(name)
    images = _images()
    native = DFineEngine(DFineModelSpec(path=checkpoint)).predict(images)

    processor = AutoImageProcessor.from_pretrained(checkpoint)
    reference_model = DFineForObjectDetection.from_pretrained(checkpoint).eval()
    inputs = processor(images=images, return_tensors="pt")
    with torch.inference_mode():
        output = reference_model(pixel_values=inputs["pixel_values"])
    reference = processor.post_process_object_detection(
        output,
        threshold=0.3,
        target_sizes=torch.tensor([(image.height, image.width) for image in images]),
    )

    for actual, expected in zip(native, reference, strict=True):
        _assert_detections_match(actual, expected)
