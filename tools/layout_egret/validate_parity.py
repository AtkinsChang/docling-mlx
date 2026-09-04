# SPDX-License-Identifier: Apache-2.0

"""Validate a converted D-FINE checkpoint against a Torch CPU capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from safetensors.numpy import load_file

from docling_mlx._models.dfine.config import DFineConfig
from docling_mlx.engines.object_detection._focal_postprocessing import (
    _postprocess_focal_detections,
)
from docling_mlx.engines.object_detection.rt_detr_v2.preprocessing import (
    RtDetrPreprocessingSpec,
    parse_preprocessing_config,
    preprocess_images,
)
from tools.layout_egret.capture_reference import (
    ARRAY_SPECS,
    CAPTURE_ARCHIVE,
    CAPTURE_SCHEMA_VERSION,
    FIXTURE_PATH,
    REFERENCE_VERSIONS,
    REPOSITORY_ROOT,
    array_sha256,
)
from tools.layout_egret.convert_weights import (
    sha256,
)

DETECTION_SCORE_THRESHOLD = 0.3
DETECTION_SCORE_MAX_ABS = 1e-2
DETECTION_BOX_MIN_IOU = 0.99
_EXPECTED_SHAPES = {
    "pixel_values_nchw_f32": [1, 3, 640, 640],
    "logits_f32": [1, 300, 17],
    "pred_boxes_cxcywh_f32": [1, 300, 4],
}
_ARTIFACT_FILES = (
    "model.safetensors",
    "config.json",
    "preprocessor_config.json",
)


def _load_reference(reference: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    reference = reference.expanduser().resolve()
    metadata = json.loads((reference / "metadata.json").read_text(encoding="utf-8"))
    source = metadata.get("source")
    if (
        metadata.get("schema_version") != CAPTURE_SCHEMA_VERSION
        or metadata.get("producer") != "tools.layout_egret.capture_reference"
        or not isinstance(source, dict)
        or not isinstance(source.get("repo_id"), str)
        or not isinstance(source.get("revision"), str)
    ):
        raise ValueError("Reference identity does not identify a pinned D-FINE oracle")

    fixture = metadata.get("fixture", {})
    if (
        fixture.get("path") != str(FIXTURE_PATH.relative_to(REPOSITORY_ROOT))
        or fixture.get("sha256") != sha256(FIXTURE_PATH)
        or fixture.get("size") != [401, 534]
    ):
        raise ValueError("Reference fixture does not match the committed coordinate image")
    if metadata.get("processor") != {
        "class": "RTDetrImageProcessor",
        "backend": "TorchvisionBackend",
    }:
        raise ValueError("Reference processor is not the pinned Torchvision backend")
    runtime = metadata.get("runtime", {})
    torch_settings = runtime.get("torch", {})
    if runtime.get("dependencies") != REFERENCE_VERSIONS or not all(
        (
            torch_settings.get("device") == "cpu",
            torch_settings.get("eval") is True,
            torch_settings.get("inference_mode") is True,
            torch_settings.get("deterministic_algorithms") is True,
        )
    ):
        raise ValueError("Reference is not the pinned deterministic Torch CPU oracle")

    declared = metadata.get("arrays")
    if not isinstance(declared, list) or len(declared) != len(ARRAY_SPECS):
        raise ValueError("Reference array schema is incomplete")
    for item, spec in zip(declared, ARRAY_SPECS, strict=True):
        expected = {
            "name": spec.name,
            "layout": spec.layout,
            "dtype": "float32",
            "shape": _EXPECTED_SHAPES[spec.name],
        }
        if not isinstance(item, dict) or any(
            item.get(key) != value for key, value in expected.items()
        ):
            raise ValueError(f"Reference array contract does not match {spec.name}")

    archive = reference / CAPTURE_ARCHIVE
    archive_metadata = metadata.get("archive", {})
    if archive_metadata != {"file": CAPTURE_ARCHIVE, "sha256": sha256(archive)}:
        raise ValueError("Reference archive hash does not match metadata")
    with np.load(archive, allow_pickle=False) as loaded:
        if loaded.files != [spec.name for spec in ARRAY_SPECS]:
            raise ValueError("Reference archive has missing or extra arrays")
        arrays = {spec.name: np.array(loaded[spec.name], copy=True) for spec in ARRAY_SPECS}
    for item, spec in zip(declared, ARRAY_SPECS, strict=True):
        value = arrays[spec.name]
        if (
            value.dtype != np.float32
            or list(value.shape) != _EXPECTED_SHAPES[spec.name]
            or not np.isfinite(value).all()
            or item.get("sha256") != array_sha256(value)
        ):
            raise ValueError(f"Reference array payload does not match {spec.name}")
    return metadata, arrays


def _load_artifact(
    artifact: Path,
) -> tuple[Any, DFineConfig, RtDetrPreprocessingSpec, dict[str, Any]]:
    artifact = artifact.expanduser().resolve()
    for name in _ARTIFACT_FILES:
        if not (artifact / name).is_file():
            raise ValueError(f"Missing Egret artifact file: {artifact / name}")

    config_raw = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    preprocessor_raw = json.loads(
        (artifact / "preprocessor_config.json").read_text(encoding="utf-8")
    )
    config = DFineConfig.from_dict(config_raw)
    preprocessor = parse_preprocessing_config(preprocessor_raw)

    weights = load_file(str(artifact / "model.safetensors"))
    if not weights or any(
        value.dtype != np.float32 or not np.isfinite(value).all() for value in weights.values()
    ):
        raise ValueError("Converted artifact weights must be nonempty finite FP32 tensors")

    import mlx.core as mx

    from docling_mlx._models.dfine.model import DFine

    model = DFine(config)
    model.load_weights([(key, mx.array(value)) for key, value in weights.items()], strict=True)
    model.eval()
    mx.eval(model.parameters())
    evidence = {
        name: {"sha256": sha256(artifact / name), "bytes": (artifact / name).stat().st_size}
        for name in _ARTIFACT_FILES
    }
    return model, config, preprocessor, evidence


def _public_detection_gates(
    actual_logits: np.ndarray,
    actual_boxes: np.ndarray,
    expected_logits: np.ndarray,
    expected_boxes: np.ndarray,
    image_size: tuple[int, int],
) -> list[dict[str, Any]]:
    """Check labels, scores, and boxes after shared focal postprocessing."""

    actual = _postprocess_focal_detections(
        actual_logits,
        actual_boxes,
        [image_size],
        score_threshold=DETECTION_SCORE_THRESHOLD,
    )[0]
    expected = _postprocess_focal_detections(
        expected_logits,
        expected_boxes,
        [image_size],
        score_threshold=DETECTION_SCORE_THRESHOLD,
    )[0]
    actual_labels, actual_scores, actual_pixel_boxes = actual
    expected_labels, expected_scores, expected_pixel_boxes = expected
    labels_match = sorted(actual_labels) == sorted(expected_labels)
    unmatched = list(range(len(actual_labels)))
    score_max_abs = 0.0
    box_iou_min = 1.0
    if labels_match:
        for label, score, box in zip(
            expected_labels, expected_scores, expected_pixel_boxes, strict=True
        ):
            candidates = [index for index in unmatched if actual_labels[index] == label]
            best = max(
                candidates,
                key=lambda index: _box_iou(np.asarray(box), np.asarray(actual_pixel_boxes[index])),
            )
            unmatched.remove(best)
            score_max_abs = max(score_max_abs, abs(score - actual_scores[best]))
            box_iou_min = min(
                box_iou_min,
                _box_iou(np.asarray(box), np.asarray(actual_pixel_boxes[best])),
            )
    return [
        {
            "name": "public_detection.labels",
            "expected_count": len(expected_labels),
            "actual_count": len(actual_labels),
            "expected_label_ids": expected_labels,
            "actual_label_ids": actual_labels,
            "passed": labels_match,
        },
        {
            "name": "public_detection.scores",
            "max_abs": score_max_abs,
            "threshold": DETECTION_SCORE_MAX_ABS,
            "passed": labels_match and score_max_abs <= DETECTION_SCORE_MAX_ABS,
        },
        {
            "name": "public_detection.boxes",
            "iou_min": box_iou_min,
            "threshold": DETECTION_BOX_MIN_IOU,
            "passed": labels_match and box_iou_min >= DETECTION_BOX_MIN_IOU,
        },
    ]


def _box_iou(left: np.ndarray, right: np.ndarray) -> float:
    top_left = np.maximum(left[:2], right[:2])
    bottom_right = np.minimum(left[2:], right[2:])
    overlap = np.maximum(bottom_right - top_left, 0.0)
    intersection = float(overlap[0] * overlap[1])
    left_area = float(np.prod(np.maximum(left[2:] - left[:2], 0.0)))
    right_area = float(np.prod(np.maximum(right[2:] - right[:2], 0.0)))
    union = left_area + right_area - intersection
    return intersection / union if union else float(np.array_equal(left, right))


def validate(artifact: Path, reference: Path) -> dict[str, Any]:
    """Strict-load the converted checkpoint and run all fixed parity gates."""

    artifact = artifact.expanduser().resolve()
    metadata, expected = _load_reference(reference)
    model, config, preprocessor, artifact_evidence = _load_artifact(artifact)

    import mlx.core as mx

    with Image.open(FIXTURE_PATH) as image:
        native_pixels = mx.array(preprocess_images([image], preprocessor), dtype=mx.float32)
    mx.eval(native_pixels)
    native_pixels_nhwc = np.array(native_pixels, dtype=np.float32, copy=True)

    output = model(mx.array(native_pixels_nhwc, dtype=mx.float32))
    mx.eval(output["pred_logits"], output["pred_boxes"])
    actual_logits = np.array(output["pred_logits"], dtype=np.float32, copy=True)
    actual_boxes = np.array(output["pred_boxes"], dtype=np.float32, copy=True)

    expected_shapes = {
        "logits": [1, config.decoder.num_queries, config.num_labels],
        "pred_boxes": [1, config.decoder.num_queries, 4],
    }
    shape_gate = {
        "name": "model.shape_dtype_finite",
        "expected_shapes": expected_shapes,
        "actual_shapes": {
            "logits": list(actual_logits.shape),
            "pred_boxes": list(actual_boxes.shape),
        },
        "float32": actual_logits.dtype == actual_boxes.dtype == np.float32,
        "finite": bool(np.isfinite(actual_logits).all() and np.isfinite(actual_boxes).all()),
    }
    shape_gate["passed"] = (
        shape_gate["actual_shapes"] == expected_shapes
        and shape_gate["float32"]
        and shape_gate["finite"]
    )
    fixture = metadata["fixture"]
    fixture_size = fixture["size"]
    if not isinstance(fixture_size, list) or len(fixture_size) != 2:
        raise ValueError("Reference fixture size is invalid")
    public_gates = _public_detection_gates(
        actual_logits,
        actual_boxes,
        expected["logits_f32"],
        expected["pred_boxes_cxcywh_f32"],
        (int(fixture_size[0]), int(fixture_size[1])),
    )
    qualification_mode = "public_detection"
    qualification_gates = [shape_gate, *public_gates]
    failed = [gate["name"] for gate in qualification_gates if not gate["passed"]]
    return {
        "schema_version": 1,
        "qualification_mode": qualification_mode,
        "passed": not failed,
        "first_failure": failed[0] if failed else None,
        "gates": [shape_gate, *public_gates],
        "qualification_gates": qualification_gates,
        "artifact": artifact_evidence,
        "reference": {
            "archive_sha256": metadata["archive"]["sha256"],
            "fixture_sha256": metadata["fixture"]["sha256"],
            "source": metadata["source"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.artifact, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "passed" if report["passed"] else "failed",
                "output": str(args.output.expanduser().resolve()),
                "first_failure": report["first_failure"],
            },
            separators=(",", ":"),
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
