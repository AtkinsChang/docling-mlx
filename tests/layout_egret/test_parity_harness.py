# SPDX-License-Identifier: Apache-2.0

"""Pure contracts for the D-FINE capture and parity report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.layout_egret.capture_reference import (
    ARRAY_SPECS,
    CAPTURE_ARCHIVE,
    CAPTURE_SCHEMA_VERSION,
    FIXTURE_PATH,
    REFERENCE_VERSIONS,
    REPOSITORY_ROOT,
    array_sha256,
    write_compressed_npz,
)
from tools.layout_egret.convert_weights import sha256
from tools.layout_egret.validate_parity import (
    _load_reference,
    _public_detection_gates,
)


def _arrays() -> dict[str, np.ndarray]:
    return {
        "pixel_values_nchw_f32": np.zeros((1, 3, 640, 640), dtype=np.float32),
        "logits_f32": np.zeros((1, 300, 17), dtype=np.float32),
        "pred_boxes_cxcywh_f32": np.zeros((1, 300, 4), dtype=np.float32),
    }


def _write_reference(path: Path) -> dict[str, object]:
    arrays = _arrays()
    path.mkdir()
    archive = path / CAPTURE_ARCHIVE
    write_compressed_npz(archive, arrays)
    metadata: dict[str, object] = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "producer": "tools.layout_egret.capture_reference",
        "source": {
            "repo_id": "example/dfine",
            "revision": "pinned-revision",
        },
        "fixture": {
            "path": str(FIXTURE_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256(FIXTURE_PATH),
            "size": [401, 534],
        },
        "processor": {
            "class": "RTDetrImageProcessor",
            "backend": "TorchvisionBackend",
        },
        "arrays": [
            {
                "name": spec.name,
                "layout": spec.layout,
                "dtype": "float32",
                "shape": list(arrays[spec.name].shape),
                "sha256": array_sha256(arrays[spec.name]),
            }
            for spec in ARRAY_SPECS
        ],
        "archive": {"file": CAPTURE_ARCHIVE, "sha256": sha256(archive)},
        "runtime": {
            "dependencies": REFERENCE_VERSIONS,
            "torch": {
                "device": "cpu",
                "eval": True,
                "inference_mode": True,
                "deterministic_algorithms": True,
            },
        },
    }
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return metadata


def test_capture_archive_is_reproducible_and_contains_only_three_arrays(tmp_path: Path) -> None:
    arrays = _arrays()
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"

    write_compressed_npz(first, arrays)
    write_compressed_npz(second, arrays)

    assert first.read_bytes() == second.read_bytes()
    with np.load(first, allow_pickle=False) as loaded:
        assert loaded.files == [spec.name for spec in ARRAY_SPECS]


def test_reference_loader_requires_pinned_source_processor_and_payload(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    metadata = _write_reference(reference)

    loaded_metadata, loaded_arrays = _load_reference(reference)

    assert loaded_metadata["processor"] == {
        "class": "RTDetrImageProcessor",
        "backend": "TorchvisionBackend",
    }
    assert list(loaded_arrays) == [spec.name for spec in ARRAY_SPECS]

    metadata["processor"] = {"class": "RTDetrImageProcessor", "backend": "PillowBackend"}
    (reference / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Torchvision backend"):
        _load_reference(reference)


def _public_arrays() -> tuple[np.ndarray, np.ndarray]:
    logits = np.full((1, 300, 17), -10.0, dtype=np.float32)
    logits[0, 0, 0] = 0.0
    boxes = np.full((1, 300, 4), 0.5, dtype=np.float32)
    return logits, boxes


def test_unretained_raw_tail_drift_does_not_change_public_contract() -> None:
    expected_logits, expected_boxes = _public_arrays()
    actual_logits = expected_logits.copy()
    actual_logits[0, 299, 16] += np.float32(0.002)

    public = _public_detection_gates(
        actual_logits, expected_boxes, expected_logits, expected_boxes, (401, 534)
    )

    assert all(gate["passed"] for gate in public)


@pytest.mark.parametrize(
    ("drift", "failed_gate"),
    [
        ("label", "public_detection.labels"),
        ("score", "public_detection.scores"),
        ("box", "public_detection.boxes"),
    ],
)
def test_public_contract_rejects_retained_detection_drift(drift: str, failed_gate: str) -> None:
    expected_logits, expected_boxes = _public_arrays()
    actual_logits = expected_logits.copy()
    actual_boxes = expected_boxes.copy()
    if drift == "label":
        actual_logits[0, 0, 0] = -10.0
        actual_logits[0, 0, 1] = 0.0
    elif drift == "score":
        actual_logits[0, 0, 0] = 1.0
    else:
        actual_boxes[0, 0, 0] += np.float32(0.01)

    gates = _public_detection_gates(
        actual_logits, actual_boxes, expected_logits, expected_boxes, (401, 534)
    )

    assert not next(gate for gate in gates if gate["name"] == failed_gate)["passed"]
