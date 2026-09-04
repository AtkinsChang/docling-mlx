# SPDX-License-Identifier: Apache-2.0

"""Pure contracts for TableFormerV2 reference capture and parity reports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from docling_mlx._models.tableformer_v2.config import TABLEFORMER_V2_TOKENS
from tools.pinned_versions import locked_versions
from tools.tableformer_v2.capture_reference import (
    ARRAY_SPECS,
    CAPTURE_SCHEMA_VERSION,
    MAX_GENERATION_STEPS,
    REFERENCE_PACKAGES,
    array_sha256,
    otsl_from_ids,
    write_compressed_npz,
)
from tools.tableformer_v2.source import SOURCE_REPO, SOURCE_REVISION, sha256
from tools.tableformer_v2.validate_parity import (
    FINAL_BBOX_MAX_ABS,
    FINAL_LOGITS_MAX_ABS,
    FINAL_LOGITS_MAX_MAE,
    PREPROCESSING_MAX_ABS,
    _exact_array_gate,
    _load_reference,
    _numeric_gate,
    _sequence_gate,
)


def _small_arrays() -> dict[str, np.ndarray]:
    return {
        "input_rgb_u8": np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
        "resized_rgb_u8": np.arange(48, dtype=np.uint8).reshape(4, 4, 3),
        "pixels_nhwc_f32": np.arange(48, dtype=np.float32).reshape(1, 4, 4, 3),
        "encoder_last_hidden_state_f32": np.arange(2, dtype=np.float32).reshape(1, 1, 2),
        "generated_ids_i64": np.array([[2, 3]], dtype=np.int64),
        "greedy_step_logits_f32": np.zeros((1, 1, 13), dtype=np.float32),
        "final_logits_f32": np.zeros((1, 2, 13), dtype=np.float32),
        "normalized_bboxes_f32": np.empty((0, 4), dtype=np.float32),
    }


def _write_reference(directory: Path) -> None:
    arrays = _small_arrays()
    archive = directory / "000-small.npz"
    write_compressed_npz(archive, arrays)
    metadata = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "producer": "tools.tableformer_v2.capture_reference",
        "profile": "tableformer_v2",
        "source": {
            "repo_id": SOURCE_REPO,
            "revision": SOURCE_REVISION,
        },
        "token_map": list(TABLEFORMER_V2_TOKENS),
        "generation": {"max_generation_steps": MAX_GENERATION_STEPS},
        "runtime": {
            "dependencies": locked_versions(REFERENCE_PACKAGES),
            "torch": {
                "device": "cpu",
                "eval": True,
                "inference_mode": True,
                "deterministic_algorithms": True,
            },
        },
        "captures": [
            {
                "name": "small",
                "source_file_sha256": "0" * 64,
                "input_rgb_sha256": array_sha256(arrays["input_rgb_u8"]),
                "encoder_spatial_size": [1, 1],
                "otsl": [],
                "arrays": [
                    {
                        "name": spec.name,
                        "dtype": spec.dtype,
                        "layout": spec.layout,
                        "shape": list(arrays[spec.name].shape),
                        "sha256": array_sha256(arrays[spec.name]),
                    }
                    for spec in ARRAY_SPECS
                ],
                "archive": {"file": archive.name, "sha256": sha256(archive)},
            }
        ],
    }
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_reference_archive_is_reproducible_and_schema_ordered(tmp_path: Path) -> None:
    arrays = _small_arrays()
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    write_compressed_npz(first, arrays)
    write_compressed_npz(second, arrays)

    assert first.read_bytes() == second.read_bytes()
    with np.load(first, allow_pickle=False) as archive:
        assert archive.files == [spec.name for spec in ARRAY_SPECS]


def test_reference_loader_verifies_every_array_and_archive_hash(tmp_path: Path) -> None:
    _write_reference(tmp_path)

    metadata, captures = _load_reference(tmp_path)

    assert metadata["captures"][0]["name"] == "small"
    assert list(captures[0]) == [spec.name for spec in ARRAY_SPECS]

    archive = tmp_path / "000-small.npz"
    archive.write_bytes(archive.read_bytes() + b"tampered")
    try:
        _load_reference(tmp_path)
    except ValueError as error:
        assert "SHA-256 mismatch" in str(error)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("tampered reference archive was accepted")


def test_numeric_gates_keep_fixed_release_caps_and_report_the_failed_boundary() -> None:
    expected = np.zeros((1, 2), dtype=np.float32)
    actual = expected.copy()
    actual[0, 0] = np.float32(FINAL_LOGITS_MAX_ABS * 1.1)

    logits = _numeric_gate(
        "generation.final_logits",
        actual,
        expected,
        max_mae=FINAL_LOGITS_MAX_MAE,
        max_abs=FINAL_LOGITS_MAX_ABS,
    )
    preprocessing = _numeric_gate(
        "preprocessing.normalized_pixels",
        np.array([PREPROCESSING_MAX_ABS], dtype=np.float32),
        np.zeros((1,), dtype=np.float32),
        max_mae=None,
        max_abs=PREPROCESSING_MAX_ABS,
    )
    boxes = _numeric_gate(
        "generation.normalized_bboxes",
        np.empty((0, 4), dtype=np.float32),
        np.empty((0, 4), dtype=np.float32),
        max_mae=None,
        max_abs=FINAL_BBOX_MAX_ABS,
    )

    assert not logits["passed"]
    assert logits["max_abs"] > FINAL_LOGITS_MAX_ABS
    assert preprocessing["passed"]
    assert boxes["passed"] and boxes["max_abs"] == 0.0


def test_exact_ids_and_otsl_expose_the_first_divergence() -> None:
    expected = np.array([[2, 10, 9, 5, 9, 3]], dtype=np.int64)
    actual = expected.copy()
    actual[0, 3] = 4

    gate = _sequence_gate("generation.ids", actual, expected)

    assert gate["first_divergence"] == 3
    assert not gate["passed"]
    assert otsl_from_ids(expected[0].tolist()) == ["ched", "nl", "fcel", "nl"]


def test_exact_preprocessing_gate_rejects_shape_and_dtype_drift() -> None:
    expected = np.zeros((2, 2, 3), dtype=np.uint8)

    assert _exact_array_gate("pixels", expected.copy(), expected)["passed"]
    assert not _exact_array_gate("pixels", expected.astype(np.float32), expected)["passed"]
    assert not _exact_array_gate("pixels", expected[:1], expected)["passed"]
