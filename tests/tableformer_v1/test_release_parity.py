# SPDX-License-Identifier: Apache-2.0

"""Pure release-capture and parity-gate contracts for TableFormer v1."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from docling.datamodel.pipeline_options import TableFormerMode

from tools.tableformer_v1.capture_reference import (
    CAPTURE_SCHEMA_VERSION,
    array_specs,
    write_compressed_npz,
)
from tools.tableformer_v1.validate_parity import (
    FINAL_BBOX_MAX_ABS,
    FINAL_LOGITS_MAX_ABS,
    FINAL_LOGITS_MAX_MAE,
    _load_reference,
    _numeric_gate,
    _sequence_gate,
    validate,
)

ROOT = Path(__file__).resolve().parents[2]


def _arrays(encoder_layers: int) -> dict[str, np.ndarray]:
    return {
        spec.name: np.zeros((1,), dtype=np.dtype(spec.dtype))
        for spec in array_specs(encoder_layers)
    }


@pytest.mark.parametrize("encoder_layers", [6, 4])
def test_reference_archive_is_reproducible_and_schema_ordered(
    tmp_path: Path, encoder_layers: int
) -> None:
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    specs = array_specs(encoder_layers)
    write_compressed_npz(first, _arrays(encoder_layers), specs=specs)
    write_compressed_npz(second, _arrays(encoder_layers), specs=specs)

    assert first.read_bytes() == second.read_bytes()
    with np.load(first, allow_pickle=False) as archive:
        assert archive.files == [spec.name for spec in specs]


def test_release_caps_and_generated_ids_are_not_relaxed() -> None:
    expected = np.zeros((1, 2), dtype=np.float32)
    actual = expected.copy()
    actual[0, 0] = FINAL_LOGITS_MAX_ABS * 1.1
    logits = _numeric_gate(
        "generation.logits",
        actual,
        expected,
        mae=FINAL_LOGITS_MAX_MAE,
        max_abs=FINAL_LOGITS_MAX_ABS,
    )
    boxes = _numeric_gate(
        "prediction.boxes",
        np.empty((0, 4), dtype=np.float32),
        np.empty((0, 4), dtype=np.float32),
        mae=None,
        max_abs=FINAL_BBOX_MAX_ABS,
    )
    ids = _sequence_gate(np.array([[2, 4, 3]]), np.array([[2, 5, 3]]))

    assert not logits["passed"]
    assert boxes["passed"]
    assert ids["first_divergence"] == 1


def test_reference_rejects_unfrozen_source_metadata(
    tmp_path: Path,
) -> None:
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": CAPTURE_SCHEMA_VERSION,
                "producer": "tools.tableformer_v1.capture_reference",
                "profile": "tableformer_v1_accurate",
                "source": {"revision": "0" * 40},
                "captures": [],
            }
        )
    )

    with pytest.raises(ValueError, match="frozen revision"):
        _load_reference(tmp_path, TableFormerMode.ACCURATE, 6)


@pytest.mark.mlx
@pytest.mark.parity
@pytest.mark.release
@pytest.mark.parametrize("mode", [TableFormerMode.ACCURATE, TableFormerMode.FAST])
def test_release_capture_passes_every_raw_and_stage_gate(mode: TableFormerMode) -> None:
    artifact = Path(
        os.environ.get(
            "DOCLING_MLX_TABLEFORMER_V1_ARTIFACT",
            ROOT / ".artifacts/tableformer-v1",
        )
    )
    reference_environment = (
        "DOCLING_MLX_TABLEFORMER_V1_REFERENCE"
        if mode is TableFormerMode.ACCURATE
        else "DOCLING_MLX_TABLEFORMER_V1_FAST_REFERENCE"
    )
    reference_default = (
        ROOT / ".artifacts/tableformer-v1-reference"
        if mode is TableFormerMode.ACCURATE
        else ROOT / ".artifacts/tableformer-v1-fast-reference"
    )
    reference = Path(os.environ.get(reference_environment, reference_default))
    if not artifact.is_dir() or not reference.is_dir():
        pytest.fail(
            "TableFormer v1 release parity requires both "
            "DOCLING_MLX_TABLEFORMER_V1_ARTIFACT and "
            f"{reference_environment}"
        )

    report = validate(artifact, reference, mode=mode)

    assert report["passed"], report
