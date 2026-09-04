# SPDX-License-Identifier: Apache-2.0

"""Real checkpoint-loader parity for the generic RT-DETR-v2 engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from docling_mlx.engines.object_detection.rt_detr_v2.engine import (
    RtDetrV2Engine,
    RtDetrV2ModelSpec,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/layout_heron/benchmark_gradient.png"
HF_CACHE = Path.home() / ".cache/huggingface/hub"


@pytest.mark.mlx
@pytest.mark.parity
@pytest.mark.parametrize(
    ("artifact", "upstream"),
    [
        (
            ROOT / ".artifacts/heron-r50",
            HF_CACHE
            / "models--docling-project--docling-layout-heron/snapshots"
            / "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8",
        ),
        (
            ROOT / ".artifacts/heron-r101",
            HF_CACHE
            / "models--docling-project--docling-layout-heron-101/snapshots"
            / "2e4993cf6bb211112084a2f80938f26138008917",
        ),
    ],
)
def test_upstream_checkpoint_matches_converted_artifact(artifact: Path, upstream: Path) -> None:
    required = ("config.json", "preprocessor_config.json", "model.safetensors")
    for directory in (artifact, upstream):
        if any(not (directory / name).is_file() for name in required):
            pytest.fail(f"RT-DETR-v2 loader parity requires local checkpoint: {directory}")
    with Image.open(FIXTURE) as image:
        artifact_engine = RtDetrV2Engine(RtDetrV2ModelSpec(path=artifact))
        upstream_engine = RtDetrV2Engine(RtDetrV2ModelSpec(path=upstream))
        artifact_result = artifact_engine.predict([image.copy()])[0]
        upstream_result = upstream_engine.predict([image.copy()])[0]

    assert upstream_result.id2label == artifact_result.id2label
    assert upstream_result.label_ids == artifact_result.label_ids
    np.testing.assert_array_equal(upstream_result.scores, artifact_result.scores)
    np.testing.assert_array_equal(upstream_result.boxes, artifact_result.boxes)
