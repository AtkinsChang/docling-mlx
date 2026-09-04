# SPDX-License-Identifier: Apache-2.0

"""MLX-artifact and upstream-HF D-FINE loader parity."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from docling_mlx.engines.object_detection.dfine.engine import (
    DFineEngine,
    DFineEngineOptions,
    DFineModelSpec,
)

pytestmark = pytest.mark.mlx

_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = _ROOT / "tests/fixtures/layout_heron/benchmark_gradient.png"


@pytest.mark.parametrize("name", ["medium", "large", "xlarge"])
def test_upstream_and_mlx_artifact_loaders_return_identical_detections(name: str) -> None:
    environment_name = f"DOCLING_MLX_EGRET_{name.upper()}_SOURCE"
    source_value = os.environ.get(environment_name)
    if source_value is None:
        pytest.fail(f"D-FINE loader parity requires {environment_name}")
    source = Path(source_value).expanduser()
    artifact = _ROOT / ".artifacts" / f"egret-{name}"
    missing = [path for path in (source, artifact, _FIXTURE) if not path.exists()]
    if missing:
        pytest.fail(f"D-FINE loader parity requires local inputs: {missing}")
    with Image.open(_FIXTURE) as image:
        images = [image.copy()]
    upstream = DFineEngine(DFineModelSpec(path=source)).predict(images)
    converted = DFineEngine(DFineModelSpec(path=artifact)).predict(images)
    assert converted == upstream


def test_engine_dtype_option_loads_float16_weights() -> None:
    import mlx.core as mx

    artifact = _ROOT / ".artifacts" / "egret-medium"
    if not artifact.is_dir() or not _FIXTURE.is_file():
        pytest.fail("D-FINE dtype test requires the local egret-medium artifact and fixture")
    with Image.open(_FIXTURE) as image:
        detections = DFineEngine(
            DFineModelSpec(path=artifact), DFineEngineOptions(dtype=mx.float16)
        ).predict([image.copy()])
    assert len(detections) == 1
