# SPDX-License-Identifier: Apache-2.0

"""Framework-free D-FINE engine API contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from docling_mlx.engines.object_detection.dfine.engine import (
    DFineEngine,
    DFineEngineOptions,
    DFineModelSpec,
)
from docling_mlx.engines.object_detection.rt_detr_v2.engine import Detections


def test_model_spec_requires_one_checkpoint_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        DFineModelSpec()
    with pytest.raises(ValueError, match="exactly one"):
        DFineModelSpec(repo_id="example/dfine", path=tmp_path)
    assert DFineModelSpec(path=tmp_path).path == tmp_path


def test_generic_engine_exposes_shared_detection_contract(tmp_path: Path) -> None:
    engine = DFineEngine(DFineModelSpec(path=tmp_path), DFineEngineOptions(dtype=None))
    assert engine.predict([]) == []
    assert Detections.__module__ == "docling_mlx.engines.object_detection._types"


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_options_validate_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="score_threshold"):
        DFineEngineOptions(score_threshold=threshold)
