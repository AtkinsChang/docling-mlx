# SPDX-License-Identifier: Apache-2.0

"""CPU-only contracts for the generic RT-DETR-v2 engine boundary."""

from pathlib import Path

import pytest

from docling_mlx.engines.object_detection.rt_detr_v2.engine import (
    RtDetrV2Engine,
    RtDetrV2EngineOptions,
    RtDetrV2ModelSpec,
)


def test_model_spec_requires_one_loading_source() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        RtDetrV2ModelSpec()
    with pytest.raises(ValueError, match="exactly one"):
        RtDetrV2ModelSpec(repo_id="example/model", path=Path("model"))


def test_engine_options_validate_threshold() -> None:
    assert RtDetrV2EngineOptions().score_threshold == pytest.approx(0.3)
    with pytest.raises(ValueError, match="score_threshold"):
        RtDetrV2EngineOptions(score_threshold=1.1)


def test_empty_prediction_is_lazy() -> None:
    engine = RtDetrV2Engine(RtDetrV2ModelSpec(path=Path("does-not-need-to-exist")))

    assert engine.predict([]) == []
