# SPDX-License-Identifier: Apache-2.0

"""Warmup preserves generic-engine outputs and is idempotent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from docling_mlx.engines.image_classification.efficientnet.engine import (
    EfficientNetEngine,
    EfficientNetModelSpec,
)
from docling_mlx.engines.object_detection.dfine.engine import DFineEngine, DFineModelSpec
from docling_mlx.engines.object_detection.rt_detr_v2.engine import (
    RtDetrV2Engine,
    RtDetrV2ModelSpec,
)
from docling_mlx.engines.table_structure.tableformer_v1.engine import (
    TableFormerV1Engine,
)
from docling_mlx.engines.table_structure.tableformer_v1.model_spec import TableFormerV1ModelSpec
from docling_mlx.engines.table_structure.tableformer_v2.engine import TableFormerV2Engine
from docling_mlx.engines.table_structure.tableformer_v2.model_spec import TableFormerV2ModelSpec

ROOT = Path(__file__).parents[2]


@pytest.mark.mlx
@pytest.mark.parametrize(
    "kind",
    ["rt-detr-v2", "dfine", "efficientnet", "tableformer-v1", "tableformer-v2"],
)
def test_warmup_is_idempotent_and_bit_identical(kind: str) -> None:
    artifact, image_path = {
        "rt-detr-v2": (
            ROOT / ".artifacts/heron-r50",
            ROOT / "tests/fixtures/layout_heron/benchmark_gradient.png",
        ),
        "dfine": (
            ROOT / ".artifacts/egret-medium",
            ROOT / "tests/fixtures/layout_heron/benchmark_gradient.png",
        ),
        "efficientnet": (
            ROOT / ".artifacts/document-figure-classifier",
            ROOT / "tests/fixtures/document_figure/reference_images/bar_chart.png",
        ),
        "tableformer-v1": (
            ROOT / ".artifacts/tableformer-v1/accurate",
            ROOT / "tests/fixtures/tableformer_v2/basin_table_1.png",
        ),
        "tableformer-v2": (
            ROOT / ".artifacts/tableformer-v2",
            ROOT / "tests/fixtures/tableformer_v2/basin_table_1.png",
        ),
    }[kind]
    if not artifact.is_dir() or not image_path.is_file():
        pytest.fail(f"warmup test requires local artifact and fixture for {kind}")

    engines = {
        "rt-detr-v2": lambda: RtDetrV2Engine(RtDetrV2ModelSpec(path=artifact)),
        "dfine": lambda: DFineEngine(DFineModelSpec(path=artifact)),
        "efficientnet": lambda: EfficientNetEngine(EfficientNetModelSpec(path=artifact)),
        "tableformer-v1": lambda: TableFormerV1Engine(TableFormerV1ModelSpec(path=artifact)),
        "tableformer-v2": lambda: TableFormerV2Engine(TableFormerV2ModelSpec(path=artifact)),
    }
    cold = engines[kind]()
    warm = engines[kind]()
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    try:
        cold_output = cold.predict([image.copy()])
        warm.initialize(warmup=True)
        warm.initialize(warmup=True)
        warm_output = warm.predict([image.copy()])
    finally:
        image.close()

    assert warm._warmup_done is True  # type: ignore[attr-defined]
    assert len(cold_output) == len(warm_output) == 1
    cold_item, warm_item = cold_output[0], warm_output[0]
    if kind == "efficientnet":
        assert cold_item.label_ids == warm_item.label_ids
        np.testing.assert_array_equal(cold_item.probabilities, warm_item.probabilities)
        return
    if kind in {"rt-detr-v2", "dfine"}:
        assert cold_item.label_ids == warm_item.label_ids
        np.testing.assert_array_equal(cold_item.scores, warm_item.scores)
        np.testing.assert_array_equal(cold_item.boxes, warm_item.boxes)
    else:
        assert cold_item == warm_item
