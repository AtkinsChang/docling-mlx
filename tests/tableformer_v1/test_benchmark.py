# SPDX-License-Identifier: Apache-2.0

"""TableFormerV1 benchmark boundary contracts."""

from __future__ import annotations

import numpy as np
from PIL import Image

from docling_mlx.engines.table_structure.tableformer_v1.engine import (
    TableFormerV1Prediction,
)
from tools.tableformer_v1.benchmark import _materialize_torch_prediction


def test_torch_engine_boundary_returns_the_same_canonical_prediction_as_mlx() -> None:
    class_logits = np.array([[0.1, 0.9, 0.2]], dtype=np.float32)
    cxcywh = np.array([[0.5, 0.5, 0.4, 0.2]], dtype=np.float32)

    prediction = _materialize_torch_prediction(
        [2, 4, 3], class_logits, cxcywh, Image.new("RGB", (1, 1))
    )
    class_logits[:] = -1
    cxcywh[:] = -1

    assert prediction == TableFormerV1Prediction(
        token_ids=(2, 4, 3),
        otsl_tokens=("ecel",),
        cell_bboxes=(
            (0.30000001192092896, 0.4000000059604645, 0.699999988079071, 0.6000000238418579),
        ),
        bbox_classes=(1,),
    )
