# SPDX-License-Identifier: Apache-2.0

"""Same-weight and generation-state tests for the TableFormer v1 bbox head."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from docling.datamodel.pipeline_options import TableFormerMode

pytestmark = [pytest.mark.mlx, pytest.mark.parity]

mx: Any
torch: Any
tree_flatten: Any
BBoxDecoder: Any
ReferenceBBoxDecoder: Any
load_bbox_weights: Any
cxcywh_to_xyxy: Any
merge_horizontal_bboxes: Any
select_bbox_state_indices: Any


@pytest.fixture(scope="module", autouse=True)
def _load_requirements() -> None:
    global mx, torch, tree_flatten, BBoxDecoder, ReferenceBBoxDecoder, load_bbox_weights
    global cxcywh_to_xyxy, merge_horizontal_bboxes
    global select_bbox_state_indices

    probe = subprocess.run(
        [sys.executable, "-c", "import mlx.core"], check=False, capture_output=True, text=True
    )
    if probe.returncode:
        pytest.fail(f"selected parity lane requires Metal: {probe.stderr.strip()}")
    try:
        mx = import_module("mlx.core")
        torch = import_module("torch")
        tree_flatten = import_module("mlx.utils").tree_flatten
        module = import_module("docling_mlx._models.tableformer_v1.bbox")
        reference_module = import_module("tools.tableformer_v1.reference_bbox")
        ReferenceBBoxDecoder = reference_module.ReferenceBBoxDecoder
        load_bbox_weights = reference_module.load_bbox_weights
    except ImportError as error:
        pytest.fail(f"selected parity lane is missing a required dependency: {error}")
    BBoxDecoder = module.BBoxDecoder
    cxcywh_to_xyxy = module.cxcywh_to_xyxy
    merge_horizontal_bboxes = module.merge_horizontal_bboxes
    select_bbox_state_indices = module.select_bbox_state_indices


def _target_key(source_key: str) -> str:
    return source_key


def _same_weight_models(reference: Any | None = None) -> tuple[Any, Any]:
    if reference is None:
        torch.manual_seed(20260830)
        reference = ReferenceBBoxDecoder().eval()
    native = BBoxDecoder()
    target_shapes = {key: tuple(value.shape) for key, value in tree_flatten(native.parameters())}
    converted = []
    for source_key, tensor in reference.state_dict().items():
        if source_key.endswith("num_batches_tracked"):
            continue
        key = _target_key(source_key)
        array = tensor.detach().cpu().numpy()
        if array.ndim == 4:
            array = array.transpose(0, 2, 3, 1)
        assert target_shapes[key] == array.shape
        converted.append((key, mx.array(array)))
    assert {key for key, _ in converted} == set(target_shapes)
    native.load_weights(converted, strict=True)
    native.eval()
    return reference, native


def test_bbox_state_selection() -> None:
    selected, merges = select_bbox_state_indices(
        ["fcel", "fcel", "nl", "fcel", "lcel", "lcel", "fcel", "nl", "<end>"]
    )
    assert selected == [1, 2, 4, 6, 7]
    assert merges == {2: 3}


def test_horizontal_endpoint_merge_and_corner_conversion() -> None:
    classes, boxes = merge_horizontal_bboxes(
        mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=mx.float32),
        mx.array(
            [[0.2, 0.5, 0.2, 0.4], [0.6, 0.5, 0.2, 0.2], [0.9, 0.2, 0.1, 0.1]],
            dtype=mx.float32,
        ),
        {0: 1},
    )
    corners = cxcywh_to_xyxy(boxes)
    mx.eval(classes, boxes, corners)
    np.testing.assert_allclose(np.array(classes), [[1.0, 2.0], [5.0, 6.0]])
    np.testing.assert_allclose(np.array(boxes)[0], [0.4, 0.45, 0.6, 0.3], atol=1e-7)
    np.testing.assert_allclose(np.array(corners)[0], [0.1, 0.3, 0.7, 0.6], atol=1e-7)


def test_bbox_decoder_matches_independent_torch_oracle() -> None:
    reference, native = _same_weight_models()
    generator = np.random.default_rng(1861)
    encoder = generator.normal(0.0, 0.2, (1, 2, 2, 256)).astype(np.float32)
    states = generator.normal(0.0, 0.2, (3, 512)).astype(np.float32)

    with torch.inference_mode():
        expected_classes, expected_boxes = reference(
            torch.from_numpy(encoder), torch.from_numpy(states)
        )
    actual_classes, actual_boxes = native(mx.array(encoder), mx.array(states))
    mx.eval(actual_classes, actual_boxes)

    np.testing.assert_allclose(
        np.array(actual_classes), expected_classes.numpy(), rtol=0.0, atol=1e-4
    )
    np.testing.assert_allclose(np.array(actual_boxes), expected_boxes.numpy(), rtol=0.0, atol=1e-4)


@pytest.mark.parametrize("mode", [TableFormerMode.ACCURATE, TableFormerMode.FAST])
def test_bbox_decoder_matches_official_weights(mode: TableFormerMode) -> None:
    source_value = os.environ.get("DOCLING_MLX_TABLEFORMER_V1_SOURCE")
    if source_value is None:
        pytest.fail(
            "selected parity lane requires the official TableFormer v1 source; "
            "set DOCLING_MLX_TABLEFORMER_V1_SOURCE"
        )
    source = Path(source_value).expanduser()
    if not source.is_dir():
        pytest.fail(f"TableFormer v1 source directory does not exist: {source}")
    reference = ReferenceBBoxDecoder().eval()
    load_bbox_weights(reference, source, mode)
    _, native = _same_weight_models(reference)
    generator = np.random.default_rng(127)
    encoder = generator.normal(0.0, 0.2, (1, 2, 2, 256)).astype(np.float32)
    states = generator.normal(0.0, 0.2, (2, 512)).astype(np.float32)

    with torch.inference_mode():
        expected_classes, expected_boxes = reference(
            torch.from_numpy(encoder), torch.from_numpy(states)
        )
    actual_classes, actual_boxes = native(mx.array(encoder), mx.array(states))
    mx.eval(actual_classes, actual_boxes)
    np.testing.assert_allclose(
        np.array(actual_classes), expected_classes.numpy(), rtol=0.0, atol=1e-4
    )
    np.testing.assert_allclose(np.array(actual_boxes), expected_boxes.numpy(), rtol=0.0, atol=1e-4)


def test_empty_state_contract() -> None:
    _, native = _same_weight_models()
    classes, boxes = native(
        mx.zeros((1, 2, 2, 256), dtype=mx.float32),
        mx.zeros((0, 512), dtype=mx.float32),
    )
    assert classes.shape == (0, 3)
    assert boxes.shape == (0, 4)
