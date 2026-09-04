# SPDX-License-Identifier: Apache-2.0

"""Assembly boundaries for the native Egret D-FINE model."""

from __future__ import annotations

import subprocess
import sys
from importlib import import_module
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from docling_mlx._models.dfine.config import DFineConfig
from tests.layout_egret.test_config import egret_config

pytestmark = pytest.mark.mlx

mx: Any
nn: Any
tree_flatten: Any
model_module: Any
DFine: Any


@pytest.fixture(scope="module", autouse=True)
def _load_requirements() -> None:
    global mx, nn, tree_flatten, model_module, DFine

    probe = subprocess.run(
        [sys.executable, "-c", "import mlx.core"],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode:
        pytest.fail(f"selected MLX lane requires Metal: {probe.stderr.strip()}")
    try:
        mx = import_module("mlx.core")
        nn = import_module("mlx.nn")
        tree_flatten = import_module("mlx.utils").tree_flatten
        model_module = import_module("docling_mlx._models.dfine.model")
    except ImportError as error:
        pytest.fail(f"selected MLX lane is missing a required dependency: {error}")
    DFine = model_module.DFine


class _FakeVisionTower:
    def __init__(self, config: DFineConfig) -> None:
        self.channels = config.encoder.hidden_dim

    def __call__(self, pixel_values: Any) -> tuple[Any, ...]:
        batch = pixel_values.shape[0]
        values = mx.arange(525, dtype=mx.float32)[None, :, None]
        values = mx.broadcast_to(values, (batch, 525, self.channels))
        return (
            values[:, :400].reshape(batch, 20, 20, self.channels),
            values[:, 400:500].reshape(batch, 10, 10, self.channels),
            values[:, 500:].reshape(batch, 5, 5, self.channels),
        )


class _FakeDecoder:
    def __init__(self, config: Any, *, num_labels: int) -> None:
        self.num_labels = num_labels
        self.corner_dim = 4 * (config.max_num_bins + 1)

    def __call__(
        self,
        *,
        encoder_hidden_states: Any,
        reference_points: Any,
        inputs_embeds: Any,
        spatial_shapes: Any,
    ) -> dict[str, Any]:
        del encoder_hidden_states, spatial_shapes
        logits = mx.broadcast_to(
            inputs_embeds[..., :1], (*inputs_embeds.shape[:2], self.num_labels)
        )
        references = mx.sigmoid(reference_points)
        corners = mx.zeros((*inputs_embeds.shape[:2], self.corner_dim), dtype=inputs_embeds.dtype)
        return {
            "last_hidden_state": inputs_embeds,
            "intermediate_hidden_states": inputs_embeds[None],
            "intermediate_logits": logits[:, None],
            "intermediate_reference_points": references[:, None],
            "intermediate_predicted_corners": corners[:, None],
            "initial_reference_points": references[:, None],
        }


class _Identity:
    def __call__(self, value: Any) -> Any:
        return value


class _ScoreHead:
    def __call__(self, memory: Any) -> Any:
        return mx.broadcast_to(memory[..., :1], (*memory.shape[:2], 17))


class _BoxHead:
    def __call__(self, memory: Any) -> Any:
        return mx.zeros((*memory.shape[:2], 4), dtype=memory.dtype)


def _model(profile: str = "medium") -> Any:
    config = DFineConfig.from_dict(egret_config(profile))
    with (
        patch.object(model_module, "DFineVisionTower", _FakeVisionTower),
        patch.object(model_module, "DFineDecoder", _FakeDecoder),
    ):
        return DFine(config)


@pytest.mark.parametrize("profile", ["medium", "large", "xlarge"])
def test_configs_construct_decoder_projections(profile: str) -> None:
    model = _model(profile)
    parameters = dict(tree_flatten(model.parameters()))
    assert not model.training
    assert len(model.decoder_input_proj) == 3
    assert parameters["enc_score_head.weight"].shape == (17, 256)
    assert "denoising_class_embed.weight" not in parameters
    if profile == "xlarge":
        assert parameters["decoder_input_proj.0.conv.weight"].shape == (256, 1, 1, 384)
    else:
        assert not any(key.startswith("decoder_input_proj.") for key in parameters)


def test_eval_forward_preserves_transformers_selection_and_output_boundaries() -> None:
    model = _model()
    model.enc_output = _Identity()
    model.enc_score_head = _ScoreHead()
    model.enc_bbox_head = _BoxHead()

    output = model(mx.zeros((1, 4, 4, 3), dtype=mx.float32))
    mx.eval(*output.values())

    assert set(output) == {
        "pred_logits",
        "pred_boxes",
        "last_hidden_state",
        "intermediate_hidden_states",
        "intermediate_logits",
        "intermediate_reference_points",
        "intermediate_predicted_corners",
        "initial_reference_points",
        "init_reference_points",
        "enc_topk_logits",
        "enc_topk_bboxes",
        "enc_outputs_class",
        "enc_outputs_coord_logits",
    }
    assert output["pred_logits"].shape == (1, 300, 17)
    assert output["pred_boxes"].shape == (1, 300, 4)
    np.testing.assert_array_equal(
        np.array(output["last_hidden_state"])[0, (0, -1), 0], [524.0, 225.0]
    )
    np.testing.assert_array_equal(
        np.array(output["pred_logits"]), np.array(output["intermediate_logits"][:, -1])
    )
    np.testing.assert_array_equal(
        np.array(output["pred_boxes"]),
        np.array(output["intermediate_reference_points"][:, -1]),
    )
    np.testing.assert_array_equal(
        np.array(output["enc_topk_bboxes"]), mx.sigmoid(output["init_reference_points"])
    )


def test_forward_rejects_training_and_invalid_inputs() -> None:
    model = _model()
    model(mx.zeros((1, 4, 4, 3), dtype=mx.float16))
    with pytest.raises(ValueError, match="float16 or float32"):
        model(mx.zeros((1, 4, 4, 3), dtype=mx.int32))
    with pytest.raises(ValueError, match="NHWC"):
        model(mx.zeros((1, 3, 4, 4), dtype=mx.float32))
    model.train()
    with pytest.raises(ValueError, match="eval"):
        model(mx.zeros((1, 4, 4, 3), dtype=mx.float32))


def test_compiled_backbone_is_idempotent_and_exact() -> None:
    vision_module = import_module("docling_mlx._models.dfine.vision")
    config = DFineConfig.from_dict(egret_config("medium"))
    tower = vision_module.DFineVisionTower(config)
    pixels = mx.arange(1 * 64 * 64 * 3, dtype=mx.float32).reshape(1, 64, 64, 3) / 255
    expected = tower(pixels)
    mx.eval(*expected)

    tower.compile_backbone()
    compiled = tower._backbone_forward
    tower.compile_backbone()
    actual = tower(pixels)
    mx.eval(*actual)

    assert tower._backbone_forward is compiled
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(np.array(left), np.array(right))
