# SPDX-License-Identifier: Apache-2.0

"""Assembly contracts for the private native RT-DETR-v2 model."""

from __future__ import annotations

import subprocess
import sys
from importlib import import_module
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.mlx

mx: Any
nn: Any
tree_flatten: Any
RtDetrV2: Any
RtDetrV2Config: Any


@pytest.fixture(scope="module", autouse=True)
def _load_requirements() -> None:
    global mx, nn, tree_flatten, RtDetrV2, RtDetrV2Config

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
        model_module = import_module("docling_mlx._models.rt_detr_v2")
    except ImportError as error:
        pytest.fail(f"selected MLX lane is missing a required dependency: {error}")
    RtDetrV2 = model_module.RtDetrV2
    RtDetrV2Config = model_module.RtDetrV2Config


def test_model_topology_and_eval_contract() -> None:
    model = RtDetrV2(RtDetrV2Config())
    parameters = dict(tree_flatten(model.parameters()))
    assert not model.training
    assert [len(stage.layers) for stage in model.vision.backbone.encoder.stages] == [3, 4, 6, 3]
    assert len(model.decoder.layers) == 6
    assert parameters["vision.backbone.embedder.embedder.0.conv.weight"].shape == (32, 3, 3, 3)
    assert parameters["enc_score_head.weight"].shape == (2, 256)
    assert parameters["denoising_class_embed.weight"].shape == (3, 256)
    assert type(model.vision.backbone.embedder.embedder[0].conv) is nn.Conv2d
    assert type(model.vision.backbone.encoder.stages[0].layers[0].layer[1].conv) is nn.Conv2d

    with pytest.raises(ValueError, match="NHWC"):
        model(mx.zeros((1, 3, 640, 640), dtype=mx.float32))
    model.train()
    with pytest.raises(ValueError, match="eval"):
        model(mx.zeros((1, 640, 640, 3), dtype=mx.float32))


def test_backbone_compile_is_idempotent_and_exact() -> None:
    np = import_module("numpy")
    model = RtDetrV2(RtDetrV2Config())
    pixels = mx.arange(1 * 64 * 64 * 3, dtype=mx.float32).reshape(1, 64, 64, 3) / 255
    expected = model.vision(pixels)
    mx.eval(*expected)

    model.compile_backbone()
    compiled = model.vision._backbone_forward
    model.compile_backbone()
    actual = model.vision(pixels)
    mx.eval(*actual)

    assert model.vision._backbone_forward is compiled
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(np.array(left), np.array(right))


def test_model_executes_fast_encoder_and_decoder_attention() -> None:
    model = RtDetrV2(RtDetrV2Config())
    x = mx.zeros((1, 4, 256), dtype=mx.float32)
    pos = mx.zeros_like(x)
    original = mx.fast.scaled_dot_product_attention

    with patch.object(mx.fast, "scaled_dot_product_attention", wraps=original) as fast_attention:
        mx.eval(
            model.vision.hybrid_encoder.aifi[0].layers[0].self_attn(x, pos),
            model.decoder.layers[0].self_attn(x, pos),
        )
        assert fast_attention.call_count == 2


def test_encoder_query_selection_preserves_forward_outputs() -> None:
    class Vision(nn.Module):
        def __call__(self, pixel_values: Any) -> tuple[Any, ...]:
            del pixel_values
            values = mx.arange(12, dtype=mx.float32).reshape(1, 6, 2)
            return values[:, :4].reshape(1, 2, 2, 2), values[:, 4:5, None], values[:, 5:, None]

    class Identity(nn.Module):
        def __call__(self, value: Any) -> Any:
            return value

    class ScoreHead(nn.Module):
        def __call__(self, memory: Any) -> Any:
            return mx.stack([memory[..., 0], -memory[..., 0]], axis=-1)

    class BoxHead(nn.Module):
        def __call__(self, memory: Any) -> Any:
            return mx.zeros((*memory.shape[:-1], 4), dtype=memory.dtype)

    class Decoder(nn.Module):
        def __call__(
            self,
            *,
            target: Any,
            reference_points_unact: Any,
            encoder_hidden_states: Any,
            spatial_shapes: Any,
        ) -> dict[str, Any]:
            del encoder_hidden_states, spatial_shapes
            return {
                "intermediate_logits": target[..., :1, None],
                "intermediate_reference_points": mx.sigmoid(reference_points_unact)[:, None],
                "last_hidden_state": target,
            }

    model = RtDetrV2.__new__(RtDetrV2)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_queries=2,
        backbone_config=SimpleNamespace(num_channels=3),
    )
    model.vision = Vision()
    model.decoder_input_proj = [Identity(), Identity(), Identity()]
    model.enc_output = Identity()
    model.enc_score_head = ScoreHead()
    model.enc_bbox_head = BoxHead()
    model.decoder = Decoder()
    model.eval()

    output = model(mx.zeros((1, 2, 2, 3), dtype=mx.float32))
    mx.eval(*output.values())

    np = import_module("numpy")
    np.testing.assert_array_equal(np.array(output["last_hidden_state"])[0, :, 0], [10.0, 8.0])
    np.testing.assert_allclose(
        np.array(output["pred_boxes"])[0],
        [[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.1, 0.1]],
        rtol=1e-6,
        atol=1e-6,
    )
