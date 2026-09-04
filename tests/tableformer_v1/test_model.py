# SPDX-License-Identifier: Apache-2.0

"""TableFormer v1 composition and source-generation contracts."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from tests.tableformer_v1.conftest import accurate_config, fast_config

pytestmark = pytest.mark.mlx


def _model(*, fast: bool = False) -> Any:
    from docling_mlx._models.tableformer_v1.config import TableFormerV1Config
    from docling_mlx._models.tableformer_v1.model import TableFormerV1

    return TableFormerV1(
        TableFormerV1Config.from_dict(fast_config() if fast else accurate_config())
    )


def test_model_preserves_the_complete_source_parameter_namespace() -> None:
    from mlx.utils import tree_flatten

    model = _model()
    parameters = dict(tree_flatten(model.parameters()))

    assert not model.training
    assert len(parameters) == 329
    assert sum(name.startswith("_encoder.") for name in parameters) == 75
    assert sum(name.startswith("_tag_transformer.") for name in parameters) == 209
    assert sum(name.startswith("_bbox_decoder.") for name in parameters) == 45
    assert parameters["_tag_transformer._embedding.weight"].shape == (13, 512)
    assert parameters["_tag_transformer._decoder.layers.5.linear1.weight"].shape == (
        1024,
        512,
    )
    assert parameters["_bbox_decoder._bbox_embed.layers.2.weight"].shape == (4, 256)


def test_fast_model_is_the_exact_lower_depth_subset() -> None:
    from mlx.utils import tree_flatten

    accurate = dict(tree_flatten(_model().parameters()))
    fast = dict(tree_flatten(_model(fast=True).parameters()))

    assert len(fast) == 233
    assert sum(name.startswith("_encoder.") for name in fast) == 75
    assert sum(name.startswith("_tag_transformer.") for name in fast) == 113
    assert sum(name.startswith("_bbox_decoder.") for name in fast) == 45
    assert set(fast) < set(accurate)
    assert all(fast[name].shape == accurate[name].shape for name in fast)
    assert all(fast[name].dtype == accurate[name].dtype for name in fast)
    assert set(accurate) - set(fast) == {
        name
        for name in accurate
        if name.startswith("_tag_transformer._encoder.layers.4.")
        or name.startswith("_tag_transformer._encoder.layers.5.")
        or name.startswith("_tag_transformer._decoder.layers.2.")
        or name.startswith("_tag_transformer._decoder.layers.3.")
        or name.startswith("_tag_transformer._decoder.layers.4.")
        or name.startswith("_tag_transformer._decoder.layers.5.")
    }
    assert len(set(accurate) - set(fast)) == 96


def test_encode_accepts_the_engine_nchw_contract_and_transposes_to_nhwc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mlx.core as mx

    model = _model()
    captured: dict[str, Any] = {}

    def image_encoder(_self: Any, pixels: Any) -> Any:
        captured["pixels"] = pixels
        return mx.zeros((1, 28, 28, 256), dtype=mx.float32)

    class TagEncoder:
        def __call__(self, _states: Any) -> Any:
            return mx.zeros((1, 784, 512), dtype=mx.float32)

    monkeypatch.setattr(type(model._encoder), "__call__", image_encoder)
    tag_transformer = cast(Any, model._tag_transformer)
    monkeypatch.setattr(tag_transformer, "_input_filter", [])
    monkeypatch.setattr(tag_transformer, "_encoder", TagEncoder())
    pixels = mx.broadcast_to(mx.arange(3, dtype=mx.float32)[None, :, None, None], (1, 3, 448, 448))

    image_features, memory = model._encode(pixels)
    mx.eval(captured["pixels"], image_features, memory)

    assert captured["pixels"].shape == (1, 448, 448, 3)
    np.testing.assert_array_equal(np.asarray(captured["pixels"])[0, 0, 0], [0.0, 1.0, 2.0])
    assert image_features.shape == (1, 28, 28, 256)
    assert memory.shape == (784, 1, 512)


def test_compiled_image_backbone_is_idempotent_and_preserves_encoder_output() -> None:
    import mlx.core as mx

    model = _model(fast=True)
    pixels = mx.arange(1 * 64 * 64 * 3, dtype=mx.float32).reshape(1, 64, 64, 3) / 255
    expected = model._encoder(pixels)
    mx.eval(expected)

    model.compile_image_backbone()
    compiled = model._encoder_forward
    model.compile_image_backbone()
    actual = model._encoder_forward(pixels)
    mx.eval(actual)

    assert model._encoder_forward is compiled
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=0, atol=3e-6)


def test_generation_matches_source_state_selection_and_horizontal_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mlx.core as mx

    from docling_mlx._models.tableformer_v1.bbox import BBoxDecoder
    from docling_mlx._models.tableformer_v1.decoder import (
        GenerationOutput,
        TableFormerV1TokenDecoder,
    )

    model = _model()
    image_features = mx.zeros((1, 28, 28, 256), dtype=mx.float32)
    memory = mx.zeros((784, 1, 512), dtype=mx.float32)
    hidden = mx.broadcast_to(mx.arange(4, dtype=mx.float32)[None, :, None], (1, 4, 512))
    generated_ids = mx.array([[2, 6, 5, 9, 3]], dtype=mx.int32)

    monkeypatch.setattr(model, "_encode", lambda _pixels: (image_features, memory))
    monkeypatch.setattr(
        TableFormerV1TokenDecoder,
        "generate",
        lambda _self, _memory, **_: GenerationOutput(
            generated_ids,
            hidden,
            mx.zeros((model.config.decoder_layers, 4, 1, 512), dtype=mx.float32),
        ),
    )

    def bbox(_self: Any, _image_features: Any, states: Any) -> tuple[Any, Any]:
        np.testing.assert_array_equal(np.asarray(states[:, 0]), [0.0, 1.0, 2.0])
        return (
            mx.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=mx.float32),
            mx.array(
                [[0.2, 0.3, 0.2, 0.2], [0.5, 0.4, 0.2, 0.2], [0.8, 0.7, 0.1, 0.2]],
                dtype=mx.float32,
            ),
        )

    monkeypatch.setattr(BBoxDecoder, "__call__", bbox)
    ids, classes, boxes = model.generate(mx.zeros((1, 3, 448, 448), dtype=mx.float32))
    mx.eval(ids, classes, boxes)

    np.testing.assert_array_equal(np.asarray(ids), [[2, 6, 5, 9, 3]])
    np.testing.assert_array_equal(np.asarray(classes), [[1, 2, 3], [7, 8, 9]])
    np.testing.assert_allclose(
        np.asarray(boxes),
        [[0.1, 0.2, 0.6, 0.5], [0.75, 0.6, 0.85, 0.8]],
        rtol=0,
        atol=1e-6,
    )


def test_generation_returns_empty_bbox_outputs_for_immediate_eos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mlx.core as mx

    from docling_mlx._models.tableformer_v1.decoder import (
        GenerationOutput,
        TableFormerV1TokenDecoder,
    )

    model = _model()
    monkeypatch.setattr(
        model,
        "_encode",
        lambda _pixels: (
            mx.zeros((1, 28, 28, 256), dtype=mx.float32),
            mx.zeros((784, 1, 512), dtype=mx.float32),
        ),
    )
    monkeypatch.setattr(
        TableFormerV1TokenDecoder,
        "generate",
        lambda _self, _memory, **_: GenerationOutput(
            mx.array([[2, 3]], dtype=mx.int32),
            mx.zeros((1, 1, 512), dtype=mx.float32),
            mx.zeros((model.config.decoder_layers, 1, 1, 512), dtype=mx.float32),
        ),
    )

    ids, classes, boxes = model.generate(mx.zeros((1, 3, 448, 448), dtype=mx.float32))
    mx.eval(ids, classes, boxes)

    assert ids.shape == (1, 2)
    assert classes.shape == (0, 3)
    assert boxes.shape == (0, 4)


def test_model_rejects_training_non_fp32_and_wrong_layout() -> None:
    import mlx.core as mx

    model = _model()
    with pytest.raises(ValueError, match="float32"):
        model.generate(mx.zeros((1, 3, 448, 448), dtype=mx.float16))
    with pytest.raises(ValueError, match="shape"):
        model.generate(mx.zeros((1, 448, 448, 3), dtype=mx.float32))
    model.train()
    with pytest.raises(ValueError, match="eval"):
        model.generate(mx.zeros((1, 3, 448, 448), dtype=mx.float32))
