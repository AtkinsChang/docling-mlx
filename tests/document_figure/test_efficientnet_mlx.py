# SPDX-License-Identifier: Apache-2.0

"""Apple MLX and pinned Torch parity checks for native EfficientNet."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.document_figure.test_efficientnet import b0_config

pytestmark = pytest.mark.mlx


def _required_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except (ImportError, ModuleNotFoundError) as error:
        pytest.fail(f"selected MLX qualification requires {name}: {error}")


def _model_type():
    return importlib.import_module("docling_mlx._models.efficientnet.model").EfficientNet


def test_native_topology_includes_batchnorm_state() -> None:
    mx = _required_module("mlx.core")
    tree_flatten = _required_module("mlx.utils").tree_flatten
    model = _model_type()(b0_config())
    parameters = dict(tree_flatten(model.parameters()))

    assert len(model.efficientnet.encoder.blocks) == 16
    assert not model.training
    assert parameters["efficientnet.embeddings.convolution.weight"].shape == (32, 3, 3, 3)
    assert parameters["efficientnet.embeddings.batchnorm.running_mean"].dtype == mx.float32
    assert parameters["classifier.weight"].shape == (26, 1280)


def test_rejects_non_floating_inputs() -> None:
    mx = _required_module("mlx.core")
    model = _model_type()(b0_config())
    with pytest.raises(ValueError, match="floating-point"):
        model(mx.zeros((1, 224, 224, 3), dtype=mx.int32))


def test_accepts_float16_inputs() -> None:
    mx = _required_module("mlx.core")
    model = _model_type()(b0_config())
    logits = model(mx.zeros((1, 224, 224, 3), dtype=mx.float16))
    mx.eval(logits)
    assert logits.shape == (1, 26)


def test_rejects_nchw_and_training() -> None:
    mx = _required_module("mlx.core")
    model = _model_type()(b0_config())
    with pytest.raises(ValueError, match="NHWC"):
        model(mx.zeros((1, 3, 224, 224)))
    model.train()
    with pytest.raises(ValueError, match="eval"):
        model(mx.zeros((1, 224, 224, 3)))


def test_compiled_forward_is_idempotent_and_within_fp32_tolerance() -> None:
    mx = _required_module("mlx.core")
    model = _model_type()(b0_config())
    pixels = mx.arange(1 * 64 * 64 * 3, dtype=mx.float32).reshape(1, 64, 64, 3) / 255
    expected = model(pixels)
    mx.eval(expected)

    model.compile_forward()
    compiled = model._compiled_forward
    model.compile_forward()
    actual = model(pixels)
    mx.eval(actual)

    assert model._compiled_forward is compiled
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=0, atol=1e-5)


@pytest.fixture(scope="module")
def real_models(pinned_source: Path):
    mx = _required_module("mlx.core")
    torch = _required_module("torch")
    transformers = _required_module("transformers")

    default_root = Path(__file__).resolve().parents[2] / ".artifacts/document-figure-classifier"
    root = Path(os.environ.get("DOCLING_MLX_ARTIFACTS", default_root))
    if not (root / "model.safetensors").exists():
        pytest.fail("selected parity lane requires the converted DocumentFigure artifact")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    reference = (
        transformers.EfficientNetForImageClassification.from_pretrained(
            pinned_source, local_files_only=True, dtype=torch.float32
        )
        .cpu()
        .eval()
    )
    native = _model_type()(json.loads((root / "config.json").read_text()))
    native.load_weights(str(root / "model.safetensors"), strict=True)
    mx.eval(native.parameters())
    return mx, torch, reference, native


@pytest.mark.parity
def test_real_weights_layer_and_logits_parity(real_models) -> None:
    mx, torch, reference, native = real_models
    generator = np.random.default_rng(814)
    rgb = generator.integers(0, 256, size=(1, 224, 224, 3), dtype=np.uint8)
    pixels = rgb.astype(np.float32) / np.float32(255)
    pixels = (pixels - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.47853944, 0.4732864, 0.47434163], dtype=np.float32
    )
    reference_layers = {}

    def capture(name):
        def hook(_module, _args, output):
            reference_layers[name] = output.detach().cpu().numpy()

        return hook

    handles = [reference.efficientnet.embeddings.register_forward_hook(capture("stem"))]
    handles += [
        block.register_forward_hook(capture(f"blocks.{index}"))
        for index, block in enumerate(reference.efficientnet.encoder.blocks)
    ]
    handles.append(
        reference.efficientnet.encoder.top_activation.register_forward_hook(capture("top"))
    )
    try:
        with torch.inference_mode():
            expected = reference(
                torch.from_numpy(pixels.transpose(0, 3, 1, 2).copy())
            ).logits.numpy()
    finally:
        for handle in handles:
            handle.remove()
    actual = native.forward_intermediates(mx.array(pixels))
    mx.eval(actual)
    for name, value in reference_layers.items():
        difference = np.abs(np.asarray(actual[name]) - value.transpose(0, 2, 3, 1))
        assert difference.mean() <= 1e-4, (name, difference.mean())
        assert difference.max() <= 1e-3, (name, difference.max())
    difference = np.abs(np.asarray(actual["logits"]) - expected)
    assert expected.shape == (1, 26)
    assert difference.mean() <= 1e-4, difference.mean()
    assert difference.max() <= 1e-3, difference.max()
    np.testing.assert_array_equal(np.asarray(actual["logits"]).argmax(-1), expected.argmax(-1))
