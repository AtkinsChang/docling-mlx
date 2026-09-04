# SPDX-License-Identifier: Apache-2.0

"""Frozen Torch CPU versus native MLX TableFormer v1 encoder gates."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from docling.datamodel.pipeline_options import TableFormerMode

from tools.tableformer_v1.source import verify_source

pytestmark = [pytest.mark.mlx, pytest.mark.parity]

_SOURCE_ENVIRONMENT_VARIABLE = "DOCLING_MLX_TABLEFORMER_V1_SOURCE"


def _caps(encoder_layers: int) -> dict[str, dict[str, float]]:
    return {
        "image_features": {"mean_abs_error": 1e-5, "max_abs_error": 1e-4},
        "input_filter": {"mean_abs_error": 2e-5, "max_abs_error": 2e-4},
        **{
            f"tag_encoder.layers.{index}": {
                "mean_abs_error": 5e-5,
                "max_abs_error": 5e-4,
            }
            for index in range(encoder_layers)
        },
    }


mx: Any
torch: Any
TableModel04Rs: Any
TableFormerV1Encoder: Any
load_torch_model: Any
encoder_state_dict: Any
load_same_weight_encoder: Any


@pytest.fixture(scope="module", autouse=True)
def _load_requirements() -> None:
    global mx, torch, TableModel04Rs, TableFormerV1Encoder, load_torch_model
    global encoder_state_dict, load_same_weight_encoder

    probe = subprocess.run(
        [sys.executable, "-c", "import mlx.core"],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode:
        pytest.fail(f"selected parity lane requires Metal: {probe.stderr.strip()}")
    try:
        mx = import_module("mlx.core")
        torch = import_module("torch")
        source = import_module("docling_ibm_models.tableformer.models.table04_rs.tablemodel04_rs")
        native = import_module("docling_mlx._models.tableformer_v1.vision")
        helpers = import_module("tools.tableformer_v1.reference_encoder")
        load_torch_model = import_module("safetensors.torch").load_model
    except ImportError as error:
        pytest.fail(f"selected parity lane is missing a required dependency: {error}")
    TableModel04Rs = source.TableModel04_rs
    TableFormerV1Encoder = native.TableFormerV1Encoder
    encoder_state_dict = helpers.encoder_state_dict
    load_same_weight_encoder = helpers.load_same_weight_encoder


@pytest.fixture(scope="module")
def source_root() -> Path:
    value = os.environ.get(_SOURCE_ENVIRONMENT_VARIABLE)
    if value is None:
        pytest.fail(
            "selected parity lane requires the pinned TableFormer v1 source; "
            f"set {_SOURCE_ENVIRONMENT_VARIABLE}"
        )
    root = Path(value).expanduser()
    verify_source(root)
    return root


def _same_weight_models(source_root: Path, mode: TableFormerMode) -> tuple[Any, Any]:
    from docling_mlx._models.tableformer_v1.config import TableFormerV1Config

    profile = source_root / "model_artifacts/tableformer" / mode.value
    raw = json.loads((profile / "tm_config.json").read_text())
    raw["model"]["save_dir"] = str(profile)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    reference = TableModel04Rs(raw, {"word_map": raw["dataset_wordmap"]}, "cpu").cpu().eval()
    checkpoint = profile / f"tableformer_{mode.value}.safetensors"
    missing, unexpected = load_torch_model(reference, checkpoint)
    assert not missing and not unexpected

    native = TableFormerV1Encoder(TableFormerV1Config.from_dict(raw))
    source_state = encoder_state_dict(reference)
    assert len(source_state) == (172 if mode is TableFormerMode.ACCURATE else 148)
    load_same_weight_encoder(source_state, native, mx)
    return reference, native


def _reference_intermediates(reference: Any, pixels: np.ndarray) -> dict[str, np.ndarray]:
    captured: dict[str, np.ndarray] = {}

    def capture(name: str):
        def hook(_module: Any, _arguments: tuple[Any, ...], output: Any) -> None:
            captured[name] = output.detach().cpu().numpy()

        return hook

    handles = [reference._encoder.register_forward_hook(capture("image_features"))]
    handles.append(
        reference._tag_transformer._input_filter.register_forward_hook(capture("input_filter"))
    )
    handles.extend(
        layer.register_forward_hook(capture(f"tag_encoder.layers.{index}"))
        for index, layer in enumerate(reference._tag_transformer._encoder.layers)
    )
    try:
        with torch.inference_mode():
            image_features = reference._encoder(
                torch.from_numpy(pixels.transpose(0, 3, 1, 2).copy())
            )
            filtered = reference._tag_transformer._input_filter(image_features.permute(0, 3, 1, 2))
            sequence = filtered.permute(0, 2, 3, 1).reshape(1, 784, 512).permute(1, 0, 2)
            reference._tag_transformer._encoder(sequence)
    finally:
        for handle in handles:
            handle.remove()
    captured["input_filter"] = captured["input_filter"].transpose(0, 2, 3, 1)
    return captured


@pytest.mark.parametrize("mode", [TableFormerMode.ACCURATE, TableFormerMode.FAST])
def test_encoder_matches_same_weight_torch_cpu(source_root: Path, mode: TableFormerMode) -> None:
    reference, native = _same_weight_models(source_root, mode)
    pixels = np.random.default_rng(947).random((1, 448, 448, 3), dtype=np.float32)

    expected = _reference_intermediates(reference, pixels)
    actual = native.forward_intermediates(mx.array(pixels))
    mx.eval(actual)

    caps_by_name = _caps(native.config.encoder_layers)
    assert set(expected) == set(caps_by_name)
    for name, caps in caps_by_name.items():
        difference = np.abs(np.asarray(actual[name]) - expected[name])
        assert float(difference.mean()) <= caps["mean_abs_error"], (name, difference.mean())
        assert float(difference.max()) <= caps["max_abs_error"], (name, difference.max())
