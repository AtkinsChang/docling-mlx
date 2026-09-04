# SPDX-License-Identifier: Apache-2.0

"""Pinned TableFormerV2 Torch CPU versus native MLX vision-encoder gates."""

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

from tools.tableformer_v2.source import verify_source

pytestmark = [pytest.mark.mlx, pytest.mark.parity]

_SOURCE_ENVIRONMENT_VARIABLE = "DOCLING_MLX_TABLEFORMER_V2_SOURCE"
_BOUNDARY_CAPS = {
    "stem": {"mean_abs_error": 1e-5, "max_abs_error": 1e-4},
    "backbone.stages.0": {"mean_abs_error": 1e-5, "max_abs_error": 1e-4},
    "backbone.stages.1": {"mean_abs_error": 1e-5, "max_abs_error": 1e-4},
    "backbone.stages.2": {"mean_abs_error": 1e-5, "max_abs_error": 1e-4},
    "backbone.stages.3": {"mean_abs_error": 1e-5, "max_abs_error": 1e-4},
    "backbone.stages.4": {"mean_abs_error": 1e-5, "max_abs_error": 1e-4},
    "backbone.stages.5": {"mean_abs_error": 1e-5, "max_abs_error": 1e-4},
    "backbone": {"mean_abs_error": 1e-5, "max_abs_error": 1e-4},
    "post_backbone_se": {"mean_abs_error": 1e-5, "max_abs_error": 1e-4},
    "spatial_mixer": {"mean_abs_error": 1e-5, "max_abs_error": 1e-4},
    "encoded": {"mean_abs_error": 1e-5, "max_abs_error": 1e-4},
}

mx: Any
torch: Any
TorchTableFormerV2: Any
TableFormerV2VisionEncoder: Any
encoder_state_dict: Any
load_same_weight_encoder: Any


@pytest.fixture(scope="module", autouse=True)
def _load_requirements() -> None:
    global mx, torch, TorchTableFormerV2, TableFormerV2VisionEncoder
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
        source = import_module("docling_ibm_models.tableformer_v2.model")
        native = import_module("docling_mlx._models.tableformer_v2.vision")
        helpers = import_module("tools.tableformer_v2.reference_encoder")
    except ImportError as error:
        pytest.fail(f"selected parity lane is missing a required dependency: {error}")
    TorchTableFormerV2 = source.TableFormerV2
    TableFormerV2VisionEncoder = native.TableFormerV2VisionEncoder
    encoder_state_dict = helpers.encoder_state_dict
    load_same_weight_encoder = helpers.load_same_weight_encoder


@pytest.fixture(scope="module")
def pinned_source() -> Path:
    source = os.environ.get(_SOURCE_ENVIRONMENT_VARIABLE)
    if source is None:
        try:
            source = import_module("huggingface_hub").snapshot_download(
                repo_id="docling-project/TableFormerV2",
                revision="51559fad3946873e26a6f9b8e912f948e8745bef",
                local_files_only=True,
                allow_patterns=[
                    "config.json",
                    "generation_config.json",
                    "model.safetensors",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                ],
            )
        except Exception as error:
            pytest.fail(
                "selected parity lane requires a pinned local TableFormerV2 source snapshot; "
                f"set {_SOURCE_ENVIRONMENT_VARIABLE}: {error}"
            )
    path = Path(source).expanduser()
    verify_source(path)
    return path


def _same_weight_models(pinned_source: Path) -> tuple[Any, Any]:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    reference = TorchTableFormerV2.from_pretrained(pinned_source, dtype=torch.float32).cpu().eval()
    native = TableFormerV2VisionEncoder(json.loads((pinned_source / "config.json").read_text()))
    source_state = encoder_state_dict(reference)
    assert len(source_state) == 695
    load_same_weight_encoder(source_state, native, mx)
    return reference, native


def _capture_boundaries(reference: Any, pixels: np.ndarray) -> dict[str, np.ndarray]:
    captured: dict[str, np.ndarray] = {}

    def capture(name: str):
        def hook(_module: Any, _arguments: tuple[Any, ...], output: Any) -> None:
            captured[name] = output.detach().cpu().numpy()

        return hook

    handles = [reference.feature_extractor.features[0].register_forward_hook(capture("stem"))]
    handles += [
        reference.feature_extractor.features[index].register_forward_hook(
            capture(f"backbone.stages.{index - 1}")
        )
        for index in range(1, 7)
    ]
    handles += [
        reference.feature_extractor.features.register_forward_hook(capture("backbone")),
        reference.se_module.register_forward_hook(capture("post_backbone_se")),
        reference.conv_mixer.register_forward_hook(capture("spatial_mixer")),
        reference.feature_to_embedding.register_forward_hook(capture("encoded")),
    ]
    try:
        with torch.inference_mode():
            reference.encode_images(torch.from_numpy(pixels.transpose(0, 3, 1, 2).copy()))
    finally:
        for handle in handles:
            handle.remove()
    return captured


def test_real_pinned_encoder_matches_torch_cpu_at_named_boundaries(pinned_source: Path) -> None:
    reference, native = _same_weight_models(pinned_source)
    pixels = np.random.default_rng(912).random((1, 64, 64, 3), dtype=np.float32)

    expected = _capture_boundaries(reference, pixels)
    actual = native.forward_intermediates(mx.array(pixels))
    mx.eval(actual)

    assert set(expected) == set(_BOUNDARY_CAPS) == set(actual)
    for name, caps in _BOUNDARY_CAPS.items():
        expected_value = expected[name]
        if expected_value.ndim == 4:
            expected_value = expected_value.transpose(0, 2, 3, 1)
        difference = np.abs(np.asarray(actual[name]) - expected_value)
        assert float(difference.mean()) <= caps["mean_abs_error"], (name, difference.mean())
        assert float(difference.max()) <= caps["max_abs_error"], (name, difference.max())
