# SPDX-License-Identifier: Apache-2.0

"""CPU Transformers versus native MLX parity for official EfficientNet models."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from PIL import Image

from docling_mlx.engines.image_classification.efficientnet import (
    EfficientNetEngine,
    EfficientNetModelSpec,
)

ROOT = Path(__file__).resolve().parents[2]
HF_CACHE = Path.home() / ".cache/huggingface/hub"
IMAGES = ROOT / "tests/fixtures/document_figure/reference_images"
OFFICIAL_ARTIFACTS = ROOT / ".artifacts/official"
_ARTIFACT_FILES = ("model.safetensors", "config.json", "preprocessor_config.json")


def _snapshot(model: str) -> Path | None:
    root = HF_CACHE / f"models--google--efficientnet-{model}" / "snapshots"
    required = {"config.json", "preprocessor_config.json", "pytorch_model.bin"}
    candidates = [path for path in root.glob("*") if required <= {p.name for p in path.iterdir()}]
    return sorted(candidates)[-1] if candidates else None


def _converted_snapshot(snapshot: Path, model: str) -> Path:
    """Cache an MLX-layout copy of the official PyTorch checkpoint for the native loader."""
    output = OFFICIAL_ARTIFACTS / f"google--efficientnet-{model}" / snapshot.name
    if all((output / name).is_file() for name in _ARTIFACT_FILES):
        return output

    import torch
    from mlx.utils import tree_flatten
    from safetensors.numpy import save_file

    from docling_mlx._models.efficientnet.model import EfficientNet
    from tools.document_figure.convert_weights import convert_state_dict

    source_state = torch.load(snapshot / "pytorch_model.bin", map_location="cpu", weights_only=True)
    source = {key: value.detach().cpu().numpy() for key, value in source_state.items()}
    config = json.loads((snapshot / "config.json").read_text())
    native = EfficientNet(config)
    flattened = cast(list[tuple[str, Any]], tree_flatten(native.parameters()))
    target_shapes = {key: tuple(value.shape) for key, value in flattened}
    converted, _, _ = convert_state_dict(source, target_shapes)
    output.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(output / "model.safetensors"), metadata={"format": "mlx"})
    for name in ("config.json", "preprocessor_config.json"):
        shutil.copyfile(snapshot / name, output / name)
    return output


@pytest.mark.mlx
@pytest.mark.parity
@pytest.mark.parametrize("model", ["b0", "b3", "b7"])
def test_official_checkpoint_matches_transformers(model: str) -> None:
    snapshot = _snapshot(model)
    if snapshot is None:
        pytest.fail(
            f"official google/efficientnet-{model} snapshot is absent; "
            f"download with: hf download google/efficientnet-{model}"
        )
    converted = _converted_snapshot(snapshot, model)

    import torch
    from transformers import EfficientNetForImageClassification, EfficientNetImageProcessor

    images = [
        Image.open(IMAGES / name).convert("RGB")
        for name in ("bar_chart.png", "geographical_map.png")
    ]
    try:
        processor = EfficientNetImageProcessor.from_pretrained(snapshot, local_files_only=True)
        reference = (
            EfficientNetForImageClassification.from_pretrained(
                snapshot, local_files_only=True, dtype=torch.float32
            )
            .cpu()
            .eval()
        )
        with torch.inference_mode():
            reference_logits = reference(**processor(images=images, return_tensors="pt")).logits
        reference_logits = reference_logits.numpy()

        actual_logits = EfficientNetEngine(EfficientNetModelSpec(path=converted)).predict_logits(
            images
        )
        reference_probabilities = np.exp(
            reference_logits - reference_logits.max(axis=1, keepdims=True)
        )
        reference_probabilities /= reference_probabilities.sum(axis=1, keepdims=True)
        actual_probabilities = np.exp(actual_logits - actual_logits.max(axis=1, keepdims=True))
        actual_probabilities /= actual_probabilities.sum(axis=1, keepdims=True)
        max_probability_delta = np.max(np.abs(reference_probabilities - actual_probabilities))
        assert max_probability_delta <= 1e-2, max_probability_delta
        for expected, actual in zip(reference_probabilities, actual_probabilities, strict=True):
            expected_ids = np.argsort(-expected, kind="stable")
            actual_ids = np.argsort(-actual, kind="stable")
            assert actual_ids[0] == expected_ids[0]
            assert set(actual_ids[:5]) == set(expected_ids[:5])
    finally:
        for image in images:
            image.close()
