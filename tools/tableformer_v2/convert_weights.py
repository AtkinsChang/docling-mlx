# SPDX-License-Identifier: Apache-2.0

"""Strict, deterministic FP32 conversion of the pinned TableFormerV2 checkpoint.

Run ``uv run --no-sync python -m tools.tableformer_v2.convert_weights --output DIR``.
The output is a local converted-model repository. Conversion never uploads or
publishes the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np

from docling_mlx.engines.table_structure.tableformer_v2 import conversion as _conversion
from tools._common.atomic import _atomic_output
from tools._common.model_card import lookup_model_license, render_model_card
from tools.tableformer_v2.source import SOURCE_REPO, SOURCE_REVISION, download_source, verify_source

_UNUSED_CLASSIFIER_KEYS = frozenset(
    {
        "feature_extractor.classifier.1.bias",
        "feature_extractor.classifier.1.weight",
    }
)
_CLASSIFIER_REASON = "unused_torchvision_classifier"
convert_state_dict = _conversion.convert_upstream_state_dict
rename_key = _conversion.rename_upstream_key

_PREPROCESSOR_CONFIG: dict[str, Any] = {
    "do_convert_rgb": True,
    "do_resize": True,
    "size": {"height": 448, "width": 448},
    "interpolation": "bilinear",
    "antialias": True,
    "do_rescale": True,
    "rescale_factor": 1 / 255,
    "do_normalize": True,
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225],
    "output_dtype": "float32",
    "output_layout": "NHWC",
}
_GENERATION_CONFIG = {"max_generation_steps": 512}


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    """Encode project-owned JSON deterministically, including one final newline."""

    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def _target_model(config_raw: Mapping[str, Any]) -> Any:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from docling_mlx._models.tableformer_v2.config import TableFormerV2Config
    from docling_mlx._models.tableformer_v2.model import TableFormerV2

    config = TableFormerV2Config.from_dict(config_raw)
    model = TableFormerV2(config)
    model.eval()
    flattened = cast(list[tuple[str, Any]], tree_flatten(model.parameters()))
    if not flattened:
        raise ValueError("Native TableFormerV2 exposes no parameters")
    if any(value.dtype != mx.float32 for _, value in flattened):
        raise ValueError("Native TableFormerV2 target parameters must all be float32")
    return model, {key: tuple(value.shape) for key, value in flattened}


def _verify_model(model: Any) -> dict[str, Any]:
    """Run fixed finite, nonzero forward and generation smokes."""

    import mlx.core as mx

    pixels_array = np.random.default_rng(8429).random((1, 64, 64, 3), dtype=np.float32)
    pixels = mx.array(pixels_array)
    token_ids = mx.array([[2]], dtype=mx.int32)
    forward = model(token_ids, pixels=pixels)
    generated = model.generate(pixels, max_generation_steps=1)
    tensors = {
        "forward.logits": forward.logits,
        "forward.hidden_states": forward.hidden_states,
        "forward.predicted_bboxes": forward.predicted_bboxes,
        "generation.generated_ids": generated.generated_ids,
        "generation.logits": generated.logits,
        "generation.hidden_states": generated.hidden_states,
        "generation.predicted_bboxes": generated.predicted_bboxes,
    }
    mx.eval(*tensors.values())
    shapes: dict[str, list[int]] = {}
    nonzero = False
    for name, value in tensors.items():
        array = np.array(value, copy=True)
        if not np.isfinite(array).all():
            raise ValueError(f"Converted model smoke returned non-finite output: {name}")
        if name not in {"forward.predicted_bboxes", "generation.predicted_bboxes"}:
            nonzero = nonzero or bool(np.any(array))
        shapes[name] = list(array.shape)
    generated_ids = np.array(generated.generated_ids, copy=True)
    if generated_ids.shape != (1, 2) or generated_ids[0, 0] != 2:
        raise ValueError("Converted model generation smoke returned invalid token IDs")
    if not nonzero:
        raise ValueError("Converted model smoke returned only all-zero outputs")
    return {
        "deterministic_input_seed": 8429,
        "forward_nonzero": True,
        "generation_nonzero": True,
        "output_shapes": shapes,
    }


def _write_artifact_files(directory: Path, source: Path) -> None:
    shutil.copyfile(source / "config.json", directory / "config.json")
    for name in ("special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(source / name, directory / name)
    (directory / "preprocessor_config.json").write_bytes(_json_bytes(_PREPROCESSOR_CONFIG))
    (directory / "generation_config.json").write_bytes(_json_bytes(_GENERATION_CONFIG))


def convert(source: Path, output: Path) -> dict[str, Any]:
    """Convert and atomically publish the frozen TableFormerV2 checkpoint."""

    verify_source(source)
    license = lookup_model_license(source, SOURCE_REPO, SOURCE_REVISION)
    config_bytes = (source / "config.json").read_bytes()
    config_raw = json.loads(config_bytes)
    if not isinstance(config_raw, dict):
        raise ValueError("Expected config.json to contain a JSON object")

    from safetensors.numpy import load_file, save_file

    model, target_shapes = _target_model(config_raw)
    source_state = load_file(str(source / "model.safetensors"))
    converted, _, _ = convert_state_dict(source_state, target_shapes)

    import mlx.core as mx

    model.load_weights([(key, mx.array(value)) for key, value in converted.items()], strict=True)
    mx.eval(model.parameters())

    with _atomic_output(output) as temporary:
        save_file(converted, str(temporary / "model.safetensors"), metadata={"format": "mlx"})
        _write_artifact_files(temporary, source)
        (temporary / "README.md").write_text(
            render_model_card(SOURCE_REPO, SOURCE_REVISION, license),
            encoding="utf-8",
        )
        verification = _verify_model(model)
        if (temporary / "config.json").read_bytes() != config_bytes:
            raise ValueError("Converted config.json did not preserve source bytes")
    return {"verification": verification}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    source = args.source or download_source(cache_dir=args.cache_dir)
    report = convert(source, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_revision": SOURCE_REVISION,
                "verification": report["verification"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
