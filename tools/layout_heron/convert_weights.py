# SPDX-License-Identifier: Apache-2.0

"""Convert an RT-DETR-v2 checkpoint into a local MLX artifact.

Run with the root reference dependency group:

``uv run --no-sync --group reference python -m tools.layout_heron.convert_weights
--source /path/to/checkpoint --repo-id PekingU/rtdetr_v2_r18vd --revision main
--output .artifacts/rtdetr-v2-r18``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np

from docling_mlx._models.rt_detr_v2.weights import rename_key, sanitize_state_dict
from docling_mlx.engines.object_detection.rt_detr_v2.preprocessing import (
    parse_preprocessing_config,
)
from tools._common.model_card import lookup_model_license, render_model_card

__all__ = ["convert", "convert_state_dict", "rename_key"]


def convert_state_dict(
    source: dict[str, np.ndarray], target_shapes: dict[str, tuple[int, ...]]
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[str]]:
    """Compatibility export for the shared runtime sanitizer."""

    return sanitize_state_dict(source, target_shapes)


def _verify_model(model: Any, config: Any, image_size: tuple[int, int]) -> dict[str, list[int]]:
    """Complete one deterministic finite, nonzero forward."""

    import mlx.core as mx

    model.eval()
    pixels = np.random.default_rng(8429).random(
        (1, image_size[0], image_size[1], config.backbone_config.num_channels), dtype=np.float32
    )
    output = model(mx.array(pixels))
    logits = output["pred_logits"]
    boxes = output["pred_boxes"]
    mx.eval(logits, boxes)
    logits_array = np.array(logits, copy=True)
    boxes_array = np.array(boxes, copy=True)
    expected_logits = (1, config.num_queries, config.num_labels)
    expected_boxes = (1, config.num_queries, 4)
    if logits_array.shape != expected_logits or boxes_array.shape != expected_boxes:
        raise ValueError("Converted model forward returned invalid output shapes")
    if not np.isfinite(logits_array).all() or not np.isfinite(boxes_array).all():
        raise ValueError("Converted model forward returned non-finite output")
    if not np.any(logits_array) or not np.any(boxes_array):
        raise ValueError("Converted model forward returned an all-zero output")
    return {
        "pred_logits_shape": list(logits_array.shape),
        "pred_boxes_shape": list(boxes_array.shape),
    }


def _verify_source(source: Path) -> None:
    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        if not (source / name).is_file():
            raise ValueError(f"Missing source file: {source / name}")


def convert(source: Path, output: Path, repo_id: str, revision: str) -> dict[str, Any]:
    """Write a fully strict-loaded FP32 artifact without overwriting an output."""

    _verify_source(source)
    license = lookup_model_license(source, repo_id, revision)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from safetensors.numpy import load_file, save_file

        from docling_mlx._models.rt_detr_v2 import RtDetrV2, RtDetrV2Config

        config_json = json.loads((source / "config.json").read_text())
        config = RtDetrV2Config.from_dict(config_json)
        preprocessing = parse_preprocessing_config(
            json.loads((source / "preprocessor_config.json").read_text())
        )
        model = RtDetrV2(config)
        flattened = cast(list[tuple[str, Any]], tree_flatten(model.parameters()))
        target_shapes = {key: tuple(value.shape) for key, value in flattened}
        converted, mappings, ignored = convert_state_dict(
            load_file(str(source / "model.safetensors")), target_shapes
        )
        model.load_weights(
            [(key, mx.array(value)) for key, value in converted.items()], strict=True
        )
        mx.eval(model.parameters())

        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.convert-", dir=output.parent))
        save_file(converted, str(temporary / "model.safetensors"), metadata={"format": "mlx"})
        for name in ("config.json", "preprocessor_config.json"):
            shutil.copyfile(source / name, temporary / name)
        (temporary / "README.md").write_text(
            render_model_card(repo_id, revision, license), encoding="utf-8"
        )
        _verify_model(model, config, preprocessing.size)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Output appeared during conversion: {output}")
        os.rename(temporary, output)
        temporary = None
        return {"converted_tensors": len(mappings), "ignored_counters": len(ignored)}
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    report = convert(args.source, args.output, args.repo_id, args.revision)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "repo_id": args.repo_id,
                "revision": args.revision,
                **report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
