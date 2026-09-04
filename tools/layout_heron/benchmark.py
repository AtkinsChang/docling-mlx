# SPDX-License-Identifier: Apache-2.0

"""Heron benchmark adapter for :mod:`tools.benchmark`."""

from __future__ import annotations

import platform
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
from PIL import Image

from tools._common.benchmark import _image_identity, measure
from tools._common.hashing import hash_named_files

DEFAULT_BATCH_SIZES = (1, 4)
REQUIRES = ()
DEFAULT_THREADS = None
Target = Literal["preprocessing", "forward_materialized", "engine"]
__all__ = ["artifact_provenance", "load", "measure", "normalize_result", "predict"]
_RUNTIME_ARTIFACT_FILES = ("model.safetensors", "config.json", "preprocessor_config.json")


def artifact_provenance(directory: Path) -> dict[str, Any]:
    return {"kind": "local", "sha256": hash_named_files(directory, _RUNTIME_ARTIFACT_FILES)}


def _peak_memory_bytes(mx: Any) -> int | None:
    getter = getattr(mx, "get_peak_memory", None)
    if getter is None:
        return None
    value = getter()
    if not isinstance(value, int):
        raise TypeError("mlx.get_peak_memory() did not return an integer")
    return value


def load(options: Any, images: list[Image.Image]) -> dict[str, Any]:
    if getattr(options, "backend", "mlx") != "mlx":
        raise ValueError("layout-heron supports only the mlx backend")
    if options.artifact is None:
        raise ValueError("layout-heron requires --artifact")
    artifact = Path(options.artifact).expanduser().resolve()
    batch_sizes = tuple(DEFAULT_BATCH_SIZES if options.batch_sizes is None else options.batch_sizes)
    if not batch_sizes or any(size <= 0 for size in batch_sizes):
        raise ValueError("batch_sizes must contain positive integers")
    if not images:
        raise ValueError("at least one image is required")
    import mlx.core as mx

    from docling_mlx.engines.object_detection.rt_detr_v2.engine import (
        RtDetrV2Engine,
        RtDetrV2ModelSpec,
    )
    from docling_mlx.engines.object_detection.rt_detr_v2.preprocessing import preprocess_images

    started = perf_counter()
    engine = RtDetrV2Engine(RtDetrV2ModelSpec(path=artifact))
    engine.initialize()
    if engine._model is None or engine._preprocessing is None or engine._dtype is None:
        raise RuntimeError("Engine initialization did not produce a model and preprocessing spec")
    model = engine._model
    preprocessing = engine._preprocessing

    def preprocess(batch: list[Image.Image]) -> Any:
        pixels = mx.array(preprocess_images(batch, preprocessing), dtype=engine._dtype)
        mx.eval(pixels)
        return pixels

    def forward(pixels: Any) -> tuple[np.ndarray, np.ndarray]:
        output = model(pixels)
        logits = output["pred_logits"]
        boxes = output["pred_boxes"]
        mx.eval(logits, boxes)
        return np.array(logits, copy=True), np.array(boxes, copy=True)

    return {
        "backend": "mlx-metal",
        "profile": artifact.name,
        "timing_boundary_version": 1,
        "timing_boundaries": {
            "initialization_ms": "Engine construction, artifact validation, model construction, "
            "strict weight loading, model.eval(), and initial parameter evaluation.",
            "preprocessing": "PIL RGB conversion, configured Pillow resize/rescale/normalization, "
            "and mx.eval().",
            "forward_materialized": "Fixed pre-evaluated NHWC FP32 MLX pixels; model call, "
            "mx.eval(), and NumPy host copies of logits and boxes.",
            "engine": "Public predict_batch(): Metal preprocessing, model forward, mx.eval(), "
            "host materialization, finite/shape checks, and CPU postprocessing.",
            "first_preprocessing_call_ms": "First materialized preprocessing call for the "
            "batch shape after initialization.",
            "first_engine_call_ms": "First public engine call for the batch shape after one "
            "preprocessing call; includes first-use dispatch/compilation but is not fully cold.",
            "stage": "Not timed. The Docling layout stage consumes Page/backend objects, while "
            "this tool intentionally accepts prepared PIL images. PDF rendering, page crop, "
            "cluster construction, and whole-pipeline work are excluded.",
        },
        "artifact": artifact_provenance(artifact),
        "operations": {
            "preprocessing": {"first_key": "first_preprocessing_call_ms"},
            "engine": {
                "first_key": "first_engine_call_ms",
                "prepare": lambda batch: batch,
            },
            "forward_materialized": {"prepare": preprocess},
        },
        "predictors": {
            "preprocessing": preprocess,
            "forward_materialized": forward,
            "engine": engine.predict,
        },
        "batch_sizes": batch_sizes,
        "version_names": ("mlx", "docling-slim", "numpy", "Pillow"),
        "reset_memory": mx.reset_peak_memory,
        "initialization_ms": (perf_counter() - started) * 1000,
        "notes": [
            "Run each R50/R101 profile in a separate fresh process for comparable "
            "first-call and RSS results.",
            "Do not run concurrent Metal workloads during measurement.",
            "MLX peak memory and RSS are process-wide values, not per-batch allocations.",
            "This is descriptive benchmark evidence and makes no upstream or Torch/MPS "
            "speed claim.",
        ],
    }


def normalize_result(
    state: dict[str, Any],
    rows: list[dict[str, Any]],
    image_paths: list[Path],
    images: list[Image.Image],
    options: Any,
) -> dict[str, Any]:
    del options
    import mlx.core as mx

    return {
        key: state[key]
        for key in (
            "backend",
            "profile",
            "timing_boundary_version",
            "timing_boundaries",
            "artifact",
            "initialization_ms",
            "notes",
        )
    } | {
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "images": _image_identity(image_paths, images),
        "mlx_peak_memory_bytes": _peak_memory_bytes(mx),
        "results": rows,
    }


def predict(state: dict[str, Any], target: str, value: Any) -> Any:
    return state["predictors"][target](value)
