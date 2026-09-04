# SPDX-License-Identifier: Apache-2.0

"""Egret benchmark adapter for :mod:`tools.benchmark`."""

from __future__ import annotations

import os
import platform
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

import numpy as np
from PIL import Image

from tools._common.benchmark import _image_identity
from tools._common.hashing import hash_named_files
from tools.layout_egret.convert_weights import verify_source

RUNTIME_ARTIFACT_FILES = ("model.safetensors", "config.json", "preprocessor_config.json")
REQUIRES = ("profile",)
DEFAULT_THREADS = 4
Backend = Literal["mlx", "torch"]
TorchDevice = Literal["cpu", "mps"]
_TIMING_BOUNDARIES = {
    "preprocessing": "Prepared PIL RGB images through the production processor and explicit "
    "backend materialization; no model forward or postprocessing.",
    "forward_materialized": "One already materialized model input reused for first, warmup, "
    "and sample forwards; output materialization and host copies included.",
    "engine": "Public Docling predict_batch(), including production preprocessing, model forward, "
    "output materialization, and postprocessing.",
    "excluded": "PDF rendering, page crops, Docling stage construction, and document pipeline "
    "work are excluded.",
}
_NOTES = (
    "Run one backend/profile/batch command in a fresh process.",
    "The first engine call follows one same-shape preprocessing call; it is shape-first dispatch "
    "evidence, not fully cold-process latency.",
    "RSS and backend peak memory are process-wide descriptive values.",
)
_SOURCE_PRESETS = {
    "medium": (
        "docling-project/docling-layout-egret-medium",
        "77ede7cc7bed96d853c58f319734803d6ea2ea5c",
    ),
    "large": (
        "docling-project/docling-layout-egret-large",
        "fff417c78abd6bab338c87706c95a8d79dc68f1e",
    ),
    "xlarge": (
        "docling-project/docling-layout-egret-xlarge",
        "23857d16596e0106716b3162d132212d733769e7",
    ),
}


def validate_request(
    backend: Backend,
    artifact: Path | None,
    source: Path | None,
    device: TorchDevice | None,
    cpu_threads: int,
) -> None:
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    if backend == "mlx":
        if artifact is None or source is not None or device is not None:
            raise ValueError("MLX requires --artifact only; omit --source and --device")
    elif backend == "torch":
        if source is None or artifact is not None or device is None:
            raise ValueError("Torch requires --source and --device; omit --artifact")
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def artifact_provenance(directory: Path, profile: str) -> dict[str, Any]:
    return {
        "sha256": hash_named_files(directory, RUNTIME_ARTIFACT_FILES),
        "profile": f"egret-{profile}",
    }


def source_provenance(source: Path, repo_id: str, revision: str) -> dict[str, Any]:
    verify_source(source)
    return {
        "repo_id": repo_id,
        "revision": revision,
        "source_files": None,
    }


def _sysctl(name: str) -> str | None:
    if platform.system() != "Darwin":
        return None
    completed = subprocess.run(["sysctl", "-n", name], check=False, capture_output=True, text=True)
    return completed.stdout.strip() if completed.returncode == 0 else None


def _host_state() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "macos_version": platform.mac_ver()[0] or None,
        "machine": platform.machine(),
        "model": _sysctl("hw.model"),
        "physical_memory_bytes": _sysctl("hw.memsize"),
        "logical_cpus": os.cpu_count(),
    }


def _torch_mps_memory(torch: Any, device: TorchDevice) -> dict[str, int | None]:
    if device != "mps":
        return {"torch_mps_allocated_at_end_bytes": None, "torch_mps_driver_at_end_bytes": None}
    current = getattr(torch.mps, "current_allocated_memory", None)
    driver = getattr(torch.mps, "driver_allocated_memory", None)
    return {
        "torch_mps_allocated_at_end_bytes": current() if current is not None else None,
        "torch_mps_driver_at_end_bytes": driver() if driver is not None else None,
    }


def _load_mlx(options: Any, profile: str, images: list[Image.Image]) -> dict[str, Any]:
    import mlx.core as mx
    from docling.models.inference_engines.object_detection.base import ObjectDetectionEngineInput

    from docling_mlx.engines.object_detection.dfine.engine import DFineEngine, DFineModelSpec
    from docling_mlx.engines.object_detection.rt_detr_v2.preprocessing import preprocess_images

    artifact = Path(options.artifact).expanduser().resolve()
    batch_sizes = tuple(options.batch_sizes or ((options.batch_size or 1),))
    if any(size not in (1, 4) for size in batch_sizes):
        raise ValueError("Egret batch sizes must be 1 or 4")
    del images
    started = perf_counter()
    engine = DFineEngine(DFineModelSpec(path=artifact))
    engine.initialize()
    if engine._model is None or engine._preprocessing is None or engine._dtype is None:
        raise RuntimeError("Engine initialization did not produce an Egret model")
    model = engine._model
    preprocessing = engine._preprocessing

    def preprocess(batch_images: list[Image.Image]) -> Any:
        pixels = mx.array(preprocess_images(batch_images, preprocessing), dtype=engine._dtype)
        mx.eval(pixels)
        return pixels

    def forward(pixels: Any) -> tuple[np.ndarray, np.ndarray]:
        output = model(pixels)
        logits, boxes = output["pred_logits"], output["pred_boxes"]
        mx.eval(logits, boxes)
        return np.array(logits, copy=True), np.array(boxes, copy=True)

    return {
        "schema_version": 1,
        "backend": "mlx-metal",
        "profile": f"egret-{profile}",
        "timing_boundary_version": 1,
        "timing_boundaries": _TIMING_BOUNDARIES,
        "artifact": artifact_provenance(artifact, profile),
        "operations": {
            "preprocessing": {"first_key": "first_preprocessing_call_ms"},
            "engine": {
                "first_key": "first_engine_call_ms",
                "prepare": lambda batch: [
                    ObjectDetectionEngineInput(image=image) for image in batch
                ],
            },
            "forward_materialized": {"prepare": preprocess},
        },
        "predictors": {
            "preprocessing": preprocess,
            "forward_materialized": forward,
            "engine": lambda batch: engine.predict([item.image for item in batch]),
        },
        "batch_sizes": batch_sizes,
        "version_names": ("mlx", "docling-slim", "numpy", "Pillow"),
        "reset_memory": mx.reset_peak_memory,
        "initialization_ms": (perf_counter() - started) * 1000,
        "notes": [*(_NOTES[:2]), "RSS and MLX peak memory are process-wide descriptive values."],
    }


def _synchronize_torch(torch: Any, device: TorchDevice) -> None:
    if device == "mps":
        torch.mps.synchronize()


def _load_torch(options: Any, profile: str, images: list[Image.Image]) -> dict[str, Any]:
    import torch
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.object_detection_engine_options import (
        TransformersObjectDetectionEngineOptions,
    )
    from docling.datamodel.stage_model_specs import EngineModelConfig
    from docling.models.inference_engines.object_detection.base import ObjectDetectionEngineInput
    from docling.models.inference_engines.object_detection.transformers_engine import (
        TransformersObjectDetectionEngine,
    )

    source = Path(options.source).expanduser().resolve()
    repo_id, revision = _SOURCE_PRESETS[profile]
    device: TorchDevice = options.device
    cpu_threads = options.cpu_threads
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is not available")
    torch.set_num_threads(cpu_threads)
    batch_sizes = tuple(options.batch_sizes or ((options.batch_size or 1),))
    if any(size not in (1, 4) for size in batch_sizes):
        raise ValueError("Egret batch sizes must be 1 or 4")
    del images
    temporary = tempfile.TemporaryDirectory(prefix="docling-mlx-egret-benchmark-")
    artifacts_root = Path(temporary.name)
    (artifacts_root / repo_id.replace("/", "--")).symlink_to(source, target_is_directory=True)
    started = perf_counter()
    engine = TransformersObjectDetectionEngine(
        options=TransformersObjectDetectionEngineOptions(
            torch_dtype="float32", compile_model=False
        ),
        model_config=EngineModelConfig(repo_id=repo_id, revision=revision),
        accelerator_options=AcceleratorOptions(device=device, num_threads=cpu_threads),
        artifacts_path=artifacts_root,
    )
    engine.initialize()
    _synchronize_torch(torch, device)
    if engine._model is None or engine._processor is None or engine._device is None:
        raise RuntimeError(
            "Torch engine initialization did not produce model, processor, and device"
        )
    model = cast(Callable[..., Any], engine._model)
    processor = engine._processor
    torch_device = engine._device

    def preprocess(batch_images: list[Image.Image]) -> Any:
        pixels = processor(images=batch_images, return_tensors="pt")["pixel_values"]
        pixels = pixels.to(torch_device, dtype=torch.float32)
        _synchronize_torch(torch, device)
        return pixels

    def forward(pixels: Any) -> tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            output = model(pixel_values=pixels)
        _synchronize_torch(torch, device)
        return (
            output.logits.detach().cpu().numpy().copy(),
            output.pred_boxes.detach().cpu().numpy().copy(),
        )

    return {
        "schema_version": 1,
        "backend": f"torch-{device}",
        "profile": f"egret-{profile}",
        "timing_boundary_version": 1,
        "timing_boundaries": _TIMING_BOUNDARIES,
        "source": source_provenance(source, repo_id, revision),
        "operations": {
            "preprocessing": {"first_key": "first_preprocessing_call_ms"},
            "engine": {
                "first_key": "first_engine_call_ms",
                "prepare": lambda batch: [
                    ObjectDetectionEngineInput(image=image) for image in batch
                ],
            },
            "forward_materialized": {"prepare": preprocess},
        },
        "predictors": {
            "preprocessing": preprocess,
            "forward_materialized": forward,
            "engine": engine.predict_batch,
        },
        "batch_sizes": batch_sizes,
        "version_names": (
            "torch",
            "torchvision",
            "transformers",
            "docling-slim",
            "numpy",
            "Pillow",
        ),
        "close": temporary.cleanup,
        "torch_cpu_threads": cpu_threads,
        "initialization_ms": (perf_counter() - started) * 1000,
        "notes": [*(_NOTES[:2]), "RSS and Torch MPS memory are process-wide descriptive values."],
    }


def load(options: Any, images: list[Image.Image]) -> dict[str, Any]:
    if not images:
        raise ValueError("at least one image is required")
    if options.profile not in _SOURCE_PRESETS:
        raise ValueError(f"unknown Egret profile: {options.profile}")
    profile = options.profile
    backend: Backend = options.backend
    device = options.device
    cpu_threads = options.cpu_threads
    validate_request(backend, options.artifact, options.source, device, cpu_threads)
    if options.batch_size is not None and options.batch_size not in (1, 4):
        raise ValueError("batch_size must be 1 or 4")
    if backend == "mlx":
        return _load_mlx(options, profile, images)
    return _load_torch(options, profile, images)


def normalize_result(
    state: dict[str, Any],
    rows: list[dict[str, Any]],
    image_paths: list[Path],
    images: list[Image.Image],
    options: Any,
) -> dict[str, Any]:
    del options
    report: dict[str, Any] = {
        key: state[key]
        for key in (
            "schema_version",
            "backend",
            "profile",
            "timing_boundary_version",
            "timing_boundaries",
            "artifact",
            "source",
            "torch_cpu_threads",
            "initialization_ms",
            "notes",
        )
        if key in state
    }
    report["results"] = rows
    report["host"] = _host_state()
    report["images"] = _image_identity(image_paths, images)
    if state["backend"] == "mlx-metal":
        import mlx.core as mx

        report["mlx_peak_memory_bytes"] = mx.get_peak_memory()
    else:
        import torch

        report["torch_mps_available"] = torch.backends.mps.is_available()
        report.update(_torch_mps_memory(torch, state["backend"].removeprefix("torch-")))
    return report


def predict(state: dict[str, Any], target: str, value: Any) -> Any:
    return state["predictors"][target](value)
