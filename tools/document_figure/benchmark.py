# SPDX-License-Identifier: Apache-2.0

"""DocumentFigure benchmark adapter for :mod:`tools.benchmark`."""

from __future__ import annotations

import json
import platform
import tempfile
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
from PIL import Image

from tools._common.hashing import hash_named_files

Target = Literal["preprocessing", "forward", "engine", "stage"]
TARGETS: tuple[Target, ...] = ("preprocessing", "forward", "engine", "stage")
REQUIRES = ("target",)
DEFAULT_THREADS = None
_RUNTIME_ARTIFACT_FILES = ("model.safetensors", "config.json", "preprocessor_config.json")
_TIMING_BOUNDARIES = {
    "preprocessing": "Production Metal preprocessor call and mx.eval() materialization.",
    "forward": "Fixed pre-evaluated MLX tensor, model call, mx.eval(), and host copy.",
    "engine": "Public engine call, including production preprocessing and host result creation.",
    "stage": (
        "Stage call with fresh document items and prepared images; "
        "PDF rendering and crops excluded."
    ),
}


def artifact_provenance(directory: Path) -> dict[str, Any]:
    return {"kind": "local", "sha256": hash_named_files(directory, _RUNTIME_ARTIFACT_FILES)}


def _make_mlx_state(options: Any) -> dict[str, Any]:
    import mlx.core as mx
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import ItemAndImageEnrichmentElement
    from docling.datamodel.stage_model_specs import ImageClassificationModelSpec
    from docling_core.types.doc import DoclingDocument

    from docling_mlx.engines.image_classification.efficientnet import (
        EfficientNetEngine,
        EfficientNetEngineOptions,
        EfficientNetModelSpec,
    )
    from docling_mlx.engines.image_classification.efficientnet.preprocessing import (
        parse_preprocessing_config,
        preprocess_images,
    )

    artifact = Path(options.artifact).expanduser().resolve()
    target: Target = options.target
    started = perf_counter()
    operations: dict[str, dict[str, Any]] = {}
    predictors: dict[str, Callable[[Any], Any]] = {}
    if target == "preprocessing":
        config = json.loads((artifact / "preprocessor_config.json").read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Expected preprocessor_config.json to contain a JSON object")
        preprocessing = parse_preprocessing_config(config)

        def run_preprocessing(batch: list[Image.Image]) -> Any:
            pixels = mx.array(preprocess_images(batch, preprocessing), dtype=mx.float32)
            mx.eval(pixels)
            return pixels

        predictors[target] = run_preprocessing
        operations[target] = {"first_key": "first_call_ms"}
    elif target == "forward":
        engine = EfficientNetEngine(
            EfficientNetModelSpec(path=artifact), EfficientNetEngineOptions()
        )
        engine.initialize()
        if engine._model is None or engine._preprocessing is None:
            raise RuntimeError("Engine initialization did not produce a model and preprocessor")
        model, preprocessing = engine._model, engine._preprocessing

        def prepare(batch: list[Image.Image]) -> Any:
            pixels = mx.array(preprocess_images(batch, preprocessing), dtype=engine._dtype)
            mx.eval(pixels)
            return pixels

        def forward(pixels: Any) -> Any:
            logits = model(pixels)
            mx.eval(logits)
            return np.array(logits, copy=True)

        predictors[target] = forward
        operations[target] = {"prepare": prepare, "first_key": "first_call_ms"}
    elif target == "engine":
        engine = EfficientNetEngine(
            EfficientNetModelSpec(path=artifact), EfficientNetEngineOptions()
        )
        engine.initialize()
        predictors[target] = engine.predict
        operations[target] = {"first_key": "first_call_ms"}
    else:
        from docling_mlx.stages.picture_classification import (
            MlxDocumentPictureClassifier,
            MlxDocumentPictureClassifierOptions,
            MlxImageClassificationEngineOptions,
        )

        stage = MlxDocumentPictureClassifier(
            True,
            artifact.parent,
            MlxDocumentPictureClassifierOptions(
                model_spec=ImageClassificationModelSpec(
                    name="Local Document Figure Classifier",
                    repo_id=artifact.name,
                    revision="local",
                ),
                engine_options=MlxImageClassificationEngineOptions(),
            ),
            AcceleratorOptions(device="auto"),
        )

        def run_stage(batch: list[Image.Image]) -> Any:
            document = DoclingDocument(name="benchmark")
            elements = [
                ItemAndImageEnrichmentElement(item=document.add_picture(), image=image)
                for image in batch
            ]
            return list(stage(document, elements))

        predictors[target] = run_stage
        operations[target] = {"first_key": "first_call_ms"}
    return {
        "target": target,
        "timing_boundary_version": 4,
        "timing_boundary": _TIMING_BOUNDARIES[target],
        "artifact": artifact_provenance(artifact),
        "operations": operations,
        "predictors": predictors,
        "batch_sizes": (1, 4, 8, 16),
        "version_names": ("mlx", "docling-slim", "numpy", "Pillow"),
        "reset_memory": mx.reset_peak_memory,
        "initialization_ms": (perf_counter() - started) * 1000,
    }


def _make_torch_state(options: Any) -> dict[str, Any]:
    import torch
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import ItemAndImageEnrichmentElement
    from docling.datamodel.image_classification_engine_options import (
        TransformersImageClassificationEngineOptions,
    )
    from docling.datamodel.picture_classification_options import DocumentPictureClassifierOptions
    from docling.datamodel.stage_model_specs import EngineModelConfig
    from docling.models.inference_engines.image_classification.base import (
        ImageClassificationEngineInput,
    )
    from docling.models.inference_engines.image_classification.transformers_engine import (
        TransformersImageClassificationEngine,
    )
    from docling.models.stages.picture_classifier.document_picture_classifier import (
        DocumentPictureClassifier,
    )
    from docling_core.types.doc import DoclingDocument
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    from tools.document_figure.source import SOURCE_REPO, SOURCE_REVISION, verify_source

    source = Path(options.source).expanduser().resolve()
    verify_source(source)
    device = options.device
    threads = options.cpu_threads
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is not available")
    if threads is not None:
        torch.set_num_threads(threads)
    target: Target = options.target
    started = perf_counter()
    temporary: tempfile.TemporaryDirectory[str] | None = None
    operations: dict[str, dict[str, Any]] = {}
    predictors: dict[str, Callable[[Any], Any]] = {}
    if target == "preprocessing":
        processor = AutoImageProcessor.from_pretrained(source)
        predictors[target] = lambda batch: processor(images=batch, return_tensors="pt")[
            "pixel_values"
        ]
        operations[target] = {"first_key": "first_call_ms"}
    elif target == "forward":
        processor = AutoImageProcessor.from_pretrained(source)
        model = AutoModelForImageClassification.from_pretrained(source).to(device).eval()
        if device == "mps":
            torch.mps.synchronize()

        def prepare(batch: list[Image.Image]) -> Any:
            pixels = processor(images=batch, return_tensors="pt")["pixel_values"].to(device)
            if device == "mps":
                torch.mps.synchronize()
            return pixels

        def forward(pixels: Any) -> Any:
            with torch.inference_mode():
                result = model(pixel_values=pixels).logits.cpu().numpy().copy()
            if device == "mps":
                torch.mps.synchronize()
            return result

        predictors[target] = forward
        operations[target] = {"prepare": prepare, "first_key": "first_call_ms"}
    else:
        temporary = tempfile.TemporaryDirectory(prefix="docling-mlx-torch-benchmark-")
        artifact_root = Path(temporary.name)
        (artifact_root / SOURCE_REPO.replace("/", "--")).symlink_to(
            source, target_is_directory=True
        )
        engine_options = TransformersImageClassificationEngineOptions(
            torch_dtype="float32", compile_model=False, top_k=None
        )
        engine = TransformersImageClassificationEngine(
            options=engine_options,
            model_config=EngineModelConfig(repo_id=SOURCE_REPO, revision=SOURCE_REVISION),
            accelerator_options=AcceleratorOptions(device=device, num_threads=threads or 4),
            artifacts_path=artifact_root,
        )
        engine.initialize()

        def engine_predict(batch: list[Image.Image]) -> Any:
            result = engine([ImageClassificationEngineInput(image=image) for image in batch])
            if device == "mps":
                torch.mps.synchronize()
            return result

        if target == "engine":
            predictors[target] = engine_predict
            operations[target] = {"first_key": "first_call_ms"}
        else:
            stage_options = DocumentPictureClassifierOptions.from_preset(
                "document_figure_classifier_v2",
                engine_options=engine_options.model_copy(deep=True),
            )
            stage_options.model_spec.revision = SOURCE_REVISION
            stage = DocumentPictureClassifier(
                enabled=True,
                artifacts_path=artifact_root,
                options=stage_options,
                accelerator_options=AcceleratorOptions(device=device, num_threads=threads or 4),
            )

            def run_stage(batch: list[Image.Image]) -> Any:
                document = DoclingDocument(name="benchmark")
                elements = [
                    ItemAndImageEnrichmentElement(item=document.add_picture(), image=image)
                    for image in batch
                ]
                result = list(stage(document, elements))
                if device == "mps":
                    torch.mps.synchronize()
                return result

            predictors[target] = run_stage
            operations[target] = {"first_key": "first_call_ms"}
    return {
        "backend": f"torch-{device}",
        "target": target,
        "timing_boundary_version": 4,
        "timing_boundary": _TIMING_BOUNDARIES[target],
        "source": {"revision": SOURCE_REVISION},
        "operations": operations,
        "predictors": predictors,
        "batch_sizes": (1, 4, 8, 16),
        "version_names": (
            "torch",
            "torchvision",
            "transformers",
            "docling-slim",
            "numpy",
            "Pillow",
        ),
        "torch_num_threads": torch.get_num_threads(),
        "close": temporary.cleanup if temporary is not None else None,
        "initialization_ms": (perf_counter() - started) * 1000,
    }


def load(options: Any, images: list[Image.Image]) -> dict[str, Any]:
    del images
    if options.target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    if options.backend == "mlx":
        if options.artifact is None or options.source is not None:
            raise ValueError("MLX DocumentFigure requires --artifact only")
        return _make_mlx_state(options)
    if options.backend == "torch":
        if options.source is None or options.artifact is not None or options.device is None:
            raise ValueError("Torch DocumentFigure requires --source and --device only")
        return _make_torch_state(options)
    raise ValueError(f"unsupported backend: {options.backend}")


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
            "backend",
            "target",
            "timing_boundary_version",
            "timing_boundary",
            "artifact",
            "source",
        )
        if key in state
    }
    report.update({"initialization_ms": state["initialization_ms"], "results": rows})
    report["hardware"] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    if "torch_num_threads" in state:
        report["torch_num_threads"] = state["torch_num_threads"]
        import torch

        report["mps_allocated_at_end_bytes"] = (
            torch.mps.current_allocated_memory() if state["backend"] == "torch-mps" else None
        )
    else:
        import mlx.core as mx

        report["mlx_peak_memory_bytes"] = mx.get_peak_memory()
    return report


def predict(state: dict[str, Any], target: str, value: Any) -> Any:
    return state["predictors"][target](value)
