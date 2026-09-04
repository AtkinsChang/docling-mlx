# SPDX-License-Identifier: Apache-2.0

"""TableFormerV1 benchmark adapter for :mod:`tools.benchmark`."""

from __future__ import annotations

import hashlib
import platform
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
from docling.datamodel.pipeline_options import TableFormerMode
from PIL import Image

from tools._common.benchmark import _image_identity
from tools.tableformer_v1.capture_reference import (
    _stage_page,
    configure_torch,
    load_reference,
    preprocess_exact,
)
from tools.tableformer_v1.source import SOURCE_REVISION, verify_source

Backend = Literal["mlx", "torch"]
REQUIRES = ("target",)
DEFAULT_THREADS = 1
Target = Literal["engine", "stage"]
_VERSIONS = ("docling", "docling-ibm-models", "mlx", "numpy", "Pillow", "torch", "torchvision")
_NOTES = (
    "Run each backend/target in a fresh process.",
    "This is descriptive timing; it does not imply a speedup claim.",
)


def _timing_boundary(target: Target) -> str:
    if target == "engine":
        return (
            "Engine: RGB crop through exact preprocessing, cached generation, bbox "
            "materialization, "
            "and result construction."
        )
    return (
        "Stage: prepared 2x full-page render through 1024px page resize, table crop, model "
        "inference, and table boundary; PDF parsing excluded."
    )


def _materialize_torch_prediction(
    generated_ids: object,
    class_logits: np.ndarray,
    boxes_cxcywh: np.ndarray,
    image: Image.Image,
) -> Any:
    boxes = np.asarray(boxes_cxcywh)
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise RuntimeError("TableFormerV1 Torch boxes must have shape [N, 4]")
    half_size = boxes[:, 2:] / np.float32(2)
    boxes_xyxy = np.concatenate((boxes[:, :2] - half_size, boxes[:, :2] + half_size), axis=1)
    from docling_mlx.engines.table_structure.tableformer_v1 import TableFormerV1Engine

    return TableFormerV1Engine._materialize_prediction(
        generated_ids, boxes_xyxy, class_logits, image
    )


def _artifact_digest(directory: Path, names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(name.encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            digest.update(hashlib.file_digest(stream, "sha256").digest())
    return digest.hexdigest()


def _load_mlx(options: Any, images: list[Image.Image], mode: TableFormerMode) -> dict[str, Any]:
    import mlx.core as mx
    from docling.datamodel.accelerator_options import AcceleratorOptions

    from docling_mlx.engines.table_structure.tableformer_v1 import (
        TableFormerV1Engine,
        TableFormerV1EngineOptions,
        TableFormerV1ModelSpec,
    )

    artifact = Path(options.artifact).expanduser().resolve()
    target: Target = options.target
    spec = TableFormerV1ModelSpec(path=artifact)
    started = perf_counter()
    if target == "engine":
        engine = TableFormerV1Engine(
            spec,
            TableFormerV1EngineOptions(checkpoint_subdirectory=mode.value),
        )
        engine.initialize()
        predictor: Callable[[Sequence[Image.Image]], Any] = engine.predict
    else:
        from docling_mlx.stages.table_structure_v1 import (
            MlxTableFormerV1Model,
            MlxTableStructureOptions,
        )

        stage = MlxTableFormerV1Model(
            True,
            artifact.parent,
            MlxTableStructureOptions(model_spec=spec, mode=mode, do_cell_matching=False),
            AcceleratorOptions(device="auto"),
        )
        if stage.engine is None:
            raise RuntimeError("Enabled TableFormerV1 stage did not construct its engine")
        stage.engine.initialize()

        def predictor(batch: Sequence[Image.Image]) -> Any:
            from docling.datamodel.document import ConversionResult

            return stage.predict_tables(
                ConversionResult.model_construct(timings={}),
                [_stage_page(image) for image in batch],
            )

    return {
        "backend": "mlx-metal",
        "target": target,
        "profile": f"tableformer_v1_{mode.value}",
        "timing_boundary_version": 2,
        "timing_boundary": _timing_boundary(target),
        "stage_adapter": "docling-public-stage" if target == "stage" else "not-applicable",
        "artifact": {
            "converted_sha256": _artifact_digest(
                artifact / mode.value,
                (
                    "model.safetensors",
                    "config.json",
                    "preprocessor_config.json",
                    "generation_config.json",
                ),
            )
        },
        "predictors": {target: predictor},
        "operations": {target: {"first_key": "first_call_ms"}},
        "batch_sizes": (len(images),),
        "version_names": _VERSIONS,
        "cpu_threads": None,
        "reset_memory": mx.reset_peak_memory,
        "initialization_ms": (perf_counter() - started) * 1000,
        "notes": list(_NOTES),
    }


def _load_torch(options: Any, images: list[Image.Image], mode: TableFormerMode) -> dict[str, Any]:
    import torch
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.document import ConversionResult

    source = Path(options.source).expanduser().resolve()
    verify_source(source)
    cpu_threads = options.cpu_threads
    configure_torch(torch, cpu_threads)
    target: Target = options.target
    started = perf_counter()
    if target == "engine":
        model, config = load_reference(source, torch, mode)

        def predictor(batch: Sequence[Image.Image]) -> Any:
            result = []
            with torch.inference_mode():
                for image in batch:
                    pixels = preprocess_exact(np.asarray(image.convert("RGB")), config)
                    ids, classes, boxes = model.predict(
                        torch.from_numpy(pixels), config["predict"]["max_steps"], 1
                    )
                    result.append(
                        _materialize_torch_prediction(
                            ids, classes.cpu().numpy(), boxes.cpu().numpy(), image
                        )
                    )
            return result

    else:
        from docling.datamodel.pipeline_options import TableStructureOptions
        from docling.models.stages.table_structure.table_structure_model import TableStructureModel

        stage_model = TableStructureModel(
            True,
            source,
            TableStructureOptions(mode=mode, do_cell_matching=False),
            AcceleratorOptions(device="cpu", num_threads=cpu_threads),
        )

        def predictor(batch: Sequence[Image.Image]) -> Any:
            return stage_model.predict_tables(
                ConversionResult.model_construct(timings={}),
                [_stage_page(image) for image in batch],
            )

    return {
        "backend": "torch-cpu",
        "target": target,
        "profile": f"tableformer_v1_{mode.value}",
        "timing_boundary_version": 2,
        "timing_boundary": _timing_boundary(target),
        "stage_adapter": "docling-public-stage" if target == "stage" else "not-applicable",
        "artifact": {"source_revision": SOURCE_REVISION},
        "cpu_threads": cpu_threads,
        "predictors": {target: predictor},
        "operations": {target: {"first_key": "first_call_ms"}},
        "batch_sizes": (len(images),),
        "version_names": _VERSIONS,
        "initialization_ms": (perf_counter() - started) * 1000,
        "notes": list(_NOTES),
    }


def load(options: Any, images: list[Image.Image]) -> dict[str, Any]:
    if not images:
        raise ValueError("images, nonnegative warmup, and positive repeats are required")
    if options.target not in ("engine", "stage"):
        raise ValueError("target must be engine or stage")
    mode = TableFormerMode(options.profile or TableFormerMode.ACCURATE.value)
    if options.backend == "mlx":
        if options.artifact is None:
            raise ValueError("MLX TableFormerV1 requires --artifact")
        return _load_mlx(options, images, mode)
    if options.source is None:
        raise ValueError("Torch TableFormerV1 requires --source")
    return _load_torch(options, images, mode)


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
            "profile",
            "timing_boundary_version",
            "timing_boundary",
            "stage_adapter",
            "artifact",
            "cpu_threads",
            "initialization_ms",
            "notes",
        )
        if key in state
    }
    row = rows[0]
    report["first_call_ms"] = row["first_call_ms"]
    report["warm"] = row[state["target"]]
    report["images"] = _image_identity(image_paths, images)
    report["hardware"] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    if state["backend"] == "mlx-metal":
        import mlx.core as mx

        report["mlx_peak_memory_bytes"] = mx.get_peak_memory()
    else:
        report["mlx_peak_memory_bytes"] = None
    return report


def predict(state: dict[str, Any], target: str, value: Any) -> Any:
    return state["predictors"][target](value)
