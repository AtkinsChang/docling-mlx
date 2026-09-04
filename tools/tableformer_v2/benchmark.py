# SPDX-License-Identifier: Apache-2.0

"""TableFormerV2 benchmark adapter for :mod:`tools.benchmark`."""

from __future__ import annotations

import platform
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

from docling.datamodel.base_models import Cluster, LayoutPrediction, Page
from docling.datamodel.pipeline_options import TableStructureV2Options
from docling_core.types.doc import BoundingBox, DocItemLabel, Size
from PIL import Image

from tools._common.benchmark import _image_identity
from tools._common.hashing import hash_named_files

Backend = Literal["mlx", "torch"]
REQUIRES = ("target",)
DEFAULT_THREADS = 4
Target = Literal["engine", "stage"]
TORCH_REQUIRED_FILES = (
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
_NOTES = (
    "Each backend and target must run in a separate fresh process.",
    "The first implementation generates each image independently; batch throughput therefore "
    "measures ordered sequential generation.",
    "No speedup claim is inferred from these descriptive measurements.",
)


def _timing_boundary(target: Target) -> str:
    if target == "engine":
        return (
            "Prepared RGB PIL images through preprocessing, autoregressive generation, bbox "
            "materialization, and engine result construction."
        )
    return (
        "Fresh Docling pages with prepared 2x renders through crop, inference, OTSL/cell "
        "construction, and table prediction attachment. PDF parsing excluded."
    )


def artifact_provenance(directory: Path, backend: Backend) -> dict[str, Any]:
    from docling_mlx.engines.table_structure.tableformer_v2.artifact import CHECKPOINT_FILES

    required = CHECKPOINT_FILES if backend == "mlx" else TORCH_REQUIRED_FILES
    return {
        "kind": "local",
        "backend": backend,
        "sha256": hash_named_files(directory, required),
        "source_revision": _source_revision(),
    }


def _source_revision() -> str:
    from tools.tableformer_v2.source import SOURCE_REVISION

    return SOURCE_REVISION


class _PageBackend:
    def __init__(self, image: Image.Image) -> None:
        self.image = image

    def is_valid(self) -> bool:
        return True

    def get_page_image(self, scale: float, cropbox: BoundingBox | None = None) -> Image.Image:
        if scale != 2.0 or cropbox is not None:
            raise AssertionError("benchmark page must use one full 2x render")
        return self.image

    def get_text_in_rect(self, bbox: BoundingBox) -> str:
        del bbox
        return ""


def _make_pages(images: Sequence[Image.Image]) -> list[Page]:
    pages: list[Page] = []
    for index, image in enumerate(images):
        width, height = image.width / 2, image.height / 2
        page = Page(page_no=index, size=Size(width=width, height=height))
        page.predictions.layout = LayoutPrediction(
            clusters=[
                Cluster(
                    id=index,
                    label=DocItemLabel.TABLE,
                    bbox=BoundingBox(l=0, t=0, r=width, b=height),
                    cells=[],
                )
            ]
        )
        page._backend = cast(Any, _PageBackend(image))
        pages.append(page)
    return pages


def _load_mlx(options: Any, target: Target) -> dict[str, Any]:
    import mlx.core as mx
    from docling.datamodel.accelerator_options import AcceleratorOptions

    artifact = Path(options.artifact).expanduser().resolve()
    from docling_mlx.engines.table_structure.tableformer_v2 import (
        TableFormerV2Engine,
        TableFormerV2ModelSpec,
    )

    spec = TableFormerV2ModelSpec(path=artifact)
    accelerator = AcceleratorOptions(device="auto")
    started = perf_counter()
    if target == "engine":
        engine = TableFormerV2Engine(spec)
        engine.initialize()
        predictor = engine.predict
    else:
        from docling_mlx.stages.table_structure_v2 import (
            MlxTableFormerV2Model,
            MlxTableStructureV2Options,
        )

        stage = MlxTableFormerV2Model(
            enabled=True,
            artifacts_path=artifact.parent,
            options=MlxTableStructureV2Options(model_spec=spec, do_cell_matching=False),
            accelerator_options=accelerator,
        )
        if stage.engine is None:
            raise RuntimeError("Enabled TableFormerV2 stage did not construct its engine")
        stage.engine.initialize()

        def predictor(images: Sequence[Image.Image]) -> Any:
            from docling.datamodel.document import ConversionResult

            return list(stage(ConversionResult.model_construct(timings={}), _make_pages(images)))

    return {
        "backend": "mlx-metal",
        "target": target,
        "timing_boundary_version": 1,
        "timing_boundary": _timing_boundary(target),
        "artifact": artifact_provenance(artifact, "mlx"),
        "predictors": {target: predictor},
        "operations": {target: {"first_key": "first_call_ms", "measurement_key": "warm"}},
        "batch_sizes": tuple((1,) if options.batch_sizes is None else options.batch_sizes),
        "version_names": ("docling-slim", "numpy", "Pillow", "mlx"),
        "cpu_threads": None,
        "reset_memory": getattr(mx, "reset_peak_memory", None),
        "initialization_ms": (perf_counter() - started) * 1000,
        "notes": list(_NOTES),
    }


def _load_torch(options: Any, target: Target) -> dict[str, Any]:
    import torch
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.document import ConversionResult

    artifact = Path(options.artifact).expanduser().resolve()
    cpu_threads = options.cpu_threads
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = perf_counter()
    if target == "stage":
        from docling.models.stages.table_structure.table_structure_model_v2 import (
            TableStructureModelV2,
        )

        stage = TableStructureModelV2(
            enabled=True,
            artifacts_path=artifact,
            options=TableStructureV2Options(do_cell_matching=False),
            accelerator_options=AcceleratorOptions(device="cpu", num_threads=cpu_threads),
        )

        def predictor(images: Sequence[Image.Image]) -> Any:
            with torch.inference_mode():
                return list(
                    stage(ConversionResult.model_construct(timings={}), _make_pages(images))
                )

    else:
        from docling_ibm_models.tableformer_v2 import TableFormerV2
        from torchvision import transforms
        from transformers import AutoTokenizer

        model: Any = TableFormerV2.from_pretrained(
            artifact, local_files_only=True, use_safetensors=True, dtype=torch.float32
        )
        model = model.to("cpu")
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(artifact, local_files_only=True)
        transform = transforms.Compose(
            [
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        def predictor(images: Sequence[Image.Image]) -> Any:
            outputs = []
            with torch.inference_mode():
                for image in images:
                    tensor = transform(image.convert("RGB")).unsqueeze(0)
                    output = model.generate(tensor, tokenizer, max_length=512)
                    generated = output["generated_ids"].detach().cpu().numpy()
                    bboxes = output["predicted_bboxes"]
                    outputs.append(
                        (generated, None if bboxes is None else bboxes.detach().cpu().numpy())
                    )
            return outputs

    return {
        "backend": "torch-cpu",
        "target": target,
        "timing_boundary_version": 1,
        "timing_boundary": _timing_boundary(target),
        "artifact": artifact_provenance(artifact, "torch"),
        "cpu_threads": cpu_threads,
        "predictors": {target: predictor},
        "operations": {target: {"first_key": "first_call_ms", "measurement_key": "warm"}},
        "batch_sizes": tuple((1,) if options.batch_sizes is None else options.batch_sizes),
        "version_names": (
            "docling-slim",
            "numpy",
            "Pillow",
            "torch",
            "torchvision",
            "transformers",
        ),
        "initialization_ms": (perf_counter() - started) * 1000,
        "notes": list(_NOTES),
    }


def load(options: Any, images: list[Image.Image]) -> dict[str, Any]:
    del images
    if options.target not in ("engine", "stage"):
        raise ValueError("target must be engine or stage")
    if options.artifact is None:
        raise ValueError("TableFormerV2 requires --artifact")
    target: Target = options.target
    if options.backend == "mlx":
        return _load_mlx(options, target)
    if options.backend == "torch":
        return _load_torch(options, target)
    raise ValueError(f"unsupported backend: {options.backend}")


def normalize_result(
    state: dict[str, Any],
    rows: list[dict[str, Any]],
    image_paths: list[Path],
    images: list[Image.Image],
    options: Any,
) -> dict[str, Any]:
    del options
    normalized_rows = []
    for row in rows:
        normalized_rows.append(
            {
                "batch_size": row["batch_size"],
                "first_call_ms": row["first_call_ms"],
                "cold_total_ms": state["initialization_ms"] + row["first_call_ms"],
                "warm": row["warm"],
            }
        )
    report = {
        key: state[key]
        for key in (
            "backend",
            "target",
            "timing_boundary_version",
            "timing_boundary",
            "artifact",
            "cpu_threads",
            "initialization_ms",
            "notes",
        )
        if key in state
    } | {"results": normalized_rows}
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
