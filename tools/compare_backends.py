# SPDX-License-Identifier: Apache-2.0

"""Reproduce MLX-versus-official backend measurements on DPBench."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cache
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

from tools._common.benchmark import peak_rss_bytes

ROOT = Path(__file__).resolve().parents[1]
HF = Path.home() / ".cache/huggingface/hub"
DATASET_REPO = "docling-project/docling-dpbench"
DATASET_REVISION = "8fb4a3b9a57ae119f27f42ce223dbc55f43f0e55"
DEFAULT_INPUTS = ROOT / "reports/backend-bench/inputs"
DEFAULT_REPORTS = ROOT / "reports/backend-bench"
# Docling 2.124.0 VlmModelSpec.max_new_tokens default.  The official Granite
# stages instead pass tokenizer.model_max_length, an effectively unbounded sentinel.
OFFICIAL_GRANITE_MAX_NEW_TOKENS = 4096
BENCHMARK_SCHEMA = 2
LAYOUT_SCORE_THRESHOLD_MARGIN = 0.02

COMPONENTS = (
    "heron_r50",
    "heron_r101",
    "egret_medium",
    "egret_large",
    "egret_xlarge",
    "figure",
    "table_v1_accurate",
    "table_v1_fast",
    "table_v2",
    "granite_table",
    "granite_chart",
    "pipeline",
)
LAYOUT_COMPONENTS = frozenset(COMPONENTS[:5])
TABLE_COMPONENTS = frozenset({"table_v1_accurate", "table_v1_fast", "table_v2"})
GRANITE_COMPONENTS = frozenset({"granite_table", "granite_chart"})

SOURCES: dict[str, tuple[str, str, Path]] = {
    "heron_r50": (
        "docling-project/docling-layout-heron",
        "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8",
        HF
        / "models--docling-project--docling-layout-heron/snapshots"
        / "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8",
    ),
    "heron_r101": (
        "docling-project/docling-layout-heron-101",
        "2e4993cf6bb211112084a2f80938f26138008917",
        HF
        / "models--docling-project--docling-layout-heron-101/snapshots"
        / "2e4993cf6bb211112084a2f80938f26138008917",
    ),
    "egret_medium": (
        "docling-project/docling-layout-egret-medium",
        "77ede7cc7bed96d853c58f319734803d6ea2ea5c",
        Path("/private/tmp/egret-medium-source-77ede7c"),
    ),
    "egret_large": (
        "docling-project/docling-layout-egret-large",
        "fff417c78abd6bab338c87706c95a8d79dc68f1e",
        Path("/private/tmp/egret-large-src"),
    ),
    "egret_xlarge": (
        "docling-project/docling-layout-egret-xlarge",
        "23857d16596e0106716b3162d132212d733769e7",
        Path("/private/tmp/egret-xlarge-src"),
    ),
    "figure": (
        "docling-project/DocumentFigureClassifier-v2.5",
        "f859dfbff5c9916cd996942d4b0db7fa25808220",
        HF
        / "models--docling-project--DocumentFigureClassifier-v2.5/snapshots"
        / "f859dfbff5c9916cd996942d4b0db7fa25808220",
    ),
    "table_v1": (
        "docling-project/docling-models",
        "fc0f2d45e2218ea24bce5045f58a389aed16dc23",
        Path("/private/tmp/tfv1-source"),
    ),
    "table_v2": (
        "docling-project/TableFormerV2",
        "51559fad3946873e26a6f9b8e912f948e8745bef",
        HF
        / "models--docling-project--TableFormerV2/snapshots"
        / "51559fad3946873e26a6f9b8e912f948e8745bef",
    ),
    "granite": (
        "ibm-granite/granite-vision-4.1-4b",
        "dd48e97503de471803850df70843cf9eb5da8712",
        ROOT / ".artifacts/granite-vlm/ibm-granite--granite-vision-4.1-4b",
    ),
}

ARTIFACTS = {
    "heron_r50": ROOT / ".artifacts/heron-r50",
    "heron_r101": ROOT / ".artifacts/heron-r101",
    "egret_medium": ROOT / ".artifacts/egret-medium",
    "egret_large": ROOT / ".artifacts/egret-large",
    "egret_xlarge": ROOT / ".artifacts/egret-xlarge",
    "figure": ROOT / ".artifacts/document-figure-classifier",
    "table_v1": ROOT / ".artifacts/tableformer-v1",
    "table_v2": ROOT / ".artifacts/tableformer-v2",
    "granite": ROOT / ".artifacts/granite-vlm",
}

PRESETS = {
    "heron_r50": "layout_heron_default",
    "heron_r101": "layout_heron_101",
    "egret_medium": "layout_egret_medium",
    "egret_large": "layout_egret_large",
    "egret_xlarge": "layout_egret_xlarge",
    "figure": "document_figure_classifier_v2",
    "table_v1_accurate": "tableformer_v1_accurate",
    "table_v1_fast": "tableformer_v1_fast",
    "table_v2": "tableformer_v2",
}


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(_json(value), encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_directory(path: Path) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _dataset_parquet() -> Path:
    snapshot = HF / "datasets--docling-project--docling-dpbench/snapshots" / DATASET_REVISION
    parquet = snapshot / "test/shard_000000_000000.parquet"
    if parquet.is_file():
        return parquet
    from huggingface_hub import snapshot_download

    downloaded = Path(
        snapshot_download(
            DATASET_REPO,
            repo_type="dataset",
            revision=DATASET_REVISION,
        )
    )
    return downloaded / "test/shard_000000_000000.parquet"


def _crop(
    image: Image.Image,
    document: dict[str, Any],
    provenance: dict[str, Any],
) -> Image.Image:
    page_no = int(provenance["page_no"])
    page = document["pages"][str(page_no)]
    page_size = page["size"]
    bbox = provenance["bbox"]
    if bbox.get("coord_origin") != "TOPLEFT":
        raise ValueError(f"Unsupported bbox origin: {bbox.get('coord_origin')!r}")
    scale_x = image.width / float(page_size["width"])
    scale_y = image.height / float(page_size["height"])
    left = max(0.0, float(bbox["l"]) * scale_x)
    top = max(0.0, float(bbox["t"]) * scale_y)
    right = min(float(image.width), float(bbox["r"]) * scale_x)
    bottom = min(float(image.height), float(bbox["b"]) * scale_y)
    if right <= left or bottom <= top:
        raise ValueError(f"Empty crop after scaling bbox {bbox}")
    return image.crop((left, top, right, bottom)).convert("RGB")


def prepare(args: argparse.Namespace) -> None:
    """Materialize the pinned DPBench pages, PDFs, and ground-truth crops."""

    import pyarrow.parquet as pq

    root = Path(args.inputs).resolve()
    pages_dir = root / "pages"
    pdfs_dir = root / "pdfs"
    pictures_dir = root / "pictures"
    tables_dir = root / "tables"
    for directory in (pages_dir, pdfs_dir, pictures_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows = pq.read_table(_dataset_parquet()).to_pylist()
    pages: list[dict[str, Any]] = []
    pictures: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    seen: set[str] = set()
    for dataset_index, row in enumerate(rows):
        document_id = str(row["document_id"])
        stem = Path(document_id).stem
        if Path(document_id).name != document_id or stem in seen:
            raise ValueError(f"Unsafe or duplicate document_id: {document_id!r}")
        seen.add(stem)
        document = json.loads(row["GroundTruthDocument"])
        ground_truth_document: Any | None = None
        if document.get("tables"):
            from docling_core.types.doc import DoclingDocument

            ground_truth_document = DoclingDocument.model_validate(document)
        page_images = row["GroundTruthPageImages"]
        if len(page_images) != 1 or len(document.get("pages", {})) != 1:
            raise ValueError(f"DPBench row {dataset_index} is not a one-page document")
        page_bytes = page_images[0]["bytes"]
        pdf_bytes = row["BinaryDocument"]
        if not isinstance(page_bytes, bytes) or not isinstance(pdf_bytes, bytes):
            raise TypeError(f"DPBench row {dataset_index} has non-byte inputs")
        page_path = pages_dir / f"{stem}.png"
        pdf_path = pdfs_dir / f"{stem}.pdf"
        page_path.write_bytes(page_bytes)
        pdf_path.write_bytes(pdf_bytes)
        with Image.open(BytesIO(page_bytes)) as source_image:
            image = source_image.convert("RGB")
        pages.append(
            {
                "dataset_index": dataset_index,
                "document_id": document_id,
                "id": stem,
                "image": str(page_path),
                "pdf": str(pdf_path),
                "image_sha256": _sha256(page_bytes),
                "pdf_sha256": _sha256(pdf_bytes),
            }
        )
        for kind, records, directory in (
            ("picture", pictures, pictures_dir),
            ("table", tables, tables_dir),
        ):
            for item_index, item in enumerate(document.get(f"{kind}s", [])):
                provenance = item.get("prov") or []
                if not provenance:
                    raise ValueError(f"{document_id} {kind} {item_index} has no provenance")
                crop = _crop(image, document, provenance[0])
                crop_id = f"{stem}-{kind}-{item_index:02d}"
                crop_path = directory / f"{crop_id}.png"
                crop.save(crop_path, format="PNG")
                record = {
                    "dataset_index": dataset_index,
                    "document_id": document_id,
                    "item_index": item_index,
                    "self_ref": item.get("self_ref"),
                    "id": crop_id,
                    "image": str(crop_path),
                    "bbox": provenance[0]["bbox"],
                }
                if kind == "table":
                    if ground_truth_document is None:
                        raise ValueError(
                            f"{document_id} has a table without a ground-truth document"
                        )
                    record["ground_truth_otsl"] = ground_truth_document.tables[
                        item_index
                    ].export_to_otsl(ground_truth_document, add_cell_location=False)
                records.append(record)

    if (len(pages), len(pictures), len(tables)) != (200, 168, 63):
        raise ValueError(
            "Unexpected DPBench counts: "
            f"pages={len(pages)}, pictures={len(pictures)}, tables={len(tables)}"
        )
    _write_json(
        root / "manifest.json",
        {
            "schema": 1,
            "dataset": {"repo_id": DATASET_REPO, "revision": DATASET_REVISION},
            "pages": pages,
            "pictures": pictures,
            "tables": tables,
            "granite_tables": tables[:5],
            "granite_charts": [],
        },
    )
    print(f"Prepared 200 pages, 168 pictures, and 63 tables in {root}")


def _package_versions() -> dict[str, str]:
    names = (
        "mlx",
        "torch",
        "transformers",
        "docling",
        "docling-ibm-models",
        "mlx-vlm",
    )
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def _sysctl(name: str) -> str:
    result = subprocess.run(
        ["sysctl", "-n", name],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "unavailable"


def _runtime_metadata() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "machine": platform.node(),
        "macos": platform.mac_ver()[0],
        "chip": _sysctl("machdep.cpu.brand_string"),
        "memory_bytes": int(value) if (value := _sysctl("hw.memsize")).isdigit() else value,
        "python": platform.python_version(),
        "versions": _package_versions(),
        "git_commit": commit,
    }


def _preset_metadata(component: str, artifact: Path) -> dict[str, Any]:
    from docling_mlx.presets import resolve_preset

    preset_name = PRESETS[component]
    preset = resolve_preset(preset_name)
    return {
        "artifact_path": str(_require_directory(artifact)),
        "artifact_repo_id": preset.repo_id,
        "artifact_revision": preset.revision,
        "preset": preset_name,
    }


def _source_metadata(key: str, path: Path | None = None) -> dict[str, str]:
    repo_id, revision, default_path = SOURCES[key]
    source = _require_directory(default_path if path is None else path)
    return {
        "artifact_path": str(source),
        "artifact_repo_id": repo_id,
        "artifact_revision": revision,
    }


def _sync_torch(device: str) -> None:
    if device == "mps":
        import torch

        torch.mps.synchronize()


def _normalize_label(label: object) -> str:
    return str(getattr(label, "value", label)).lower().replace("-", "_").replace(" ", "_")


def _canonical_detection(
    boxes: Iterable[Iterable[float]],
    scores: Iterable[float],
    label_ids: Iterable[int],
    labels: dict[int, str],
) -> dict[str, Any]:
    detections = []
    for box, score, label_id in zip(boxes, scores, label_ids, strict=True):
        left, top, right, bottom = (float(value) for value in box)
        detections.append(
            {
                "l": left,
                "t": top,
                "r": right,
                "b": bottom,
                "label": _normalize_label(labels[int(label_id)]),
                "score": float(score),
            }
        )
    return {"detections": detections}


def _canonical_table(
    tokens: Iterable[str],
    boxes: Iterable[Iterable[float]],
    image: Image.Image,
    *,
    normalized: bool,
) -> dict[str, Any]:
    scale = (image.width, image.height, image.width, image.height)
    return {
        "otsl": [str(token).strip("<>") for token in tokens],
        "cell_bboxes": [
            [
                float(value) * scale[index] if normalized else float(value)
                for index, value in enumerate(box)
            ]
            for box in boxes
        ],
    }


Predictor = Callable[[Image.Image], dict[str, Any]]


def _mlx_predictor(component: str) -> tuple[Predictor, dict[str, Any]]:
    engine: Any
    if component in {"heron_r50", "heron_r101"}:
        from docling_mlx.engines.object_detection.rt_detr_v2.engine import (
            RtDetrV2Engine,
            RtDetrV2ModelSpec,
        )

        engine = RtDetrV2Engine(RtDetrV2ModelSpec(path=ARTIFACTS[component]))

        def predict_heron(image: Image.Image) -> dict[str, Any]:
            value = engine.predict([image])[0]
            return _canonical_detection(value.boxes, value.scores, value.label_ids, value.id2label)

        return predict_heron, _preset_metadata(component, ARTIFACTS[component]) | {
            "score_threshold": engine.options.score_threshold
        }
    if component in {"egret_medium", "egret_large", "egret_xlarge"}:
        from docling_mlx.engines.object_detection.dfine.engine import (
            DFineEngine,
            DFineModelSpec,
        )

        engine = DFineEngine(DFineModelSpec(path=ARTIFACTS[component]))

        def predict_egret(image: Image.Image) -> dict[str, Any]:
            value = engine.predict([image])[0]
            return _canonical_detection(value.boxes, value.scores, value.label_ids, value.id2label)

        return predict_egret, _preset_metadata(component, ARTIFACTS[component]) | {
            "score_threshold": engine.options.score_threshold
        }
    if component == "figure":
        from docling_mlx.engines.image_classification.efficientnet.engine import (
            EfficientNetEngine,
            EfficientNetEngineOptions,
            EfficientNetModelSpec,
        )

        engine = EfficientNetEngine(
            EfficientNetModelSpec(path=ARTIFACTS[component]),
            EfficientNetEngineOptions(top_k=None),
        )

        def predict_figure(image: Image.Image) -> dict[str, Any]:
            value = engine.predict([image])[0]
            return {
                "probabilities": {
                    value.id2label[label_id]: float(probability)
                    for label_id, probability in zip(
                        value.label_ids, value.probabilities, strict=True
                    )
                }
            }

        return predict_figure, _preset_metadata(component, ARTIFACTS[component])
    if component.startswith("table_v1_"):
        from docling_mlx.engines.table_structure.tableformer_v1.engine import (
            TableFormerV1Engine,
            TableFormerV1EngineOptions,
        )
        from docling_mlx.engines.table_structure.tableformer_v1.model_spec import (
            TableFormerV1ModelSpec,
        )

        mode = component.rsplit("_", 1)[1]
        engine = TableFormerV1Engine(
            TableFormerV1ModelSpec(path=ARTIFACTS["table_v1"]),
            TableFormerV1EngineOptions(checkpoint_subdirectory=mode),
        )

        def predict_table_v1(image: Image.Image) -> dict[str, Any]:
            value = engine.predict([image])[0]
            return _canonical_table(value.otsl_tokens, value.cell_bboxes, image, normalized=False)

        return predict_table_v1, _preset_metadata(component, ARTIFACTS["table_v1"])
    if component == "table_v2":
        from docling_mlx.engines.table_structure.tableformer_v2.engine import (
            TableFormerV2Engine,
        )
        from docling_mlx.engines.table_structure.tableformer_v2.model_spec import (
            TableFormerV2ModelSpec,
        )

        engine = TableFormerV2Engine(TableFormerV2ModelSpec(path=ARTIFACTS["table_v2"]))

        def predict_table_v2(image: Image.Image) -> dict[str, Any]:
            value = engine.predict([image])[0]
            return _canonical_table(value.otsl_tokens, value.cell_bboxes, image, normalized=False)

        return predict_table_v2, _preset_metadata(component, ARTIFACTS["table_v2"])
    if component in GRANITE_COMPONENTS:
        return _mlx_granite_predictor(component)
    raise ValueError(f"Unsupported MLX component: {component}")


def _official_artifact_root(key: str) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    repo_id, _revision, source = SOURCES[key]
    _require_directory(source)
    temporary = tempfile.TemporaryDirectory(prefix=f"docling-backend-{key}-")
    root = Path(temporary.name)
    (root / repo_id.replace("/", "--")).symlink_to(source, target_is_directory=True)
    return temporary, root


def _official_vision_predictor(component: str, device: str) -> tuple[Predictor, dict[str, Any]]:
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.stage_model_specs import EngineModelConfig

    temporary, artifacts_root = _official_artifact_root(component)
    repo_id, revision, source = SOURCES[component]
    model_config = EngineModelConfig(repo_id=repo_id, revision=revision)
    accelerator = AcceleratorOptions(device=device, num_threads=4)
    metadata: dict[str, Any] = _source_metadata(component, source)
    metadata["_temporary"] = temporary
    if component == "figure":
        from docling.datamodel.image_classification_engine_options import (
            TransformersImageClassificationEngineOptions,
        )
        from docling.models.inference_engines.image_classification.base import (
            ImageClassificationEngineInput,
        )
        from docling.models.inference_engines.image_classification.transformers_engine import (
            TransformersImageClassificationEngine,
        )

        engine: Any = TransformersImageClassificationEngine(
            options=TransformersImageClassificationEngineOptions(
                top_k=None, torch_dtype="float32", compile_model=False
            ),
            model_config=model_config,
            accelerator_options=accelerator,
            artifacts_path=artifacts_root,
        )
        engine.initialize()
        labels = engine.get_label_mapping()

        def predict_figure(image: Image.Image) -> dict[str, Any]:
            value = engine.predict_batch([ImageClassificationEngineInput(image=image)])[0]
            _sync_torch(device)
            return {
                "probabilities": {
                    labels[label_id]: float(score)
                    for label_id, score in zip(value.label_ids, value.scores, strict=True)
                }
            }

        return predict_figure, metadata

    from docling.datamodel.object_detection_engine_options import (
        TransformersObjectDetectionEngineOptions,
    )
    from docling.models.inference_engines.object_detection.base import (
        ObjectDetectionEngineInput,
    )
    from docling.models.inference_engines.object_detection.transformers_engine import (
        TransformersObjectDetectionEngine,
    )

    engine = TransformersObjectDetectionEngine(
        options=TransformersObjectDetectionEngineOptions(
            score_threshold=0.3, torch_dtype="float32", compile_model=False
        ),
        model_config=model_config,
        accelerator_options=accelerator,
        artifacts_path=artifacts_root,
    )
    engine.initialize()
    labels = engine.get_label_mapping()
    metadata["score_threshold"] = engine.options.score_threshold

    def predict_detection(image: Image.Image) -> dict[str, Any]:
        value = engine.predict_batch([ObjectDetectionEngineInput(image=image)])[0]
        _sync_torch(device)
        return _canonical_detection(value.bboxes, value.scores, value.label_ids, labels)

    return predict_detection, metadata


def _official_table_predictor(component: str, device: str) -> tuple[Predictor, dict[str, Any]]:
    import torch

    if component.startswith("table_v1_"):
        from docling_ibm_models.tableformer.data_management.tf_predictor import (
            TFPredictor,
        )
        from docling_ibm_models.tableformer.utils import utils as table_utils

        mode = component.rsplit("_", 1)[1]
        source = SOURCES["table_v1"][2]
        profile = source / "model_artifacts/tableformer" / mode
        config = _read_json(profile / "tm_config.json")
        config["model"]["save_dir"] = str(profile)
        predictor = TFPredictor(config, device=device, num_threads=4)

        def predict_v1(image: Image.Image) -> dict[str, Any]:
            pixels = predictor._prepare_image(np.asarray(image.convert("RGB")))
            with torch.inference_mode():
                ids, _classes, boxes = predictor._model.predict(
                    pixels,
                    config["predict"]["max_steps"],
                    config["predict"]["beam_size"],
                )
            _sync_torch(device)
            xyxy = (
                table_utils.box_cxcywh_to_xyxy(boxes).detach().cpu().numpy()
                if len(boxes)
                else np.empty((0, 4))
            )
            return _canonical_table(predictor._get_html_tags(ids), xyxy, image, normalized=True)

        metadata = _source_metadata("table_v1") | {"profile": mode}
        return predict_v1, metadata

    from docling_ibm_models.tableformer_v2 import TableFormerV2
    from torchvision import transforms
    from transformers import AutoTokenizer

    source = _require_directory(SOURCES["table_v2"][2])
    model_factory: Any = TableFormerV2
    model: Any = (
        model_factory.from_pretrained(
            source,
            local_files_only=True,
            use_safetensors=True,
            dtype=torch.float32,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    transform = transforms.Compose(
        [
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    cell_tokens = {"fcel", "ecel", "ched", "rhed", "srow"}

    def predict_v2(image: Image.Image) -> dict[str, Any]:
        pixels = transform(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.inference_mode():
            output = model.generate(pixels, tokenizer, max_length=512)
        _sync_torch(device)
        ids = output["generated_ids"][0].detach().cpu().tolist()
        tokens = [str(tokenizer.convert_ids_to_tokens(token_id)).strip("<>") for token_id in ids]
        otsl = [token for token in tokens if token not in {"pad", "[UNK]", "start", "end"}]
        count = sum(token in cell_tokens for token in otsl)
        boxes = output["predicted_bboxes"][0, :count].detach().cpu().numpy()
        return _canonical_table(otsl, boxes, image, normalized=True)

    return predict_v2, _source_metadata("table_v2")


def _canonical_granite_task(prompt: str, text: str) -> Any:
    del prompt
    return text


def _granite_otsl_text(text: str) -> str:
    value = text.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1].strip()
    if value.startswith("<otsl>") and value.endswith("</otsl>"):
        value = value.removeprefix("<otsl>").removesuffix("</otsl>")
    return value


def _granite_otsl_tokens(text: str) -> list[str]:
    return re.findall(r"<(ecel|fcel|lcel|ucel|xcel|ched|rhed|srow|nl)>", text)


def _capture_mlx_generated_tokens(engine: Any, prompt_input: Any) -> tuple[Any, list[int]]:
    original_stream_generate = engine.stream_generate
    generated_token_ids: list[int] = []

    def recording_stream(*args: Any, **kwargs: Any) -> Iterable[Any]:
        for chunk in original_stream_generate(*args, **kwargs):
            if chunk.token is not None:
                if chunk.generation_tokens == len(generated_token_ids) + 1:
                    generated_token_ids.append(int(chunk.token))
                elif chunk.generation_tokens != len(generated_token_ids):
                    raise RuntimeError(
                        "Unexpected mlx-vlm token stream: "
                        f"generation_tokens={chunk.generation_tokens}, "
                        f"captured={len(generated_token_ids)}"
                    )
            yield chunk

    engine.stream_generate = recording_stream
    try:
        return engine.predict(prompt_input), generated_token_ids
    finally:
        engine.stream_generate = original_stream_generate


def _first_divergent_granite_token(
    reference_ids: list[int], candidate_ids: list[int]
) -> int | None:
    for position, (expected, actual) in enumerate(
        zip(reference_ids, candidate_ids, strict=False), 1
    ):
        if expected != actual:
            return position
    if len(reference_ids) != len(candidate_ids):
        return min(len(reference_ids), len(candidate_ids)) + 1
    return None


def _granite_prompts(component: str) -> tuple[str, ...]:
    if component == "granite_table":
        return ("<tables_otsl>",)
    return ("<chart2csv>", "<chart2summary>", "<chart2code>")


def _mlx_granite_predictor(component: str) -> tuple[Predictor, dict[str, Any]]:
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.models.inference_engines.vlm.base import VlmEngineType

    from docling_mlx.stages._granite_vision import build_granite_vision_input
    from docling_mlx.stages.chart_extraction import MlxChartExtractionModelOptions
    from docling_mlx.stages.granite_vision_engine import MlxGraniteVision41Engine
    from docling_mlx.stages.table_structure import (
        MlxGraniteVisionTableStructureOptions,
    )

    options: Any
    if component == "granite_table":
        options = MlxGraniteVisionTableStructureOptions()
    else:
        options = MlxChartExtractionModelOptions(
            chart2csv=True, chart2summary=True, chart2code=True
        )
    spec = options.model_spec
    engine = MlxGraniteVision41Engine(
        options.engine_options,
        artifacts_path=_require_directory(ARTIFACTS["granite"]),
        model_config=spec.get_engine_config(VlmEngineType.MLX),
        accelerator_options=AcceleratorOptions(device="mps"),
    )
    engine.initialize()

    def predict(image: Image.Image) -> dict[str, Any]:
        tasks: dict[str, Any] = {}
        task_timings: dict[str, float] = {}
        for prompt in _granite_prompts(component):
            started = time.perf_counter()
            output, generated_token_ids = _capture_mlx_generated_tokens(
                engine, build_granite_vision_input(spec, image, prompt=prompt)
            )
            wall_ms = (time.perf_counter() - started) * 1000
            token_count = len(generated_token_ids)
            generation_seconds = float(output.metadata.get("generation_time", 0.0))
            tasks[prompt] = {
                "text": output.text,
                "canonical": _canonical_granite_task(prompt, output.text),
                "generated_tokens": token_count,
                "generated_token_ids": generated_token_ids,
                "generation_seconds": generation_seconds,
                "tokens_per_second": (
                    token_count / generation_seconds if generation_seconds else None
                ),
                "stop_reason": output.stop_reason,
            }
            if component == "granite_table":
                tasks[prompt] |= {
                    "otsl": _granite_otsl_text(output.text),
                    "otsl_structure_tokens": _granite_otsl_tokens(output.text),
                }
            task_timings[prompt] = wall_ms
        return {"tasks": tasks, "_task_timings_ms": task_timings}

    metadata = _source_metadata("granite") | {
        "artifact_root": str(ARTIFACTS["granite"]),
        "max_new_tokens": spec.max_new_tokens,
        "api": "MlxGraniteVision41Engine",
    }
    return predict, metadata


def _official_granite_predictor(component: str, device: str) -> tuple[Predictor, dict[str, Any]]:
    import torch

    stage: Any | None = None
    processor: Any
    model: Any
    if device == "cpu":
        from docling.datamodel.accelerator_options import AcceleratorOptions

        if component == "granite_table":
            from docling.datamodel.pipeline_options import GraniteVisionTableStructureOptions
            from docling.models.stages.table_structure.table_structure_model_granite_vision import (
                GraniteVisionTableStructureModel,
            )

            stage = GraniteVisionTableStructureModel(
                enabled=True,
                artifacts_path=ARTIFACTS["granite"],
                options=GraniteVisionTableStructureOptions(),
                accelerator_options=AcceleratorOptions(device="cpu", num_threads=4),
            )
        else:
            from docling.datamodel.chart_extraction_options import ChartExtractionModelOptions
            from docling.models.stages.chart_extraction.granite_vision import (
                ChartExtractionModelGraniteVisionV4,
            )

            stage = ChartExtractionModelGraniteVisionV4(
                enabled=True,
                artifacts_path=ARTIFACTS["granite"],
                options=ChartExtractionModelOptions(
                    chart2csv=True, chart2summary=True, chart2code=True
                ),
                accelerator_options=AcceleratorOptions(device="cpu", num_threads=4),
            )
        processor = stage._processor
        model = stage._model
    elif device == "mps":
        import warnings

        from transformers import AutoModelForImageTextToText, AutoProcessor

        artifact = _require_directory(SOURCES["granite"][2])
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*torch_dtype.*deprecated.*", category=UserWarning
            )
            warnings.filterwarnings(
                "ignore", message=".*incorrect regex pattern.*", category=UserWarning
            )
            processor = AutoProcessor.from_pretrained(artifact, trust_remote_code=True)
            model = AutoModelForImageTextToText.from_pretrained(
                artifact,
                device_map="mps",
                dtype=torch.bfloat16,
                _attn_implementation="sdpa",
                trust_remote_code=True,
            )
        if hasattr(model, "merge_lora_adapters"):
            cast(Any, model).merge_lora_adapters()
        model.eval()
    else:
        raise ValueError(f"Unsupported official Granite device: {device}")

    def predict(image: Image.Image) -> dict[str, Any]:
        tasks: dict[str, Any] = {}
        task_timings: dict[str, float] = {}
        for prompt in _granite_prompts(component):
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text],
                images=[image.convert("RGB")],
                return_tensors="pt",
                padding=True,
                do_pad=True,
            ).to(device)
            input_length = int(inputs["input_ids"].shape[1])
            started = time.perf_counter()
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=OFFICIAL_GRANITE_MAX_NEW_TOKENS,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                )
            _sync_torch(device)
            generation_seconds = time.perf_counter() - started
            generated_ids = output_ids[0, input_length:]
            generated_tokens = int(generated_ids.shape[0])
            generated_token_ids = [int(token_id) for token_id in generated_ids.tolist()]
            output_text = processor.decode(generated_ids, skip_special_tokens=True)
            task_timings[prompt] = generation_seconds * 1000
            tasks[prompt] = {
                "text": output_text,
                "canonical": _canonical_granite_task(prompt, output_text),
                "generated_tokens": generated_tokens,
                "generated_token_ids": generated_token_ids,
                "generation_seconds": generation_seconds,
                "tokens_per_second": (
                    generated_tokens / generation_seconds if generation_seconds else None
                ),
                "stop_reason": None,
            }
            if component == "granite_table":
                tasks[prompt] |= {
                    "otsl": _granite_otsl_text(output_text),
                    "otsl_structure_tokens": _granite_otsl_tokens(output_text),
                }
        return {"tasks": tasks, "_task_timings_ms": task_timings}

    metadata: dict[str, Any] = _source_metadata("granite") | {
        "artifact_root": str(ARTIFACTS["granite"]),
        "max_new_tokens": OFFICIAL_GRANITE_MAX_NEW_TOKENS,
        "generation": "greedy",
    }
    if stage is not None:
        metadata |= {
            "official_stage_max_new_tokens": str(stage._model_max_length),
            "api": type(stage).__name__,
            "device_restriction": "official Docling Granite stages support CPU/CUDA only",
        }
    else:
        metadata |= {
            "official_stage_max_new_tokens": str(processor.tokenizer.model_max_length),
            "api": "AutoModelForImageTextToText",
            "official_stage_device_gate_bypassed": True,
            "official_stage_supported_devices": "cpu,cuda",
        }
    return predict, metadata


def _make_predictor(
    component: str, implementation: str, device: str
) -> tuple[Predictor, dict[str, Any]]:
    if implementation == "mlx":
        if device != "mps":
            raise ValueError("The MLX implementation requires --device mps")
        return _mlx_predictor(component)
    if implementation == "official-mps":
        if component not in GRANITE_COMPONENTS or device != "mps":
            raise ValueError("official-mps is available only for Granite with --device mps")
        return _official_granite_predictor(component, device)
    if component in LAYOUT_COMPONENTS or component == "figure":
        return _official_vision_predictor(component, device)
    if component in TABLE_COMPONENTS:
        return _official_table_predictor(component, device)
    if component in GRANITE_COMPONENTS:
        if device != "cpu":
            raise ValueError("Use official-mps for the bypassed Granite MPS path")
        return _official_granite_predictor(component, device)
    raise ValueError(f"Unsupported component: {component}")


def _pipeline_snapshot(result: Any) -> dict[str, Any]:
    exported = result.document.export_to_dict()
    clusters = []
    pages = getattr(result, "pages", {})
    page_values = pages.values() if hasattr(pages, "values") else pages
    for page in page_values:
        layout = getattr(getattr(page, "predictions", None), "layout", None)
        for cluster in getattr(layout, "clusters", []) if layout is not None else []:
            bbox = getattr(cluster, "bbox", None)
            clusters.append(
                {
                    "label": _normalize_label(getattr(cluster, "label", None)),
                    "bbox": (
                        None
                        if bbox is None
                        else [float(bbox.l), float(bbox.t), float(bbox.r), float(bbox.b)]
                    ),
                }
            )
    structures = []
    structural_keys = (
        "row_span",
        "col_span",
        "start_row_offset_idx",
        "end_row_offset_idx",
        "start_col_offset_idx",
        "end_col_offset_idx",
        "column_header",
        "row_header",
        "row_section",
    )
    for table in exported.get("tables", []):
        provenance = (table.get("prov") or [{}])[0]
        data = table.get("data") or {}
        structures.append(
            {
                "bbox": _ordered_box(provenance.get("bbox")),
                "num_rows": data.get("num_rows"),
                "num_cols": data.get("num_cols"),
                "cells": [
                    {key: cell.get(key) for key in structural_keys}
                    for cell in data.get("table_cells", [])
                ],
            }
        )
    return {
        "status": getattr(getattr(result, "status", None), "value", str(result.status)),
        "markdown": result.document.export_to_markdown(),
        "layout_clusters": clusters,
        "tables": structures,
    }


def _pipeline_predictor(
    implementation: str, device: str
) -> tuple[Callable[[dict[str, Any]], dict[str, Any]], dict[str, Any]]:
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import LayoutObjectDetectionOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
    from docling_core.types.io import DocumentStream

    from docling_mlx.pipeline import MlxStandardPdfPipeline
    from docling_mlx.stages.layout import (
        MlxLayoutObjectDetectionOptions,
        MlxObjectDetectionEngineOptions,
    )
    from tools.compare_dpbench import (
        build_mlx_pipeline_options,
        build_official_pipeline_options,
    )

    accelerator = AcceleratorOptions(device=device, num_threads=4)
    common: dict[str, Any] = {
        "accelerator_options": accelerator,
        "do_ocr": False,
        "do_chart_extraction": False,
        "do_code_enrichment": False,
        "do_formula_enrichment": False,
        "do_picture_description": False,
    }
    if implementation == "mlx":
        if device != "mps":
            raise ValueError("The MLX pipeline requires --device mps")
        options = build_mlx_pipeline_options(ROOT / ".artifacts")
        common["layout_options"] = MlxLayoutObjectDetectionOptions.from_preset(
            "layout_heron_default",
            engine_options=MlxObjectDetectionEngineOptions(),
        )
        pipeline_class: type[Any] = MlxStandardPdfPipeline
        artifact_metadata: dict[str, Any] = {
            "artifacts_path": str(ROOT / ".artifacts"),
            "layout": _preset_metadata("heron_r50", ARTIFACTS["heron_r50"]),
            "table": _preset_metadata("table_v2", ARTIFACTS["table_v2"]),
            "figure": _preset_metadata("figure", ARTIFACTS["figure"]),
        }
    else:
        temporary = tempfile.TemporaryDirectory(prefix="docling-backend-pipeline-")
        artifacts_root = Path(temporary.name)
        for key in ("heron_r50", "table_v2", "figure"):
            repo_id, _revision, source = SOURCES[key]
            (artifacts_root / repo_id.replace("/", "--")).symlink_to(
                _require_directory(source), target_is_directory=True
            )
        options = build_official_pipeline_options(artifacts_root)
        common["layout_options"] = LayoutObjectDetectionOptions.from_preset("layout_heron_default")
        pipeline_class = StandardPdfPipeline
        artifact_metadata = {
            "artifacts_path": str(artifacts_root),
            "layout": _source_metadata("heron_r50"),
            "table": _source_metadata("table_v2"),
            "figure": _source_metadata("figure"),
            "_temporary": temporary,
        }
    options = options.model_copy(deep=True, update=common)
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=pipeline_class,
                pipeline_options=options,
            )
        },
    )

    def predict(record: dict[str, Any]) -> dict[str, Any]:
        pdf_path = Path(record["pdf"])
        result = converter.convert(
            DocumentStream(name=pdf_path.name, stream=BytesIO(pdf_path.read_bytes()))
        )
        return _pipeline_snapshot(result)

    return predict, {
        "configuration": (
            "layout_heron_default + TableFormerV2 + DocumentFigure; "
            "OCR/chart/code/formula/picture-description disabled"
        ),
        **artifact_metadata,
    }


def _ensure_table_ground_truth_otsl(inputs_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    table_records = [
        record
        for collection in ("tables", "granite_tables")
        for record in manifest.get(collection, [])
    ]
    if all(isinstance(record.get("ground_truth_otsl"), str) for record in table_records):
        return manifest

    import pyarrow.parquet as pq
    from docling_core.types.doc import DoclingDocument

    documents = {
        str(row["document_id"]): json.loads(row["GroundTruthDocument"])
        for row in pq.read_table(_dataset_parquet()).to_pylist()
    }
    parsed: dict[str, Any] = {}
    for record in table_records:
        if isinstance(record.get("ground_truth_otsl"), str):
            continue
        document_id = str(record["document_id"])
        if document_id not in parsed:
            try:
                parsed[document_id] = DoclingDocument.model_validate(documents[document_id])
            except KeyError as error:
                raise ValueError(f"DPBench ground truth missing {document_id}") from error
        document = parsed[document_id]
        record["ground_truth_otsl"] = document.tables[int(record["item_index"])].export_to_otsl(
            document, add_cell_location=False
        )
    _write_json(inputs_root / "manifest.json", manifest)
    return manifest


def _manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path / "manifest.json")
    if manifest.get("dataset", {}).get("revision") != DATASET_REVISION:
        raise ValueError(f"Manifest is not pinned to DPBench {DATASET_REVISION}")
    return _ensure_table_ground_truth_otsl(path, manifest)


def _select_granite_charts(
    inputs_root: Path,
    manifest: dict[str, Any],
    figure_outputs: dict[str, Any],
) -> None:
    supported = {"bar_chart", "line_chart", "pie_chart"}
    ranked: list[tuple[float, int, dict[str, Any], str]] = []
    for position, record in enumerate(manifest["pictures"]):
        probabilities = figure_outputs[record["id"]]["probabilities"]
        label = max(probabilities, key=probabilities.get)
        if label in supported:
            ranked.append((float(probabilities[label]), position, record, label))
    ranked.sort(key=lambda value: (-value[0], value[1]))
    if len(ranked) < 5:
        raise ValueError(f"DocumentFigure selected only {len(ranked)} chart pictures")
    manifest["granite_charts"] = [
        record
        | {
            "classifier_top1": label,
            "classifier_confidence": confidence,
            "selection_rank": rank,
        }
        for rank, (confidence, _position, record, label) in enumerate(ranked[:5], 1)
    ]
    manifest["granite_chart_selection"] = {
        "implementation": "mlx",
        "supported_top1_labels": sorted(supported),
        "rule": "supported top-1 predictions ordered by descending confidence, then dataset order",
    }
    _write_json(inputs_root / "manifest.json", manifest)


def _input_records(
    component: str, inputs_root: Path, manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    if component in LAYOUT_COMPONENTS or component == "pipeline":
        return list(manifest["pages"])
    if component == "figure":
        return list(manifest["pictures"])
    if component in TABLE_COMPONENTS:
        return list(manifest["tables"])
    if component == "granite_table":
        return list(manifest["granite_tables"])
    records = list(manifest.get("granite_charts", []))
    if not records:
        figure_path = DEFAULT_REPORTS / "figure.mlx.json"
        if figure_path.is_file():
            _select_granite_charts(inputs_root, manifest, _read_json(figure_path)["outputs"])
            records = list(manifest["granite_charts"])
    if not records:
        raise ValueError("Run the MLX figure component before granite_chart")
    return records


def _clean_output(output: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    task_timings = output.pop("_task_timings_ms", {})
    return output, task_timings


def _run(args: argparse.Namespace) -> None:
    """Run one component/backend pair in this fresh process."""

    inputs_root = Path(args.inputs).resolve()
    manifest = _manifest(inputs_root)
    records = _input_records(args.component, inputs_root, manifest)
    if not records:
        raise ValueError(f"No inputs for {args.component}")

    images: dict[str, Image.Image] = {}
    if args.component != "pipeline":
        for record in records:
            with Image.open(record["image"]) as source:
                images[record["id"]] = source.convert("RGB")

    overall_started = time.perf_counter()
    predictor: Callable[[Any], dict[str, Any]]
    first_started = time.perf_counter()
    if args.component == "pipeline":
        pipeline_predictor, artifact_metadata = _pipeline_predictor(
            args.implementation, args.device
        )
        predictor = cast(Callable[[Any], dict[str, Any]], pipeline_predictor)
        first_output = predictor(records[0])
    else:
        image_predictor, artifact_metadata = _make_predictor(
            args.component, args.implementation, args.device
        )
        predictor = cast(Callable[[Any], dict[str, Any]], image_predictor)
        first_output = predictor(images[records[0]["id"]])
    first_call_ms = (time.perf_counter() - first_started) * 1000
    del first_output

    outputs: dict[str, Any] = {}
    rounds: list[dict[str, Any]] = []
    all_timings: list[float] = []
    for round_index in range(args.rounds):
        round_started = time.perf_counter()
        item_timings = []
        for record in records:
            started = time.perf_counter()
            if args.component == "pipeline":
                output = predictor(record)
            else:
                output = predictor(images[record["id"]])
            elapsed_ms = (time.perf_counter() - started) * 1000
            clean, task_timings = _clean_output(output)
            if args.component == "granite_table":
                clean["ground_truth_otsl"] = record["ground_truth_otsl"]
            outputs[record["id"]] = clean
            timing: dict[str, Any] = {"id": record["id"], "wall_ms": elapsed_ms}
            if task_timings:
                timing["task_ms"] = task_timings
            item_timings.append(timing)
            all_timings.append(elapsed_ms)
        rounds.append(
            {
                "round": round_index + 1,
                "wall_ms": (time.perf_counter() - round_started) * 1000,
                "items": item_timings,
            }
        )

    report = {
        "schema": BENCHMARK_SCHEMA,
        "status": "complete",
        "component": args.component,
        "implementation": args.implementation,
        "device": "mlx-metal" if args.implementation == "mlx" else f"torch-{args.device}",
        "input_count": len(records),
        "input_ids": [record["id"] for record in records],
        "warmup_calls": 1,
        "timed_rounds": args.rounds,
        "first_call_ms": first_call_ms,
        "warm_per_item_ms": {
            "median": statistics.median(all_timings),
            "mean": statistics.fmean(all_timings),
            "min": min(all_timings),
            "max": max(all_timings),
        },
        "process_peak_rss_bytes": peak_rss_bytes(),
        "wall_clock_seconds": time.perf_counter() - overall_started,
        "rounds": rounds,
        "outputs": outputs,
        "metadata": _runtime_metadata()
        | {key: value for key, value in artifact_metadata.items() if not key.startswith("_")},
    }
    _write_json(Path(args.output), report)
    if args.component == "figure" and args.implementation == "mlx":
        _select_granite_charts(inputs_root, manifest, outputs)
    print(f"Wrote {args.component}/{args.implementation} ({len(records)} inputs) to {args.output}")


def run(args: argparse.Namespace) -> None:
    """Run one component/backend pair, retaining optional MPS Granite failures."""

    try:
        _run(args)
    except Exception as error:
        if args.component not in GRANITE_COMPONENTS or args.implementation != "official-mps":
            raise
        report = {
            "schema": BENCHMARK_SCHEMA,
            "status": "failed",
            "component": args.component,
            "implementation": args.implementation,
            "device": f"torch-{args.device}",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "metadata": _runtime_metadata()
            | {
                "max_new_tokens": OFFICIAL_GRANITE_MAX_NEW_TOKENS,
                "official_stage_device_gate_bypassed": True,
            },
        }
        _write_json(Path(args.output), report)
        print(
            f"Wrote failed {args.component}/{args.implementation} to {args.output}: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )


def _ordered_box(value: Any) -> list[float] | None:
    if not isinstance(value, (dict, list, tuple)):
        return None
    try:
        if isinstance(value, dict):
            left, top, right, bottom = (float(value[key]) for key in ("l", "t", "r", "b"))
        else:
            left, top, right, bottom = (float(item) for item in value)
    except (KeyError, TypeError, ValueError):
        return None
    return [min(left, right), min(top, bottom), max(left, right), max(top, bottom)]


def _iou(left_value: Any, right_value: Any) -> float:
    left = _ordered_box(left_value)
    right = _ordered_box(right_value)
    if left is None or right is None:
        return 0.0
    overlap = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )
    union = (
        (left[2] - left[0]) * (left[3] - left[1])
        + (right[2] - right[0]) * (right[3] - right[1])
        - overlap
    )
    return overlap / union if union else 0.0


def _match_boxes(
    candidate: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    *,
    require_label: bool,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any], float]], int, int]:
    matches = []
    available = set(range(len(candidate)))
    for expected in reference:
        choices = (
            (
                _iou(
                    candidate[index].get("bbox", candidate[index]), expected.get("bbox", expected)
                ),
                index,
            )
            for index in available
            if not require_label or candidate[index].get("label") == expected.get("label")
        )
        score, index = max(choices, default=(0.0, -1))
        if score >= 0.5:
            matches.append((candidate[index], expected, score))
            available.remove(index)
    return matches, len(available), len(reference) - len(matches)


def _match_detections(
    candidate: list[dict[str, Any]], reference: list[dict[str, Any]]
) -> tuple[
    list[tuple[dict[str, Any], dict[str, Any], float]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    matches = []
    available = set(range(len(candidate)))
    unmatched_reference = []
    for expected in reference:
        score, _negative_index, index = max(
            ((_iou(candidate[item], expected), -item, item) for item in available),
            default=(0.0, 0, -1),
        )
        if score >= 0.5:
            matches.append((candidate[index], expected, score))
            available.remove(index)
        else:
            unmatched_reference.append(expected)
    return matches, [candidate[index] for index in sorted(available)], unmatched_reference


def _same_box_label_swap(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    candidates: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> bool:
    if candidate["label"] == reference["label"]:
        return False
    return any(
        other is not candidate
        and other["label"] == reference["label"]
        and _iou(other, reference) >= 0.9
        for other in candidates
    )


def _threshold_boundary(detection: dict[str, Any], threshold: float) -> bool:
    return abs(float(detection["score"]) - threshold) <= LAYOUT_SCORE_THRESHOLD_MARGIN


def _detection_detail(detection: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": detection["label"],
        "score": detection["score"],
        "bbox": [detection[key] for key in ("l", "t", "r", "b")],
    }


def detection_quality(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    *,
    candidate_threshold: float,
    reference_threshold: float,
) -> dict[str, Any]:
    matches = []
    unmatched_candidate: list[dict[str, Any]] = []
    unmatched_reference: list[dict[str, Any]] = []
    same_label_matches = same_box_label_swaps = label_disagreements = 0
    for name, expected in reference.items():
        detections = candidate.get(name, {}).get("detections", [])
        references = expected["detections"]
        found, extra, missing = _match_detections(detections, references)
        matches.extend(found)
        unmatched_candidate.extend(extra)
        unmatched_reference.extend(missing)
        for actual, expected_detection, _score in found:
            if actual["label"] == expected_detection["label"]:
                same_label_matches += 1
            elif _same_box_label_swap(actual, expected_detection, detections, references):
                same_box_label_swaps += 1
            else:
                label_disagreements += 1

    boundary_candidate = [
        detection
        for detection in unmatched_candidate
        if _threshold_boundary(detection, candidate_threshold)
    ]
    boundary_reference = [
        detection
        for detection in unmatched_reference
        if _threshold_boundary(detection, reference_threshold)
    ]
    other_candidate = [
        detection
        for detection in unmatched_candidate
        if not _threshold_boundary(detection, candidate_threshold)
    ]
    other_reference = [
        detection
        for detection in unmatched_reference
        if not _threshold_boundary(detection, reference_threshold)
    ]
    non_swap_matches = len(matches) - same_box_label_swaps
    ious = [match[2] for match in matches]
    score_deltas = [abs(match[0]["score"] - match[1]["score"]) for match in matches]
    return {
        "label_agreement_iou50_excluding_same_box_swaps": (
            same_label_matches / non_swap_matches if non_swap_matches else None
        ),
        "mean_iou_matched": statistics.fmean(ious) if ious else None,
        "mean_abs_delta_score": statistics.fmean(score_deltas) if score_deltas else None,
        "matched": len(matches),
        "same_box_label_swaps": same_box_label_swaps,
        "label_disagreements": label_disagreements,
        "score_threshold_mlx": candidate_threshold,
        "score_threshold_official": reference_threshold,
        "threshold_boundary_unmatched_mlx": len(boundary_candidate),
        "threshold_boundary_unmatched_official": len(boundary_reference),
        "other_unmatched_mlx": len(other_candidate),
        "other_unmatched_official": len(other_reference),
        "other_unmatched_details": {
            "mlx": [_detection_detail(detection) for detection in other_candidate],
            "official": [_detection_detail(detection) for detection in other_reference],
        },
    }


def classification_quality(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    top1 = 0
    deltas: list[float] = []
    for name, expected in reference.items():
        expected_probabilities = expected["probabilities"]
        probabilities = candidate[name]["probabilities"]
        top1 += max(probabilities, key=probabilities.get) == max(
            expected_probabilities, key=expected_probabilities.get
        )
        deltas.extend(
            abs(probabilities.get(label, 0.0) - expected_probabilities.get(label, 0.0))
            for label in set(probabilities) | set(expected_probabilities)
        )
    count = len(reference)
    return {
        "top1_agreement": top1 / count if count else None,
        "mean_abs_delta_prob": statistics.fmean(deltas) if deltas else None,
        "max_abs_delta_prob": max(deltas) if deltas else None,
    }


@dataclass(frozen=True)
class _Node:
    label: str
    children: tuple[_Node, ...] = ()


def _otsl_tree(tokens: list[str]) -> _Node:
    rows: list[_Node] = []
    cells: list[_Node] = []
    for token in tokens:
        if token == "nl":
            rows.append(_Node("tr", tuple(cells)))
            cells = []
        else:
            cells.append(_Node(token))
    if cells:
        rows.append(_Node("tr", tuple(cells)))
    return _Node("table", tuple(rows))


def _tree_size(node: _Node) -> int:
    return 1 + sum(_tree_size(child) for child in node.children)


def _tree_distance(left: _Node, right: _Node) -> int:
    @cache
    def distance(first: _Node, second: _Node) -> int:
        rows, columns = len(first.children), len(second.children)
        costs = [[0] * (columns + 1) for _ in range(rows + 1)]
        for row in range(1, rows + 1):
            costs[row][0] = costs[row - 1][0] + _tree_size(first.children[row - 1])
        for column in range(1, columns + 1):
            costs[0][column] = costs[0][column - 1] + _tree_size(second.children[column - 1])
        for row in range(1, rows + 1):
            for column in range(1, columns + 1):
                costs[row][column] = min(
                    costs[row - 1][column] + _tree_size(first.children[row - 1]),
                    costs[row][column - 1] + _tree_size(second.children[column - 1]),
                    costs[row - 1][column - 1]
                    + distance(first.children[row - 1], second.children[column - 1]),
                )
        return int(first.label != second.label) + costs[rows][columns]

    return distance(left, right)


def _tree_teds(left_tokens: list[str], right_tokens: list[str]) -> float:
    left, right = _otsl_tree(left_tokens), _otsl_tree(right_tokens)
    return 1.0 - _tree_distance(left, right) / max(_tree_size(left), _tree_size(right))


def table_quality(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    exact = 0
    teds = []
    ious = []
    unmatched_candidate = unmatched_reference = 0
    for name, expected in reference.items():
        value = candidate[name]
        exact += value["otsl"] == expected["otsl"]
        teds.append(_tree_teds(value["otsl"], expected["otsl"]))
        available = set(range(len(value["cell_bboxes"])))
        for box in expected["cell_bboxes"]:
            score, index = max(
                ((_iou(value["cell_bboxes"][item], box), item) for item in available),
                default=(0.0, -1),
            )
            if score >= 0.5:
                ious.append(score)
                available.remove(index)
            else:
                unmatched_reference += 1
        unmatched_candidate += len(available)
    count = len(reference)
    return {
        "otsl_exact_sequence_agreement": exact / count if count else None,
        "tree_teds": statistics.fmean(teds) if teds else None,
        "cell_bbox_mean_iou": statistics.fmean(ious) if ious else None,
        "cell_bbox_unmatched_mlx": unmatched_candidate,
        "cell_bbox_unmatched_official": unmatched_reference,
    }


def _granite_pair_quality(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    *,
    candidate_name: str,
    reference_name: str,
) -> tuple[dict[str, Any], list[str]]:
    prompts = tuple(next(iter(reference.values()))["tasks"])
    matches = {prompt: 0 for prompt in prompts}
    differences = []
    positions = []
    for name, expected in reference.items():
        for prompt in prompts:
            actual_task = candidate[name]["tasks"][prompt]
            expected_task = expected["tasks"][prompt]
            actual_text = actual_task["text"]
            expected_text = expected_task["text"]
            matches[prompt] += actual_text == expected_text
            if actual_text != expected_text:
                if isinstance(expected_task.get("generated_token_ids"), list) and isinstance(
                    actual_task.get("generated_token_ids"), list
                ):
                    position = _first_divergent_granite_token(
                        expected_task["generated_token_ids"], actual_task["generated_token_ids"]
                    )
                    position_method = "generated_token_id"
                else:
                    position = next(
                        (
                            index
                            for index, (expected_char, actual_char) in enumerate(
                                zip(expected_text, actual_text, strict=False), 1
                            )
                            if expected_char != actual_char
                        ),
                        min(len(expected_text), len(actual_text)) + 1,
                    )
                    position_method = "character"
                positions.append(
                    {
                        "id": name,
                        "prompt": prompt,
                        "position": position,
                        "method": position_method,
                    }
                )
                divergence = (
                    f"First divergent {position_method.replace('_', ' ')} (1-based): "
                    f"{position if position is not None else 'unavailable'}."
                )
                differences.append(
                    divergence
                    + "\n"
                    + "\n".join(
                        difflib.unified_diff(
                            expected_text.splitlines(),
                            actual_text.splitlines(),
                            fromfile=f"{reference_name}/{name}/{prompt}",
                            tofile=f"{candidate_name}/{name}/{prompt}",
                            lineterm="",
                        )
                    )
                )
    count = len(reference)
    return (
        {
            "identity_rates": {
                prompt.strip("<>") + "_identity_rate": matches[prompt] / count for prompt in prompts
            },
            "first_divergent_positions": positions,
        },
        differences,
    )


def _granite_table_ground_truth_quality(outputs: dict[str, Any]) -> dict[str, Any]:
    exact = 0
    teds = []
    per_item = []
    for name, output in outputs.items():
        expected = _granite_otsl_text(str(output["ground_truth_otsl"]))
        task = output["tasks"]["<tables_otsl>"]
        actual = str(task.get("otsl", _granite_otsl_text(task["text"])))
        expected_tokens = _granite_otsl_tokens(expected)
        actual_tokens = list(task.get("otsl_structure_tokens", _granite_otsl_tokens(actual)))
        is_exact = actual == expected
        teds_value = _tree_teds(actual_tokens, expected_tokens)
        exact += is_exact
        teds.append(teds_value)
        per_item.append({"id": name, "exact": is_exact, "tree_teds": teds_value})
    count = len(outputs)
    return {
        "exact_gt_otsl_identity": exact / count if count else None,
        "gt_tree_teds": statistics.fmean(teds) if teds else None,
        "per_item": per_item,
    }


def granite_quality(
    component: str,
    candidate: dict[str, Any],
    reference: dict[str, Any],
    official_mps: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    mlx_quality, differences = _granite_pair_quality(
        candidate,
        reference,
        candidate_name="mlx",
        reference_name="official-cpu",
    )
    quality: dict[str, Any] = {"mlx_vs_official_cpu": mlx_quality}
    if component == "granite_table":
        quality["ground_truth"] = {
            "mlx": _granite_table_ground_truth_quality(candidate),
            "official_cpu": _granite_table_ground_truth_quality(reference),
        }
    if official_mps is None:
        return quality, differences
    if official_mps.get("status") == "failed":
        quality["official_mps_error"] = (
            f"{official_mps['error_type']}: {official_mps['error_message']}"
        )
        return quality, differences

    mps_quality, mps_differences = _granite_pair_quality(
        official_mps["outputs"],
        reference,
        candidate_name="official-mps",
        reference_name="official-cpu",
    )
    quality["official_mps_vs_official_cpu"] = mps_quality
    if component == "granite_table":
        quality["ground_truth"]["official_mps"] = _granite_table_ground_truth_quality(
            official_mps["outputs"]
        )
    return quality, differences + mps_differences


def pipeline_quality(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    markdown_matches = table_matches = matched_tables = 0
    layout_matches = layout_reference = 0
    layout_unmatched_mlx = layout_unmatched_official = 0
    table_unmatched_mlx = table_unmatched_official = 0
    for name, expected in reference.items():
        value = candidate[name]
        markdown_matches += value["markdown"] == expected["markdown"]
        found, extra, missing = _match_boxes(
            value["layout_clusters"], expected["layout_clusters"], require_label=True
        )
        layout_matches += len(found)
        layout_reference += len(expected["layout_clusters"])
        layout_unmatched_mlx += extra
        layout_unmatched_official += missing
        tables, extra, missing = _match_boxes(
            value["tables"], expected["tables"], require_label=False
        )
        matched_tables += len(tables)
        table_matches += sum(left["cells"] == right["cells"] for left, right, _ in tables)
        table_unmatched_mlx += extra
        table_unmatched_official += missing
    count = len(reference)
    return {
        "markdown_identity_rate": markdown_matches / count if count else None,
        "layout_cluster_agreement_iou50": (
            layout_matches / layout_reference if layout_reference else None
        ),
        "layout_unmatched_mlx": layout_unmatched_mlx,
        "layout_unmatched_official": layout_unmatched_official,
        "table_structure_exact_agreement": (
            table_matches / matched_tables if matched_tables else None
        ),
        "table_unmatched_mlx": table_unmatched_mlx,
        "table_unmatched_official": table_unmatched_official,
    }


def _quality_for(
    component: str,
    mlx: dict[str, Any],
    official: dict[str, Any],
    official_mps: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    if component in LAYOUT_COMPONENTS:
        return (
            detection_quality(
                mlx["outputs"],
                official["outputs"],
                candidate_threshold=float(mlx["metadata"]["score_threshold"]),
                reference_threshold=float(official["metadata"]["score_threshold"]),
            ),
            [],
        )
    if component == "figure":
        return classification_quality(mlx["outputs"], official["outputs"]), []
    if component in TABLE_COMPONENTS:
        return table_quality(mlx["outputs"], official["outputs"]), []
    if component in GRANITE_COMPONENTS:
        return granite_quality(component, mlx["outputs"], official["outputs"], official_mps)
    return pipeline_quality(mlx["outputs"], official["outputs"]), []


def _format_rss(value: int) -> str:
    return f"{value / (1024**3):.2f} GiB"


def _granite_generation(report: dict[str, Any]) -> tuple[float, float]:
    tasks = [task for output in report["outputs"].values() for task in output["tasks"].values()]
    tokens = [float(task["generated_tokens"]) for task in tasks]
    seconds = sum(float(task["generation_seconds"]) for task in tasks)
    return statistics.median(tokens), sum(tokens) / seconds if seconds else 0.0


def _machine_line(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    versions = metadata["versions"]
    version_text = ", ".join(f"{name} {version}" for name, version in versions.items())
    return (
        f"Machine: {metadata['chip']}, {int(metadata['memory_bytes']) / (1024**3):.0f} GiB, "
        f"macOS {metadata['macos']}; Python {metadata['python']}; {version_text}; "
        f"commit `{metadata['git_commit']}`."
    )


def summarize(args: argparse.Namespace) -> None:
    """Compare paired JSON results and render the reusable Markdown tables."""

    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for path_text in args.inputs:
        path = Path(path_text)
        result = _read_json(path)
        if "outputs" not in result:
            if not (
                result.get("component") in GRANITE_COMPONENTS
                and result.get("implementation") == "official-mps"
                and result.get("status") == "failed"
            ):
                raise ValueError(f"Incomplete backend result: {path}")
        grouped.setdefault(result["component"], {})[result["implementation"]] = result

    lines = [
        "# Backend comparison",
        "",
        "One fresh Python process was used for every component and implementation at batch size 1; "
        "the measured construction plus first inference was the single warm-up call, followed by "
        "three timed rounds over the full input set (one round for official CPU Granite).",
        "",
    ]
    for component in COMPONENTS:
        pair = grouped.get(component, {})
        if not pair:
            continue
        if not {"mlx", "official"}.issubset(pair):
            raise ValueError(f"Missing MLX/official pair for {component}: {sorted(pair)}")
        mlx, official = pair["mlx"], pair["official"]
        official_mps = pair.get("official-mps")
        quality, differences = _quality_for(component, mlx, official, official_mps)
        lines += [f"## {component}", "", _machine_line(mlx), ""]
        if component in GRANITE_COMPONENTS:
            lines += [
                "| implementation | device | warm ms/item (median) | first-call ms | "
                "peak RSS | generated tokens/request (median) | tokens/s | quality vs official |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        else:
            lines += [
                "| implementation | device | warm ms/item (median) | first-call ms | "
                "peak RSS | quality vs official |",
                "| --- | --- | ---: | ---: | ---: | --- |",
            ]
        reports = [("mlx", mlx), ("official", official)]
        if (
            component in GRANITE_COMPONENTS
            and official_mps is not None
            and "outputs" in official_mps
        ):
            reports.append(("official-mps", official_mps))
        for implementation, report in reports:
            if implementation == "mlx":
                quality_text = json.dumps(quality, sort_keys=True)
            elif implementation == "official":
                quality_text = "reference"
            else:
                quality_text = json.dumps(
                    quality.get("official_mps_vs_official_cpu"), sort_keys=True
                )
            values = [
                implementation,
                report["device"],
                f"{report['warm_per_item_ms']['median']:.3f}",
                f"{report['first_call_ms']:.3f}",
                _format_rss(int(report["process_peak_rss_bytes"])),
            ]
            if component in GRANITE_COMPONENTS:
                generated_tokens, tokens_per_second = _granite_generation(report)
                values += [f"{generated_tokens:.0f}", f"{tokens_per_second:.3f}"]
            values.append(quality_text)
            lines.append("| " + " | ".join(values) + " |")
        if component in GRANITE_COMPONENTS and official_mps is not None:
            if official_mps.get("status") == "failed":
                lines += [
                    "",
                    "official-mps error: "
                    f"{official_mps['error_type']}: {official_mps['error_message']}",
                ]
        if component in LAYOUT_COMPONENTS:
            other = quality["other_unmatched_details"]
            for implementation, values in other.items():
                if values:
                    lines += [
                        "",
                        f"Other unmatched {implementation}: "
                        f"`{json.dumps(values, sort_keys=True)}`.",
                    ]
        if component == "granite_chart":
            chosen = [output_id.rsplit("-picture-", 1)[0] for output_id in mlx["input_ids"]]
            lines += [
                "",
                "Chosen chart document ids: " + ", ".join(f"`{item}`" for item in chosen) + ".",
            ]
        for difference in differences:
            lines += ["", "```diff", difference, "```"]
        lines.append("")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote summary to {args.output}")


def _successful_result(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        result = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        result.get("schema") == BENCHMARK_SCHEMA
        and result.get("status") == "complete"
        and isinstance(result.get("outputs"), dict)
    )


def run_all(args: argparse.Namespace) -> None:
    """Run the approved matrix serially, resuming completed JSON outputs."""

    inputs_root = Path(args.inputs).resolve()
    output_root = Path(args.output_dir).resolve()
    if not (inputs_root / "manifest.json").is_file():
        prepare(argparse.Namespace(inputs=inputs_root))
    output_root.mkdir(parents=True, exist_ok=True)
    failures = []
    results = []
    environment = os.environ | {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    for component in COMPONENTS:
        if component == "granite_chart":
            manifest = _manifest(inputs_root)
            if not manifest.get("granite_charts"):
                figure_result = output_root / "figure.mlx.json"
                if not _successful_result(figure_result):
                    failures.append("granite_chart: missing successful figure.mlx.json")
                    continue
                _select_granite_charts(inputs_root, manifest, _read_json(figure_result)["outputs"])
        implementations = (
            ("mlx", "official", "official-mps")
            if component in GRANITE_COMPONENTS
            else ("mlx", "official")
        )
        for implementation in implementations:
            output = output_root / f"{component}.{implementation}.json"
            results.append(output)
            if _successful_result(output):
                print(f"SKIP {component}/{implementation}: {output}", flush=True)
                continue
            device = (
                "cpu" if implementation == "official" and component in GRANITE_COMPONENTS else "mps"
            )
            rounds = (
                1
                if implementation == "official" and component in GRANITE_COMPONENTS
                else args.rounds
            )
            command = [
                sys.executable,
                "-m",
                "tools.compare_backends",
                "run",
                "--component",
                component,
                "--implementation",
                implementation,
                "--device",
                device,
                "--rounds",
                str(rounds),
                "--inputs",
                str(inputs_root),
                "--output",
                str(output),
            ]
            print(f"RUN {component}/{implementation} ({rounds} round(s))", flush=True)
            completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
            if completed.returncode:
                failures.append(f"{component}/{implementation}: exit {completed.returncode}")
    paired_inputs = [
        str(output_root / f"{component}.{implementation}.json")
        for component in COMPONENTS
        if all(
            _successful_result(output_root / f"{component}.{side}.json")
            for side in ("mlx", "official")
        )
        for implementation in ("mlx", "official")
    ]
    paired_inputs.extend(
        str(output_root / f"{component}.official-mps.json")
        for component in GRANITE_COMPONENTS
        if (output_root / f"{component}.official-mps.json").is_file()
    )
    if paired_inputs:
        summarize(argparse.Namespace(inputs=paired_inputs, output=args.summary))
    if failures:
        raise RuntimeError("Backend queue failures:\n" + "\n".join(failures))


def _positive_rounds(value: str) -> int:
    rounds = int(value)
    if rounds < 1:
        raise argparse.ArgumentTypeError("rounds must be positive")
    return rounds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare", help="materialize pinned DPBench inputs")
    prepare_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    prepare_parser.set_defaults(func=prepare)

    run_parser = commands.add_parser("run", help="run one fresh component/backend process")
    run_parser.add_argument("--component", choices=COMPONENTS, required=True)
    run_parser.add_argument(
        "--implementation", choices=("mlx", "official", "official-mps"), required=True
    )
    run_parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    run_parser.add_argument("--rounds", type=_positive_rounds, default=3)
    run_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.set_defaults(func=run)

    summary_parser = commands.add_parser("summarize", help="compare JSON pairs")
    summary_parser.add_argument("--inputs", nargs="+", required=True)
    summary_parser.add_argument("--output", type=Path, required=True)
    summary_parser.set_defaults(func=summarize)

    all_parser = commands.add_parser("all", help="run or resume the approved matrix")
    all_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    all_parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORTS)
    all_parser.add_argument("--rounds", type=_positive_rounds, default=3)
    all_parser.add_argument("--summary", type=Path, default=DEFAULT_REPORTS / "summary.md")
    all_parser.set_defaults(func=run_all)

    args = parser.parse_args()
    if (
        args.command == "run"
        and args.implementation == "official"
        and args.component in GRANITE_COMPONENTS
        and args.device == "cpu"
        and args.rounds != 1
    ):
        parser.error("official Granite runs exactly one timed round")
    args.func(args)


if __name__ == "__main__":
    main()
