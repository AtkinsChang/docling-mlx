# SPDX-License-Identifier: Apache-2.0

"""Compare official Docling and docling-mlx on Docling DPBench."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from collections import Counter
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions

from docling_mlx.pipeline import MlxStandardPdfPipeline
from docling_mlx.stages.chart_extraction import (
    MlxChartExtractionModelOptions,
)
from docling_mlx.stages.layout import (
    MlxLayoutObjectDetectionOptions,
    MlxObjectDetectionEngineOptions,
)
from docling_mlx.stages.picture_classification import MlxDocumentPictureClassifierOptions
from docling_mlx.stages.table_structure_v2 import MlxTableStructureV2Options

ROOT = Path(__file__).resolve().parents[1]
DATASET_REPO = "docling-project/docling-dpbench"
DATASET_REVISION = "8fb4a3b9a57ae119f27f42ce223dbc55f43f0e55"
DEFAULT_LIMIT = 10
EVAL_MODALITIES = (
    "layout",
    "table_structure",
    "document_structure",
    "markdown_text",
    "bboxes_text",
)


def build_official_pipeline_options(artifacts_path: Path | None) -> ThreadedPdfPipelineOptions:
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.chart_extraction_options import ChartExtractionModelOptions
    from docling.datamodel.picture_classification_options import (
        DocumentPictureClassifierOptions,
    )
    from docling.datamodel.pipeline_options import (
        CodeFormulaVlmOptions,
        LayoutObjectDetectionOptions,
        OcrMacOptions,
        PictureDescriptionVlmEngineOptions,
        TableStructureV2Options,
    )
    from docling.datamodel.vlm_engine_options import MlxVlmEngineOptions

    return ThreadedPdfPipelineOptions(
        artifacts_path=artifacts_path,
        accelerator_options=AcceleratorOptions(device="auto"),
        allow_external_plugins=False,
        do_ocr=True,
        ocr_options=OcrMacOptions(),
        layout_options=LayoutObjectDetectionOptions.from_preset("layout_heron_101"),
        do_table_structure=True,
        table_structure_options=TableStructureV2Options(do_cell_matching=True),
        do_picture_classification=True,
        picture_classification_options=DocumentPictureClassifierOptions.from_preset(
            "document_figure_classifier_v2"
        ),
        do_code_enrichment=True,
        do_formula_enrichment=True,
        code_formula_options=CodeFormulaVlmOptions.from_preset(
            "codeformulav2",
            engine_options=MlxVlmEngineOptions(),
        ),
        do_chart_extraction=True,
        chart_extraction_options=ChartExtractionModelOptions(
            chart2csv=True,
            chart2summary=True,
            chart2code=True,
        ),
        do_picture_description=True,
        picture_description_options=PictureDescriptionVlmEngineOptions.from_preset(
            "smolvlm",
            engine_options=MlxVlmEngineOptions(),
        ),
    )


def build_mlx_pipeline_options(
    artifacts_path: Path | None,
) -> ThreadedPdfPipelineOptions:
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.pipeline_options import (
        CodeFormulaVlmOptions,
        OcrMacOptions,
        PictureDescriptionVlmEngineOptions,
    )
    from docling.datamodel.vlm_engine_options import MlxVlmEngineOptions

    return ThreadedPdfPipelineOptions(
        artifacts_path=artifacts_path,
        accelerator_options=AcceleratorOptions(device="auto"),
        allow_external_plugins=True,
        do_ocr=True,
        ocr_options=OcrMacOptions(),
        layout_options=MlxLayoutObjectDetectionOptions.from_preset(
            "layout_heron_101",
            engine_options=MlxObjectDetectionEngineOptions(),
        ),
        do_table_structure=True,
        table_structure_options=MlxTableStructureV2Options(do_cell_matching=True),
        do_picture_classification=True,
        picture_classification_options=MlxDocumentPictureClassifierOptions(),
        do_code_enrichment=True,
        do_formula_enrichment=True,
        code_formula_options=CodeFormulaVlmOptions.from_preset(
            "codeformulav2",
            engine_options=MlxVlmEngineOptions(),
        ),
        do_chart_extraction=True,
        chart_extraction_options=MlxChartExtractionModelOptions(
            chart2csv=True,
            chart2summary=True,
            chart2code=True,
        ),
        do_picture_description=True,
        picture_description_options=PictureDescriptionVlmEngineOptions.from_preset(
            "smolvlm",
            engine_options=MlxVlmEngineOptions(),
        ),
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _snapshot_path() -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface-hub is required") from exc
    return Path(
        snapshot_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            revision=DATASET_REVISION,
        )
    )


def _prepare_benchmark(snapshot: Path, workspace: Path, limit: int) -> Path:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from datasets import load_dataset
        from docling_eval.datamodels.dataset_record import DatasetRecord
    except ImportError as exc:
        raise SystemExit("docling-eval is required; run with --with 'docling-eval==1.4.2'") from exc

    parquet_files = sorted((snapshot / "test").glob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No test parquet files found in {snapshot}")

    pdf_dir = workspace / "pdfs"
    gt_dir = workspace / "gt_dataset" / "test"
    pdf_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)
    manifest: list[dict[str, str]] = []
    ground_truth_rows: list[dict[str, Any]] = []

    dataset = load_dataset(
        "parquet",
        data_files={"test": [str(path) for path in parquet_files]},
    )["test"]
    for row in dataset.select(range(min(limit, len(dataset)))):
        doc_id = str(row["document_id"])
        if Path(doc_id).name != doc_id:
            raise SystemExit(f"Unsupported document_id path: {doc_id}")
        pdf_bytes = row["BinaryDocument"]
        if not isinstance(pdf_bytes, bytes):
            raise SystemExit(f"BinaryDocument is not bytes for {doc_id}")
        pdf_path = pdf_dir / f"{len(manifest):04d}.pdf"
        pdf_path.write_bytes(pdf_bytes)
        manifest.append(
            {
                "document_id": doc_id,
                "path": str(pdf_path),
                "sha256": _sha256_bytes(pdf_bytes),
            }
        )
        ground_truth_rows.append(DatasetRecord.model_validate(row).as_record_dict())

    if not manifest:
        raise SystemExit("Docling DPBench is empty")
    table = pa.Table.from_pylist(ground_truth_rows, schema=DatasetRecord.pyarrow_schema())
    pq.write_table(table, gt_dir / "shard_000000_000000.parquet")
    manifest_path = workspace / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _component_snapshot(document: dict[str, Any]) -> dict[str, Any]:
    pictures = []
    charts = []
    for picture in document.get("pictures", []):
        meta = picture.get("meta") or {}
        predictions = (meta.get("classification") or {}).get("predictions") or []
        if predictions:
            pictures.append(
                {
                    "self_ref": picture.get("self_ref"),
                    "predictions": [
                        {
                            key: prediction.get(key)
                            for key in ("class_name", "confidence", "created_by")
                        }
                        for prediction in predictions
                    ],
                }
            )

        chart: dict[str, Any] = {"self_ref": picture.get("self_ref")}
        tabular = meta.get("tabular_chart")
        if isinstance(tabular, dict):
            chart_data = tabular.get("chart_data")
            if isinstance(chart_data, dict):
                chart["tabular_chart"] = {
                    key: value for key, value in chart_data.items() if key != "grid"
                }
        for field in ("description", "code"):
            value = meta.get(field)
            if value is not None:
                chart[field] = value
        if len(chart) > 1:
            charts.append(chart)

    code_formula = [
        {
            "self_ref": item.get("self_ref"),
            "label": item.get("label"),
            "text": item.get("text"),
            "code_language": item.get("code_language"),
        }
        for item in document.get("texts", [])
        if item.get("label") in {"code", "formula"}
    ]
    return {
        "picture_classification": sorted(pictures, key=lambda item: str(item["self_ref"])),
        "chart_extraction": sorted(charts, key=lambda item: str(item["self_ref"])),
        "code_formula": sorted(code_formula, key=lambda item: str(item["self_ref"])),
    }


def _run_worker(
    side: Literal["official", "mlx"],
    manifest_path: Path,
    output_dir: Path,
    artifacts_path: Path | None,
) -> dict[str, Any]:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.settings import settings
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
    from docling_core.types.doc import ImageRefMode
    from docling_core.types.io import DocumentStream

    settings.debug.profile_pipeline_timings = True

    if side == "official":
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

        pipeline_cls: type[StandardPdfPipeline] = StandardPdfPipeline
        options = build_official_pipeline_options(artifacts_path)
    else:
        pipeline_cls = MlxStandardPdfPipeline
        options = build_mlx_pipeline_options(artifacts_path)

    overall_started = time.perf_counter()
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=pipeline_cls,
                pipeline_options=options,
            )
        },
    )
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    side_dir = output_dir / side
    predictions_dir = side_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    labels: Counter[str] = Counter()
    documents: list[dict[str, Any]] = []
    components: dict[str, Any] = {}
    total_pages = 0
    total_items = 0

    for record in records:
        doc_id = record["document_id"]
        pdf_bytes = Path(record["path"]).read_bytes()
        started = time.perf_counter()
        result = converter.convert(DocumentStream(name=f"{doc_id}.pdf", stream=BytesIO(pdf_bytes)))
        elapsed = time.perf_counter() - started
        exported = result.document.export_to_dict()
        result.document.save_as_json(
            predictions_dir / f"{doc_id}.json",
            image_mode=ImageRefMode.EMBEDDED,
        )
        components[doc_id] = _component_snapshot(exported)
        item_count = 0
        for item, _level in result.document.iterate_items():
            item_count += 1
            label = getattr(item, "label", None)
            labels[str(getattr(label, "value", label))] += 1
        page_count = len(result.document.pages)
        total_pages += page_count
        total_items += item_count
        documents.append(
            {
                "document_id": doc_id,
                "source_sha256": record["sha256"],
                "status": getattr(result.status, "value", str(result.status)),
                "wall_seconds": elapsed,
                "timings": {
                    key: {
                        "scope": timing.scope.value,
                        "count": timing.count,
                        "total_seconds": float(timing.total()),
                    }
                    for key, timing in sorted(result.timings.items())
                },
                "pages": page_count,
                "items": item_count,
                "errors": [
                    error.model_dump(mode="json") if hasattr(error, "model_dump") else str(error)
                    for error in result.errors
                ],
            }
        )

    wall_seconds = time.perf_counter() - overall_started
    summary = {
        "side": side,
        "dataset": {"repo_id": DATASET_REPO, "revision": DATASET_REVISION},
        "documents_processed": len(documents),
        "wall_seconds": wall_seconds,
        "documents_per_second": len(documents) / wall_seconds,
        "peak_rss_bytes": _peak_rss_bytes(),
        "pages": total_pages,
        "items": total_items,
        "labels": dict(sorted(labels.items())),
        "documents": documents,
    }
    (side_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (side_dir / "components.json").write_text(
        json.dumps(components, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _child_command(args: argparse.Namespace, side: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--output-dir",
        str(args.output_dir),
        "--side",
        side,
        "--manifest",
        str(args.manifest),
    ]
    if args.artifacts_path is not None:
        command.extend(("--artifacts-path", str(args.artifacts_path)))
    return command


def _evaluate(
    side: str,
    output_dir: Path,
    gt_dir: Path,
    workspace: Path,
    concurrency: int,
) -> None:
    eval_workspace = workspace / f"eval-{side}"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "docling_eval.cli.main",
            "create-eval",
            "--benchmark",
            "DoclingDPBench",
            "--output-dir",
            str(eval_workspace),
            "--prediction-provider",
            "File",
            "--gt-dir",
            str(gt_dir),
            "--file-prediction-format",
            "json",
            "--file-source-path",
            str(output_dir / side / "predictions"),
            "--no-do-visualization",
        ],
        check=True,
        cwd=ROOT,
    )
    command = [
        sys.executable,
        "-m",
        "docling_eval.cli.main",
        "evaluate",
        "--benchmark",
        "DoclingDPBench",
        "--input-dir",
        str(eval_workspace / "eval_dataset"),
        "--output-dir",
        str(output_dir / side / "docling-eval"),
        "--concurrency",
        str(concurrency),
    ]
    evaluation_dir = output_dir / side / "docling-eval"
    statuses: list[dict[str, Any]] = []
    for modality in EVAL_MODALITIES:
        modality_command = [*command, "--modality", modality]
        try:
            subprocess.run(modality_command, check=True, cwd=ROOT)
        except subprocess.CalledProcessError as exc:
            statuses.append(
                {
                    "modality": modality,
                    "status": "failed",
                    "returncode": exc.returncode,
                }
            )
        else:
            statuses.append({"modality": modality, "status": "passed"})
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    (evaluation_dir / "evaluation-status.json").write_text(
        json.dumps(statuses, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _indexed(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item["self_ref"]): item for item in items}


def _group_diff(
    official: list[dict[str, Any]],
    mlx: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], dict[str, Any]]]]:
    official_by_ref = _indexed(official)
    mlx_by_ref = _indexed(mlx)
    shared = sorted(official_by_ref.keys() & mlx_by_ref.keys())
    return (
        {
            "official": len(official_by_ref),
            "mlx": len(mlx_by_ref),
            "matched_refs": len(shared),
            "missing_from_mlx": sorted(official_by_ref.keys() - mlx_by_ref.keys()),
            "missing_from_official": sorted(mlx_by_ref.keys() - official_by_ref.keys()),
        },
        [(official_by_ref[ref], mlx_by_ref[ref]) for ref in shared],
    )


def _compare_components(output_dir: Path) -> Path:
    official = json.loads((output_dir / "official" / "components.json").read_text())
    mlx = json.loads((output_dir / "mlx" / "components.json").read_text())
    result: dict[str, Any] = {"documents": {}}
    for doc_id in sorted(official.keys() | mlx.keys()):
        official_doc = official.get(doc_id, {})
        mlx_doc = mlx.get(doc_id, {})
        document_result: dict[str, Any] = {}
        for group in ("picture_classification", "chart_extraction", "code_formula"):
            counts, pairs = _group_diff(official_doc.get(group, []), mlx_doc.get(group, []))
            counts["exact_matches"] = sum(left == right for left, right in pairs)
            if group == "picture_classification":
                counts["top1_class_matches"] = sum(
                    bool(left["predictions"])
                    and bool(right["predictions"])
                    and left["predictions"][0]["class_name"]
                    == right["predictions"][0]["class_name"]
                    for left, right in pairs
                )
                errors = [
                    abs(float(lp["confidence"]) - float(rp["confidence"]))
                    for left, right in pairs
                    for lp, rp in zip(left["predictions"], right["predictions"], strict=False)
                    if lp["class_name"] == rp["class_name"]
                ]
                counts["max_aligned_confidence_error"] = max(errors, default=None)
            document_result[group] = counts
        result["documents"][doc_id] = document_result
    result["note"] = (
        "Component matching uses self_ref only; layout differences can change downstream crops."
    )
    path = output_dir / "component-diff.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_comparison(output_dir: Path) -> Path:
    official = json.loads((output_dir / "official" / "summary.json").read_text())
    mlx = json.loads((output_dir / "mlx" / "summary.json").read_text())
    comparison = {
        "schema_version": 3,
        "dataset": official["dataset"],
        "documents": official["documents_processed"],
        "features": {
            "layout": "Heron R101",
            "ocr": "MacOCR",
            "table_structure": "TableFormerV2",
            "picture_classification": True,
            "code_enrichment": True,
            "formula_enrichment": True,
            "chart_csv": True,
            "chart_summary": True,
            "chart_code": True,
            "picture_description": True,
        },
        "runs": {"official": official, "mlx": mlx},
        "wall_ratio_mlx_over_official": (
            mlx["wall_seconds"] / official["wall_seconds"] if official["wall_seconds"] else None
        ),
        "throughput_ratio_mlx_over_official": (
            mlx["documents_per_second"] / official["documents_per_second"]
            if official["documents_per_second"]
            else None
        ),
        "quality": {
            side: str(Path(side) / "docling-eval" / "evaluations") for side in ("official", "mlx")
        },
        "component_diff": "component-diff.json",
    }
    path = output_dir / "comparison.json"
    path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--artifacts-path", type=Path)
    parser.add_argument("--eval-concurrency", type=int, default=4)
    parser.add_argument("--side", choices=("official", "mlx"), help=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.artifacts_path is not None and not args.artifacts_path.is_dir():
        raise SystemExit(f"Artifact root does not exist: {args.artifacts_path}")

    if args.side is not None:
        if args.manifest is None or not args.manifest.is_file():
            raise SystemExit("Worker manifest does not exist")
        summary = _run_worker(
            args.side,
            args.manifest,
            args.output_dir,
            args.artifacts_path,
        )
        print(json.dumps(summary, sort_keys=True))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="docling-mlx-dpbench-") as temporary:
        workspace = Path(temporary)
        args.manifest = _prepare_benchmark(_snapshot_path(), workspace, args.limit)
        for side in ("official", "mlx"):
            subprocess.run(_child_command(args, side), check=True, cwd=ROOT)
            _evaluate(
                side,
                args.output_dir,
                workspace / "gt_dataset",
                workspace,
                args.eval_concurrency,
            )
    _compare_components(args.output_dir)
    print(_write_comparison(args.output_dir))


if __name__ == "__main__":
    main()
