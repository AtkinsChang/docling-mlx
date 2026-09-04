# SPDX-License-Identifier: Apache-2.0

"""Run a Torch-free PDF pipeline with docling-mlx models."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from collections import Counter
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import OcrMacOptions, ThreadedPdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from docling_mlx.pipeline import MlxStandardPdfPipeline
from docling_mlx.stages.layout import (
    MlxLayoutObjectDetectionOptions,
    MlxObjectDetectionEngineOptions,
)
from docling_mlx.stages.picture_classification import (
    MlxDocumentPictureClassifierOptions,
)
from docling_mlx.stages.table_structure_v2 import MlxTableStructureV2Options


def build_options(
    *,
    artifacts_path: Path | None = None,
    enable_ocr: bool = True,
) -> ThreadedPdfPipelineOptions:
    """Build the Torch-free Heron R101, TableFormerV2, and Figure profile."""

    return ThreadedPdfPipelineOptions(
        artifacts_path=artifacts_path,
        accelerator_options=AcceleratorOptions(device="auto"),
        allow_external_plugins=True,
        do_ocr=enable_ocr,
        ocr_options=OcrMacOptions(),
        layout_options=MlxLayoutObjectDetectionOptions.from_preset(
            "layout_heron_101", engine_options=MlxObjectDetectionEngineOptions()
        ),
        do_table_structure=True,
        table_structure_options=MlxTableStructureV2Options(do_cell_matching=True),
        do_picture_classification=True,
        picture_classification_options=MlxDocumentPictureClassifierOptions(),
        do_code_enrichment=False,
        do_formula_enrichment=False,
        do_chart_extraction=False,
        do_picture_description=False,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def summarize(result: Any, source: Path, wall_seconds: float, side: str) -> dict[str, Any]:
    """Return the small common report used by the example and comparison."""

    labels: Counter[str] = Counter()
    item_count = 0
    for item, _level in result.document.iterate_items():
        item_count += 1
        label = getattr(item, "label", None)
        labels[str(getattr(label, "value", label))] += 1
    return {
        "side": side,
        "status": getattr(result.status, "value", str(result.status)),
        "source_sha256": _sha256(source),
        "wall_seconds": wall_seconds,
        "peak_rss_bytes": _peak_rss_bytes(),
        "pages": len(result.document.pages),
        "items": item_count,
        "labels": dict(sorted(labels.items())),
        "errors": [
            error.model_dump(mode="json") if hasattr(error, "model_dump") else str(error)
            for error in result.errors
        ],
    }


def run(
    source: Path,
    output_dir: Path,
    options: ThreadedPdfPipelineOptions,
    *,
    side: str = "docling-mlx",
) -> dict[str, Any]:
    """Convert one PDF and write its document and summary."""

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=MlxStandardPdfPipeline,
                pipeline_options=options,
            )
        },
    )
    started = time.perf_counter()
    result = converter.convert(source)
    wall_seconds = time.perf_counter() - started

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "document.json").write_text(
        json.dumps(result.document.export_to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = summarize(result, source, wall_seconds, side)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _assert_torch_free() -> None:
    installed = []
    for name in ("torch", "torchvision"):
        try:
            distribution(name)
        except PackageNotFoundError:
            continue
        installed.append(name)
    if installed:
        raise RuntimeError(f"Torch-free example found installed packages: {', '.join(installed)}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifacts-path", type=Path)
    parser.add_argument(
        "--allow-torch",
        action="store_true",
        help="allow this smoke run in an environment that has Torch installed",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="disable the OCRMac stage for a host smoke run",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not args.source.is_file():
        raise SystemExit(f"Input PDF does not exist: {args.source}")
    if args.artifacts_path is not None and not args.artifacts_path.is_dir():
        raise SystemExit(f"Artifact root does not exist: {args.artifacts_path}")
    if not args.allow_torch:
        _assert_torch_free()
    summary = run(
        args.source,
        args.output_dir,
        build_options(artifacts_path=args.artifacts_path, enable_ocr=not args.no_ocr),
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
