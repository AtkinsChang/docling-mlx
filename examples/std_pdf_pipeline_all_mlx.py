# SPDX-License-Identifier: Apache-2.0

"""Run Docling's standard PDF pipeline with the MLX layout, table, and picture stages."""

from __future__ import annotations

import argparse
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from docling_mlx.pipeline import MlxStandardPdfPipeline
from docling_mlx.stages.layout import (
    MlxLayoutObjectDetectionOptions,
    MlxObjectDetectionEngineOptions,
)
from docling_mlx.stages.picture_classification import MlxDocumentPictureClassifierOptions
from docling_mlx.stages.table_structure_v2 import MlxTableStructureV2Options

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = ROOT / "tests/fixtures/tableformer_v2/basin_table_1.pdf"


def build_options(artifacts_path: Path | None) -> ThreadedPdfPipelineOptions:
    return ThreadedPdfPipelineOptions(
        artifacts_path=artifacts_path,
        accelerator_options=AcceleratorOptions(device="auto"),
        allow_external_plugins=True,
        do_ocr=False,
        layout_options=MlxLayoutObjectDetectionOptions.from_preset(
            "layout_heron_default", engine_options=MlxObjectDetectionEngineOptions()
        ),
        do_table_structure=True,
        table_structure_options=MlxTableStructureV2Options(),
        do_picture_classification=True,
        picture_classification_options=MlxDocumentPictureClassifierOptions(),
        do_code_enrichment=False,
        do_formula_enrichment=False,
        do_chart_extraction=False,
        do_picture_description=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", nargs="?", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--artifacts-path", type=Path)
    args = parser.parse_args()
    source = args.pdf if args.pdf.is_absolute() else ROOT / args.pdf
    result = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=MlxStandardPdfPipeline,
                pipeline_options=build_options(args.artifacts_path),
            )
        },
    ).convert(source)
    print(result.document.export_to_markdown())


if __name__ == "__main__":
    main()
