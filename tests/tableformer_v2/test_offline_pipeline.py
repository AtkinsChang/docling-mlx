# SPDX-License-Identifier: Apache-2.0

"""Offline standard-pipeline qualification for the TableFormerV2 plugin."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.mlx, pytest.mark.release]

ROOT = Path(__file__).resolve().parents[2]
PDF_FIXTURE = ROOT / "tests/fixtures/tableformer_v2/basin_table_1.pdf"


def _required_artifact(environment_name: str, default: Path, required: tuple[str, ...]) -> Path:
    artifact = Path(os.environ.get(environment_name, default)).expanduser()
    missing = [name for name in required if not (artifact / name).is_file()]
    if missing:
        pytest.fail(f"release artifact {artifact} is missing {missing}")
    return artifact.resolve()


def test_offline_pdf_uses_heron_and_tableformer_plugins(tmp_path: Path) -> None:
    heron = _required_artifact(
        "DOCLING_MLX_HERON_R50_ARTIFACT",
        ROOT / ".artifacts/heron-r50",
        ("model.safetensors", "config.json", "preprocessor_config.json"),
    )
    tableformer = _required_artifact(
        "DOCLING_MLX_TABLEFORMER_V2_ARTIFACT",
        ROOT / ".artifacts/tableformer-v2",
        (
            "model.safetensors",
            "config.json",
            "preprocessor_config.json",
            "generation_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ),
    )
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir()
    (artifacts_root / "test-heron--r50").symlink_to(heron, target_is_directory=True)
    (artifacts_root / "test-tableformer--v2").symlink_to(
        tableformer,
        target_is_directory=True,
    )

    script = r"""
import json
import os
from pathlib import Path

from docling.datamodel.backend_options import ThreadedDoclingParseBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import ObjectDetectionModelSpec, ThreadedPdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from docling_mlx.stages.layout import MlxObjectDetectionEngineOptions
from docling_mlx.engines.table_structure.tableformer_v2 import TableFormerV2ModelSpec
from docling_mlx.stages.layout import MlxLayoutObjectDetectionOptions
from docling_mlx.stages.table_structure_v2 import MlxTableStructureV2Options

root = Path(os.environ["TABLEFORMER_ARTIFACTS_ROOT"])
options = ThreadedPdfPipelineOptions(
    artifacts_path=root,
    allow_external_plugins=True,
    do_ocr=False,
    do_table_structure=True,
    do_code_enrichment=False,
    do_formula_enrichment=False,
    do_picture_classification=False,
    layout_options=MlxLayoutObjectDetectionOptions(
        model_spec=ObjectDetectionModelSpec(
            name="Test Heron R50",
            repo_id="test-heron/r50",
            revision="local",
        ),
        engine_options=MlxObjectDetectionEngineOptions(),
    ),
    table_structure_options=MlxTableStructureV2Options(
        model_spec=TableFormerV2ModelSpec(
            repo_id="test-tableformer/v2",
            revision="local",
        ),
        do_cell_matching=False,
    ),
)
converter = DocumentConverter(
    allowed_formats=[InputFormat.PDF],
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=options,
            backend_options=ThreadedDoclingParseBackendOptions(parser_threads=1),
        )
    },
)
result = converter.convert(Path(os.environ["TABLEFORMER_PDF"]))
pipeline = next(iter(converter.initialized_pipelines.values()))
print("TABLEFORMER_SMOKE=" + json.dumps({
    "status": result.status.value,
    "errors": [str(error) for error in result.errors],
    "layout_stage": type(pipeline.layout_model).__name__,
    "table_stage": type(pipeline.table_model).__name__,
    "tables": [
        {
            "rows": table.data.num_rows,
            "cols": table.data.num_cols,
            "cells": len(table.data.table_cells),
        }
        for table in result.document.tables
    ],
}, separators=(",", ":")))
"""
    environment = os.environ | {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TABLEFORMER_ARTIFACTS_ROOT": str(artifacts_root),
        "TABLEFORMER_PDF": str(PDF_FIXTURE),
    }
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    marker = next(
        line.removeprefix("TABLEFORMER_SMOKE=")
        for line in completed.stdout.splitlines()
        if line.startswith("TABLEFORMER_SMOKE=")
    )
    payload = json.loads(marker)
    assert payload == {
        "status": "success",
        "errors": [],
        "layout_stage": "MlxLayoutObjectDetectionModel",
        "table_stage": "MlxTableFormerV2Model",
        "tables": [{"rows": 6, "cols": 8, "cells": 48}],
    }
