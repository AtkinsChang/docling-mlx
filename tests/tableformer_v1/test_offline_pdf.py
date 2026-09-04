# SPDX-License-Identifier: Apache-2.0

"""Offline standard-pipeline qualification for the TableFormerV1 plugin."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from docling.datamodel.pipeline_options import TableFormerMode

from tools.tableformer_v1.source import SOURCE_REVISION

pytestmark = [pytest.mark.mlx, pytest.mark.release]

ROOT = Path(__file__).resolve().parents[2]
PDF_FIXTURE = ROOT / "tests/fixtures/tableformer_v1/basin_table_1_text.pdf"
VISIBLE_SOURCE_FIXTURE = ROOT / "tests/fixtures/tableformer_v2/basin_table_1.pdf"
FAST_ORACLE_ENVIRONMENT_VARIABLE = "DOCLING_MLX_TABLEFORMER_V1_FAST_PDF_ORACLE"
FAST_ORACLE_DEFAULT = ROOT / "tests/golden/tableformer_v1/fast_basin_table_1.json"


def _artifact(environment_name: str, default: Path, required: tuple[str, ...]) -> Path:
    artifact = Path(os.environ.get(environment_name, default)).expanduser()
    missing = [name for name in required if not (artifact / name).is_file()]
    if missing:
        pytest.fail(f"release artifact {artifact} is missing {missing}")
    return artifact.resolve()


def _fast_oracle() -> list[object]:
    path = Path(os.environ.get(FAST_ORACLE_ENVIRONMENT_VARIABLE, FAST_ORACLE_DEFAULT)).expanduser()
    if not path.is_file():
        pytest.fail(
            "Fast PDF qualification requires the pinned Torch-reference oracle at "
            f"{path}; set {FAST_ORACLE_ENVIRONMENT_VARIABLE} to an equivalent capture"
        )
    oracle = json.loads(path.read_text(encoding="utf-8"))
    with PDF_FIXTURE.open("rb") as stream:
        fixture_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    expected_metadata = {
        "schema_version": 1,
        "profile": "tableformer_v1_fast",
        "pdf_sha256": fixture_sha256,
        "source_revision": SOURCE_REVISION,
    }
    if not isinstance(oracle, dict) or any(
        oracle.get(name) != value for name, value in expected_metadata.items()
    ):
        pytest.fail(f"Fast PDF oracle {path} does not match the pinned fixture/profile")
    tables = oracle.get("tables")
    if set(oracle) != set(expected_metadata) | {"tables"} or not isinstance(tables, list):
        pytest.fail(f"Fast PDF oracle {path} has an invalid closed schema")
    return tables


@pytest.mark.parametrize("mode", [TableFormerMode.ACCURATE, TableFormerMode.FAST])
def test_offline_pdf_uses_heron_and_tableformer_v1_plugins(
    tmp_path: Path, mode: TableFormerMode
) -> None:
    heron = _artifact(
        "DOCLING_MLX_HERON_R50_ARTIFACT",
        ROOT / ".artifacts/heron-r50",
        ("model.safetensors", "config.json", "preprocessor_config.json"),
    )
    tableformer = _artifact(
        "DOCLING_MLX_TABLEFORMER_V1_ARTIFACT",
        ROOT / ".artifacts/tableformer-v1",
        tuple(
            f"{mode.value}/{name}"
            for name in (
                "model.safetensors",
                "config.json",
                "preprocessor_config.json",
                "generation_config.json",
            )
        ),
    )
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir()
    (artifacts_root / "test-heron--r50").symlink_to(heron, target_is_directory=True)
    (artifacts_root / "test-tableformer--v1").symlink_to(
        tableformer,
        target_is_directory=True,
    )

    script = r"""
import json
import os
from pathlib import Path

from docling.datamodel.backend_options import ThreadedDoclingParseBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    ObjectDetectionModelSpec,
    TableFormerMode,
    ThreadedPdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from docling_mlx.stages.layout import MlxObjectDetectionEngineOptions
from docling_mlx.engines.table_structure.tableformer_v1 import TableFormerV1ModelSpec
from docling_mlx.stages.layout import MlxLayoutObjectDetectionOptions
from docling_mlx.stages.table_structure_v1 import MlxTableStructureOptions

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
    table_structure_options=MlxTableStructureOptions(
        model_spec=TableFormerV1ModelSpec(
            repo_id="test-tableformer/v1",
            revision="local",
        ),
        mode=TableFormerMode(os.environ["TABLEFORMER_MODE"]),
        do_cell_matching=True,
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
print("TABLEFORMER_V1_SMOKE=" + json.dumps({
    "status": result.status.value,
    "errors": [str(error) for error in result.errors],
    "layout_stage": type(pipeline.layout_model).__name__,
    "table_stage": type(pipeline.table_model).__name__,
    "tables": [
        {
            "rows": table.data.num_rows,
            "cols": table.data.num_cols,
            "cells": [
                {
                    "row": cell.start_row_offset_idx,
                    "col": cell.start_col_offset_idx,
                    "text": cell.text,
                    "bbox": {
                        "l": cell.bbox.l,
                        "t": cell.bbox.t,
                        "r": cell.bbox.r,
                        "b": cell.bbox.b,
                    },
                }
                for cell in table.data.table_cells
            ],
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
        "TABLEFORMER_MODE": mode.value,
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
        line.removeprefix("TABLEFORMER_V1_SMOKE=")
        for line in completed.stdout.splitlines()
        if line.startswith("TABLEFORMER_V1_SMOKE=")
    )
    payload = json.loads(marker)
    assert payload["status"] == "success"
    assert payload["errors"] == []
    assert payload["layout_stage"] == "MlxLayoutObjectDetectionModel"
    assert payload["table_stage"] == "MlxTableFormerV1Model"
    assert payload["tables"]
    if mode is TableFormerMode.FAST:
        assert payload["tables"] == _fast_oracle()
        return

    assert len(payload["tables"]) == 1, payload
    table = payload["tables"][0]
    assert (table["rows"], table["cols"], len(table["cells"])) == (4, 5, 12)

    expected_positions = {
        "A1": (0, 2),
        "B1 OV": (0, 1),
        "C1": (0, 3),
        "D1": (0, 4),
        "A2": (2, 2),
        "B2": (2, 1),
        "C2": (2, 3),
        "D2": (1, 4),
        "A3": (3, 0),
        "B3": (3, 1),
        "C3": (3, 3),
        "D3": (3, 4),
    }
    assert {
        cell["text"]: (cell["row"], cell["col"]) for cell in table["cells"]
    } == expected_positions
    assert sorted({cell["row"] for cell in table["cells"]}) == list(range(table["rows"]))
    assert sorted({cell["col"] for cell in table["cells"]}) == list(range(table["cols"]))

    tokens = [token for cell in table["cells"] for token in cell["text"].split()]
    assert len(tokens) == len(set(tokens)) == 13
    assert set(tokens) == {"OV"} | {f"{column}{row}" for row in range(1, 4) for column in "ABCD"}
    repaired = next(cell for cell in table["cells"] if cell["text"] == "B1 OV")["bbox"]
    assert repaired == pytest.approx(
        {"l": 420.0, "t": 110.984, "r": 522.34, "b": 122.084},
        abs=1e-6,
    )


def test_matching_fixture_keeps_visible_pixels_identical() -> None:
    import pypdfium2 as pdfium

    def render(path: Path) -> tuple[tuple[int, int], bytes]:
        document = pdfium.PdfDocument(path)
        image = document[0].render(scale=2).to_pil().convert("RGB")
        return image.size, image.tobytes()

    assert render(PDF_FIXTURE) == render(VISIBLE_SOURCE_FIXTURE)
