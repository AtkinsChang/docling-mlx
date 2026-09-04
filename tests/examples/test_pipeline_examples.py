# SPDX-License-Identifier: Apache-2.0

"""Portable contracts for the two pipeline examples."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path
from subprocess import CalledProcessError

import pypdfium2  # noqa: F401
import pytest
from packaging.requirements import Requirement

EXAMPLES = Path(__file__).parents[2] / "examples"
PIPELINE_DIR = EXAMPLES / "mlx_pipeline"
TOOLS = Path(__file__).parents[2] / "tools"
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(TOOLS))

import compare_dpbench as compare  # noqa: E402
import pipeline  # noqa: E402
from docling.models.stages.picture_classifier.document_picture_classifier import (  # noqa: E402
    DocumentPictureClassifier,
)

from docling_mlx.pipeline import MlxStandardPdfPipeline, configure  # noqa: E402
from docling_mlx.presets import resolve_preset  # noqa: E402
from docling_mlx.stages.chart_extraction import MlxChartExtractionModelOptions  # noqa: E402
from docling_mlx.stages.layout import MlxLayoutObjectDetectionOptions  # noqa: E402
from docling_mlx.stages.picture_classification import (  # noqa: E402
    MlxDocumentPictureClassifierOptions,
)
from docling_mlx.stages.table_structure_v2 import MlxTableStructureV2Options  # noqa: E402


def test_mlx_project_declares_only_torch_free_docling_dependencies() -> None:
    project = tomllib.loads((PIPELINE_DIR / "pyproject.toml").read_text())
    dependencies = [Requirement(entry) for entry in project["project"]["dependencies"]]
    names = {requirement.name for requirement in dependencies}

    assert "docling-mlx" in names
    assert "docling-slim" in names
    assert "docling" not in names
    lock = tomllib.loads((PIPELINE_DIR / "uv.lock").read_text())
    locked_packages = {package["name"] for package in lock["package"]}
    assert not {"torch", "torchvision", "transformers", "docling-ibm-models"} & locked_packages


def test_torch_free_profile_selects_heron_101_and_native_components() -> None:
    heron_101 = resolve_preset("layout_heron_101")
    options = pipeline.build_options()

    assert options.allow_external_plugins
    assert isinstance(options.layout_options, MlxLayoutObjectDetectionOptions)
    assert options.layout_options.model_spec.repo_id == heron_101.repo_id
    assert options.layout_options.model_spec.revision == heron_101.revision
    assert isinstance(options.table_structure_options, MlxTableStructureV2Options)
    assert isinstance(options.picture_classification_options, MlxDocumentPictureClassifierOptions)
    assert options.do_ocr
    assert options.do_table_structure
    assert options.do_picture_classification
    assert not options.do_code_enrichment
    assert not options.do_formula_enrichment
    assert not options.do_chart_extraction
    assert not options.do_picture_description


def test_component_snapshot_keeps_only_owned_outputs() -> None:
    snapshot = compare._component_snapshot(
        {
            "pictures": [
                {
                    "self_ref": "#/pictures/0",
                    "meta": {
                        "classification": {
                            "predictions": [
                                {
                                    "class_name": "bar_chart",
                                    "confidence": 0.9,
                                    "created_by": "DocumentPictureClassifier",
                                }
                            ]
                        },
                        "tabular_chart": {
                            "chart_data": {
                                "num_rows": 1,
                                "num_cols": 1,
                                "grid": [["ignored"]],
                            }
                        },
                    },
                }
            ],
            "texts": [
                {
                    "self_ref": "#/texts/0",
                    "label": "code",
                    "text": "print(1)",
                    "code_language": "python",
                }
            ],
        }
    )

    assert snapshot["picture_classification"][0]["predictions"][0]["class_name"] == ("bar_chart")
    assert snapshot["chart_extraction"][0]["tabular_chart"] == {
        "num_rows": 1,
        "num_cols": 1,
    }
    assert snapshot["code_formula"][0]["text"] == "print(1)"


def test_compare_records_one_failed_docling_eval_modality_without_aborting(
    tmp_path: Path, monkeypatch: object
) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> None:
        commands.append(command)
        if command[-1] == "bboxes_text":
            raise CalledProcessError(1, command)

    monkeypatch.setattr(compare.subprocess, "run", fake_run)  # type: ignore[attr-defined]

    compare._evaluate("official", tmp_path, tmp_path / "gt", tmp_path / "work", 2)

    assert "create-eval" in commands[0]
    assert "File" in commands[0]
    assert len(commands) == len(compare.EVAL_MODALITIES) + 1
    assert [command[-1] for command in commands[1:]] == list(compare.EVAL_MODALITIES)
    statuses = (tmp_path / "official" / "docling-eval" / "evaluation-status.json").read_text()
    assert '"modality": "bboxes_text"' in statuses
    assert '"status": "failed"' in statuses


def test_standard_pipeline_preserves_official_chart_during_mlx_picture_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDoclingPicture:
        def __init__(self, **_kwargs: object) -> None:
            pass

    class FakeDoclingChart:
        def __init__(self, *, enabled: bool, **_kwargs: object) -> None:
            self.enabled = enabled

    class FakeDoclingChartV4(FakeDoclingChart):
        pass

    class FakeMlxPicture:
        def __init__(self, **_kwargs: object) -> None:
            pass

    monkeypatch.setattr(MlxStandardPdfPipeline.__mro__[1], "_init_models", lambda _self: None)
    monkeypatch.setattr(
        "docling.pipeline.base_pipeline.ConvertPipeline._get_picture_description_model",
        lambda _self, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "docling.pipeline.base_pipeline.DocumentPictureClassifier",
        FakeDoclingPicture,
    )
    monkeypatch.setattr(
        "docling.models.stages.picture_classifier.document_picture_classifier."
        "DocumentPictureClassifier",
        FakeDoclingPicture,
    )
    monkeypatch.setattr(
        "docling.models.stages.chart_extraction.granite_vision.ChartExtractionModelGraniteVision",
        FakeDoclingChart,
    )
    monkeypatch.setattr(
        "docling.models.stages.chart_extraction.granite_vision.ChartExtractionModelGraniteVisionV4",
        FakeDoclingChartV4,
    )
    monkeypatch.setattr(
        "docling_mlx.stages.picture_classification.MlxDocumentPictureClassifier",
        FakeMlxPicture,
    )

    options = pipeline.build_options()
    options.do_chart_extraction = True
    instance = MlxStandardPdfPipeline(options)

    assert any(type(stage) is FakeMlxPicture for stage in instance.enrichment_pipe)
    assert any(
        type(stage) is FakeDoclingChartV4 and stage.enabled for stage in instance.enrichment_pipe
    )


def test_configure_replaces_the_non_pluggable_chart_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = pipeline.build_options()
    options.do_chart_extraction = True
    options.chart_extraction_options = MlxChartExtractionModelOptions()

    class FakeMlxPicture:
        def __init__(self, **_kwargs: object) -> None:
            pass

    class FakeMlxChart:
        def __init__(self, **_kwargs: object) -> None:
            pass

    fake_pipeline = type("Pipeline", (), {})()
    fake_pipeline.pipeline_options = options
    fake_pipeline.artifacts_path = None
    fake_pipeline.enrichment_pipe = [
        object.__new__(DocumentPictureClassifier),
        type("ChartExtractionModelGraniteVision", (), {})(),
        type("ChartExtractionModelGraniteVisionV4", (), {})(),
    ]
    monkeypatch.setattr(
        "docling_mlx.stages.picture_classification.MlxDocumentPictureClassifier",
        FakeMlxPicture,
    )
    monkeypatch.setattr(
        "docling_mlx.stages.chart_extraction.MlxGraniteVisionChartExtractionModel",
        FakeMlxChart,
    )

    configure(fake_pipeline)  # type: ignore[arg-type]
    configure(fake_pipeline)  # type: ignore[arg-type]

    assert [type(stage) for stage in fake_pipeline.enrichment_pipe] == [
        FakeMlxPicture,
        FakeMlxChart,
    ]
    assert fake_pipeline.keep_backend is True


def test_shipped_examples_import_and_compile_without_running_models() -> None:
    tracked = (
        subprocess.run(
            ["git", "ls-files", "-z", "--", "examples"],
            cwd=EXAMPLES.parent,
            check=True,
            capture_output=True,
        )
        .stdout.decode()
        .split("\0")
    )
    sources = sorted(str(EXAMPLES.parent / path) for path in tracked if path.endswith(".py"))
    assert sources
    script = """
import py_compile
import runpy
import sys

for index, path in enumerate(sys.argv[1:]):
    py_compile.compile(path, doraise=True)
    runpy.run_path(path, run_name=f"_example_{index}")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, *sources],
        cwd=EXAMPLES.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
