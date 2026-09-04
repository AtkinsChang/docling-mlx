# SPDX-License-Identifier: Apache-2.0

"""Serializable options and the Docling-shaped artifact resolution boundary."""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
from docling.datamodel.stage_model_specs import ImageClassificationModelSpec
from docling.utils.pipeline_cache import create_pipeline_options_hash
from pydantic import ValidationError

from docling_mlx.engines._shared import resolve_artifact_checkpoint
from docling_mlx.stages.picture_classification import (
    MlxDocumentPictureClassifierOptions,
    MlxImageClassificationEngineOptions,
)


@pytest.mark.parametrize(
    "options",
    [
        MlxImageClassificationEngineOptions(top_k=3),
        MlxDocumentPictureClassifierOptions(
            model_spec=ImageClassificationModelSpec(
                name="Document Figure Classifier",
                repo_id="docling-project/DocumentFigureClassifier-v2.5",
                revision="refs/pr/123",
            ),
            engine_options=MlxImageClassificationEngineOptions(top_k=26),
        ),
    ],
)
def test_options_round_trip_through_json(
    options: MlxImageClassificationEngineOptions | MlxDocumentPictureClassifierOptions,
) -> None:
    restored = type(options).model_validate_json(options.model_dump_json())

    assert restored == options


def test_figure_options_have_only_the_docling_shaped_serializable_surface() -> None:
    assert set(MlxImageClassificationEngineOptions.model_fields) == {"top_k", "dtype", "warmup"}
    assert set(MlxDocumentPictureClassifierOptions.model_fields) == {
        "model_spec",
        "engine_options",
    }
    with pytest.raises(ValidationError):
        MlxImageClassificationEngineOptions.model_validate({"artifact": "/legacy"})
    with pytest.raises(ValidationError):
        MlxDocumentPictureClassifierOptions.model_validate(
            {
                "model_spec": {
                    "name": "figure",
                    "repo_id": "example/figure",
                },
                "engine_options": {},
                "unknown": True,
            }
        )


def test_application_pipeline_cache_includes_all_figure_component_settings() -> None:
    """An app-owned pipeline type can serialize the manually injected component."""

    class ApplicationPipelineOptions(ThreadedPdfPipelineOptions):
        picture_classification_options: MlxDocumentPictureClassifierOptions

    def pipeline_options(
        *,
        repo_id: str = "example/document-figure-mlx",
        revision: str = "main",
        top_k: int | None = None,
        warmup: bool = False,
    ) -> ApplicationPipelineOptions:
        return ApplicationPipelineOptions(
            picture_classification_options=MlxDocumentPictureClassifierOptions(
                model_spec=ImageClassificationModelSpec(
                    name="Document Figure MLX",
                    repo_id=repo_id,
                    revision=revision,
                ),
                engine_options=MlxImageClassificationEngineOptions(top_k=top_k, warmup=warmup),
            )
        )

    baseline = create_pipeline_options_hash(pipeline_options())
    variants = (
        pipeline_options(repo_id="example/other-document-figure-mlx"),
        pipeline_options(revision="release-v1"),
        pipeline_options(top_k=5),
        pipeline_options(warmup=True),
    )

    assert all(create_pipeline_options_hash(variant) != baseline for variant in variants)


@pytest.mark.parametrize("as_string", [False, True])
def test_artifact_resolver_uses_prefetched_artifact_without_hugging_face(
    tmp_path: Path, as_string: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_id = "docling-project/docling-layout-heron"
    root = tmp_path / "artifacts"
    directory = root / repo_id.replace("/", "--")
    directory.mkdir(parents=True)
    blocked_hub = ModuleType("huggingface_hub")
    blocked_hub.snapshot_download = Mock(side_effect=AssertionError("network resolver called"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", blocked_hub)

    artifacts_path = str(root) if as_string else root
    assert resolve_artifact_checkpoint(repo_id, "main", artifacts_path, files=()) == directory
    blocked_hub.snapshot_download.assert_not_called()


def test_artifact_resolver_downloads_when_no_prefetched_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "snapshot"
    snapshot_download = Mock(return_value=str(directory))
    hub = ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    assert (
        resolve_artifact_checkpoint(
            "docling-project/docling-layout-heron",
            "refs/pr/123",
            None,
            files=("config.json", "model.safetensors"),
        )
        == directory
    )
    snapshot_download.assert_called_once_with(
        repo_id="docling-project/docling-layout-heron",
        revision="refs/pr/123",
        allow_patterns=["config.json", "model.safetensors"],
    )


def test_artifact_resolver_requires_prefetched_artifact_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_artifact_checkpoint(
            "docling-project/docling-layout-heron", "main", tmp_path, files=()
        )


def test_artifact_resolver_prefers_revision_directory_when_flat_directory_exists(
    tmp_path: Path,
) -> None:
    repo_id = "docling-project/docling-layout-heron"
    revision = "refs/pr/123"
    base = tmp_path / repo_id.replace("/", "--")
    revision_directory = base / revision
    base.mkdir(parents=True)
    (base / "legacy-marker").touch()
    revision_directory.mkdir(parents=True)

    assert resolve_artifact_checkpoint(repo_id, revision, tmp_path, files=()) == revision_directory


def test_artifact_resolver_uses_flat_directory(tmp_path: Path) -> None:
    repo_id = "docling-project/docling-layout-heron"
    revision = "refs/pr/123"
    base = tmp_path / repo_id.replace("/", "--")
    base.mkdir(parents=True)

    assert resolve_artifact_checkpoint(repo_id, revision, tmp_path, files=()) == base


def test_artifact_resolver_errors_when_revision_and_flat_directories_are_missing(
    tmp_path: Path,
) -> None:
    repo_id = "docling-project/docling-layout-heron"
    base = tmp_path / repo_id.replace("/", "--")

    with pytest.raises(
        FileNotFoundError,
        match=f"Model artifact directory does not exist: {base}",
    ):
        resolve_artifact_checkpoint(repo_id, "main", tmp_path, files=())
