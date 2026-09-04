# SPDX-License-Identifier: Apache-2.0

"""Real-artifact release contracts for the Egret Docling integration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Literal
from unittest.mock import patch

import numpy as np
import pytest
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import ObjectDetectionModelSpec, ThreadedPdfPipelineOptions
from docling.models.inference_engines.object_detection.base import (
    ObjectDetectionEngineInput,
    ObjectDetectionEngineOutput,
)
from docling.utils.pipeline_cache import create_pipeline_options_hash
from PIL import Image

from docling_mlx.stages.layout import (
    MlxLayoutObjectDetectionModel,
    MlxLayoutObjectDetectionOptions,
    MlxObjectDetectionEngineOptions,
)

pytestmark = pytest.mark.release

ROOT = Path(__file__).resolve().parents[2]
PDF_FIXTURE = ROOT / "tests/fixtures/document_figure/picture_classification.pdf"
ProfileName = Literal["medium", "large", "xlarge"]
MEDIUM_REPO_ID = "test-egret/medium"


def _provision_artifact(tmp_path: Path, profile: ProfileName) -> tuple[Path, str]:
    environment_name = f"DOCLING_MLX_EGRET_{profile.upper()}_ARTIFACT"
    artifact = Path(
        os.environ.get(
            environment_name,
            ROOT / ".artifacts" / f"egret-{profile}",
        )
    ).expanduser()
    required = ["config.json", "preprocessor_config.json", "model.safetensors"]
    if not artifact.is_dir() or any(not (artifact / name).is_file() for name in required):
        pytest.fail(
            f"provision the Egret {profile} artifact at {artifact} or set "
            f"{environment_name}; release runs require zero skips"
        )
    repo_id = f"test-egret/{profile}"
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir()
    (artifacts_root / repo_id.replace("/", "--")).symlink_to(
        artifact.resolve(), target_is_directory=True
    )
    return artifacts_root, repo_id


@pytest.fixture(params=["medium", "large", "xlarge"], ids=["medium", "large", "xlarge"])
def egret_profile_artifact(
    request: pytest.FixtureRequest, tmp_path: Path
) -> tuple[ProfileName, Path, str]:
    profile = request.param
    assert profile in {"medium", "large", "xlarge"}
    artifacts_root, repo_id = _provision_artifact(tmp_path, profile)
    return profile, artifacts_root, repo_id


@pytest.fixture
def egret_medium_artifact(tmp_path: Path) -> tuple[Path, str]:
    return _provision_artifact(tmp_path, "medium")


def _engine(artifacts_root: Path, repo_id: str, *, threshold: float = 0.3):
    return MlxLayoutObjectDetectionModel(
        artifacts_path=artifacts_root,
        accelerator_options=AcceleratorOptions(device="auto"),
        options=MlxLayoutObjectDetectionOptions(
            model_spec=ObjectDetectionModelSpec(
                name="release-egret", repo_id=repo_id, revision="local-test-revision"
            ),
            engine_options=MlxObjectDetectionEngineOptions(score_threshold=threshold),
        ),
    ).engine


def _require_metal() -> None:
    probe = subprocess.run(
        [sys.executable, "-c", "import mlx.core"],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        stderr = probe.stderr.strip()
        message = stderr.splitlines()[-1] if stderr else "MLX import failed"
        pytest.fail(
            f"MLX Metal is unavailable; Apple-Silicon release runs require zero skips: {message}"
        )


def _input(index: int = 0) -> ObjectDetectionEngineInput:
    image = Image.new("RGB", (401, 534), color=(16 + index, 32 + index, 64 + index))
    return ObjectDetectionEngineInput(image=image, metadata={"fixture_index": index})


def _assert_same_output(
    actual: ObjectDetectionEngineOutput,
    expected: ObjectDetectionEngineOutput,
    profile: ProfileName,
) -> None:
    assert actual.label_ids == expected.label_ids
    assert actual.metadata == expected.metadata
    # XLarge raw batch-shape variation is bounded by its public detection contract.
    score_atol, bbox_atol = (1e-3, 0.5) if profile == "xlarge" else (1e-6, 5e-4)
    np.testing.assert_allclose(actual.scores, expected.scores, rtol=0, atol=score_atol)
    np.testing.assert_allclose(actual.bboxes, expected.bboxes, rtol=0, atol=bbox_atol)


@pytest.mark.mlx
def test_real_profiles_preserve_batch_order_and_metadata(
    egret_profile_artifact: tuple[ProfileName, Path, str],
) -> None:
    _require_metal()
    profile, artifacts_root, repo_id = egret_profile_artifact
    engine = _engine(artifacts_root, repo_id)
    items = [_input(index) for index in range(4)]

    singles = [engine.predict_batch([item])[0] for item in items]
    batched = engine.predict_batch(items)

    assert len(batched) == 4
    for index, (output, single) in enumerate(zip(batched, singles, strict=True)):
        _assert_same_output(output, single, profile)
        assert output.metadata == {"fixture_index": index}
        assert len(output.label_ids) == len(output.scores) == len(output.bboxes)
        assert np.isfinite(output.scores).all()


@pytest.mark.mlx
def test_real_medium_initializes_shared_engine_once(
    egret_medium_artifact: tuple[Path, str],
) -> None:
    _require_metal()
    artifacts_root, repo_id = egret_medium_artifact
    from docling_mlx._models.dfine.model import DFine

    with patch("docling_mlx._models.dfine.model.DFine", wraps=DFine) as constructor:
        engine = _engine(artifacts_root, repo_id)
        barrier = Barrier(2)

        def predict_in_parallel(index: int) -> ObjectDetectionEngineOutput:
            barrier.wait(timeout=30)
            engine.initialize()
            return engine.predict_batch([_input(index)])[0]

        with ThreadPoolExecutor(max_workers=2) as pool:
            outputs = list(pool.map(predict_in_parallel, range(2)))

    assert engine._initialized
    assert constructor.call_count == 1
    assert [output.metadata for output in outputs] == [
        {"fixture_index": 0},
        {"fixture_index": 1},
    ]
    for output in outputs:
        assert np.isfinite(output.scores).all()


def _pipeline_options(
    artifacts_root: Path,
    repo_id: str,
    *,
    revision: str = "local-test-revision",
    threshold: float = 0.3,
) -> ThreadedPdfPipelineOptions:
    return ThreadedPdfPipelineOptions(
        artifacts_path=artifacts_root,
        allow_external_plugins=True,
        do_ocr=False,
        do_table_structure=False,
        do_code_enrichment=False,
        do_formula_enrichment=False,
        do_picture_classification=False,
        layout_options=MlxLayoutObjectDetectionOptions(
            model_spec=ObjectDetectionModelSpec(
                name="Test Egret", repo_id=repo_id, revision=revision
            ),
            engine_options=MlxObjectDetectionEngineOptions(score_threshold=threshold),
        ),
    )


@pytest.mark.parametrize(
    ("repo_id", "revision", "threshold"),
    [
        ("test-egret/other", "local-test-revision", 0.3),
        (MEDIUM_REPO_ID, "refs/pr/17", 0.3),
        (MEDIUM_REPO_ID, "local-test-revision", 0.4),
    ],
    ids=["repo-id", "revision", "threshold"],
)
def test_pipeline_option_variants_produce_distinct_hashes(
    tmp_path: Path, repo_id: str, revision: str, threshold: float
) -> None:
    baseline = _pipeline_options(tmp_path, MEDIUM_REPO_ID)
    variant = _pipeline_options(tmp_path, repo_id, revision=revision, threshold=threshold)

    assert create_pipeline_options_hash(baseline) != create_pipeline_options_hash(variant)


@pytest.mark.mlx
def test_offline_pdf_plugin_smoke_uses_selected_egret_profile(
    egret_profile_artifact: tuple[ProfileName, Path, str],
) -> None:
    _require_metal()
    profile, artifacts_root, repo_id = egret_profile_artifact
    assert PDF_FIXTURE.is_file()
    script = r"""
import json
import os
from pathlib import Path
from unittest.mock import patch

from docling.datamodel.base_models import InputFormat
from docling.datamodel.backend_options import ThreadedDoclingParseBackendOptions
from docling.datamodel.pipeline_options import ObjectDetectionModelSpec, ThreadedPdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from docling_mlx.stages.layout import MlxObjectDetectionEngineOptions
from docling_mlx.stages.layout import MlxLayoutObjectDetectionModel, MlxLayoutObjectDetectionOptions
import docling_mlx.stages.layout as engine_module

artifacts_root = Path(os.environ["EGRET_ARTIFACTS_ROOT"]).resolve()
repo_id = os.environ["EGRET_REPO_ID"]
fixture = Path(os.environ["EGRET_PDF"]).resolve()
original_resolve = engine_module.resolve_artifact_checkpoint

def offline_resolve(request_repo_id, revision, request_artifacts_path, **kwargs):
    if request_repo_id != repo_id or Path(request_artifacts_path).resolve() != artifacts_root:
        raise AssertionError("offline smoke attempted an unexpected artifact resolution")
    return original_resolve(request_repo_id, revision, request_artifacts_path, **kwargs)

options = ThreadedPdfPipelineOptions(
    artifacts_path=artifacts_root,
    allow_external_plugins=True,
    do_ocr=False,
    do_table_structure=False,
    do_code_enrichment=False,
    do_formula_enrichment=False,
    do_picture_classification=False,
    layout_options=MlxLayoutObjectDetectionOptions(
        model_spec=ObjectDetectionModelSpec(
            name="Test Egret", repo_id=repo_id, revision="local-test-revision"
        ),
        engine_options=MlxObjectDetectionEngineOptions(),
    ),
)
converter = DocumentConverter(
    allowed_formats=[InputFormat.PDF],
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=options,
            # One parser thread is equal-side stability containment for this release smoke.
            backend_options=ThreadedDoclingParseBackendOptions(parser_threads=1),
        )
    },
)
with patch.object(engine_module, "resolve_artifact_checkpoint", offline_resolve):
    result = converter.convert(fixture)
pipeline = next(iter(converter.initialized_pipelines.values()))
payload = {
    "status": result.status.value,
    "errors": [str(error) for error in result.errors],
    "stage": type(pipeline.layout_model).__name__,
    "engine": type(pipeline.layout_model.engine).__name__,
    "artifact": str(pipeline.layout_model.engine.artifact_path),
}
print("EGRET_SMOKE=" + json.dumps(payload, separators=(",", ":")))
"""
    environment = os.environ | {
        "EGRET_ARTIFACTS_ROOT": str(artifacts_root),
        "EGRET_REPO_ID": repo_id,
        "EGRET_PDF": str(PDF_FIXTURE),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    marker = next(
        line.removeprefix("EGRET_SMOKE=")
        for line in completed.stdout.splitlines()
        if line.startswith("EGRET_SMOKE=")
    )
    payload = json.loads(marker)
    assert payload["status"] == "success", payload["errors"]
    assert payload["stage"] == MlxLayoutObjectDetectionModel.__name__
    assert payload["engine"] == "_MlxObjectDetectionEngine"
    assert payload["artifact"] == str(artifacts_root / repo_id.replace("/", "--"))
    assert repo_id == f"test-egret/{profile}"
