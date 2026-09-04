# SPDX-License-Identifier: Apache-2.0

"""Real-artifact release contracts for the Heron Docling integration.

These tests intentionally use converted local artifacts.  They are skipped in a
developer checkout only when those ignored artifacts have not been provisioned;
the Apple-Silicon release environment provisions both profiles and must run them
without skips.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from typing import Literal
from unittest.mock import patch

import numpy as np
import pytest
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.pipeline_options import ObjectDetectionModelSpec, ThreadedPdfPipelineOptions
from docling.models.inference_engines.object_detection.base import ObjectDetectionEngineInput
from docling.utils.pipeline_cache import create_pipeline_options_hash
from docling_core.types.doc import Size
from PIL import Image

from docling_mlx.stages.layout import (
    MlxLayoutObjectDetectionModel,
    MlxLayoutObjectDetectionOptions,
    MlxObjectDetectionEngineOptions,
)

pytestmark = pytest.mark.release

ROOT = Path(__file__).resolve().parents[2]
PDF_FIXTURE = ROOT / "tests/fixtures/document_figure/picture_classification.pdf"
ProfileName = Literal["r50", "r101"]


def _repo_id(profile: ProfileName) -> str:
    return f"test-heron/{profile}"


def _artifact_path(profile: ProfileName) -> Path:
    environment_name = f"DOCLING_MLX_HERON_{profile.upper()}_ARTIFACT"
    if override := os.environ.get(environment_name):
        return Path(override).expanduser()
    return ROOT / ".artifacts" / f"heron-{profile}"


@pytest.fixture(params=["r50", "r101"], ids=["r50", "r101"])
def heron_artifact(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[ProfileName, Path, str]:
    profile = request.param
    assert profile in {"r50", "r101"}
    artifact = _artifact_path(profile)
    required = ["config.json", "preprocessor_config.json", "model.safetensors"]
    if not artifact.is_dir() or any(not (artifact / name).is_file() for name in required):
        pytest.fail(
            f"provision {profile} local artifact at {artifact} or set "
            f"DOCLING_MLX_HERON_{profile.upper()}_ARTIFACT; release runs require zero skips"
        )
    repo_id = _repo_id(profile)
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir()
    (artifacts_root / repo_id.replace("/", "--")).symlink_to(
        artifact.resolve(), target_is_directory=True
    )
    return profile, artifacts_root, repo_id


def _engine(artifacts_root: Path, repo_id: str, *, threshold: float = 0.3):
    return MlxLayoutObjectDetectionModel(
        artifacts_path=artifacts_root,
        accelerator_options=AcceleratorOptions(device="auto"),
        options=MlxLayoutObjectDetectionOptions(
            model_spec=ObjectDetectionModelSpec(
                name="release-heron", repo_id=repo_id, revision="local-test-revision"
            ),
            engine_options=MlxObjectDetectionEngineOptions(score_threshold=threshold),
        ),
    ).engine


def _require_metal() -> None:
    """Fail a selected release lane when Metal is unavailable."""

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
    """Use a non-square RGB input so processor and original-pixel metadata matter."""

    image = Image.new("RGB", (401, 534), color=(16 + index, 32 + index, 64 + index))
    return ObjectDetectionEngineInput(image=image, metadata={"fixture_index": index})


def _assert_same_output(actual, expected) -> None:
    assert actual.label_ids == expected.label_ids
    assert actual.metadata == expected.metadata
    np.testing.assert_allclose(actual.scores, expected.scores, rtol=0, atol=1e-6)
    np.testing.assert_allclose(actual.bboxes, expected.bboxes, rtol=0, atol=5e-4)


@pytest.mark.mlx
def test_real_profiles_preserve_batch_order_and_metadata(heron_artifact) -> None:
    """Both released profiles must agree between batch one and batch four."""

    _require_metal()
    _, artifacts_root, repo_id = heron_artifact
    engine = _engine(artifacts_root, repo_id)
    items = [_input(index) for index in range(4)]

    single = engine.predict_batch(items[:1])[0]
    batched = engine.predict_batch(items)

    assert len(batched) == 4
    _assert_same_output(batched[0], single)
    for index, output in enumerate(batched):
        assert output.metadata == {"fixture_index": index}
        assert len(output.label_ids) == len(output.scores) == len(output.bboxes)
        assert np.isfinite(output.scores).all()


@pytest.mark.mlx
def test_real_profiles_initialize_shared_engine_once(heron_artifact) -> None:
    """One engine instance must initialize once and serve two concurrent callers."""

    _require_metal()
    _, artifacts_root, repo_id = heron_artifact
    from docling_mlx._models.rt_detr_v2 import RtDetrV2

    with patch("docling_mlx._models.rt_detr_v2.RtDetrV2", wraps=RtDetrV2) as constructor:
        engine = _engine(artifacts_root, repo_id)
        barrier = Barrier(2)

        def predict_in_parallel(index: int):
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


@pytest.mark.mlx
def test_real_stage_filters_invalid_pages_and_preserves_docling_coordinates(heron_artifact) -> None:
    """Exercise inherited page batching, scaling, clipping and raw clusters."""

    _require_metal()
    _, artifacts_root, repo_id = heron_artifact
    options = MlxLayoutObjectDetectionOptions(
        model_spec=ObjectDetectionModelSpec(
            name="Test Heron", repo_id=repo_id, revision="local-test-revision"
        ),
        engine_options=MlxObjectDetectionEngineOptions(score_threshold=0.0),
        keep_empty_clusters=True,
    )
    stage = MlxLayoutObjectDetectionModel(
        artifacts_path=artifacts_root,
        accelerator_options=AcceleratorOptions(device="auto"),
        options=options,
    )

    class PageBackend:
        def __init__(self, *, valid: bool, image: Image.Image) -> None:
            self._valid = valid
            self._image = image

        def is_valid(self) -> bool:
            return self._valid

        def get_page_image(self, scale: float) -> Image.Image:
            assert scale == 1.0
            return self._image

    valid = Page(page_no=1, size=Size(width=802, height=1068))
    valid._backend = PageBackend(valid=True, image=_input().image)
    invalid = Page(page_no=2, size=Size(width=802, height=1068))
    invalid._backend = PageBackend(valid=False, image=_input().image)

    captured = []
    original_predict = stage.engine.predict_batch

    def capture_batch(inputs):
        captured.extend(inputs)
        return original_predict(inputs)

    stage.engine.predict_batch = capture_batch  # type: ignore[method-assign]
    predictions = stage.predict_layout(SimpleNamespace(timings={}), [valid, invalid])

    assert [item.metadata for item in captured] == [{"page_no": 1}]
    assert len(predictions) == 2
    assert predictions[1].clusters == []
    assert predictions[0].clusters
    for cluster in predictions[0].clusters:
        left, top, right, bottom = cluster.bbox.as_tuple()
        assert 0.0 <= left <= right <= 802.0
        assert 0.0 <= top <= bottom <= 1068.0


def _pipeline_options(
    artifacts_root: Path,
    repo_id: str,
    *,
    revision: str = "local-test-revision",
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
                name="Test Heron", repo_id=repo_id, revision=revision
            ),
            engine_options=MlxObjectDetectionEngineOptions(),
        ),
    )


@pytest.mark.parametrize(
    ("repo_id", "revision"),
    [
        ("test-heron/r101", "local-test-revision"),
        ("test-heron/r50", "main"),
    ],
    ids=["profile", "revision"],
)
def test_pipeline_option_variants_produce_distinct_hashes(
    tmp_path: Path, repo_id: str, revision: str
) -> None:
    baseline = _pipeline_options(tmp_path, _repo_id("r50"))
    variant = _pipeline_options(tmp_path, repo_id, revision=revision)

    assert create_pipeline_options_hash(baseline) != create_pipeline_options_hash(variant)


@pytest.mark.mlx
def test_offline_pdf_plugin_smoke_uses_the_selected_profile(heron_artifact) -> None:
    """Run a cleanly configured standard PDF pipeline through external plugin dispatch."""

    _require_metal()
    profile, artifacts_root, repo_id = heron_artifact
    assert PDF_FIXTURE.is_file()
    # Docling's parser owns native worker threads. Run each profile in a clean
    # interpreter so earlier Metal/concurrency tests cannot contaminate parser
    # lifetime, and make an unexpected non-local artifact resolution fatal.
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

artifacts_root = Path(os.environ["HERON_ARTIFACTS_ROOT"]).resolve()
repo_id = os.environ["HERON_REPO_ID"]
fixture = Path(os.environ["HERON_PDF"]).resolve()
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
            name="Test Heron", repo_id=repo_id, revision="local-test-revision"
        ),
        engine_options=MlxObjectDetectionEngineOptions()
    ),
)
converter = DocumentConverter(
    allowed_formats=[InputFormat.PDF],
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=options,
            # This smoke qualifies the plugin boundary, not docling-parse's native
            # multi-thread scheduler. Keep the default backend class but use one
            # parser thread to avoid an unrelated native parser race.
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
print("HERON_SMOKE=" + json.dumps(payload, separators=(",", ":")))
"""
    environment = os.environ | {
        "HERON_ARTIFACTS_ROOT": str(artifacts_root),
        "HERON_REPO_ID": repo_id,
        "HERON_PDF": str(PDF_FIXTURE),
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
        line.removeprefix("HERON_SMOKE=")
        for line in completed.stdout.splitlines()
        if line.startswith("HERON_SMOKE=")
    )
    payload = json.loads(marker)
    assert payload["status"] == "success", payload["errors"]
    assert payload["stage"] == MlxLayoutObjectDetectionModel.__name__
    assert payload["engine"] == "_MlxObjectDetectionEngine"
    assert payload["artifact"] == str(artifacts_root / repo_id.replace("/", "--")), profile
