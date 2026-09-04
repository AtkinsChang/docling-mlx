# SPDX-License-Identifier: Apache-2.0

"""Black-box checks for the distributable package boundary.

These checks build a fresh wheel instead of inspecting ``dist/`` so a stale
local build cannot hide a newly included source directory.  The clean install
uses the wheel as the only project input; running Python with ``-I`` keeps the
checkout off ``sys.path`` while it verifies the intentionally lightweight base
import.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tomllib
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path

import pytest
from packaging.requirements import Requirement

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
PROJECT_VERSION = PYPROJECT["project"]["version"]
FORBIDDEN_PATH_PARTS = frozenset(
    {
        ".artifacts",
        ".reference",
        "docs",
        "reports",
        "tests",
        "tools",
        "reference",
        "bundle",
    }
)
FORBIDDEN_SUFFIXES = frozenset({".npz", ".onnx", ".pt", ".pth", ".safetensors"})


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
    )


def _install_wheel_venv(
    workspace: Path,
    wheel: Path,
    *,
    uv: str,
    env: dict[str, str],
) -> Path:
    venv = workspace / "venv"
    _run([uv, "venv", "--python", sys.executable, str(venv)], cwd=workspace, env=env)
    python = venv / "bin" / "python"
    _run([uv, "pip", "install", "--python", str(python), str(wheel)], cwd=workspace, env=env)
    _run([uv, "pip", "check", "--python", str(python)], cwd=workspace, env=env)
    return python


def _run_isolated_json(
    python: Path, script: str, *, cwd: Path, env: dict[str, str]
) -> dict[str, object]:
    completed = _run([str(python), "-I", "-c", script], cwd=cwd, env=env)
    return json.loads(completed.stdout)


@pytest.fixture(scope="session")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    uv = shutil.which("uv")
    if uv is None:
        pytest.fail("Packaging validation requires uv on PATH")

    workspace = tmp_path_factory.mktemp("wheel-build")
    output_dir = workspace / "wheel"
    cache_dir = workspace / "uv-cache"
    environment = os.environ | {"UV_CACHE_DIR": str(cache_dir)}
    _run(
        [uv, "build", "--wheel", "--out-dir", str(output_dir)],
        cwd=REPOSITORY_ROOT,
        env=environment,
    )

    wheels = list(output_dir.glob("docling_mlx-*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def _wheel_members(wheel: Path) -> tuple[str, ...]:
    with zipfile.ZipFile(wheel) as archive:
        return tuple(archive.namelist())


def _read_wheel_member(wheel: Path, suffix: str) -> bytes:
    with zipfile.ZipFile(wheel) as archive:
        members = [member for member in archive.namelist() if member.endswith(suffix)]
        assert len(members) == 1
        return archive.read(members[0])


def test_wheel_contains_only_runtime_package_and_license_material(built_wheel: Path) -> None:
    members = _wheel_members(built_wheel)

    assert "docling_mlx/__init__.py" in members
    assert "docling_mlx/py.typed" in members
    license_members = {
        member.split(".dist-info/licenses/", 1)[1]
        for member in members
        if ".dist-info/licenses/" in member
    }
    assert license_members == {
        "LICENSE",
        "NOTICE",
        "LICENSES/Apache-2.0.txt",
        "LICENSES/MIT.txt",
    }

    forbidden_members = [
        member
        for member in members
        if FORBIDDEN_PATH_PARTS.intersection(Path(member).parts)
        or Path(member).suffix.lower() in FORBIDDEN_SUFFIXES
    ]
    assert forbidden_members == []


def test_wheel_declares_docling_plugin_entry_point(built_wheel: Path) -> None:
    entry_points = _read_wheel_member(built_wheel, ".dist-info/entry_points.txt").decode("utf-8")

    assert "[docling]" in entry_points
    assert "docling_mlx = docling_mlx.plugins" in entry_points


def test_wheel_declares_docling_dependency_boundaries(built_wheel: Path) -> None:
    """Keep Docling's VLM stack transitive instead of pinning it ourselves."""

    metadata = BytesParser(policy=default).parsebytes(
        _read_wheel_member(built_wheel, ".dist-info/METADATA")
    )

    requirements = metadata.get_all("Requires-Dist", [])
    parsed = [Requirement(requirement) for requirement in requirements]
    unconditional = {item.name for item in parsed if item.marker is None}
    assert "docling-slim" in unconditional
    assert "docling" not in unconditional
    assert not any(
        requirement.startswith(("mlx-vlm", "mlx-lm", "transformers"))
        for requirement in requirements
    )


@pytest.fixture(scope="session")
def clean_wheel_python(
    tmp_path_factory: pytest.TempPathFactory, built_wheel: Path
) -> tuple[Path, Path]:
    """Install the fresh wheel and its declared runtime dependencies in a new venv."""

    uv = shutil.which("uv")
    if uv is None:
        pytest.fail("Clean-wheel validation requires uv on PATH")

    workspace = tmp_path_factory.mktemp("wheel-install")
    cache_dir = workspace / "uv-cache"
    environment = os.environ | {"UV_CACHE_DIR": str(cache_dir)}
    python = _install_wheel_venv(workspace, built_wheel, uv=uv, env=environment)
    return python, workspace


def test_clean_wheel_import_has_no_reference_frameworks(
    clean_wheel_python: tuple[Path, Path],
) -> None:
    python, workspace = clean_wheel_python
    script = """
import importlib.metadata
import importlib.abc
import json
import sys

import docling_mlx
import docling_mlx.plugins

class ReferenceFrameworkBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {
            'docling_ibm_models', 'mlx', 'torch', 'torchvision', 'transformers'
        }:
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, ReferenceFrameworkBlocker())
plugin_stages = {
    'layout': [
        stage.__name__
        for stage in docling_mlx.plugins.layout_engines()['layout_engines']
    ],
    'table': [
        stage.__name__
        for stage in docling_mlx.plugins.table_structure_engines()['table_structure_engines']
    ],
}

forbidden = (
    "cv2", "docling_ibm_models", "torch", "torchvision", "transformers", "mlx_vlm"
)
loaded = sorted(
    name
    for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
)
entry_points = [
    entry_point.value
    for entry_point in importlib.metadata.entry_points(group="docling")
    if entry_point.name == "docling_mlx"
]
installed_names = {
    distribution.metadata['Name'].lower()
    for distribution in importlib.metadata.distributions()
    if distribution.metadata['Name']
}
installed_reference_frameworks = sorted(
    name for name in (
        'docling-ibm-models', 'opencv-python', 'opencv-python-headless',
        'torch', 'torchvision', 'transformers', 'mlx-vlm', 'mlx-lm'
    )
    if name in installed_names
)
print(
    json.dumps(
        {
            "version": docling_mlx.__version__,
            "entry_points": entry_points,
            "loaded": loaded,
            "installed_reference_frameworks": installed_reference_frameworks,
            "plugin_stages": plugin_stages,
        }
    )
)
"""
    payload = _run_isolated_json(python, script, cwd=workspace, env=os.environ.copy())

    assert payload["version"] == PROJECT_VERSION
    assert payload["entry_points"] == ["docling_mlx.plugins"]
    assert payload["loaded"] == []
    assert payload["installed_reference_frameworks"] == []
    assert payload["plugin_stages"] == {
        "layout": ["MlxLayoutObjectDetectionModel"],
        "table": [
            "MlxGraniteVisionTableStructureModel",
            "MlxTableFormerV2Model",
            "MlxTableFormerV1Model",
        ],
    }


@pytest.mark.mlx
@pytest.mark.release
def test_clean_base_wheel_runs_egret_offline(
    clean_wheel_python: tuple[Path, Path],
) -> None:
    python, workspace = clean_wheel_python
    artifact = Path(
        os.environ.get(
            "DOCLING_MLX_EGRET_MEDIUM_ARTIFACT",
            REPOSITORY_ROOT / ".artifacts/egret-medium",
        )
    ).expanduser()
    required = ("model.safetensors", "config.json", "preprocessor_config.json")
    missing = [name for name in required if not (artifact / name).is_file()]
    if missing:
        pytest.fail(f"clean-wheel Egret medium artifact {artifact} is missing {missing}")

    script = f"""
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

import numpy as np
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.stage_model_specs import ObjectDetectionModelSpec
from docling.models.inference_engines.object_detection.base import ObjectDetectionEngineInput
from PIL import Image

import docling_mlx.plugins
from docling_mlx.stages.layout import (
    MlxLayoutObjectDetectionModel,
    MlxLayoutObjectDetectionOptions,
    MlxObjectDetectionEngineOptions,
)

artifact = Path({str(artifact.resolve())!r})
stage = MlxLayoutObjectDetectionModel(
    artifacts_path=artifact.parent,
    accelerator_options=AcceleratorOptions(device='auto'),
    options=MlxLayoutObjectDetectionOptions(
        model_spec=ObjectDetectionModelSpec(
            name='clean-wheel-egret', repo_id=artifact.name, revision='local'
        ),
        engine_options=MlxObjectDetectionEngineOptions(),
    ),
)
prediction = stage.engine.predict_batch([
    ObjectDetectionEngineInput(
        image=Image.new('RGB', (401, 534), color=(16, 32, 64)),
        metadata={{'fixture': 'clean-wheel'}},
    )
])[0]

installed = {{
    distribution.metadata['Name'].lower()
    for distribution in metadata.distributions()
    if distribution.metadata['Name']
}}
forbidden_installed = sorted(
    name for name in ('torch', 'torchvision', 'transformers') if name in installed
)
forbidden_loaded = sorted(
    name for name in sys.modules
    if name.split('.')[0] in {{'torch', 'torchvision', 'transformers'}}
)
print(json.dumps({{
    'layout_engines': [
        stage.__name__
        for stage in docling_mlx.plugins.layout_engines()['layout_engines']
    ],
    'metadata': prediction.metadata,
    'label_count': len(prediction.label_ids),
    'score_count': len(prediction.scores),
    'bbox_count': len(prediction.bboxes),
    'scores_finite': bool(np.isfinite(prediction.scores).all()),
    'bboxes_finite': bool(np.isfinite(prediction.bboxes).all()),
    'forbidden_installed': forbidden_installed,
    'forbidden_loaded': forbidden_loaded,
}}))
"""
    environment = os.environ | {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    payload = _run_isolated_json(python, script, cwd=workspace, env=environment)
    assert payload["layout_engines"] == ["MlxLayoutObjectDetectionModel"]
    assert payload["metadata"] == {"fixture": "clean-wheel"}
    assert payload["label_count"] == payload["score_count"] == payload["bbox_count"]
    assert payload["scores_finite"] is True
    assert payload["bboxes_finite"] is True
    assert payload["forbidden_installed"] == []
    assert payload["forbidden_loaded"] == []
