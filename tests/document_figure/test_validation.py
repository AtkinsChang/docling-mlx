# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.document_figure.validate_parity import validate

pytestmark = [pytest.mark.release, pytest.mark.mlx]


@pytest.fixture(scope="module", autouse=True)
def _require_mlx_and_artifact() -> None:
    probe = subprocess.run(
        [sys.executable, "-c", "import mlx.core"],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode:
        pytest.fail(f"selected release MLX lane requires Metal: {probe.stderr.strip()}")
    artifact = Path(__file__).resolve().parents[2] / ".artifacts/document-figure-classifier"
    required = ("model.safetensors", "config.json", "preprocessor_config.json")
    if any(not (artifact / name).is_file() for name in required):
        pytest.fail(f"selected release MLX lane requires the converted artifact at {artifact}")


def test_empty_reference_cannot_report_success(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    metadata = json.loads((root / "tests/golden/document_figure/metadata.json").read_text())
    metadata["captures"] = []
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="at least one"):
        validate(root / ".artifacts/document-figure-classifier", tmp_path)
