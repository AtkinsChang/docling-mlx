# SPDX-License-Identifier: Apache-2.0

"""Release contract against compact pinned Torch CPU TableFormerV2 goldens."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from docling_mlx._models.tableformer_v2.config import TABLEFORMER_V2_TOKENS
from docling_mlx.engines.table_structure.tableformer_v2 import (
    TableFormerV2Engine,
    TableFormerV2ModelSpec,
)
from docling_mlx.engines.table_structure.tableformer_v2.artifact import (
    CHECKPOINT_FILES,
)
from tools.tableformer_v2.source import SOURCE_REPO, SOURCE_REVISION

pytestmark = [pytest.mark.mlx, pytest.mark.release]

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests/golden/tableformer_v2/release.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _otsl(token_ids: tuple[int, ...]) -> list[str]:
    return [TABLEFORMER_V2_TOKENS[token_id][1:-1] for token_id in token_ids if token_id > 3]


def _required_artifact() -> Path:
    artifact = Path(
        os.environ.get(
            "DOCLING_MLX_TABLEFORMER_V2_ARTIFACT",
            ROOT / ".artifacts/tableformer-v2",
        )
    ).expanduser()
    missing = [name for name in CHECKPOINT_FILES if not (artifact / name).is_file()]
    if missing:
        pytest.fail(f"TableFormerV2 release artifact {artifact} is missing {missing}")
    return artifact.resolve()


def test_real_table_generation_matches_pinned_torch_cpu_golden() -> None:
    golden: dict[str, Any] = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert golden["schema_version"] == 1
    assert golden["profile"] == "tableformer_v2"
    assert golden["source"] == {"repo_id": SOURCE_REPO, "revision": SOURCE_REVISION}

    artifact = _required_artifact()
    engine = TableFormerV2Engine(TableFormerV2ModelSpec(path=artifact))
    images = []
    for fixture in golden["fixtures"]:
        image_path = ROOT / fixture["image"]
        assert _sha256(image_path) == fixture["image_sha256"]
        with Image.open(image_path) as source:
            images.append(source.convert("RGB"))

    predictions = engine.predict(images)

    assert len(predictions) == len(golden["fixtures"])
    for prediction, fixture, image in zip(predictions, golden["fixtures"], images, strict=True):
        expected_ids = tuple(fixture["token_ids"])
        assert prediction.token_ids == expected_ids
        assert _otsl(prediction.token_ids) == fixture["otsl"]
        np.testing.assert_allclose(
            np.asarray(prediction.cell_bboxes)
            / np.asarray((image.width, image.height, image.width, image.height)),
            fixture["normalized_bboxes"],
            rtol=0,
            atol=1e-4,
        )
