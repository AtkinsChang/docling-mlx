# SPDX-License-Identifier: Apache-2.0

"""Frozen TableFormer v1 source download helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

from docling.datamodel.pipeline_options import TableFormerMode

SOURCE_REPO = "docling-project/docling-models"
SOURCE_REVISION = "fc0f2d45e2218ea24bce5045f58a389aed16dc23"
SOURCE_FILES = (
    "README.md",
    "model_artifacts/tableformer/accurate/tableformer_accurate.safetensors",
    "model_artifacts/tableformer/accurate/tm_config.json",
    "model_artifacts/tableformer/fast/tableformer_fast.safetensors",
    "model_artifacts/tableformer/fast/tm_config.json",
)


def profile_id(mode: TableFormerMode) -> str:
    """Return the stable capture identity for a selected v1 checkpoint."""

    return f"tableformer_v1_{mode.value}"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one regular source file."""

    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_source(source: Path) -> None:
    """Require the files consumed by the converter."""

    for name in SOURCE_FILES:
        path = source / name
        if not path.is_file():
            raise ValueError(f"Missing source file: {path}")


def download_source(*, cache_dir: Path | None = None) -> Path:
    """Fetch both frozen TableFormer v1 profiles."""

    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=SOURCE_REPO,
            revision=SOURCE_REVISION,
            cache_dir=cache_dir,
            allow_patterns=list(SOURCE_FILES),
        )
    )
    verify_source(snapshot)
    return snapshot
