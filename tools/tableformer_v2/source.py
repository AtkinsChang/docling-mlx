# SPDX-License-Identifier: Apache-2.0

"""Pinned TableFormerV2 source download helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

SOURCE_REPO = "docling-project/TableFormerV2"
SOURCE_REVISION = "51559fad3946873e26a6f9b8e912f948e8745bef"
SOURCE_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_source(source: Path) -> None:
    """Require the files consumed by the converter."""

    for name in SOURCE_FILES:
        path = source / name
        if not path.is_file():
            raise ValueError(f"Missing source file: {path}")


def download_source(*, cache_dir: Path | None = None) -> Path:
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
