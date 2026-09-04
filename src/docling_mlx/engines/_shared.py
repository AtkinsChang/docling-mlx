# SPDX-License-Identifier: Apache-2.0

"""Checkpoint resolution shared by framework-free engines and Docling adaptors."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


class ModelSpec(Protocol):
    """The common local-or-Hub checkpoint identity used by native engines."""

    @property
    def repo_id(self) -> str | None: ...

    @property
    def revision(self) -> str | None: ...

    @property
    def path(self) -> Path | str | None: ...


def resolve_checkpoint(
    model_spec: ModelSpec,
    *,
    files: Sequence[str],
    component: str,
) -> Path:
    """Resolve one plain checkpoint directory without importing Docling or MLX."""

    if model_spec.path is not None:
        directory = Path(model_spec.path)
        if not directory.is_dir():
            raise FileNotFoundError(f"{component} model directory does not exist: {directory}")
        return directory
    if model_spec.repo_id is None:
        raise RuntimeError(f"{component} model spec has neither a path nor a repo_id")

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model_spec.repo_id,
            revision=model_spec.revision,
            allow_patterns=list(files),
        )
    )


def resolve_artifact_checkpoint(
    repo_id: str,
    revision: str,
    artifacts_path: Path | str | None,
    *,
    files: Sequence[str],
) -> Path:
    """Resolve Docling's optional artifact cache or the immutable Hub snapshot."""

    if artifacts_path is None:
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                allow_patterns=list(files),
            )
        )

    base = Path(artifacts_path) / repo_id.replace("/", "--")
    revision_directory = base / revision
    if revision_directory.is_dir():
        return revision_directory
    if base.is_dir():
        return base
    raise FileNotFoundError(f"Model artifact directory does not exist: {base}")


def require_checkpoint_files(directory: Path, files: Sequence[str]) -> None:
    """Fail with the first missing runtime file before parsing a checkpoint."""

    for name in files:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing model artifact: {path}")


def read_json_object(path: Path) -> dict[str, object]:
    """Read a checkpoint metadata object without importing Docling."""

    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return value
