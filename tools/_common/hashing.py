# SPDX-License-Identifier: Apache-2.0

"""Shared hashing and package-version helpers for repository-only tools."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _package_versions(names: Sequence[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def hash_named_files(directory: Path, names: Sequence[str]) -> str:
    """Hash a named-file set without exposing local paths in reports."""

    digest = hashlib.sha256()
    for name in names:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing benchmark artifact file: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()
