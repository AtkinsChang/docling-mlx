# SPDX-License-Identifier: Apache-2.0

"""Versions that ``uv.lock`` resolves, for tools that must run against exactly those.

The reference group in ``pyproject.toml`` pins the packages that produce reference outputs, and
``uv.lock`` records the version of everything else. Tools read the lock instead of restating
versions so a re-pin changes one place.
"""

from __future__ import annotations

import importlib.metadata
import tomllib
from collections.abc import Iterable
from pathlib import Path

LOCKFILE = Path(__file__).resolve().parents[1] / "uv.lock"


def locked_versions(names: Iterable[str]) -> dict[str, str]:
    """Return the version ``uv.lock`` resolves for each named package."""

    with LOCKFILE.open("rb") as handle:
        packages = tomllib.load(handle)["package"]
    resolved: dict[str, set[str]] = {}
    for package in packages:
        resolved.setdefault(package["name"], set()).add(package["version"])
    versions: dict[str, str] = {}
    for name in names:
        candidates = resolved.get(name)
        if candidates is None:
            raise KeyError(f"uv.lock does not resolve {name!r}")
        if len(candidates) != 1:
            raise KeyError(f"uv.lock resolves {name!r} to several versions: {sorted(candidates)}")
        versions[name] = next(iter(candidates))
    return versions


def require_locked_versions(names: Iterable[str], *, context: str) -> dict[str, str]:
    """Fail unless every named package is installed at the version ``uv.lock`` resolves."""

    expected = locked_versions(names)
    actual = {name: importlib.metadata.version(name) for name in expected}
    if actual != expected:
        raise RuntimeError(
            f"{context} requires the locked reference versions: expected {expected}, got {actual}"
        )
    return actual


__all__ = ["LOCKFILE", "locked_versions", "require_locked_versions"]
