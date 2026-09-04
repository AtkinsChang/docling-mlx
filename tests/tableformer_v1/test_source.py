# SPDX-License-Identifier: Apache-2.0

"""Tests for the frozen TableFormer v1 source boundary."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from tools.tableformer_v1 import source


def test_verify_source_rejects_missing_source_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(source, "SOURCE_FILES", ("accurate/file",))
    with pytest.raises(ValueError, match="Missing source file"):
        source.verify_source(tmp_path)


def test_download_is_limited_to_the_frozen_source_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def snapshot_download(**kwargs: object) -> str:
        captured.update(kwargs)
        return str(tmp_path)

    hub = ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(source, "verify_source", lambda _source: None)

    assert source.download_source(cache_dir=tmp_path / "cache") == tmp_path
    assert captured == {
        "repo_id": source.SOURCE_REPO,
        "revision": source.SOURCE_REVISION,
        "cache_dir": tmp_path / "cache",
        "allow_patterns": list(source.SOURCE_FILES),
    }
