# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import huggingface_hub
import pytest

from tools._common.model_card import lookup_model_license, render_model_card


def test_render_model_card_has_the_exact_shared_template() -> None:
    assert (
        render_model_card("owner/model", "abc123", "apache-2.0")
        == """---
license: apache-2.0
library_name: docling-mlx
tags:
  - mlx
base_model:
  - owner/model
---

# model - MLX

MLX-converted weights of [`owner/model`](https://huggingface.co/owner/model) at revision `abc123`.
"""
    )


@pytest.mark.parametrize(
    ("repo", "revision", "license", "expected_license"),
    [
        (
            "docling-project/DocumentFigureClassifier-v2.5",
            "f859dfbff5c9916cd996942d4b0db7fa25808220",
            "mit",
            "license: mit",
        ),
        (
            "docling-project/docling-layout-heron",
            "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8",
            "apache-2.0",
            "license: apache-2.0",
        ),
        (
            "docling-project/docling-layout-egret-medium",
            "77ede7cc7bed96d853c58f319734803d6ea2ea5c",
            "apache-2.0",
            "license: apache-2.0",
        ),
        (
            "docling-project/docling-models",
            "fc0f2d45e2218ea24bce5045f58a389aed16dc23",
            ["cdla-permissive-2.0", "apache-2.0"],
            "license:\n  - cdla-permissive-2.0\n  - apache-2.0",
        ),
        (
            "docling-project/TableFormerV2",
            "51559fad3946873e26a6f9b8e912f948e8745bef",
            None,
            "",
        ),
    ],
)
def test_model_card_samples(repo: str, revision: str, license, expected_license: str) -> None:
    card = render_model_card(repo, revision, license)
    assert "library_name: docling-mlx" in card
    if license is None:
        assert "license:" not in card
    else:
        assert expected_license in card
    assert f"# {repo.rsplit('/', 1)[-1]} - MLX" in card
    assert f"at revision `{revision}`." in card


def test_lookup_uses_hub_card_license_first(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []

    def model_info(repo_id: str, *, revision: str):
        calls.append((repo_id, revision))
        return SimpleNamespace(card_data=SimpleNamespace(license="apache-2.0"))

    monkeypatch.setattr(huggingface_hub, "model_info", model_info)

    assert lookup_model_license(tmp_path, "owner/model", "abc123") == "apache-2.0"
    assert calls == [("owner/model", "abc123")]


def test_lookup_falls_back_to_local_front_matter(monkeypatch, tmp_path: Path) -> None:
    def model_info(repo_id: str, *, revision: str):
        return SimpleNamespace(card_data=SimpleNamespace(license=None))

    monkeypatch.setattr(huggingface_hub, "model_info", model_info)
    (tmp_path / "README.md").write_text(
        "---\nlicense:\n  - cdla-permissive-2.0\n  - apache-2.0\n---\n"
    )

    assert lookup_model_license(tmp_path, "owner/model", "abc123") == [
        "cdla-permissive-2.0",
        "apache-2.0",
    ]


def test_lookup_reports_missing_license(monkeypatch, tmp_path: Path, capsys) -> None:
    def model_info(repo_id: str, *, revision: str):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(huggingface_hub, "model_info", model_info)

    assert lookup_model_license(tmp_path, "owner/model", "abc123") is None
    error = capsys.readouterr().err
    assert "Hub license lookup failed for owner/model at abc123: offline" in error
    assert "No license metadata for owner/model at abc123" in error
