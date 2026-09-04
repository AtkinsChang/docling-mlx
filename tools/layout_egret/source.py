# SPDX-License-Identifier: Apache-2.0

"""Pinned upstream Egret source metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class EgretSource:
    """One official Egret repository at the revision a conversion records."""

    repo_id: str
    revision: str


SOURCES: Mapping[str, EgretSource] = MappingProxyType(
    {
        "medium": EgretSource(
            "docling-project/docling-layout-egret-medium",
            "77ede7cc7bed96d853c58f319734803d6ea2ea5c",
        ),
        "large": EgretSource(
            "docling-project/docling-layout-egret-large",
            "fff417c78abd6bab338c87706c95a8d79dc68f1e",
        ),
        "xlarge": EgretSource(
            "docling-project/docling-layout-egret-xlarge",
            "23857d16596e0106716b3162d132212d733769e7",
        ),
    }
)

__all__ = ["EgretSource", "SOURCES"]
