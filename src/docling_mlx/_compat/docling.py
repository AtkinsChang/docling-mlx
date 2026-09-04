# SPDX-License-Identifier: Apache-2.0

"""Isolate upstream Docling private APIs; the supported lower bound lives in pyproject.toml."""

from __future__ import annotations

from typing import Any

from docling.models.inference_engines.vlm._utils import resolve_model_artifacts_path


def require_page_backend(page: object, stage_name: str) -> Any:
    """Return Docling's internal page backend or preserve the stage error contract."""

    backend = getattr(page, "_backend", None)
    if backend is None:
        raise RuntimeError(f"{stage_name} requires an initialized page backend")
    return backend


__all__ = [
    "require_page_backend",
    "resolve_model_artifacts_path",
]
