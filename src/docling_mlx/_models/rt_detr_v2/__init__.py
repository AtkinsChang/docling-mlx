# SPDX-License-Identifier: Apache-2.0

"""Minimal config-driven native MLX RT-DETR-v2 implementation."""

from typing import TYPE_CHECKING, Any

from .config import RtDetrV2Config

if TYPE_CHECKING:
    from .model import RtDetrV2, RtDetrV2Output

__all__ = ["RtDetrV2", "RtDetrV2Config", "RtDetrV2Output"]


def __getattr__(name: str) -> Any:
    """Keep config-only imports free of MLX and Metal initialization."""
    if name in {"RtDetrV2", "RtDetrV2Output"}:
        from .model import RtDetrV2, RtDetrV2Output

        return {"RtDetrV2": RtDetrV2, "RtDetrV2Output": RtDetrV2Output}[name]
    raise AttributeError(name)
