# SPDX-License-Identifier: Apache-2.0

"""Native MLX TableFormerV2 architecture."""

from typing import TYPE_CHECKING, Any

from docling_mlx._models.tableformer_v2.config import (
    TABLEFORMER_V2_TOKENS,
    TableFormerV2Config,
)

if TYPE_CHECKING:
    from docling_mlx._models.tableformer_v2.vision import TableFormerV2VisionEncoder

__all__ = [
    "TABLEFORMER_V2_TOKENS",
    "TableFormerV2Config",
    "TableFormerV2VisionEncoder",
    "is_ignored_source_key",
]


def __getattr__(name: str) -> Any:
    if name in {"TableFormerV2VisionEncoder", "is_ignored_source_key"}:
        from docling_mlx._models.tableformer_v2.vision import (
            TableFormerV2VisionEncoder,
            is_ignored_source_key,
        )

        return {
            "TableFormerV2VisionEncoder": TableFormerV2VisionEncoder,
            "is_ignored_source_key": is_ignored_source_key,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
