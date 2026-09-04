# SPDX-License-Identifier: Apache-2.0

"""Native MLX TableFormer v1 architecture."""

from typing import TYPE_CHECKING, Any

from docling_mlx._models.tableformer_v1.config import (
    TABLEFORMER_V1_DATA_CELL_TOKEN_IDS,
    TABLEFORMER_V1_TOKENS,
    TableFormerV1Config,
)

if TYPE_CHECKING:
    from docling_mlx._models.tableformer_v1.vision import (
        TableFormerV1Encoder,
        TableFormerV1ImageEncoder,
        TableFormerV1TagEncoder,
    )

__all__ = [
    "TABLEFORMER_V1_TOKENS",
    "TABLEFORMER_V1_DATA_CELL_TOKEN_IDS",
    "TableFormerV1Config",
    "TableFormerV1Encoder",
    "TableFormerV1ImageEncoder",
    "TableFormerV1TagEncoder",
]


def __getattr__(name: str) -> Any:
    if name in {"TableFormerV1Encoder", "TableFormerV1ImageEncoder", "TableFormerV1TagEncoder"}:
        from docling_mlx._models.tableformer_v1.vision import (
            TableFormerV1Encoder,
            TableFormerV1ImageEncoder,
            TableFormerV1TagEncoder,
        )

        return {
            "TableFormerV1Encoder": TableFormerV1Encoder,
            "TableFormerV1ImageEncoder": TableFormerV1ImageEncoder,
            "TableFormerV1TagEncoder": TableFormerV1TagEncoder,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
