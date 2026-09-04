# Implemented after docling-ibm-models (docling_ibm_models/tableformer_v2); module
# structure, parameter names, and forward-pass order follow it so the published
# checkpoint loads unchanged.
# SPDX-License-Identifier: Apache-2.0
"""Closed configuration for the pinned TableFormerV2 checkpoint."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

TABLEFORMER_V2_TOKENS = (
    "<pad>",
    "[UNK]",
    "<start>",
    "<end>",
    "<ecel>",
    "<fcel>",
    "<lcel>",
    "<ucel>",
    "<xcel>",
    "<nl>",
    "<ched>",
    "<rhed>",
    "<srow>",
)
TABLEFORMER_V2_DATA_CELL_TOKEN_IDS = frozenset({4, 5, 10, 11, 12})


def _integer(raw: Mapping[str, Any], name: str) -> int:
    value = raw.get(name)
    if type(value) is not int:
        raise TypeError(f"TableFormerV2 {name} must be an integer")
    return value


def _real(raw: Mapping[str, Any], name: str) -> float:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"TableFormerV2 {name} must be a real number")
    return float(value)


@dataclass(frozen=True, slots=True)
class TableFormerV2Config:
    """The one architecture profile supported by the first native port."""

    model_type: str
    embed_dim: int
    num_heads: int
    ff_dim: int
    num_decoder_layers: int
    vocab_size: int
    conv_mixer_expansion: float
    data_cell_token_ids: frozenset[int]
    pad_token_id: int
    eos_token_id: int

    def __post_init__(self) -> None:
        hard = {
            "model_type": "TableFormerV2",
            "embed_dim": 512,
            "num_heads": 8,
            "ff_dim": 2048,
            "num_decoder_layers": 4,
            "vocab_size": len(TABLEFORMER_V2_TOKENS),
            "conv_mixer_expansion": 1.0,
            "data_cell_token_ids": TABLEFORMER_V2_DATA_CELL_TOKEN_IDS,
            "pad_token_id": 0,
            "eos_token_id": 3,
        }
        for name, value in hard.items():
            if getattr(self, name) != value:
                raise ValueError(
                    f"Unsupported TableFormerV2 {name}: {getattr(self, name)!r}; expected {value!r}"
                )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TableFormerV2Config:
        if not isinstance(raw, Mapping):
            raise TypeError("TableFormerV2 configuration must be a mapping")
        raw_cells = raw.get("data_cells")
        if not isinstance(raw_cells, Sequence) or isinstance(raw_cells, (str, bytes)):
            raise TypeError("TableFormerV2 data_cells must be a sequence")
        if any(type(value) is not int for value in raw_cells):
            raise TypeError("TableFormerV2 data_cells must contain integers")
        if len(set(raw_cells)) != len(raw_cells):
            raise ValueError("TableFormerV2 data_cells must not contain duplicates")
        return cls(
            model_type=str(raw.get("model_type", "")),
            embed_dim=_integer(raw, "embed_dim"),
            num_heads=_integer(raw, "num_heads"),
            ff_dim=_integer(raw, "ff_dim"),
            num_decoder_layers=_integer(raw, "num_decoder_layers"),
            vocab_size=_integer(raw, "vocab_size"),
            conv_mixer_expansion=_real(raw, "conv_mixer_expansion"),
            data_cell_token_ids=frozenset(raw_cells),
            pad_token_id=_integer(raw, "pad_token_id"),
            eos_token_id=_integer(raw, "eos_token_id"),
        )
