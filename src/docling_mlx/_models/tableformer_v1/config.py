# Implemented after docling-ibm-models (docling_ibm_models/tableformer/models/table04_rs);
# module structure, parameter names, and forward-pass order follow it so the published
# checkpoint loads unchanged.
# SPDX-License-Identifier: Apache-2.0
"""Closed configuration for the native TableFormer v1 profiles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

TABLEFORMER_V1_TOKENS = (
    "<pad>",
    "<unk>",
    "<start>",
    "<end>",
    "ecel",
    "fcel",
    "lcel",
    "ucel",
    "xcel",
    "nl",
    "ched",
    "rhed",
    "srow",
)
TABLEFORMER_V1_DATA_CELL_TOKEN_IDS = frozenset({4, 5, 10, 11, 12})


def _integer(raw: Mapping[str, Any], name: str) -> int:
    value = raw.get(name)
    if type(value) is not int:
        raise TypeError(f"TableFormer v1 {name} must be an integer")
    return value


def _string(raw: Mapping[str, Any], name: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str):
        raise TypeError(f"TableFormer v1 {name} must be a string")
    return value


def _sequence(raw: Mapping[str, Any], name: str) -> Sequence[Any]:
    value = raw.get(name)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"TableFormer v1 {name} must be a sequence")
    return value


def _mapping(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"TableFormer v1 {name} must be a mapping")
    return value


@dataclass(frozen=True, slots=True)
class TableFormerV1Config:
    """The shared architecture metadata for the Accurate and Fast profiles."""

    model_type: str
    architectures: tuple[str, ...]
    architecture: str
    backbone: str
    image_size: int
    encoded_image_size: int
    embed_dim: int
    num_encoder_layers: int
    num_decoder_layers: int
    num_heads: int
    ff_dim: int
    tag_embed_dim: int
    tag_decoder_dim: int
    bbox_embed_dim: int
    tag_attention_dim: int
    bbox_attention_dim: int
    bbox_classes: int
    vocab_size: int
    vocab: tuple[str, ...]
    data_cell_token_ids: frozenset[int]
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    dtype: str = "float32"

    def __post_init__(self) -> None:
        expected = {
            "model_type": "tableformer_v1",
            "architectures": ("TableFormerV1",),
            "architecture": "TableModel04_rs",
            "backbone": "resnet18",
            "image_size": 448,
            "encoded_image_size": 28,
            "embed_dim": 512,
            "num_heads": 8,
            "ff_dim": 1024,
            "tag_embed_dim": 16,
            "tag_decoder_dim": 512,
            "bbox_embed_dim": 256,
            "tag_attention_dim": 256,
            "bbox_attention_dim": 512,
            "bbox_classes": 2,
            "vocab_size": len(TABLEFORMER_V1_TOKENS),
            "vocab": TABLEFORMER_V1_TOKENS,
            "data_cell_token_ids": TABLEFORMER_V1_DATA_CELL_TOKEN_IDS,
            "pad_token_id": 0,
            "bos_token_id": 2,
            "eos_token_id": 3,
            "dtype": "float32",
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(
                    f"Unsupported TableFormer v1 {name}: "
                    f"{getattr(self, name)!r}; expected {value!r}"
                )
        if (self.num_encoder_layers, self.num_decoder_layers) not in {(6, 6), (4, 2)}:
            raise ValueError(
                "Unsupported TableFormer v1 encoder/decoder layer depths: "
                f"{(self.num_encoder_layers, self.num_decoder_layers)!r}; "
                "expected Accurate (6, 6) or Fast (4, 2)"
            )

    @property
    def hidden_dim(self) -> int:
        """Legacy name retained by parity tooling."""

        return self.embed_dim

    @property
    def encoder_layers(self) -> int:
        """Legacy name retained by parity tooling."""

        return self.num_encoder_layers

    @property
    def decoder_layers(self) -> int:
        """Legacy name retained by parity tooling."""

        return self.num_decoder_layers

    @property
    def max_steps(self) -> int:
        """Legacy generation limit; runtime metadata lives in generation_config.json."""

        return 1024

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TableFormerV1Config:
        if not isinstance(raw, Mapping):
            raise TypeError("TableFormer v1 configuration must be a mapping")
        if "model_type" in raw:
            return cls._from_flat(raw)
        # Source/oracle tooling still reads the upstream nested tm_config.json.
        # Artifact validation never calls this branch because tm_config.json is not required.
        return cls._from_legacy(raw)

    @classmethod
    def _from_flat(cls, raw: Mapping[str, Any]) -> TableFormerV1Config:
        raw_architectures = _sequence(raw, "architectures")
        if any(not isinstance(value, str) for value in raw_architectures):
            raise TypeError("TableFormer v1 architectures must contain strings")
        raw_vocab = _sequence(raw, "vocab")
        if any(not isinstance(value, str) for value in raw_vocab):
            raise TypeError("TableFormer v1 vocab must contain strings")
        if len(set(raw_vocab)) != len(raw_vocab):
            raise ValueError("TableFormer v1 vocab must not contain duplicates")
        raw_cells = _sequence(raw, "data_cells")
        if any(type(value) is not int for value in raw_cells):
            raise TypeError("TableFormer v1 data_cells must contain integers")
        if len(set(raw_cells)) != len(raw_cells):
            raise ValueError("TableFormer v1 data_cells must not contain duplicates")
        return cls(
            model_type=_string(raw, "model_type"),
            architectures=tuple(raw_architectures),
            architecture=_string(raw, "architecture"),
            backbone=_string(raw, "backbone"),
            image_size=_integer(raw, "image_size"),
            encoded_image_size=_integer(raw, "encoded_image_size"),
            embed_dim=_integer(raw, "embed_dim"),
            num_encoder_layers=_integer(raw, "num_encoder_layers"),
            num_decoder_layers=_integer(raw, "num_decoder_layers"),
            num_heads=_integer(raw, "num_heads"),
            ff_dim=_integer(raw, "ff_dim"),
            tag_embed_dim=_integer(raw, "tag_embed_dim"),
            tag_decoder_dim=_integer(raw, "tag_decoder_dim"),
            bbox_embed_dim=_integer(raw, "bbox_embed_dim"),
            tag_attention_dim=_integer(raw, "tag_attention_dim"),
            bbox_attention_dim=_integer(raw, "bbox_attention_dim"),
            bbox_classes=_integer(raw, "bbox_classes"),
            vocab_size=_integer(raw, "vocab_size"),
            vocab=tuple(raw_vocab),
            data_cell_token_ids=frozenset(raw_cells),
            pad_token_id=_integer(raw, "pad_token_id"),
            bos_token_id=_integer(raw, "bos_token_id"),
            eos_token_id=_integer(raw, "eos_token_id"),
            dtype=str(raw.get("torch_dtype", raw.get("dtype", ""))),
        )

    @classmethod
    def _from_legacy(cls, raw: Mapping[str, Any]) -> TableFormerV1Config:
        dataset = _mapping(raw, "dataset")
        model = _mapping(raw, "model")
        word_map = _mapping(_mapping(raw, "dataset_wordmap"), "word_map_tag")
        expected_map = {token: index for index, token in enumerate(TABLEFORMER_V1_TOKENS)}
        if model.get("type") != "TableModel04_rs":
            raise ValueError("Expected the TableModel04_rs architecture")
        if model.get("backbone") != "resnet18":
            raise ValueError("Expected the TableFormer v1 resnet18 backbone")
        if dict(word_map) != expected_map:
            raise ValueError("TableFormer v1 tag vocabulary does not match the frozen checkpoint")
        return cls(
            model_type="tableformer_v1",
            architectures=("TableFormerV1",),
            architecture="TableModel04_rs",
            backbone="resnet18",
            image_size=_integer(dataset, "resized_image"),
            encoded_image_size=_integer(model, "enc_image_size"),
            embed_dim=_integer(model, "hidden_dim"),
            num_encoder_layers=_integer(model, "enc_layers"),
            num_decoder_layers=_integer(model, "dec_layers"),
            num_heads=_integer(model, "nheads"),
            ff_dim=1024,
            tag_embed_dim=_integer(model, "tag_embed_dim"),
            tag_decoder_dim=_integer(model, "tag_decoder_dim"),
            bbox_embed_dim=_integer(model, "bbox_embed_dim"),
            tag_attention_dim=_integer(model, "tag_attention_dim"),
            bbox_attention_dim=_integer(model, "bbox_attention_dim"),
            bbox_classes=_integer(model, "bbox_classes"),
            vocab_size=len(word_map),
            vocab=TABLEFORMER_V1_TOKENS,
            data_cell_token_ids=TABLEFORMER_V1_DATA_CELL_TOKEN_IDS,
            pad_token_id=0,
            bos_token_id=2,
            eos_token_id=3,
        )


__all__ = [
    "TABLEFORMER_V1_DATA_CELL_TOKEN_IDS",
    "TABLEFORMER_V1_TOKENS",
    "TableFormerV1Config",
]
