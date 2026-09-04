# SPDX-License-Identifier: Apache-2.0

"""Pure conversion of the upstream TableFormer v1 checkpoint layout."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np


def flatten_upstream_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Translate upstream ``tm_config.json`` into the native config schema."""

    dataset = config.get("dataset")
    model = config.get("model")
    predict = config.get("predict")
    wordmap = config.get("dataset_wordmap")
    if not all(isinstance(value, Mapping) for value in (dataset, model, predict, wordmap)):
        raise ValueError("Expected tm_config.json to contain dataset, model, predict, and wordmap")
    dataset = cast(Mapping[str, Any], dataset)
    model = cast(Mapping[str, Any], model)
    wordmap = cast(Mapping[str, Any], wordmap)
    normalization = dataset.get("image_normalization")
    tag_map = wordmap.get("word_map_tag")
    if not isinstance(normalization, Mapping) or not isinstance(tag_map, Mapping):
        raise ValueError("Expected tm_config.json image normalization and tag vocabulary")
    entries = sorted(tag_map.items(), key=lambda item: item[1])
    indices = [index for _, index in entries]
    if any(type(index) is not int for index in indices) or indices != list(range(len(entries))):
        raise ValueError("TableFormer v1 tag vocabulary IDs must be contiguous")
    vocab = [token for token, _ in entries]
    return {
        "model_type": "tableformer_v1",
        "architectures": ["TableFormerV1"],
        "architecture": model["type"],
        "backbone": model["backbone"],
        "image_size": dataset["resized_image"],
        "encoded_image_size": model["enc_image_size"],
        "embed_dim": model["hidden_dim"],
        "num_encoder_layers": model["enc_layers"],
        "num_decoder_layers": model["dec_layers"],
        "num_heads": model["nheads"],
        "ff_dim": 1024,
        "tag_embed_dim": model["tag_embed_dim"],
        "tag_decoder_dim": model["tag_decoder_dim"],
        "bbox_embed_dim": model["bbox_embed_dim"],
        "tag_attention_dim": model["tag_attention_dim"],
        "bbox_attention_dim": model["bbox_attention_dim"],
        "bbox_classes": model["bbox_classes"],
        "vocab_size": len(vocab),
        "vocab": vocab,
        "data_cells": [4, 5, 10, 11, 12],
        "pad_token_id": vocab.index("<pad>"),
        "bos_token_id": vocab.index("<start>"),
        "eos_token_id": vocab.index("<end>"),
        "torch_dtype": "float32",
    }


def upstream_preprocessor_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return native preprocessing metadata derived from upstream settings."""

    dataset = config.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("Expected tm_config.json to contain a dataset mapping")
    normalization = dataset.get("image_normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError("Expected tm_config.json image normalization")
    return {
        "do_convert_rgb": True,
        "do_resize": True,
        "size": {"height": dataset["resized_image"], "width": dataset["resized_image"]},
        "interpolation": "bilinear",
        "resample": "bilinear",
        "antialias": False,
        "do_rescale": True,
        "rescale_factor": 1 / 255,
        "do_normalize": True,
        "normalize_before_resize": True,
        "image_mean": normalization["mean"],
        "image_std": normalization["std"],
        "page_height": 1024,
        "page_interpolation": "area",
        "output_dtype": "float32",
        "output_layout": "CWH",
    }


def _ignored_reason(key: str, tensor: np.ndarray, source: Mapping[str, np.ndarray]) -> str | None:
    if not key.endswith(".num_batches_tracked"):
        return None
    stem = key.removesuffix("num_batches_tracked")
    if stem + "running_mean" not in source or stem + "running_var" not in source:
        raise ValueError(f"Unpaired BatchNorm training counter is not ignorable: {key}")
    if tensor.dtype != np.int64 or tensor.shape != ():
        raise ValueError(f"Ignored BatchNorm counter must be an int64 scalar: {key}")
    return "batch_norm_training_counter"


def convert_upstream_state_dict(
    source: Mapping[str, np.ndarray], target_shapes: Mapping[str, tuple[int, ...]]
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, str]]:
    """Convert every upstream inference tensor to its native MLX layout."""

    ignored: dict[str, str] = {}
    renamed: dict[str, tuple[str, np.ndarray]] = {}
    for source_key, tensor in source.items():
        reason = _ignored_reason(source_key, tensor, source)
        if reason is not None:
            ignored[source_key] = reason
            continue
        if source_key in renamed:
            raise ValueError(f"Source keys map to the same target key: {source_key}")
        renamed[source_key] = source_key, tensor
    if missing := target_shapes.keys() - renamed.keys():
        raise ValueError(f"Source state missing target keys: {sorted(missing)}")
    if extra := renamed.keys() - target_shapes.keys():
        raise ValueError(
            "Source state has unexpected inference keys: "
            f"{sorted(renamed[key][0] for key in extra)}"
        )
    converted: dict[str, np.ndarray] = {}
    mappings: list[dict[str, Any]] = []
    for target_key, target_shape in sorted(target_shapes.items()):
        source_key, tensor = renamed[target_key]
        if tensor.dtype != np.float32 or not np.isfinite(tensor).all():
            raise ValueError(f"Inference tensor must be finite float32: {source_key}")
        value = tensor.transpose(0, 2, 3, 1) if tensor.ndim == 4 else tensor
        if value.shape != target_shape:
            raise ValueError(
                f"Target shape mismatch for {source_key} -> {target_key}: "
                f"{value.shape} != {target_shape}"
            )
        converted[target_key] = np.ascontiguousarray(value)
        mappings.append(
            {
                "source_key": source_key,
                "target_key": target_key,
                "source_shape": list(tensor.shape),
                "target_shape": list(target_shape),
                "source_dtype": "float32",
                "target_dtype": "float32",
                "transform": "OIHW_to_OHWI" if tensor.ndim == 4 else "identity",
            }
        )
    return converted, mappings, dict(sorted(ignored.items()))


__all__ = [
    "convert_upstream_state_dict",
    "flatten_upstream_config",
    "upstream_preprocessor_config",
]
