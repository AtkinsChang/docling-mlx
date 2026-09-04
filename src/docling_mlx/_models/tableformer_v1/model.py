# Implemented after docling-ibm-models (docling_ibm_models/tableformer/models/table04_rs);
# module structure, parameter names, and forward-pass order follow it so the published
# checkpoint loads unchanged.
# SPDX-License-Identifier: Apache-2.0
"""Native TableFormer v1 model composition and generation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn

from ._source import source_parameter_filter
from .bbox import (
    BBoxDecoder,
    cxcywh_to_xyxy,
    merge_horizontal_bboxes,
    select_bbox_state_indices,
)
from .config import TABLEFORMER_V1_TOKENS, TableFormerV1Config
from .decoder import TableFormerV1TokenDecoder
from .vision import TableFormerV1Encoder


class TableFormerV1(nn.Module):
    """The shared closed-profile image-to-structure model."""

    def __init__(self, config: TableFormerV1Config | Mapping[str, Any]) -> None:
        super().__init__()
        self.config = (
            config
            if isinstance(config, TableFormerV1Config)
            else TableFormerV1Config.from_dict(config)
        )
        encoder = TableFormerV1Encoder(self.config)
        self._encoder = encoder._encoder
        self._encoder_forward: Callable[[mx.array], mx.array] | None = None
        self._tag_transformer = TableFormerV1TokenDecoder(self.config)
        tag_transformer = cast(Any, self._tag_transformer)
        tag_transformer._input_filter = encoder._tag_transformer._input_filter
        tag_transformer._encoder = encoder._tag_transformer._encoder
        self._bbox_decoder = BBoxDecoder(self.config)
        self.eval()

    valid_parameter_filter = staticmethod(source_parameter_filter)

    def compile_image_backbone(self) -> None:
        if self._encoder_forward is None:
            self._encoder_forward = mx.compile(self._encoder, inputs=self._encoder.state)

    def _encode(self, pixels: mx.array) -> tuple[mx.array, mx.array]:
        size = self.config.image_size
        if self.training:
            raise ValueError("TableFormer v1 native generation requires eval mode")
        if pixels.dtype != mx.float32:
            raise ValueError("TableFormer v1 generation requires float32 pixels")
        if tuple(pixels.shape) != (1, 3, size, size):
            raise ValueError(f"TableFormer v1 pixels must have shape (1, 3, {size}, {size})")

        encoder_forward = self._encoder_forward or self._encoder
        image_features = encoder_forward(pixels.transpose(0, 2, 3, 1))
        tag_transformer = cast(Any, self._tag_transformer)
        filtered = image_features
        for block in tag_transformer._input_filter:
            filtered = block(filtered)
        memory = tag_transformer._encoder(
            filtered.reshape(filtered.shape[0], -1, filtered.shape[-1])
        ).transpose(1, 0, 2)
        return image_features, memory

    def generate(
        self,
        pixels: mx.array,
        *,
        max_generation_steps: int | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Return corrected IDs, bbox class logits, and normalized xyxy boxes."""

        image_features, memory = self._encode(pixels)
        generation = self._tag_transformer.generate(
            memory, max_generation_steps=max_generation_steps
        )
        token_ids = cast(list[int], generation.generated_ids[0].tolist())
        tokens = [TABLEFORMER_V1_TOKENS[token_id] for token_id in token_ids[1:]]
        state_indices, merge_endpoints = select_bbox_state_indices(tokens)
        tag_states = (
            mx.take(
                generation.hidden_states[0],
                mx.array(state_indices, dtype=mx.int32),
                axis=0,
            )
            if state_indices
            else mx.zeros((0, self.config.embed_dim), dtype=mx.float32)
        )
        class_logits, boxes = self._bbox_decoder(image_features, tag_states)
        class_logits, boxes = merge_horizontal_bboxes(class_logits, boxes, merge_endpoints)
        return generation.generated_ids, class_logits, cxcywh_to_xyxy(boxes)


__all__ = ["TableFormerV1"]
