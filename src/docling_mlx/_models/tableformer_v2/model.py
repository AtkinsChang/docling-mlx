# Implemented after docling-ibm-models (docling_ibm_models/tableformer_v2); module
# structure, parameter names, and forward-pass order follow it so the published
# checkpoint loads unchanged.
# SPDX-License-Identifier: Apache-2.0
"""Native FP32 TableFormerV2 composition and greedy generation."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import mlx.core as mx
import mlx.nn as nn

from docling_mlx._models.tableformer_v2.bbox import BboxHead
from docling_mlx._models.tableformer_v2.config import TableFormerV2Config
from docling_mlx._models.tableformer_v2.decoder import (
    CrossAttentionCache,
    DecoderProjectedCache,
    TableFormerV2TokenDecoder,
    TokenDecoderOutput,
)
from docling_mlx._models.tableformer_v2.vision import TableFormerV2VisionEncoder

_BOS_TOKEN_ID = 2
_MAX_GENERATION_STEPS = 512


@dataclass(frozen=True, slots=True)
class TableFormerV2EncoderOutput:
    """Image memory and its pre-flattening spatial dimensions."""

    last_hidden_state: mx.array
    spatial_size: tuple[int, int]


@dataclass(frozen=True, slots=True)
class TableFormerV2Output:
    """One token-decoder pass plus exact cell-token bbox correspondence."""

    logits: mx.array
    hidden_states: mx.array
    predicted_bboxes: mx.array
    cell_positions: mx.array
    cell_batch_indices: mx.array
    encoder_outputs: TableFormerV2EncoderOutput
    past_key_values: DecoderProjectedCache | None


@dataclass(frozen=True, slots=True)
class TableFormerV2GenerationOutput:
    """Greedy IDs, final full-sequence states, and one box per cell token."""

    generated_ids: mx.array
    logits: mx.array
    hidden_states: mx.array
    predicted_bboxes: mx.array
    cell_positions: mx.array
    cell_batch_indices: mx.array
    encoder_outputs: TableFormerV2EncoderOutput


class TableFormerV2(nn.Module):
    """Source-namespaced native model with cache-aware greedy generation.

    Vision and decoder component modules are promoted to the original
    checkpoint keys instead of gaining ``vision.`` or ``token_decoder.``
    wrappers. The temporary components are not retained as child modules.
    """

    _POSITIONAL_ENCODING_LENGTH = 512

    def __init__(self, config: TableFormerV2Config) -> None:
        super().__init__()
        self.config = config
        vision_encoder = TableFormerV2VisionEncoder(config)
        self.feature_extractor = vision_encoder.feature_extractor
        self._feature_forward: Callable[[mx.array], mx.array] | None = None
        self.se_module = vision_encoder.se_module
        self.conv_mixer = vision_encoder.conv_mixer
        self.feature_to_embedding = vision_encoder.feature_to_embedding
        token_decoder = TableFormerV2TokenDecoder(config)
        self.input_embedding = token_decoder.input_embedding
        self.positional_encoding = token_decoder.positional_encoding
        self.transformer_decoder = token_decoder.transformer_decoder
        self.output_projection = token_decoder.output_projection
        self.bbox_head = BboxHead(config.embed_dim, config.num_heads)
        self.data_cells = config.data_cell_token_ids
        self.eval()

    def encode_images(self, pixels: mx.array) -> TableFormerV2EncoderOutput:
        """Encode one NHWC FP32 image batch exactly once for subsequent decoding."""

        if self.training:
            raise ValueError("TableFormerV2 supports eval mode only; call model.eval()")
        if pixels.dtype != mx.float32:
            raise ValueError("TableFormerV2 image inputs must be float32")
        if pixels.ndim != 4 or pixels.shape[-1] != 3:
            raise ValueError("Expected NHWC pixels with three channels")
        if min(pixels.shape[:3]) <= 0:
            raise ValueError("TableFormerV2 requires nonempty images")
        feature_forward = self._feature_forward or self.feature_extractor
        features = feature_forward(pixels)
        features = self.se_module(features)
        features = self.conv_mixer(features)
        batch_size, height, width, channels = features.shape
        flattened = mx.reshape(features, (batch_size, height * width, channels))
        last_hidden_state = self.feature_to_embedding(flattened)
        return TableFormerV2EncoderOutput(last_hidden_state, (height, width))

    def compile_image_backbone(self) -> None:
        if self._feature_forward is None:
            self._feature_forward = mx.compile(
                self.feature_extractor,
                inputs=self.feature_extractor.state,
            )

    def _positional_encoding(
        self, batch_size: int, sequence_length: int, *, offset: int = 0
    ) -> mx.array:
        total_length = offset + sequence_length
        if total_length <= self._POSITIONAL_ENCODING_LENGTH:
            positions = self.positional_encoding[:, offset:total_length]
        else:
            repeats = math.ceil(total_length / self._POSITIONAL_ENCODING_LENGTH)
            repeated = mx.tile(self.positional_encoding, (1, repeats, 1))
            positions = repeated[:, offset:total_length]
        return mx.broadcast_to(positions, (batch_size, sequence_length, self.config.embed_dim))

    def _validate_decoder_inputs(
        self,
        input_ids: mx.array,
        encoder_hidden_states: mx.array,
        past_key_values: DecoderProjectedCache | None,
    ) -> int:
        if self.training:
            raise ValueError("TableFormerV2 supports eval mode only; call model.eval()")
        if input_ids.ndim != 2:
            raise ValueError("TableFormerV2 token IDs must have shape (batch, sequence)")
        if input_ids.dtype not in (mx.int32, mx.int64):
            raise TypeError("TableFormerV2 token IDs must use an integer dtype")
        if encoder_hidden_states.ndim != 3:
            raise ValueError(
                "TableFormerV2 encoder states must have shape (batch, sequence, embed)"
            )
        if encoder_hidden_states.dtype != mx.float32:
            raise TypeError("TableFormerV2 encoder states must be float32")
        if encoder_hidden_states.shape[0] != input_ids.shape[0]:
            raise ValueError("TableFormerV2 token IDs and encoder states must share a batch size")
        if encoder_hidden_states.shape[-1] != self.config.embed_dim:
            raise ValueError("TableFormerV2 encoder state width does not match embed_dim")
        if past_key_values is None:
            return 0
        if len(past_key_values) != self.config.num_decoder_layers:
            raise ValueError("TableFormerV2 cache must contain one entry per decoder layer")
        lengths: set[int] = set()
        for cache in past_key_values:
            if (
                cache.keys.shape[0] != input_ids.shape[0]
                or cache.keys.shape[1] != self.config.num_heads
                or cache.keys.shape[-1] != self.config.embed_dim // self.config.num_heads
            ):
                raise ValueError("TableFormerV2 projected cache has an invalid shape")
            if cache.keys.dtype != mx.float32 or cache.values.dtype != mx.float32:
                raise TypeError("TableFormerV2 projected cache must be float32")
            lengths.add(cache.offset)
        if len(lengths) != 1:
            raise ValueError("TableFormerV2 cache layers must have the same sequence length")
        return lengths.pop()

    def prepare_cross_attention_cache(self, encoder_hidden_states: mx.array) -> CrossAttentionCache:
        """Project encoder memory once for every decoder layer."""

        return self.transformer_decoder.prepare_cross_attention_cache(encoder_hidden_states)

    def decode_tokens(
        self,
        input_ids: mx.array,
        encoder_hidden_states: mx.array,
        *,
        past_key_values: DecoderProjectedCache | None = None,
        cross_attention_cache: CrossAttentionCache | None = None,
        use_cache: bool = False,
    ) -> TokenDecoderOutput:
        """Run either full-sequence decoding or one cached token step."""

        past_length = 0 if past_key_values is None else past_key_values[0].offset
        sequence_length = input_ids.shape[1]
        if sequence_length < 1:
            raise ValueError("TableFormerV2 sequence length must be positive")
        if use_cache and sequence_length != 1:
            raise ValueError("TableFormerV2 cached decoding accepts exactly one token per step")
        if past_key_values is not None and not use_cache:
            raise ValueError("Projected cache requires use_cache=True")
        target = self.input_embedding(input_ids) + self._positional_encoding(
            input_ids.shape[0], sequence_length, offset=past_length
        )
        causal_mask = "causal" if not use_cache else None
        cross_attention_cache = (
            cross_attention_cache
            if cross_attention_cache is not None
            else self.prepare_cross_attention_cache(encoder_hidden_states)
        )
        hidden_states, present = self.transformer_decoder(
            target,
            encoder_hidden_states,
            causal_mask=causal_mask,
            past_key_values=past_key_values,
            cross_attention_cache=cross_attention_cache,
            use_cache=use_cache,
        )
        return TokenDecoderOutput(
            logits=self.output_projection(hidden_states),
            hidden_states=hidden_states,
            past_key_values=present,
        )

    @staticmethod
    def _cell_positions(input_ids: mx.array, data_cells: frozenset[int]) -> mx.array:
        token_rows = cast(list[list[int]], input_ids.tolist())
        positions = [
            (batch_index, token_index)
            for batch_index, token_ids in enumerate(token_rows)
            for token_index, token_id in enumerate(token_ids)
            if token_id in data_cells
        ]
        return mx.array(positions, dtype=mx.int32).reshape(-1, 2)

    def _predict_cell_bboxes(
        self,
        input_ids: mx.array,
        hidden_states: mx.array,
        encoder_outputs: TableFormerV2EncoderOutput,
    ) -> tuple[mx.array, mx.array, mx.array]:
        cell_positions = self._cell_positions(input_ids, self.data_cells)
        cell_batch_indices = cell_positions[:, 0]
        cell_embeddings = hidden_states[cell_positions[:, 0], cell_positions[:, 1]]
        predicted_bboxes = self.bbox_head(
            cell_embeddings,
            encoder_outputs.last_hidden_state,
            cell_batch_indices,
        )
        if predicted_bboxes.ndim != 2 or tuple(predicted_bboxes.shape) != (
            cell_positions.shape[0],
            4,
        ):
            raise RuntimeError(
                "TableFormerV2 bbox head must return exactly one box per data-cell token"
            )
        return predicted_bboxes, cell_positions, cell_batch_indices

    def __call__(
        self,
        input_ids: mx.array,
        *,
        pixels: mx.array | None = None,
        encoder_outputs: TableFormerV2EncoderOutput | None = None,
        past_key_values: DecoderProjectedCache | None = None,
        use_cache: bool = False,
    ) -> TableFormerV2Output:
        """Decode supplied IDs and predict exact-cardinality cell boxes."""

        if encoder_outputs is None:
            if pixels is None:
                raise ValueError("Either pixels or encoder_outputs must be provided")
            encoder_outputs = self.encode_images(pixels)
        self._validate_decoder_inputs(input_ids, encoder_outputs.last_hidden_state, past_key_values)
        decoded = self.decode_tokens(
            input_ids,
            encoder_outputs.last_hidden_state,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        predicted_bboxes, cell_positions, cell_batch_indices = self._predict_cell_bboxes(
            input_ids, decoded.hidden_states, encoder_outputs
        )
        return TableFormerV2Output(
            logits=decoded.logits,
            hidden_states=decoded.hidden_states,
            predicted_bboxes=predicted_bboxes,
            cell_positions=cell_positions,
            cell_batch_indices=cell_batch_indices,
            encoder_outputs=encoder_outputs,
            past_key_values=decoded.past_key_values,
        )

    def _greedy_token_step(
        self,
        input_ids: mx.array,
        encoder_hidden_states: mx.array,
        past_key_values: DecoderProjectedCache | None,
        cross_attention_cache: CrossAttentionCache,
    ) -> tuple[mx.array, DecoderProjectedCache | None]:
        decoded = self.decode_tokens(
            input_ids,
            encoder_hidden_states,
            past_key_values=past_key_values,
            cross_attention_cache=cross_attention_cache,
            use_cache=True,
        )
        next_token = mx.argmax(decoded.logits[:, -1, :], axis=-1).astype(mx.int32)[:, None]
        return next_token, decoded.past_key_values

    def generate(
        self,
        pixels: mx.array,
        *,
        max_generation_steps: int = _MAX_GENERATION_STEPS,
    ) -> TableFormerV2GenerationOutput:
        """Greedily generate after BOS, then decode final states and boxes once."""

        if self.training:
            raise ValueError("TableFormerV2 supports eval mode only; call model.eval()")
        if not 1 <= max_generation_steps <= _MAX_GENERATION_STEPS:
            raise ValueError("TableFormerV2 generation steps must be between 1 and 512")

        encoder_outputs = self.encode_images(pixels)
        memory = encoder_outputs.last_hidden_state
        mx.eval(memory)
        cross_attention_cache = self.prepare_cross_attention_cache(memory)
        tokens = [mx.full((pixels.shape[0], 1), _BOS_TOKEN_ID, dtype=mx.int32)]
        next_token, past_key_values = self._greedy_token_step(
            tokens[0], memory, None, cross_attention_cache
        )
        mx.async_eval(next_token)

        for step in range(max_generation_steps):
            pending = None
            if step + 1 < max_generation_steps:
                following_token, following_cache = self._greedy_token_step(
                    next_token, memory, past_key_values, cross_attention_cache
                )
                mx.async_eval(following_token)
                pending = (following_token, following_cache)
            tokens.append(next_token)
            emitted = cast(list[int], next_token.flatten().tolist())
            if pending is None or all(token_id == self.config.eos_token_id for token_id in emitted):
                break
            next_token, past_key_values = pending

        generated_ids = mx.concatenate(tokens, axis=1)

        final_decoded = self.decode_tokens(
            generated_ids,
            memory,
            cross_attention_cache=cross_attention_cache,
            use_cache=False,
        )
        predicted_bboxes, cell_positions, cell_batch_indices = self._predict_cell_bboxes(
            generated_ids, final_decoded.hidden_states, encoder_outputs
        )
        return TableFormerV2GenerationOutput(
            generated_ids=generated_ids,
            logits=final_decoded.logits,
            hidden_states=final_decoded.hidden_states,
            predicted_bboxes=predicted_bboxes,
            cell_positions=cell_positions,
            cell_batch_indices=cell_batch_indices,
            encoder_outputs=encoder_outputs,
        )


__all__ = [
    "TableFormerV2",
    "TableFormerV2EncoderOutput",
    "TableFormerV2GenerationOutput",
    "TableFormerV2Output",
]
