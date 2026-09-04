# Implemented after docling-ibm-models (docling_ibm_models/tableformer/models/table04_rs);
# module structure, parameter names, and forward-pass order follow it so the published
# checkpoint loads unchanged.
# SPDX-License-Identifier: Apache-2.0
"""Native FP32 token decoding for the closed TableFormer v1 profiles.

The decoder keeps the source tensor layouts and raw per-layer hidden-state
cache. Projected K/V caches are deliberately outside this implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import mlx.core as mx
import mlx.nn as nn

from ._source import source_parameter_filter
from .config import TABLEFORMER_V1_TOKENS, TableFormerV1Config

type DecoderCache = mx.array
_TOKENS = TABLEFORMER_V1_TOKENS


@dataclass(frozen=True, slots=True)
class DecoderStepOutput:
    logits: mx.array
    hidden_state: mx.array
    cache: DecoderCache


@dataclass(frozen=True, slots=True)
class GenerationOutput:
    generated_ids: mx.array
    hidden_states: mx.array
    cache: DecoderCache


class _SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_length: int = 1024) -> None:
        super().__init__()
        position = mx.arange(max_length, dtype=mx.float32)[:, None]
        frequencies = mx.exp(
            mx.arange(0, embed_dim, 2, dtype=mx.float32) * (-math.log(10000.0) / embed_dim)
        )
        pe = mx.zeros((max_length, 1, embed_dim), dtype=mx.float32)
        pe[:, :, 0::2] = mx.sin(position * frequencies)[:, None, :]
        pe[:, :, 1::2] = mx.cos(position * frequencies)[:, None, :]
        self.pe = pe

    def __call__(self, states: mx.array) -> mx.array:
        if states.shape[0] > self.pe.shape[0]:
            raise ValueError("TableFormer v1 position limit exceeded")
        return states + self.pe[: states.shape[0]]


class _FusedMultiHeadAttention(nn.Module):
    """PyTorch MultiheadAttention-compatible fused projections."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.in_proj_weight = mx.random.normal((3 * embed_dim, embed_dim), dtype=mx.float32)
        self.in_proj_bias = mx.zeros((3 * embed_dim,), dtype=mx.float32)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    @staticmethod
    def _linear(values: mx.array, weight: mx.array, bias: mx.array) -> mx.array:
        return values @ weight.T + bias

    def _split_heads(self, values: mx.array) -> mx.array:
        sequence_length, batch_size, _ = values.shape
        return values.reshape(sequence_length, batch_size, self.num_heads, self.head_dim).transpose(
            1, 2, 0, 3
        )

    def _project_query(self, query: mx.array) -> mx.array:
        return self._split_heads(
            self._linear(
                query,
                self.in_proj_weight[: self.embed_dim],
                self.in_proj_bias[: self.embed_dim],
            )
        )

    def _project_key_values(self, key_value: mx.array) -> tuple[mx.array, mx.array]:
        projected = self._linear(
            key_value,
            self.in_proj_weight[self.embed_dim :],
            self.in_proj_bias[self.embed_dim :],
        )
        key, value = mx.split(projected, 2, axis=-1)
        return self._split_heads(key), self._split_heads(value)

    def __call__(self, query: mx.array, key_value: mx.array) -> mx.array:
        key, value = self._project_key_values(key_value)
        output = mx.fast.scaled_dot_product_attention(
            self._project_query(query), key, value, scale=self.scale
        )
        batch_size, _, sequence_length, _ = output.shape
        output = output.transpose(2, 0, 1, 3).reshape(sequence_length, batch_size, self.embed_dim)
        return self.out_proj(output)


class _TransformerDecoderLayer(nn.Module):
    """One source-compatible post-norm layer that emits the current token only."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int) -> None:
        super().__init__()
        self.self_attn = _FusedMultiHeadAttention(embed_dim, num_heads)
        self.multihead_attn = _FusedMultiHeadAttention(embed_dim, num_heads)
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def __call__(self, target: mx.array, memory: mx.array) -> mx.array:
        current = target[-1:]
        current = self.norm1(current + self.self_attn(current, target))
        current = self.norm2(current + self.multihead_attn(current, memory))
        return self.norm3(current + self.linear2(nn.relu(self.linear1(current))))


class _TransformerDecoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, num_layers: int) -> None:
        super().__init__()
        self.layers = [
            _TransformerDecoderLayer(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ]

    def __call__(
        self, target: mx.array, memory: mx.array, cache: DecoderCache | None = None
    ) -> tuple[mx.array, DecoderCache]:
        if cache is not None:
            if cache.ndim != 4 or cache.shape[0] != len(self.layers):
                raise ValueError("TableFormer v1 decoder cache has an invalid shape")
            if cache.shape[2] != target.shape[1] or cache.shape[3] != target.shape[2]:
                raise ValueError("TableFormer v1 decoder cache is incompatible with the target")
            if cache.shape[1] + 1 != target.shape[0]:
                raise ValueError("TableFormer v1 decoder cache length must trail the target by one")

        output = target
        current_states = []
        for index, layer in enumerate(self.layers):
            current = layer(output, memory)
            current_states.append(current)
            output = current if cache is None else mx.concatenate([cache[index], current], axis=0)
        stacked = mx.stack(current_states, axis=0)
        next_cache = stacked if cache is None else mx.concatenate([cache, stacked], axis=1)
        return current_states[-1], next_cache


class TableFormerV1TokenDecoder(nn.Module):
    """TableFormer v1 embedding, decoder stack, head, and greedy loop."""

    def __init__(self, config: TableFormerV1Config) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = self.config.vocab_size
        self.hidden_dim = self.config.embed_dim
        self.num_heads = self.config.num_heads
        self.ff_dim = self.config.ff_dim
        self.num_decoder_layers = self.config.num_decoder_layers
        self.bos_token_id = self.config.bos_token_id
        self.eos_token_id = self.config.eos_token_id
        self.max_generation_steps = 1024
        self._embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self._positional_encoding = _SinusoidalPositionalEncoding(
            self.hidden_dim, self.max_generation_steps
        )
        self._decoder = _TransformerDecoder(
            self.hidden_dim,
            self.num_heads,
            self.ff_dim,
            self.num_decoder_layers,
        )
        self._fc = nn.Linear(self.hidden_dim, self.vocab_size)
        self.eval()

    valid_parameter_filter = staticmethod(source_parameter_filter)

    def step(
        self,
        input_ids: mx.array,
        memory: mx.array,
        cache: DecoderCache | None = None,
    ) -> DecoderStepOutput:
        if input_ids.ndim != 2 or input_ids.shape[1] != memory.shape[1]:
            raise ValueError("TableFormer v1 input IDs must have shape (sequence, batch)")
        if memory.ndim != 3 or memory.shape[2] != self.hidden_dim:
            raise ValueError("TableFormer v1 memory must have shape (sequence, batch, 512)")
        if input_ids.shape[0] > self.max_generation_steps:
            raise ValueError("TableFormer v1 input exceeds the positional encoding limit")
        embedded = self._positional_encoding(self._embedding(input_ids))
        hidden_state, next_cache = self._decoder(embedded, memory, cache)
        return DecoderStepOutput(self._fc(hidden_state[-1]), hidden_state, next_cache)

    def _correct_token(self, token_id: int, generated: list[int]) -> int:
        token = _TOKENS[token_id]
        if token == "xcel":
            token = "lcel"
        if generated[-1] == 7 and token == "lcel":
            token = "fcel"
        return _TOKENS.index(token)

    def generate(
        self, memory: mx.array, *, max_generation_steps: int | None = None
    ) -> GenerationOutput:
        if memory.ndim != 3 or memory.shape[1] != 1 or memory.shape[2] != self.hidden_dim:
            raise ValueError("TableFormer v1 generation requires memory shape (sequence, 1, 512)")
        steps = self.max_generation_steps if max_generation_steps is None else max_generation_steps
        if not 0 < steps <= self.max_generation_steps:
            raise ValueError("max_generation_steps must be between 1 and the profile limit")

        generated = [self.bos_token_id]
        hidden_states: list[mx.array] = []
        cache = None
        for _ in range(steps):
            output = self.step(mx.array(generated, dtype=mx.int32)[:, None], memory, cache)
            mx.eval(output.logits)
            token_id = self._correct_token(
                cast(int, mx.argmax(output.logits, axis=-1).item()), generated
            )
            generated.append(token_id)
            hidden_states.append(output.hidden_state)
            cache = output.cache
            if token_id == self.eos_token_id:
                break

        if cache is None:
            raise RuntimeError("TableFormer v1 generation did not produce a decoder cache")
        return GenerationOutput(
            generated_ids=mx.array(generated, dtype=mx.int32)[None, :],
            hidden_states=mx.concatenate(hidden_states, axis=0).transpose(1, 0, 2),
            cache=cache,
        )


__all__ = [
    "DecoderCache",
    "DecoderStepOutput",
    "GenerationOutput",
    "TableFormerV1TokenDecoder",
]
