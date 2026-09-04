# Implemented after docling-ibm-models (docling_ibm_models/tableformer_v2); module
# structure, parameter names, and forward-pass order follow it so the published
# checkpoint loads unchanged.
# SPDX-License-Identifier: Apache-2.0
"""Native FP32 token decoding for the closed TableFormerV2 profile.

The parameter names intentionally match the token-decoding portion of
``docling_ibm_models.tableformer_v2.model.TableFormerV2``. PyTorch's fused
``MultiheadAttention.in_proj_weight`` and ``in_proj_bias`` remain fused, while
runtime generation caches their projected keys and values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .config import TableFormerV2Config

type ProjectedKeyValues = tuple[mx.array, mx.array]
CrossAttentionCache = tuple[ProjectedKeyValues, ...]
_MAX_CACHE_LENGTH = 512


class ProjectedKVCache:
    """Fixed-capacity projected self-attention keys and values."""

    def __init__(
        self,
        *,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        capacity: int = _MAX_CACHE_LENGTH,
        dtype: mx.Dtype = mx.float32,
    ) -> None:
        if min(batch_size, num_heads, head_dim, capacity) <= 0:
            raise ValueError("Projected K/V cache dimensions and capacity must be positive")
        self.capacity = capacity
        self.offset = 0
        shape = (batch_size, num_heads, capacity, head_dim)
        self.keys = mx.zeros(shape, dtype=dtype)
        self.values = mx.zeros(shape, dtype=dtype)

    @property
    def state(self) -> ProjectedKeyValues:
        """Return only the initialized prefix of each backing buffer."""

        return self.keys[:, :, : self.offset, :], self.values[:, :, : self.offset, :]

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> ProjectedKeyValues:
        """Append projected states and return the initialized cache prefix."""

        if keys.dtype != self.keys.dtype or values.dtype != self.values.dtype:
            raise TypeError("Projected K/V cache update dtype does not match the cache")
        if keys.ndim != 4 or tuple(keys.shape) != tuple(values.shape):
            raise ValueError("Projected K/V cache keys and values must share a 4-D shape")
        expected = (self.keys.shape[0], self.keys.shape[1], keys.shape[2], self.keys.shape[3])
        if tuple(keys.shape) != expected or keys.shape[2] < 1:
            raise ValueError(f"Projected K/V cache update has incompatible shape {keys.shape}")
        stop = self.offset + keys.shape[2]
        if stop > self.capacity:
            raise ValueError("Projected K/V cache capacity exceeded")
        self.keys[:, :, self.offset : stop, :] = keys
        self.values[:, :, self.offset : stop, :] = values
        self.offset = stop
        return self.state


DecoderProjectedCache = tuple[ProjectedKVCache, ...]


@dataclass(frozen=True, slots=True)
class TokenDecoderOutput:
    """Logits, final states, and optional projected self-attention cache."""

    logits: mx.array
    hidden_states: mx.array
    past_key_values: DecoderProjectedCache | None


class FusedMultiHeadAttention(nn.Module):
    """PyTorch-``MultiheadAttention``-compatible fused-projection attention."""

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
        batch_size, sequence_length, _ = values.shape
        return values.reshape(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

    def project_self(self, states: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Project source-compatible fused Q/K/V states once."""

        projected = self._linear(states, self.in_proj_weight, self.in_proj_bias)
        query, key, value = mx.split(projected, 3, axis=-1)
        return self._split_heads(query), self._split_heads(key), self._split_heads(value)

    def project_query(self, query: mx.array) -> mx.array:
        """Project only the query portion of the fused source state."""

        weight = self.in_proj_weight[: self.embed_dim]
        bias = self.in_proj_bias[: self.embed_dim]
        return self._split_heads(self._linear(query, weight, bias))

    def project_key_values(self, key_value: mx.array) -> ProjectedKeyValues:
        """Project and split the key/value portions of the fused source state."""

        weight = self.in_proj_weight[self.embed_dim :]
        bias = self.in_proj_bias[self.embed_dim :]
        projected = self._linear(key_value, weight, bias)
        key, value = mx.split(projected, 2, axis=-1)
        return self._split_heads(key), self._split_heads(value)

    def _scaled_dot_product_attention(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: str | mx.array | None,
    ) -> mx.array:
        return mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=self.scale,
            mask=mask,
        )

    def _attend(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: str | mx.array | None,
    ) -> mx.array:
        output = self._scaled_dot_product_attention(query, key, value, mask)
        batch_size, _, sequence_length, _ = output.shape
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, self.embed_dim)
        return self.out_proj(output)

    def self_attention(self, states: mx.array, mask: str | mx.array | None = None) -> mx.array:
        """Attend to one complete set of decoder states."""

        query, key, value = self.project_self(states)
        return self._attend(query, key, value, mask)

    def self_attention_step(self, states: mx.array, cache: ProjectedKVCache) -> mx.array:
        """Project one token and attend to the cached projected prefix."""

        query, key, value = self.project_self(states)
        cached_key, cached_value = cache.update_and_fetch(key, value)
        return self._attend(query, cached_key, cached_value, None)

    def cross_attention(
        self,
        query: mx.array,
        memory: mx.array | None = None,
        *,
        projected_memory: ProjectedKeyValues | None = None,
    ) -> mx.array:
        """Cross-attend with either memory or its already projected K/V state."""

        if (memory is None) == (projected_memory is None):
            raise ValueError("Provide exactly one of memory or projected_memory")
        if projected_memory is None:
            if memory is None:
                raise ValueError("Provide memory when projected_memory is not set")
            key, value = self.project_key_values(memory)
        else:
            key, value = projected_memory
        return self._attend(self.project_query(query), key, value, None)


class CachedTransformerDecoderLayer(nn.Module):
    """One source-compatible post-norm decoder layer."""

    def __init__(self, config: TableFormerV2Config) -> None:
        super().__init__()
        self.self_attn = FusedMultiHeadAttention(config.embed_dim, config.num_heads)
        self.multihead_attn = FusedMultiHeadAttention(config.embed_dim, config.num_heads)
        self.linear1 = nn.Linear(config.embed_dim, config.ff_dim)
        self.linear2 = nn.Linear(config.ff_dim, config.embed_dim)
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.norm3 = nn.LayerNorm(config.embed_dim)

    def __call__(
        self,
        target: mx.array,
        memory: mx.array,
        *,
        causal_mask: str | mx.array | None = None,
        self_cache: ProjectedKVCache | None = None,
        cross_cache: ProjectedKeyValues | None = None,
        use_cache: bool = False,
    ) -> mx.array:
        if use_cache:
            if self_cache is None:
                raise ValueError("Cached decoding requires a projected self-attention cache")
            self_attention = self.self_attn.self_attention_step(target, self_cache)
        else:
            if self_cache is not None:
                raise ValueError("Full-sequence decoding cannot consume a projected cache")
            self_attention = self.self_attn.self_attention(target, causal_mask)
        target = self.norm1(target + self_attention)
        target = self.norm2(
            target
            + self.multihead_attn.cross_attention(
                target,
                memory if cross_cache is None else None,
                projected_memory=cross_cache,
            )
        )
        target = self.norm3(target + self.linear2(nn.relu(self.linear1(target))))
        return target


class CachedTransformerDecoder(nn.Module):
    """Four source-compatible decoder layers with projected runtime caches."""

    def __init__(self, config: TableFormerV2Config) -> None:
        super().__init__()
        self.config = config
        self.layers = [
            CachedTransformerDecoderLayer(config) for _ in range(config.num_decoder_layers)
        ]

    def prepare_cross_attention_cache(self, memory: mx.array) -> CrossAttentionCache:
        """Project encoder memory once for every decoder layer."""

        return tuple(layer.multihead_attn.project_key_values(memory) for layer in self.layers)

    def _new_self_attention_cache(self, batch_size: int) -> DecoderProjectedCache:
        return tuple(
            ProjectedKVCache(
                batch_size=batch_size,
                num_heads=self.config.num_heads,
                head_dim=self.config.embed_dim // self.config.num_heads,
                capacity=_MAX_CACHE_LENGTH,
                dtype=mx.float32,
            )
            for _ in self.layers
        )

    def __call__(
        self,
        target: mx.array,
        memory: mx.array,
        *,
        causal_mask: str | mx.array | None = None,
        past_key_values: DecoderProjectedCache | None = None,
        cross_attention_cache: CrossAttentionCache | None = None,
        use_cache: bool = False,
    ) -> tuple[mx.array, DecoderProjectedCache | None]:
        if cross_attention_cache is not None and len(cross_attention_cache) != len(self.layers):
            raise ValueError("Cross-attention cache must contain one entry per decoder layer")
        if use_cache:
            caches = past_key_values or self._new_self_attention_cache(target.shape[0])
            if len(caches) != len(self.layers):
                raise ValueError("Self-attention cache must contain one entry per decoder layer")
        else:
            if past_key_values is not None:
                raise ValueError("Full-sequence decoding cannot consume a projected cache")
            caches = None

        for index, layer in enumerate(self.layers):
            target = layer(
                target,
                memory,
                causal_mask=causal_mask,
                self_cache=caches[index] if caches is not None else None,
                cross_cache=(
                    cross_attention_cache[index] if cross_attention_cache is not None else None
                ),
                use_cache=use_cache,
            )
        return target, caches


class TableFormerV2TokenDecoder(nn.Module):
    """Token embedding, positional encoding, decoder stack, and output head."""

    _POSITIONAL_ENCODING_LENGTH = 512

    def __init__(self, config: TableFormerV2Config) -> None:
        super().__init__()
        self.config = config
        self.input_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.positional_encoding = mx.random.normal(
            (1, self._POSITIONAL_ENCODING_LENGTH, config.embed_dim), dtype=mx.float32
        )
        self.transformer_decoder = CachedTransformerDecoder(config)
        self.output_projection = nn.Linear(config.embed_dim, config.vocab_size)
        self.eval()

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

    def _cache_offset(
        self,
        input_ids: mx.array,
        past_key_values: DecoderProjectedCache | None,
    ) -> int:
        if past_key_values is None:
            return 0
        if len(past_key_values) != self.config.num_decoder_layers:
            raise ValueError("TableFormerV2 cache must contain one entry per decoder layer")
        offsets = {cache.offset for cache in past_key_values}
        if len(offsets) != 1:
            raise ValueError("TableFormerV2 cache layers must have the same sequence length")
        for cache in past_key_values:
            if (
                cache.keys.shape[0] != input_ids.shape[0]
                or cache.keys.shape[1] != self.config.num_heads
                or cache.keys.shape[-1] != self.config.embed_dim // self.config.num_heads
                or cache.keys.dtype != mx.float32
                or cache.values.dtype != mx.float32
            ):
                raise ValueError("TableFormerV2 projected cache has an invalid contract")
        return offsets.pop()

    def prepare_cross_attention_cache(self, memory: mx.array) -> CrossAttentionCache:
        return self.transformer_decoder.prepare_cross_attention_cache(memory)

    def decode(
        self,
        input_ids: mx.array,
        encoder_hidden_states: mx.array,
        *,
        past_key_values: DecoderProjectedCache | None = None,
        cross_attention_cache: CrossAttentionCache | None = None,
        use_cache: bool = False,
    ) -> TokenDecoderOutput:
        """Decode a full sequence or exactly one cached autoregressive token."""

        past_length = self._cache_offset(input_ids, past_key_values)
        sequence_length = input_ids.shape[1]
        if sequence_length < 1:
            raise ValueError("TableFormerV2 sequence length must be positive")
        if past_length and sequence_length != 1:
            raise ValueError("TableFormerV2 cached decoding accepts exactly one token per step")
        if past_key_values is not None and not use_cache:
            raise ValueError("Projected cache requires use_cache=True")
        target = self.input_embedding(input_ids) + self._positional_encoding(
            input_ids.shape[0], sequence_length, offset=past_length
        )
        causal_mask = "causal" if not use_cache else None
        cross_attention_cache = cross_attention_cache or self.prepare_cross_attention_cache(
            encoder_hidden_states
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

    def full_sequence(
        self, input_ids: mx.array, encoder_hidden_states: mx.array
    ) -> TokenDecoderOutput:
        return self.decode(input_ids, encoder_hidden_states, use_cache=False)

    def cached_token_step(
        self,
        input_ids: mx.array,
        encoder_hidden_states: mx.array,
        past_key_values: DecoderProjectedCache | None,
        *,
        cross_attention_cache: CrossAttentionCache | None = None,
    ) -> TokenDecoderOutput:
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("TableFormerV2 cached token step requires shape (batch, 1)")
        return self.decode(
            input_ids,
            encoder_hidden_states,
            past_key_values=past_key_values,
            cross_attention_cache=cross_attention_cache,
            use_cache=True,
        )

    def greedy_token_step(
        self,
        input_ids: mx.array,
        encoder_hidden_states: mx.array,
        past_key_values: DecoderProjectedCache | None,
        *,
        cross_attention_cache: CrossAttentionCache | None = None,
    ) -> tuple[mx.array, TokenDecoderOutput]:
        output = self.cached_token_step(
            input_ids,
            encoder_hidden_states,
            past_key_values,
            cross_attention_cache=cross_attention_cache,
        )
        next_token = mx.argmax(output.logits[:, -1, :], axis=-1)[:, None]
        return next_token, output

    def __call__(self, input_ids: mx.array, encoder_hidden_states: mx.array) -> TokenDecoderOutput:
        return self.full_sequence(input_ids, encoder_hidden_states)

    @staticmethod
    def causal_mask(sequence_length: int, *, dtype: mx.Dtype) -> mx.array:
        if sequence_length < 1:
            raise ValueError("TableFormerV2 sequence length must be positive")
        positions = mx.arange(sequence_length)
        blocked = positions[:, None] < positions[None, :]
        return mx.where(
            blocked,
            mx.array(-mx.inf, dtype=dtype),
            mx.array(0.0, dtype=dtype),
        )


__all__ = [
    "CachedTransformerDecoder",
    "CrossAttentionCache",
    "DecoderProjectedCache",
    "FusedMultiHeadAttention",
    "ProjectedKVCache",
    "ProjectedKeyValues",
    "TableFormerV2TokenDecoder",
    "TokenDecoderOutput",
]
