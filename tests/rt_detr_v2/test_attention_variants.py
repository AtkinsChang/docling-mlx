# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for RT-DETR-v2 attention variants."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from docling_mlx._models.rt_detr_v2.config import RtDetrV2TransformerConfig
from docling_mlx._models.rt_detr_v2.transformer import MSDeformableAttention


def _attention() -> MSDeformableAttention:
    attention = MSDeformableAttention(
        RtDetrV2TransformerConfig(
            d_model=2,
            decoder_layers=1,
            decoder_attention_heads=1,
            decoder_ffn_dim=2,
            decoder_in_channels=(2, 2),
            decoder_activation_function="relu",
            decoder_method="discrete",
            decoder_n_levels=2,
            points_per_level=(1, 2),
            decoder_offset_scale=0.5,
            num_feature_levels=2,
            num_queries=1,
            num_labels=1,
            learn_initial_query=False,
            layer_norm_eps=1e-5,
            with_box_refine=True,
            use_focal_loss=True,
        )
    )
    attention.value_proj.weight = mx.eye(2)
    attention.value_proj.bias = mx.zeros(2)
    attention.output_proj.weight = mx.eye(2)
    attention.output_proj.bias = mx.zeros(2)
    attention.sampling_offsets.weight = mx.zeros((6, 2))
    attention.sampling_offsets.bias = mx.zeros(6)
    attention.attention_weights.weight = mx.zeros((3, 2))
    attention.attention_weights.bias = mx.zeros(3)
    return attention


@pytest.mark.mlx
@pytest.mark.parametrize(
    "reference_points",
    [
        mx.array([[[[1.2, -0.1, 0.5, 0.5]]]]),
        mx.array([[[[1.2, -0.1]]]]),
    ],
)
def test_discrete_sampling_clamps_each_axis_and_supports_point_lists(
    reference_points: mx.array,
) -> None:
    attention = _attention()
    values = mx.array(
        [
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
                [5.0, 50.0],
                [6.0, 60.0],
                [7.0, 70.0],
                [8.0, 80.0],
            ]
        ]
    )
    output = attention(mx.zeros((1, 1, 2)), reference_points, values, ((2, 3), (1, 2)))
    mx.eval(output)

    # HF's discrete path obtains (x, y) by int(location * [W, H] + .5),
    # clamps axes independently, then gathers nearest values.  The first
    # level gathers 3 and the second gathers 8 twice, with uniform weights.
    np.testing.assert_allclose(np.array(output), [[[19 / 3, 190 / 3]]], rtol=0, atol=1e-5)
