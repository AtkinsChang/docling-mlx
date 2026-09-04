# Implemented after docling-ibm-models (docling_ibm_models/tableformer/models/table04_rs)
# and Torchvision's ResNet-18 (torchvision/models/resnet.py); module structure,
# parameter names, and forward-pass order follow them so the published checkpoint
# loads unchanged.
# SPDX-License-Identifier: Apache-2.0
"""Native MLX bounding-box decoder for TableFormer v1."""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn

from ._source import source_parameter_filter
from .config import TableFormerV1Config

_CELL_TOKENS = frozenset({"ecel", "fcel", "ched", "rhed", "srow"})
_STATE_CLOSING_TOKENS = _CELL_TOKENS | {"nl", "ucel"}
_STATE_SKIPPING_TOKENS = frozenset({"nl", "ucel", "xcel"})


def cxcywh_to_xyxy(boxes: mx.array) -> mx.array:
    """Convert normalized center boxes to the source model's corner format."""
    if boxes.ndim < 1 or boxes.shape[-1] != 4:
        raise ValueError("Expected bounding boxes with a final dimension of four")
    cx, cy, width, height = mx.split(boxes, 4, axis=-1)
    return mx.concatenate(
        [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2], axis=-1
    )


def select_bbox_state_indices(tokens: Sequence[str]) -> tuple[list[int], dict[int, int]]:
    """Select decoder states and horizontal-span endpoints like the Torch model.

    Each index addresses the decoder state that predicted the token at the same
    position.  The merge mapping addresses the compact selected-state list.
    """
    selected: list[int] = []
    merge_endpoints: dict[int, int] = {}
    skip_next = True
    first_lcel = True
    current_merge = -1

    for state_index, token in enumerate(tokens):
        if token == "<end>":
            break

        if not skip_next and token in _STATE_CLOSING_TOKENS:
            selected.append(state_index)
            if not first_lcel:
                merge_endpoints[current_merge] = len(selected) - 1

        if token != "lcel":
            first_lcel = True
        elif first_lcel:
            selected.append(state_index)
            first_lcel = False
            current_merge = len(selected) - 1
            merge_endpoints[current_merge] = -1

        skip_next = token in _STATE_SKIPPING_TOKENS

    return selected, merge_endpoints


def merge_horizontal_bboxes(
    classes: mx.array,
    boxes: mx.array,
    merge_endpoints: dict[int, int],
) -> tuple[mx.array, mx.array]:
    """Merge the first and last cxcywh predictions for horizontal spans."""
    if classes.ndim != 2 or boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError("Expected class logits [N, C] and cxcywh boxes [N, 4]")
    if classes.shape[0] != boxes.shape[0]:
        raise ValueError("Class logits and bounding boxes must have the same length")

    merged_classes: list[mx.array] = []
    merged_boxes: list[mx.array] = []
    skipped: set[int] = set()
    for index in range(boxes.shape[0]):
        if index in merge_endpoints:
            endpoint = merge_endpoints[index]
            if not -boxes.shape[0] <= endpoint < boxes.shape[0]:
                raise ValueError(f"Horizontal merge endpoint {endpoint} is out of range")
            first = boxes[index]
            last = boxes[endpoint]
            width = (last[0] + last[2] / 2) - (first[0] - first[2] / 2)
            height = (last[1] + last[3] / 2) - (first[1] - first[3] / 2)
            left = first[0] - first[2] / 2
            top = mx.minimum(last[1] - last[3] / 2, first[1] - first[3] / 2)
            merged_boxes.append(mx.stack([left + width / 2, top + height / 2, width, height]))
            merged_classes.append(classes[index])
            skipped.add(endpoint % boxes.shape[0])
        elif index not in skipped:
            merged_boxes.append(boxes[index])
            merged_classes.append(classes[index])

    if not merged_boxes:
        return (
            mx.zeros((0, classes.shape[1]), dtype=classes.dtype),
            mx.zeros((0, 4), dtype=boxes.dtype),
        )
    return mx.stack(merged_classes), mx.stack(merged_boxes)


class _BasicBlock(nn.Module):
    """The two-convolution Torchvision BasicBlock used by the bbox filter."""

    def __init__(self, in_channels: int, out_channels: int, *, project: bool) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(out_channels)
        self.downsample = (
            [
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm(out_channels),
            ]
            if project
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = nn.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.downsample is not None:
            residual = self.downsample[1](self.downsample[0](residual))
        return nn.relu(x + residual)


class CellAttention(nn.Module):
    """Additive attention over the filtered image for each selected cell state."""

    def __init__(
        self,
        encoder_dim: int = 512,
        tag_decoder_dim: int = 512,
        language_dim: int = 512,
        attention_dim: int = 512,
    ) -> None:
        super().__init__()
        self._encoder_att = nn.Linear(encoder_dim, attention_dim)
        self._tag_decoder_att = nn.Linear(tag_decoder_dim, attention_dim)
        self._language_att = nn.Linear(language_dim, attention_dim)
        self._full_att = nn.Linear(attention_dim, 1)

    valid_parameter_filter = staticmethod(source_parameter_filter)

    def __call__(
        self, encoder_out: mx.array, decoder_hidden: mx.array, language_out: mx.array
    ) -> tuple[mx.array, mx.array]:
        if encoder_out.ndim != 3 or decoder_hidden.ndim != 2 or language_out.ndim != 2:
            raise ValueError("Expected encoder [1, S, D] and cell states [N, D]")
        attention = self._full_att(
            nn.relu(
                self._encoder_att(encoder_out)
                + self._tag_decoder_att(decoder_hidden)[:, None, :]
                + self._language_att(language_out)[:, None, :]
            )
        ).squeeze(-1)
        alpha = mx.softmax(attention, axis=1)
        return mx.sum(encoder_out * alpha[:, :, None], axis=1), alpha


class _BboxMlp(nn.Module):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256) -> None:
        super().__init__()
        self.layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 4),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.layers[0](x))
        x = nn.relu(self.layers[1](x))
        return self.layers[2](x)


class BBoxDecoder(nn.Module):
    """TableFormer v1 bbox head, including its three-class output."""

    def __init__(self, config: TableFormerV1Config | None = None) -> None:
        super().__init__()
        decoder_dim = config.tag_decoder_dim if config is not None else 512
        attention_dim = config.bbox_attention_dim if config is not None else 512
        embed_dim = config.bbox_embed_dim if config is not None else 256
        bbox_classes = config.bbox_classes + 1 if config is not None else 3
        self._input_filter = [
            _BasicBlock(256, attention_dim, project=True),
            _BasicBlock(attention_dim, attention_dim, project=False),
        ]
        self._attention = CellAttention(
            encoder_dim=attention_dim,
            tag_decoder_dim=decoder_dim,
            language_dim=decoder_dim,
            attention_dim=attention_dim,
        )
        self._init_h = nn.Linear(attention_dim, decoder_dim)
        self._f_beta = nn.Linear(decoder_dim, decoder_dim)
        self._class_embed = nn.Linear(decoder_dim, bbox_classes)
        self._bbox_embed = _BboxMlp(decoder_dim, embed_dim)
        self._attention_dim = attention_dim
        self._decoder_dim = decoder_dim
        self._bbox_classes = bbox_classes

    valid_parameter_filter = staticmethod(source_parameter_filter)

    def __call__(self, encoder_out: mx.array, tag_states: mx.array) -> tuple[mx.array, mx.array]:
        """Return class logits and normalized cxcywh boxes for selected states."""
        if encoder_out.ndim != 4 or encoder_out.shape[-1] != 256:
            raise ValueError("Expected encoder output [1, H, W, 256]")
        if (
            encoder_out.shape[0] != 1
            or tag_states.ndim != 2
            or tag_states.shape[-1] != self._decoder_dim
        ):
            raise ValueError(
                "TableFormer v1 bbox decoding requires one image and states "
                f"[N, {self._decoder_dim}]"
            )
        if encoder_out.dtype != mx.float32 or tag_states.dtype != mx.float32:
            raise TypeError("TableFormer v1 bbox decoding requires float32 inputs")
        if tag_states.shape[0] == 0:
            return mx.zeros((0, self._bbox_classes), dtype=mx.float32), mx.zeros(
                (0, 4), dtype=mx.float32
            )

        for block in self._input_filter:
            encoder_out = block(encoder_out)
        encoder_out = encoder_out.reshape(1, -1, self._attention_dim)
        hidden = mx.broadcast_to(
            self._init_h(mx.mean(encoder_out, axis=1)), (tag_states.shape[0], self._decoder_dim)
        )
        attended, _ = self._attention(encoder_out, tag_states, hidden)
        hidden = attended * mx.sigmoid(self._f_beta(hidden)) * hidden
        return self._class_embed(hidden), mx.sigmoid(self._bbox_embed(hidden))


__all__ = [
    "BBoxDecoder",
    "CellAttention",
    "cxcywh_to_xyxy",
    "merge_horizontal_bboxes",
    "select_bbox_state_indices",
]
