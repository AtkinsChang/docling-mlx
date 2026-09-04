# Implemented after docling-ibm-models (docling_ibm_models/tableformer/models/table04_rs);
# module structure and parameter names follow it so the source checkpoint loads
# unchanged.
# SPDX-License-Identifier: Apache-2.0
"""Independent Torch oracle for the TableFormer v1 bbox decoder."""

from __future__ import annotations

from pathlib import Path

import torch
from docling.datamodel.pipeline_options import TableFormerMode
from safetensors import safe_open
from torch import nn
from torchvision.models.resnet import BasicBlock, conv1x1


def _input_filter() -> nn.Sequential:
    downsample = nn.Sequential(conv1x1(256, 512), nn.BatchNorm2d(512))
    return nn.Sequential(BasicBlock(256, 512, downsample=downsample), BasicBlock(512, 512))


class _CellAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._encoder_att = nn.Linear(512, 512)
        self._tag_decoder_att = nn.Linear(512, 512)
        self._language_att = nn.Linear(512, 512)
        self._full_att = nn.Linear(512, 1)

    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_hidden: torch.Tensor,
        language_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention = self._full_att(
            torch.relu(
                self._encoder_att(encoder_out)
                + self._tag_decoder_att(decoder_hidden).unsqueeze(1)
                + self._language_att(language_out).unsqueeze(1)
            )
        ).squeeze(2)
        alpha = torch.softmax(attention, dim=1)
        return (encoder_out * alpha.unsqueeze(2)).sum(dim=1), alpha


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(512, 256), nn.Linear(256, 256), nn.Linear(256, 4)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.relu(self.layers[0](value))
        value = torch.relu(self.layers[1](value))
        return self.layers[2](value)


class ReferenceBBoxDecoder(nn.Module):
    """Source-equivalent inference module with no docling-ibm-models import."""

    def __init__(self) -> None:
        super().__init__()
        self._input_filter = _input_filter()
        self._attention = _CellAttention()
        self._init_h = nn.Linear(512, 512)
        self._f_beta = nn.Linear(512, 512)
        self._class_embed = nn.Linear(512, 3)
        self._bbox_embed = _Mlp()

    def forward(
        self, encoder_out: torch.Tensor, tag_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_out = self._input_filter(encoder_out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        encoder_out = encoder_out.reshape(1, -1, 512)
        hidden = self._init_h(encoder_out.mean(dim=1)).expand(len(tag_states), -1)
        attended, _ = self._attention(encoder_out, tag_states, hidden)
        hidden = attended * torch.sigmoid(self._f_beta(hidden)) * hidden
        return self._class_embed(hidden), torch.sigmoid(self._bbox_embed(hidden))


def load_bbox_weights(
    model: ReferenceBBoxDecoder,
    source: Path,
    mode: TableFormerMode = TableFormerMode.ACCURATE,
) -> None:
    """Strict-load the bbox subtree from an official source snapshot."""
    profile = mode.value
    checkpoint = (
        source / "model_artifacts" / "tableformer" / profile / f"tableformer_{profile}.safetensors"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing {profile} TableFormer checkpoint: {checkpoint}")
    prefix = "_bbox_decoder."
    state: dict[str, torch.Tensor] = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as archive:
        for key in archive.keys():
            if key.startswith(prefix):
                state[key.removeprefix(prefix)] = archive.get_tensor(key)
    model.load_state_dict(state, strict=True)


__all__ = ["ReferenceBBoxDecoder", "load_bbox_weights"]
