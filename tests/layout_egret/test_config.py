# SPDX-License-Identifier: Apache-2.0

"""Hugging Face-shaped D-FINE configuration contracts."""

from __future__ import annotations

import pytest

from docling_mlx._models.dfine.config import DFineConfig
from docling_mlx.engines.object_detection.rt_detr_v2.preprocessing import (
    parse_preprocessing_config,
)

PROFILES = {
    "medium": {
        "stem_channels": [3, 24, 32],
        "stage_in_channels": [32, 96, 384, 768],
        "stage_mid_channels": [32, 64, 128, 256],
        "stage_out_channels": [96, 384, 768, 1536],
        "stage_num_blocks": [1, 1, 3, 1],
        "stage_num_layers": [4, 4, 4, 4],
        "use_learnable_affine_block": True,
        "encoder_in_channels": [384, 768, 1536],
        "encoder_hidden_dim": 256,
        "encoder_ffn_dim": 1024,
        "decoder_in_channels": [256, 256, 256],
        "decoder_layers": 4,
        "depth_mult": 0.67,
    },
    "large": {
        "stem_channels": [3, 32, 48],
        "stage_in_channels": [48, 128, 512, 1024],
        "stage_mid_channels": [48, 96, 192, 384],
        "stage_out_channels": [128, 512, 1024, 2048],
        "stage_num_blocks": [1, 1, 3, 1],
        "stage_num_layers": [6, 6, 6, 6],
        "use_learnable_affine_block": False,
        "encoder_in_channels": [512, 1024, 2048],
        "encoder_hidden_dim": 256,
        "encoder_ffn_dim": 1024,
        "decoder_in_channels": [256, 256, 256],
        "decoder_layers": 6,
        "depth_mult": 1.0,
    },
    "xlarge": {
        "stem_channels": [3, 32, 64],
        "stage_in_channels": [64, 128, 512, 1024],
        "stage_mid_channels": [64, 128, 256, 512],
        "stage_out_channels": [128, 512, 1024, 2048],
        "stage_num_blocks": [1, 2, 5, 2],
        "stage_num_layers": [6, 6, 6, 6],
        "use_learnable_affine_block": False,
        "encoder_in_channels": [512, 1024, 2048],
        "encoder_hidden_dim": 384,
        "encoder_ffn_dim": 2048,
        "decoder_in_channels": [384, 384, 384],
        "decoder_layers": 6,
        "depth_mult": 1.0,
    },
}


def egret_config(profile: str = "medium") -> dict[str, object]:
    if profile not in PROFILES:
        raise ValueError(profile)
    topology = PROFILES[profile]

    labels = tuple(f"label_{index}" for index in range(17))
    id2label = {str(index): label for index, label in enumerate(labels)}
    return {
        "activation_function": "silu",
        "anchor_image_size": None,
        "attention_dropout": 0.0,
        "backbone_config": {
            "depths": [3, 4, 6, 3],
            "downsample_in_bottleneck": False,
            "downsample_in_first_stage": False,
            "embedding_size": 32,
            "hidden_act": "relu",
            "hidden_sizes": [192, 384, 768, 1536]
            if profile == "medium"
            else [256, 512, 1024, 2048],
            "layer_type": "basic",
            "model_type": "hgnet_v2",
            "num_channels": 3,
            "out_features": ["stage2", "stage3", "stage4"],
            "out_indices": [2, 3, 4],
            "stage_downsample": [False, True, True, True],
            "stage_in_channels": topology["stage_in_channels"],
            "stage_kernel_size": [3, 3, 5, 5],
            "stage_light_block": [False, False, True, True],
            "stage_mid_channels": topology["stage_mid_channels"],
            "stage_names": ["stem", "stage1", "stage2", "stage3", "stage4"],
            "stage_num_blocks": topology["stage_num_blocks"],
            "stage_numb_of_layers": topology["stage_num_layers"],
            "stage_out_channels": topology["stage_out_channels"],
            "stage_downsample_strides": [2, 2, 2, 2],
            "stem_channels": topology["stem_channels"],
            "stem_strides": [2, 1, 1, 2, 1],
            "use_learnable_affine_block": topology["use_learnable_affine_block"],
        },
        "batch_norm_eps": 1e-5,
        "d_model": 256,
        "decoder_activation_function": "relu",
        "decoder_attention_heads": 8,
        "decoder_ffn_dim": 1024,
        "decoder_in_channels": topology["decoder_in_channels"],
        "decoder_layers": topology["decoder_layers"],
        "decoder_method": "default",
        "decoder_n_points": [3, 6, 3],
        "decoder_offset_scale": 0.5,
        "depth_mult": topology["depth_mult"],
        "dropout": 0.0,
        "encode_proj_layers": [2],
        "encoder_activation_function": "gelu",
        "encoder_attention_heads": 8,
        "encoder_ffn_dim": topology["encoder_ffn_dim"],
        "encoder_hidden_dim": topology["encoder_hidden_dim"],
        "encoder_in_channels": topology["encoder_in_channels"],
        "encoder_layers": 1,
        "eval_idx": -1,
        "eval_size": None,
        "feat_strides": [8, 16, 32],
        "freeze_backbone_batch_norms": True,
        "hidden_expansion": 1.0,
        "id2label": id2label,
        "label2id": {label: int(index) for index, label in id2label.items()},
        "layer_norm_eps": 1e-5,
        "layer_scale": 1,
        "learn_initial_query": False,
        "lqe_hidden_dim": 64,
        "lqe_layers": 2,
        "max_num_bins": 32,
        "model_type": "d_fine",
        "normalize_before": False,
        "num_denoising": 100,
        "num_feature_levels": 3,
        "num_queries": 300,
        "positional_encoding_temperature": 10000,
        "reg_scale": 4.0,
        "top_prob_values": 4,
        "torch_dtype": "float32",
        "up": 0.5,
        "use_focal_loss": True,
        "with_box_refine": True,
    }


def preprocessor_config() -> dict[str, object]:
    return {
        "do_normalize": False,
        "do_pad": False,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.485, 0.456, 0.406],
        "image_processor_type": "RTDetrImageProcessor",
        "image_std": [0.229, 0.224, 0.225],
        "pad_size": None,
        "resample": 2,
        "rescale_factor": 1 / 255,
        "size": {"height": 640, "width": 640},
    }


@pytest.mark.parametrize("source_profile", ["medium", "large", "xlarge"])
def test_hf_configs_construct_their_declared_inference_topology(source_profile: str) -> None:
    config = DFineConfig.from_dict(egret_config(source_profile))
    topology = PROFILES[source_profile]
    stage_out_channels = tuple(topology["stage_out_channels"])
    assert config.encoder.hidden_dim == topology["encoder_hidden_dim"]
    assert config.decoder.layers == topology["decoder_layers"]
    assert config.backbone.stage_out_channels == stage_out_channels
    assert config.encoder.in_channels == stage_out_channels[1:]
    assert config.backbone.stem_strides == (2, 1, 1, 2, 1)
    assert config.backbone.stage_downsample_strides == (2, 2, 2, 2)
    assert config.decoder.points_per_level == (3, 6, 3)
    assert config.labels == tuple(f"label_{index}" for index in range(17))
    assert config.num_labels == 17


def test_source_configs_receive_explicit_stride_defaults() -> None:
    raw = egret_config()
    backbone = raw["backbone_config"]
    assert isinstance(backbone, dict)
    del backbone["stem_strides"]
    del backbone["stage_downsample_strides"]
    config = DFineConfig.from_dict(raw)
    assert config.backbone.stem_strides == (2, 1, 1, 2, 1)
    assert config.backbone.stage_downsample_strides == (2, 2, 2, 2)


def test_parser_accepts_supported_variants_and_rejects_inconsistent_shapes() -> None:
    raw = egret_config("large")
    raw["d_model"] = 384
    raw["decoder_in_channels"] = [384, 384, 384]
    raw["decoder_n_points"] = 4
    raw["decoder_method"] = "discrete"
    config = DFineConfig.from_dict(raw)
    assert config.decoder.hidden_dim == 384
    assert config.decoder.points_per_level == (4, 4, 4)
    assert config.decoder.method == "discrete"

    invalid = egret_config()
    backbone = invalid["backbone_config"]
    assert isinstance(backbone, dict)
    backbone["out_indices"] = [0, 2, 3]
    with pytest.raises(ValueError, match="out_indices"):
        DFineConfig.from_dict(invalid)

    labels = egret_config()
    id2label = labels["id2label"]
    assert isinstance(id2label, dict)
    id2label["6"] = "label_0"
    with pytest.raises(ValueError, match="unique"):
        DFineConfig.from_dict(labels)

    unknown = egret_config()
    unknown["unreviewed_shape"] = 17
    DFineConfig.from_dict(unknown)


def test_parser_copies_mutable_source_values() -> None:
    raw = egret_config("xlarge")
    config = DFineConfig.from_dict(raw)
    raw["decoder_in_channels"] = [1, 2, 3]
    backbone = raw["backbone_config"]
    assert isinstance(backbone, dict)
    backbone["stage_out_channels"] = [1, 2, 3, 4]
    assert config.decoder.in_channels == (384, 384, 384)
    assert config.backbone.stage_out_channels == (128, 512, 1024, 2048)


def test_preprocessor_uses_checkpoint_values() -> None:
    parsed = parse_preprocessing_config(preprocessor_config())
    assert parsed.size == (640, 640)
    assert parsed.rescale_factor == pytest.approx(1 / 255)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rescale_factor", 0.5),
        ("size", {"height": 640, "width": 641}),
    ],
)
def test_accepts_preprocessor_variants(field: str, value: object) -> None:
    raw = preprocessor_config()
    raw[field] = value
    parsed = parse_preprocessing_config(raw)
    assert parsed.rescale_factor == value or parsed.size == (640, 641)
