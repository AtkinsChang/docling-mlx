# SPDX-License-Identifier: Apache-2.0

"""Strict conversion of the two pinned TableFormer v1 checkpoints.

Run ``python -m tools.tableformer_v1.convert_weights --output DIR``. The output
is a local converted-model repository; conversion never uploads or publishes it.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from docling_mlx.engines.table_structure.tableformer_v1 import conversion as _conversion
from tools._common.atomic import _atomic_output
from tools._common.model_card import lookup_model_license, render_model_card
from tools.tableformer_v1.source import (
    SOURCE_REPO,
    SOURCE_REVISION,
    download_source,
    verify_source,
)

_BN_COUNTER_REASON = "batch_norm_training_counter"
convert_state_dict = _conversion.convert_upstream_state_dict
_flat_config = _conversion.flatten_upstream_config
_preprocessor_config = _conversion.upstream_preprocessor_config
_PROFILES = {
    "accurate": {
        "checkpoint": "tableformer_accurate.safetensors",
        "profile_id": "tableformer_v1_accurate",
    },
    "fast": {
        "checkpoint": "tableformer_fast.safetensors",
        "profile_id": "tableformer_v1_fast",
    },
}

TM_CONFIG_FIELD_MAPPING = {
    "dataset.resized_image": "image_size",
    "dataset.image_normalization.state": "preprocessor_config.do_normalize",
    "dataset.image_normalization.mean/std": "preprocessor_config.image_mean/image_std",
    "model.type": "architecture",
    "model.backbone": "backbone",
    "model.enc_image_size": "encoded_image_size",
    "model.hidden_dim": "embed_dim",
    "model.enc_layers": "num_encoder_layers",
    "model.dec_layers": "num_decoder_layers",
    "model.nheads": "num_heads",
    "model.tag_embed_dim": "tag_embed_dim",
    "model.tag_decoder_dim": "tag_decoder_dim",
    "model.bbox_embed_dim": "bbox_embed_dim",
    "model.tag_attention_dim": "tag_attention_dim",
    "model.bbox_attention_dim": "bbox_attention_dim",
    "model.bbox_classes": "bbox_classes",
    "predict.max_steps": "generation_config.max_generation_steps",
    "dataset_wordmap.word_map_tag": "vocab (id ordered)",
}
TM_CONFIG_DROPPED_FIELDS = (
    "dataset.type",
    "dataset.name",
    "dataset.raw_data_dir",
    "dataset.load_cells",
    "dataset.bbox_format",
    "dataset.keep_AR",
    "dataset.up_scaling_enabled",
    "dataset.down_scaling_enabled",
    "dataset.padding_mode",
    "dataset.padding_color",
    "dataset.color_jitter",
    "dataset.rand_crop",
    "dataset.rand_pad",
    "dataset.image_grayscale",
    "model.name",
    "model.dropout",
    "train.*",
    "predict.beam_size",
    "predict.bbox",
    "predict.pdf_cell_iou_thres",
    "predict.padding",
    "predict.padding_size",
    "predict.disable_post_process",
    "predict.profiling",
    "debug.*",
    "dataset_wordmap.word_map_cell",
)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _target_model(config_raw: Mapping[str, Any]) -> tuple[Any, dict[str, tuple[int, ...]]]:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from docling_mlx._models.tableformer_v1.model import TableFormerV1

    model = TableFormerV1(config_raw)
    model.eval()
    flattened = cast(list[tuple[str, Any]], tree_flatten(model.parameters()))
    if not flattened:
        raise ValueError("Native TableFormer v1 exposes no parameters")
    if any(value.dtype != mx.float32 for _, value in flattened):
        raise ValueError("Native TableFormer v1 target parameters must all be float32")
    return model, {key: tuple(value.shape) for key, value in flattened}


def convert(source: Path, output: Path) -> dict[str, Any]:
    """Convert and atomically publish both frozen profiles."""

    verify_source(source)
    license = lookup_model_license(source, SOURCE_REPO, SOURCE_REVISION)
    import mlx.core as mx
    from safetensors.numpy import load_file, save_file

    report: dict[str, Any] = {
        "output": str(output),
        "source_revision": SOURCE_REVISION,
        "field_mapping": TM_CONFIG_FIELD_MAPPING,
        "dropped_fields": TM_CONFIG_DROPPED_FIELDS,
        "profiles": {},
    }
    with _atomic_output(output) as temporary:
        (temporary / "README.md").write_text(
            render_model_card(SOURCE_REPO, SOURCE_REVISION, license), encoding="utf-8"
        )
        for profile_name, spec in _PROFILES.items():
            source_profile = source / "model_artifacts/tableformer" / profile_name
            config_path = source_profile / "tm_config.json"
            config_raw = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(config_raw, dict):
                raise ValueError("Expected tm_config.json to contain a JSON object")
            flat_config = _flat_config(config_raw)

            model, target_shapes = _target_model(flat_config)
            source_state = load_file(str(source_profile / str(spec["checkpoint"])))
            converted, _, _ = convert_state_dict(source_state, target_shapes)
            model.load_weights(
                [(key, mx.array(value)) for key, value in converted.items()], strict=True
            )
            mx.eval(model.parameters())

            profile = temporary / profile_name
            profile.mkdir()
            save_file(
                converted,
                str(profile / "model.safetensors"),
                metadata={"format": "mlx"},
            )
            (profile / "config.json").write_bytes(_json_bytes(flat_config))
            (profile / "generation_config.json").write_bytes(
                _json_bytes({"max_generation_steps": config_raw["predict"]["max_steps"]})
            )
            (profile / "preprocessor_config.json").write_bytes(
                _json_bytes(_preprocessor_config(config_raw))
            )
            report["profiles"][profile_name] = {
                "files": sorted(path.name for path in profile.iterdir()),
                "layer_depths": [
                    flat_config["num_encoder_layers"],
                    flat_config["num_decoder_layers"],
                ],
            }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    source = args.source or download_source(cache_dir=args.cache_dir)
    print(json.dumps(convert(source, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
