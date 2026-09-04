# SPDX-License-Identifier: Apache-2.0

"""Pure conversion checks for the strict TableFormerV2 artifact contract."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from tools.tableformer_v2.convert_weights import (
    _CLASSIFIER_REASON,
    _GENERATION_CONFIG,
    _PREPROCESSOR_CONFIG,
    _UNUSED_CLASSIFIER_KEYS,
    _atomic_output,
    _json_bytes,
    _write_artifact_files,
    convert,
    convert_state_dict,
    rename_key,
)


def _source_state() -> dict[str, np.ndarray]:
    return {
        "conv.weight": np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2),
        "linear.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "bn.running_mean": np.zeros(2, dtype=np.float32),
        "bn.running_var": np.ones(2, dtype=np.float32),
        "bn.num_batches_tracked": np.array(7, dtype=np.int64),
        "feature_extractor.classifier.1.weight": np.ones((1000, 1280), dtype=np.float32),
        "feature_extractor.classifier.1.bias": np.ones(1000, dtype=np.float32),
    }


@pytest.mark.parametrize(
    ("source_key", "target_key"),
    [
        (
            "bbox_head.bbox_mlp.0.weight",
            "bbox_head.bbox_mlp.layers.0.weight",
        ),
        (
            "bbox_head.bbox_mlp.6.bias",
            "bbox_head.bbox_mlp.layers.6.bias",
        ),
        (
            "bbox_head.layers.1.ffn.3.weight",
            "bbox_head.layers.1.ffn.layers.3.weight",
        ),
        ("bbox_head.input_proj.weight", "bbox_head.input_proj.weight"),
    ],
)
def test_rename_key_adds_only_the_required_mlx_sequential_namespaces(
    source_key: str, target_key: str
) -> None:
    assert rename_key(source_key) == target_key


def test_conversion_transposes_only_oihw_and_accounts_for_each_ignore_reason() -> None:
    source = _source_state()
    target = {
        "conv.weight": (2, 2, 2, 3),
        "linear.weight": (2, 3),
        "bn.running_mean": (2,),
        "bn.running_var": (2,),
    }
    converted, mappings, ignored = convert_state_dict(source, target)

    assert set(converted) == set(target)
    np.testing.assert_array_equal(
        converted["conv.weight"], source["conv.weight"].transpose(0, 2, 3, 1)
    )
    np.testing.assert_array_equal(converted["linear.weight"], source["linear.weight"])
    assert {item["source_key"] for item in mappings} == set(target)
    assert {item["target_key"] for item in mappings} == set(target)
    assert {item["transform"] for item in mappings} == {"OIHW_to_OHWI", "identity"}
    assert ignored["bn.num_batches_tracked"] == "batch_norm_training_counter"
    assert {key for key, reason in ignored.items() if reason == _CLASSIFIER_REASON} == set(
        _UNUSED_CLASSIFIER_KEYS
    )
    assert set(ignored).isdisjoint(converted)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda source: source.pop("linear.weight"), "missing"),
        (lambda source: source.__setitem__("extra", np.zeros(1, dtype=np.float32)), "unexpected"),
        (
            lambda source: source.__setitem__("linear.weight", np.zeros((2, 3), dtype=np.float16)),
            "float32",
        ),
        (
            lambda source: source.__setitem__(
                "linear.weight", np.full((2, 3), np.nan, dtype=np.float32)
            ),
            "finite",
        ),
        (
            lambda source: source.__setitem__("linear.weight", np.zeros((3, 2), dtype=np.float32)),
            "shape",
        ),
    ],
)
def test_conversion_rejects_missing_extra_or_invalid_inference_tensor(
    mutation: Callable[[dict[str, np.ndarray]], object], message: str
) -> None:
    source = _source_state()
    mutation(source)
    target = {
        "conv.weight": (2, 2, 2, 3),
        "linear.weight": (2, 3),
        "bn.running_mean": (2,),
        "bn.running_var": (2,),
    }
    with pytest.raises(ValueError, match=message):
        convert_state_dict(source, target)


def test_conversion_rejects_unpaired_or_invalid_bn_counter() -> None:
    source = _source_state()
    source["bn.num_batches_tracked"] = np.array(0, dtype=np.int32)
    with pytest.raises(ValueError, match="int64 scalar"):
        convert_state_dict(source, {})

    source = _source_state()
    source["other.num_batches_tracked"] = np.array(0, dtype=np.int64)
    with pytest.raises(ValueError, match="Unpaired"):
        convert_state_dict(source, {})


def test_conversion_rejects_invalid_ignored_classifier_tensor() -> None:
    source = _source_state()
    source["feature_extractor.classifier.1.bias"] = np.full(1000, np.inf, dtype=np.float32)
    with pytest.raises(ValueError, match="finite float32"):
        convert_state_dict(source, {})


def test_synthesized_json_contract_is_exact_and_deterministic() -> None:
    assert _PREPROCESSOR_CONFIG == {
        "do_convert_rgb": True,
        "do_resize": True,
        "size": {"height": 448, "width": 448},
        "interpolation": "bilinear",
        "antialias": True,
        "do_rescale": True,
        "rescale_factor": 0.00392156862745098,
        "do_normalize": True,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "output_dtype": "float32",
        "output_layout": "NHWC",
    }
    assert _GENERATION_CONFIG == {"max_generation_steps": 512}
    for value in (_PREPROCESSOR_CONFIG, _GENERATION_CONFIG):
        encoded = _json_bytes(value)
        assert encoded.endswith(b"\n")
        assert json.loads(encoded) == value
        assert _json_bytes(value) == encoded


def test_artifact_files_copy_tokenizers_verbatim_and_omit_license_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "config.json").write_bytes(b"{}\n")
    tokenizer_files = {
        "special_tokens_map.json": b'{"pad_token":{"content":"<pad>"}}\n',
        "tokenizer.json": b'{"added_tokens":[]}\n',
        "tokenizer_config.json": b'{"pad_token":"<pad>"}\n',
    }
    for name, contents in tokenizer_files.items():
        (source / name).write_bytes(contents)

    _write_artifact_files(output, source)

    assert not (output / "LICENSES").exists()
    assert {path.name for path in output.iterdir()} == {
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        *tokenizer_files,
    }
    for name, contents in tokenizer_files.items():
        assert (output / name).read_bytes() == contents


def test_atomic_output_publishes_complete_directory(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    with _atomic_output(output) as staging:
        assert staging != output
        (staging / "complete").write_text("yes")
        assert not output.exists()
    assert (output / "complete").read_text() == "yes"
    assert not list(tmp_path.glob(".artifact.convert-*"))


def test_atomic_output_cleans_staging_after_failure(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    with pytest.raises(RuntimeError, match="conversion failed"):
        with _atomic_output(output) as staging:
            (staging / "partial").write_text("no")
            raise RuntimeError("conversion failed")
    assert not output.exists()
    assert not list(tmp_path.glob(".artifact.convert-*"))


@pytest.mark.mlx
@pytest.mark.parity
def test_real_pinned_checkpoint_converts_and_smokes(tmp_path: Path) -> None:
    source_value = os.environ.get("DOCLING_MLX_TABLEFORMER_V2_SOURCE")
    if source_value is None:
        pytest.fail(
            "selected parity lane requires DOCLING_MLX_TABLEFORMER_V2_SOURCE "
            "to point to the pinned offline snapshot"
        )
    source = Path(source_value).expanduser()
    output = tmp_path / "converted"
    report = convert(source, output)

    assert report["verification"]["forward_nonzero"] is True
    assert report["verification"]["generation_nonzero"] is True
    assert (output / "config.json").read_bytes() == (source / "config.json").read_bytes()
    for name in ("special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"):
        assert (output / name).read_bytes() == (source / name).read_bytes()
    output_files = {
        path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
    }
    assert output_files == {
        "README.md",
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }

    from safetensors.numpy import load_file

    from docling_mlx.engines.table_structure.tableformer_v2.artifact import (
        validate_tableformer_v2_artifact,
    )

    validate_tableformer_v2_artifact(output)
    weights = load_file(output / "model.safetensors")
    assert all(tensor.dtype == np.float32 for tensor in weights.values())
