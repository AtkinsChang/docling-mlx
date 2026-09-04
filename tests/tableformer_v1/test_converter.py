# SPDX-License-Identifier: Apache-2.0

"""Strict TableFormer v1 conversion and publication checks."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from tools.tableformer_v1.convert_weights import (
    _BN_COUNTER_REASON,
    _atomic_output,
    convert,
    convert_state_dict,
)


def _source_state() -> dict[str, np.ndarray]:
    return {
        "conv.weight": np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2),
        "linear.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "bn.running_mean": np.zeros(2, dtype=np.float32),
        "bn.running_var": np.ones(2, dtype=np.float32),
        "bn.num_batches_tracked": np.array(7, dtype=np.int64),
    }


def _target_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "conv.weight": (2, 2, 2, 3),
        "linear.weight": (2, 3),
        "bn.running_mean": (2,),
        "bn.running_var": (2,),
    }


def test_conversion_transposes_oihw_and_retains_batch_norm_running_state() -> None:
    source = _source_state()
    converted, mappings, ignored = convert_state_dict(source, _target_shapes())

    assert set(converted) == set(_target_shapes())
    np.testing.assert_array_equal(
        converted["conv.weight"], source["conv.weight"].transpose(0, 2, 3, 1)
    )
    np.testing.assert_array_equal(converted["bn.running_mean"], source["bn.running_mean"])
    np.testing.assert_array_equal(converted["bn.running_var"], source["bn.running_var"])
    assert {item["source_key"] for item in mappings} == set(_target_shapes())
    assert {item["target_key"] for item in mappings} == set(_target_shapes())
    assert {item["transform"] for item in mappings} == {"OIHW_to_OHWI", "identity"}
    assert ignored == {"bn.num_batches_tracked": _BN_COUNTER_REASON}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda state: state.pop("linear.weight"), "missing"),
        (lambda state: state.__setitem__("extra", np.zeros(1, dtype=np.float32)), "unexpected"),
        (
            lambda state: state.__setitem__("linear.weight", np.zeros((2, 3), dtype=np.float16)),
            "float32",
        ),
        (
            lambda state: state.__setitem__(
                "linear.weight", np.full((2, 3), np.nan, dtype=np.float32)
            ),
            "finite",
        ),
        (
            lambda state: state.__setitem__("linear.weight", np.zeros((3, 2), dtype=np.float32)),
            "shape",
        ),
    ],
)
def test_conversion_rejects_incomplete_or_invalid_inference_state(
    mutation: Callable[[dict[str, np.ndarray]], object], message: str
) -> None:
    source = _source_state()
    mutation(source)
    with pytest.raises(ValueError, match=message):
        convert_state_dict(source, _target_shapes())


def test_only_paired_int64_scalar_batch_norm_counters_are_ignored() -> None:
    source = _source_state()
    source["bn.num_batches_tracked"] = np.array(0, dtype=np.int32)
    with pytest.raises(ValueError, match="int64 scalar"):
        convert_state_dict(source, _target_shapes())

    source = _source_state()
    source["other.num_batches_tracked"] = np.array(0, dtype=np.int64)
    with pytest.raises(ValueError, match="Unpaired"):
        convert_state_dict(source, _target_shapes())


def test_model_card_records_profiles_and_source_attribution() -> None:
    from tools._common.model_card import render_model_card

    card = render_model_card(
        "docling-project/docling-models",
        "fc0f2d45e2218ea24bce5045f58a389aed16dc23",
        [
            "cdla-permissive-2.0",
            "apache-2.0",
        ],
    )
    assert "  - cdla-permissive-2.0" in card
    assert "  - apache-2.0" in card
    assert "# docling-models - MLX" in card
    assert "  - docling-project/docling-models" in card
    assert "https://huggingface.co/docling-project/docling-models" in card


def test_atomic_output_publishes_complete_directory(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    with _atomic_output(output) as staging:
        (staging / "complete").write_text("yes")
        assert not output.exists()
    assert (output / "complete").read_text() == "yes"
    assert not list(tmp_path.glob(".artifact.convert-*"))


def test_atomic_output_cleans_partial_state_after_failure(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    with pytest.raises(RuntimeError, match="failed"):
        with _atomic_output(output) as staging:
            (staging / "partial").write_text("no")
            raise RuntimeError("failed")
    assert not output.exists()
    assert not list(tmp_path.glob(".artifact.convert-*"))


@pytest.mark.mlx
@pytest.mark.parity
def test_real_closed_profiles_convert(tmp_path: Path) -> None:
    from safetensors.numpy import load_file

    source_value = os.environ.get("DOCLING_MLX_TABLEFORMER_V1_SOURCE")
    if source_value is None:
        pytest.fail(
            "selected parity lane requires the pinned TableFormer v1 source; "
            "set DOCLING_MLX_TABLEFORMER_V1_SOURCE"
        )
    source = Path(source_value).expanduser()
    accurate_config = json.loads(
        (source / "model_artifacts/tableformer/accurate/tm_config.json").read_text()
    )
    fast_config = json.loads(
        (source / "model_artifacts/tableformer/fast/tm_config.json").read_text()
    )
    assert (accurate_config["model"]["enc_layers"], accurate_config["model"]["dec_layers"]) == (
        6,
        6,
    )
    assert (fast_config["model"]["enc_layers"], fast_config["model"]["dec_layers"]) == (4, 2)
    fast_config["model"]["enc_layers"] = 6
    fast_config["model"]["dec_layers"] = 6
    assert fast_config == accurate_config

    output = tmp_path / "converted"
    convert(source, output)
    assert set(path.name for path in output.iterdir()) == {"accurate", "fast", "README.md"}
    for name in ("accurate", "fast"):
        assert set(path.name for path in (output / name).iterdir()) == {
            "model.safetensors",
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
        }
        config = json.loads((output / name / "config.json").read_text())
        assert config["model_type"] == "tableformer_v1"
        assert config["architectures"] == ["TableFormerV1"]
        assert config["vocab"] == [
            "<pad>",
            "<unk>",
            "<start>",
            "<end>",
            "ecel",
            "fcel",
            "lcel",
            "ucel",
            "xcel",
            "nl",
            "ched",
            "rhed",
            "srow",
        ]
        assert json.loads((output / name / "generation_config.json").read_text()) == {
            "max_generation_steps": 1024
        }
        assert "tm_config.json" not in {path.name for path in (output / name).iterdir()}
        weights = load_file(output / name / "model.safetensors")
        assert all(tensor.dtype == np.float32 for tensor in weights.values())
