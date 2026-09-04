# SPDX-License-Identifier: Apache-2.0

"""Validate native MLX TableFormerV2 against a pinned CPU reference capture."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from docling_mlx._models.tableformer_v2.config import TABLEFORMER_V2_TOKENS
from docling_mlx.engines.table_structure.tableformer_v2.artifact import (
    CHECKPOINT_FILES,
    validate_tableformer_v2_artifact,
)
from docling_mlx.engines.table_structure.tableformer_v2.preprocessing import (
    preprocess_images,
    resize_pil_rgb_uint8,
)
from tools.tableformer_v2.capture_reference import (
    ARRAY_SPECS,
    CAPTURE_SCHEMA_VERSION,
    MAX_GENERATION_STEPS,
    REFERENCE_VERSIONS,
    array_sha256,
    otsl_from_ids,
)
from tools.tableformer_v2.source import SOURCE_REPO, SOURCE_REVISION, sha256

PREPROCESSING_MAX_ABS = 1e-6
ENCODER_MAX_MAE = 1e-5
ENCODER_MAX_ABS = 1e-4
FINAL_LOGITS_MAX_MAE = 1e-4
FINAL_LOGITS_MAX_ABS = 1e-3
FINAL_BBOX_MAX_ABS = 1e-4


def _artifact_file_evidence(artifact: Path) -> dict[str, dict[str, int | str]]:
    """Identify every present runtime artifact file used by the report."""

    return {
        name: {"sha256": sha256(artifact / name), "bytes": (artifact / name).stat().st_size}
        for name in CHECKPOINT_FILES
        if (artifact / name).is_file()
    }


def _numeric_gate(
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    max_mae: float | None,
    max_abs: float,
) -> dict[str, Any]:
    """Report shape, dtype, finite, and numeric bounds without masking boundaries."""

    shape_match = actual.shape == expected.shape
    dtype_match = actual.dtype == expected.dtype
    finite = bool(np.isfinite(actual).all()) and bool(np.isfinite(expected).all())
    gate: dict[str, Any] = {
        "name": name,
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "actual_dtype": str(actual.dtype),
        "expected_dtype": str(expected.dtype),
        "shape_match": shape_match,
        "dtype_match": dtype_match,
        "finite": finite,
        "thresholds": {"max_abs": max_abs},
    }
    if max_mae is not None:
        gate["thresholds"]["max_mae"] = max_mae
    if not shape_match or not dtype_match or not finite:
        gate["mae"] = None
        gate["max_abs"] = None
        gate["passed"] = False
        return gate
    if actual.size:
        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        mae = float(difference.mean())
        observed_max_abs = float(difference.max())
    else:
        mae = 0.0
        observed_max_abs = 0.0
    gate["mae"] = mae
    gate["max_abs"] = observed_max_abs
    gate["passed"] = observed_max_abs <= max_abs and (max_mae is None or mae <= max_mae)
    return gate


def _exact_array_gate(name: str, actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    shape_match = actual.shape == expected.shape
    dtype_match = actual.dtype == expected.dtype
    exact = shape_match and dtype_match and bool(np.array_equal(actual, expected))
    return {
        "name": name,
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "actual_dtype": str(actual.dtype),
        "expected_dtype": str(expected.dtype),
        "shape_match": shape_match,
        "dtype_match": dtype_match,
        "exact": exact,
        "passed": exact,
    }


def _first_sequence_divergence(actual: Sequence[Any], expected: Sequence[Any]) -> int | None:
    for index, (actual_id, expected_id) in enumerate(zip(actual, expected, strict=False)):
        if actual_id != expected_id:
            return index
    return None if len(actual) == len(expected) else min(len(actual), len(expected))


def _sequence_gate(name: str, actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    actual_ids = [int(value) for value in actual.reshape(-1).tolist()]
    expected_ids = [int(value) for value in expected.reshape(-1).tolist()]
    exact = actual.shape == expected.shape and actual_ids == expected_ids
    return {
        "name": name,
        "actual_count": len(actual_ids),
        "expected_count": len(expected_ids),
        "actual_ids": actual_ids,
        "expected_ids": expected_ids,
        "first_divergence": _first_sequence_divergence(actual_ids, expected_ids),
        "exact": exact,
        "passed": exact,
    }


def _otsl_gate(actual_ids: np.ndarray, expected_otsl: list[str]) -> dict[str, Any]:
    actual = otsl_from_ids([int(value) for value in actual_ids.reshape(-1).tolist()])
    return {
        "name": "generation.otsl",
        "actual": actual,
        "expected": expected_otsl,
        "first_divergence": _first_sequence_divergence(actual, expected_otsl),
        "exact": actual == expected_otsl,
        "passed": actual == expected_otsl,
    }


def _load_reference(reference: Path) -> tuple[dict[str, Any], list[dict[str, np.ndarray]]]:
    metadata = json.loads((reference / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != CAPTURE_SCHEMA_VERSION:
        raise ValueError("Unsupported TableFormerV2 reference schema_version")
    if metadata.get("producer") != "tools.tableformer_v2.capture_reference":
        raise ValueError("Reference was not produced by the TableFormerV2 capture tool")
    if metadata.get("profile") != "tableformer_v2":
        raise ValueError("Reference profile does not match the closed TableFormerV2 profile")
    expected_source = {
        "repo_id": SOURCE_REPO,
        "revision": SOURCE_REVISION,
    }
    source = metadata.get("source", {})
    if not isinstance(source, dict) or any(
        source.get(key) != value for key, value in expected_source.items()
    ):
        raise ValueError("Reference source identity does not match the closed profile")
    if metadata.get("token_map") != list(TABLEFORMER_V2_TOKENS):
        raise ValueError("Reference token map does not match the closed profile")
    if metadata.get("generation") != {"max_generation_steps": MAX_GENERATION_STEPS}:
        raise ValueError("Reference generation contract does not match TableFormerV2")
    runtime = metadata.get("runtime", {})
    if runtime.get("dependencies") != REFERENCE_VERSIONS:
        raise ValueError("Reference does not use the pinned TableFormerV2 dependencies")
    torch_settings = runtime.get("torch", {})
    if (
        torch_settings.get("device") != "cpu"
        or not torch_settings.get("eval")
        or not torch_settings.get("inference_mode")
        or not torch_settings.get("deterministic_algorithms")
    ):
        raise ValueError("Reference is not the deterministic PyTorch CPU eval oracle")
    captures = metadata.get("captures")
    if not isinstance(captures, list) or not captures:
        raise ValueError("Reference must contain at least one TableFormerV2 capture")

    expected_specs = [
        {"name": spec.name, "dtype": spec.dtype, "layout": spec.layout} for spec in ARRAY_SPECS
    ]
    loaded_captures: list[dict[str, np.ndarray]] = []
    for capture in captures:
        if not isinstance(capture, dict):
            raise ValueError("Reference capture metadata must be an object")
        declared_arrays = capture.get("arrays")
        if not isinstance(declared_arrays, list) or len(declared_arrays) != len(ARRAY_SPECS):
            raise ValueError("Reference capture has an incomplete array schema")
        for declared, expected in zip(declared_arrays, expected_specs, strict=True):
            if not isinstance(declared, dict) or any(
                declared.get(key) != value for key, value in expected.items()
            ):
                raise ValueError("Reference array schema does not match the validator")
            if not isinstance(declared.get("shape"), list) or not isinstance(
                declared.get("sha256"), str
            ):
                raise ValueError("Reference arrays must declare shape and SHA-256")
        archive_metadata = capture.get("archive", {})
        archive = reference / str(archive_metadata.get("file", ""))
        if not archive.is_file() or sha256(archive) != archive_metadata.get("sha256"):
            raise ValueError(f"Reference archive SHA-256 mismatch: {archive.name}")
        with np.load(archive, allow_pickle=False) as loaded:
            if loaded.files != [spec.name for spec in ARRAY_SPECS]:
                raise ValueError("Reference archive has missing, extra, or reordered arrays")
            arrays = {spec.name: np.array(loaded[spec.name], copy=True) for spec in ARRAY_SPECS}
        for declared, spec in zip(declared_arrays, ARRAY_SPECS, strict=True):
            array = arrays[spec.name]
            if (
                str(array.dtype) != spec.dtype
                or list(array.shape) != declared["shape"]
                or array_sha256(array) != declared["sha256"]
            ):
                raise ValueError(f"Reference array does not match metadata: {spec.name}")
            if not np.isfinite(array).all():
                raise ValueError(f"Reference array is non-finite: {spec.name}")
        if array_sha256(arrays["input_rgb_u8"]) != capture.get("input_rgb_sha256"):
            raise ValueError("Reference input RGB identity does not match capture metadata")
        expected_otsl = capture.get("otsl")
        expected_ids = arrays["generated_ids_i64"][0].tolist()
        if expected_otsl != otsl_from_ids(expected_ids):
            raise ValueError("Reference OTSL does not match its generated token IDs")
        loaded_captures.append(arrays)
    return metadata, loaded_captures


def _mlx_cached_trace(model: Any, encoder_outputs: Any) -> tuple[np.ndarray, np.ndarray]:
    """Replay the native cached loop and retain every step logit for diagnostics."""

    import mlx.core as mx

    generated_ids = mx.full((1, 1), 2, dtype=mx.int32)
    current_input = generated_ids
    past = None
    cross_cache = model.prepare_cross_attention_cache(encoder_outputs.last_hidden_state)
    mx.eval(*(array for entry in cross_cache for array in entry))
    step_logits: list[Any] = []
    for _ in range(MAX_GENERATION_STEPS):
        decoded = model.decode_tokens(
            current_input,
            encoder_outputs.last_hidden_state,
            past_key_values=past,
            cross_attention_cache=cross_cache,
            use_cache=True,
        )
        logits = decoded.logits[:, -1:, :]
        next_token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)[:, None]
        generated_ids = mx.concatenate([generated_ids, next_token], axis=1)
        past = decoded.past_key_values
        materialize = [generated_ids, logits]
        if past is not None:
            materialize.extend(array for cache in past for array in (cache.keys, cache.values))
        mx.eval(*materialize)
        step_logits.append(logits)
        current_input = next_token
        if bool(mx.all(next_token == model.config.eos_token_id).item()):
            break
    concatenated = mx.concatenate(step_logits, axis=1)
    mx.eval(generated_ids, concatenated)
    return (
        np.array(generated_ids, dtype=np.int64, copy=True),
        np.array(concatenated, dtype=np.float32, copy=True),
    )


def _first_token_diagnostic(
    actual_ids: np.ndarray,
    expected_ids: np.ndarray,
    actual_step_logits: np.ndarray,
    expected_step_logits: np.ndarray,
) -> dict[str, Any] | None:
    actual = [int(value) for value in actual_ids.reshape(-1).tolist()]
    expected = [int(value) for value in expected_ids.reshape(-1).tolist()]
    divergence = _first_sequence_divergence(actual, expected)
    if divergence is None:
        return None
    result: dict[str, Any] = {
        "token_index": divergence,
        "actual_id": actual[divergence] if divergence < len(actual) else None,
        "expected_id": expected[divergence] if divergence < len(expected) else None,
    }
    step = divergence - 1
    if (
        step >= 0
        and step < actual_step_logits.shape[1]
        and step < expected_step_logits.shape[1]
        and divergence < len(actual)
        and divergence < len(expected)
    ):
        target = actual_step_logits[0, step]
        reference = expected_step_logits[0, step]
        result["decision_step"] = step
        result["actual_winner_logit"] = float(target[actual[divergence]])
        result["actual_expected_id_logit"] = float(target[expected[divergence]])
        result["reference_expected_id_logit"] = float(reference[expected[divergence]])
        result["step_logit_max_abs"] = float(
            np.max(np.abs(target.astype(np.float64) - reference.astype(np.float64)))
        )
    return result


def _validate_capture(
    model: Any,
    artifact_spec: Any,
    capture: dict[str, Any],
    expected: dict[str, np.ndarray],
) -> dict[str, Any]:
    import mlx.core as mx

    image = Image.fromarray(expected["input_rgb_u8"], mode="RGB")
    resized = resize_pil_rgb_uint8(image, artifact_spec.preprocessing)
    pixels = preprocess_images([image], artifact_spec.preprocessing)
    mx.eval(pixels)
    pixels_np = np.array(pixels, dtype=np.float32, copy=True)
    encoder = model.encode_images(pixels)
    mx.eval(encoder.last_hidden_state)
    encoder_np = np.array(encoder.last_hidden_state, dtype=np.float32, copy=True)

    generation = model.generate(pixels, max_generation_steps=MAX_GENERATION_STEPS)
    mx.eval(generation.generated_ids, generation.logits, generation.predicted_bboxes)
    generated_ids = np.array(generation.generated_ids, dtype=np.int64, copy=True)
    generated_logits = np.array(generation.logits, dtype=np.float32, copy=True)
    generated_bboxes = np.array(generation.predicted_bboxes, dtype=np.float32, copy=True)

    trace_ids, trace_logits = _mlx_cached_trace(model, encoder)
    reference_ids_mlx = mx.array(expected["generated_ids_i64"], dtype=mx.int32)
    teacher = model(reference_ids_mlx, encoder_outputs=encoder, use_cache=False)
    mx.eval(teacher.logits, teacher.predicted_bboxes)
    teacher_logits = np.array(teacher.logits, dtype=np.float32, copy=True)
    teacher_bboxes = np.array(teacher.predicted_bboxes, dtype=np.float32, copy=True)

    gates = [
        _exact_array_gate("preprocessing.resized_rgb", resized, expected["resized_rgb_u8"]),
        _numeric_gate(
            "preprocessing.normalized_pixels",
            pixels_np,
            expected["pixels_nhwc_f32"],
            max_mae=None,
            max_abs=PREPROCESSING_MAX_ABS,
        ),
        _numeric_gate(
            "encoder.last_hidden_state",
            encoder_np,
            expected["encoder_last_hidden_state_f32"],
            max_mae=ENCODER_MAX_MAE,
            max_abs=ENCODER_MAX_ABS,
        ),
        _sequence_gate("generation.ids", generated_ids, expected["generated_ids_i64"]),
        _sequence_gate("generation.runtime_trace_ids", generated_ids, trace_ids),
        _otsl_gate(generated_ids, capture["otsl"]),
        _numeric_gate(
            "generation.greedy_step_logits",
            trace_logits,
            expected["greedy_step_logits_f32"],
            max_mae=FINAL_LOGITS_MAX_MAE,
            max_abs=FINAL_LOGITS_MAX_ABS,
        ),
        _numeric_gate(
            "generation.final_logits",
            generated_logits,
            expected["final_logits_f32"],
            max_mae=FINAL_LOGITS_MAX_MAE,
            max_abs=FINAL_LOGITS_MAX_ABS,
        ),
        _numeric_gate(
            "generation.normalized_bboxes",
            generated_bboxes,
            expected["normalized_bboxes_f32"],
            max_mae=None,
            max_abs=FINAL_BBOX_MAX_ABS,
        ),
        _numeric_gate(
            "teacher_forced.final_logits",
            teacher_logits,
            expected["final_logits_f32"],
            max_mae=FINAL_LOGITS_MAX_MAE,
            max_abs=FINAL_LOGITS_MAX_ABS,
        ),
        _numeric_gate(
            "teacher_forced.normalized_bboxes",
            teacher_bboxes,
            expected["normalized_bboxes_f32"],
            max_mae=None,
            max_abs=FINAL_BBOX_MAX_ABS,
        ),
    ]
    spatial_size_actual = list(encoder.spatial_size)
    spatial_size_expected = capture["encoder_spatial_size"]
    spatial_gate = {
        "name": "encoder.spatial_size",
        "actual": spatial_size_actual,
        "expected": spatial_size_expected,
        "exact": spatial_size_actual == spatial_size_expected,
        "passed": spatial_size_actual == spatial_size_expected,
    }
    gates.insert(3, spatial_gate)
    passed = all(gate["passed"] for gate in gates)
    return {
        "name": capture["name"],
        "status": "passed" if passed else "failed",
        "source_file_sha256": capture["source_file_sha256"],
        "first_token_divergence": _first_token_diagnostic(
            generated_ids,
            expected["generated_ids_i64"],
            trace_logits,
            expected["greedy_step_logits_f32"],
        ),
        "gates": gates,
    }


def validate(artifact: Path, reference: Path) -> dict[str, Any]:
    """Strict-load one artifact and evaluate every captured divergence boundary."""

    import mlx.core as mx

    from docling_mlx._models.tableformer_v2.model import TableFormerV2

    artifact = artifact.expanduser().resolve()
    reference = reference.expanduser().resolve()
    artifact_spec = validate_tableformer_v2_artifact(artifact)
    metadata, expected_captures = _load_reference(reference)
    model = TableFormerV2(artifact_spec.config)
    weights = mx.load(str(artifact / "model.safetensors"))
    if not isinstance(weights, dict) or any(
        value.dtype != mx.float32 for value in weights.values()
    ):
        raise ValueError("All TableFormerV2 parity artifact tensors must be float32")
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())

    captures = [
        _validate_capture(model, artifact_spec, capture, expected)
        for capture, expected in zip(metadata["captures"], expected_captures, strict=True)
    ]
    passed = all(capture["status"] == "passed" for capture in captures)
    return {
        "status": "passed" if passed else "failed",
        "profile": "tableformer_v2",
        "source_revision": SOURCE_REVISION,
        "artifact": {"files": _artifact_file_evidence(artifact)},
        "thresholds": {
            "preprocessing_max_abs": PREPROCESSING_MAX_ABS,
            "encoder_max_mae": ENCODER_MAX_MAE,
            "encoder_max_abs": ENCODER_MAX_ABS,
            "final_logits_max_mae": FINAL_LOGITS_MAX_MAE,
            "final_logits_max_abs": FINAL_LOGITS_MAX_ABS,
            "final_bbox_max_abs": FINAL_BBOX_MAX_ABS,
            "generated_ids": "exact",
            "otsl": "exact",
        },
        "captures": captures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional compact JSON report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate(args.artifact, args.reference)
    text = json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
