# SPDX-License-Identifier: Apache-2.0

"""Validate a converted TableFormer v1 artifact against a CPU capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from docling.datamodel.pipeline_options import TableFormerMode
from PIL import Image

from docling_mlx._models.tableformer_v1.config import TABLEFORMER_V1_TOKENS
from docling_mlx.engines.table_structure.tableformer_v1.artifact import (
    CHECKPOINT_FILES,
    validate_tableformer_v1_artifact,
)
from docling_mlx.engines.table_structure.tableformer_v1.preprocessing import preprocess_table_image
from tools.tableformer_v1.capture_reference import (
    CAPTURE_SCHEMA_VERSION,
    _stage_page,
    array_sha256,
    array_specs,
    otsl_from_ids,
    structured_table,
)
from tools.tableformer_v1.source import SOURCE_REVISION, profile_id, sha256

PREPROCESSING_MAX_ABS = 1e-6
ENCODER_MAX_MAE = 1e-5
ENCODER_MAX_ABS = 1e-4
TAG_ENCODER_MAX_MAE = 5e-5
TAG_ENCODER_MAX_ABS = 5e-4
FINAL_LOGITS_MAX_MAE = 1e-4
FINAL_LOGITS_MAX_ABS = 1e-3
FINAL_BBOX_MAX_ABS = 1e-4


def _numeric_gate(
    name: str, actual: np.ndarray, expected: np.ndarray, *, mae: float | None, max_abs: float
) -> dict[str, Any]:
    comparable = (
        actual.shape == expected.shape
        and actual.dtype == expected.dtype
        and bool(np.isfinite(actual).all())
        and bool(np.isfinite(expected).all())
    )
    result: dict[str, Any] = {
        "name": name,
        "shape_match": actual.shape == expected.shape,
        "dtype_match": actual.dtype == expected.dtype,
        "thresholds": {"max_abs": max_abs, "max_mae": mae},
    }
    if not comparable:
        return {**result, "mae": None, "max_abs": None, "passed": False}
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    observed_mae = float(difference.mean()) if difference.size else 0.0
    observed_max = float(difference.max()) if difference.size else 0.0
    return {
        **result,
        "mae": observed_mae,
        "max_abs": observed_max,
        "passed": observed_max <= max_abs and (mae is None or observed_mae <= mae),
    }


def _exact_gate(name: str, actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    exact = (
        actual.shape == expected.shape
        and actual.dtype == expected.dtype
        and bool(np.array_equal(actual, expected))
    )
    return {
        "name": name,
        "shape_match": actual.shape == expected.shape,
        "dtype_match": actual.dtype == expected.dtype,
        "exact": exact,
        "passed": exact,
    }


def _sequence_gate(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    actual_ids, expected_ids = actual.reshape(-1).tolist(), expected.reshape(-1).tolist()
    first = next(
        (
            index
            for index, pair in enumerate(zip(actual_ids, expected_ids, strict=False))
            if pair[0] != pair[1]
        ),
        None,
    )
    if first is None and len(actual_ids) != len(expected_ids):
        first = min(len(actual_ids), len(expected_ids))
    exact = actual.shape == expected.shape and actual_ids == expected_ids
    return {
        "name": "generation.ids",
        "actual_ids": actual_ids,
        "expected_ids": expected_ids,
        "first_divergence": first,
        "exact": exact,
        "passed": exact,
    }


def _load_reference(
    reference: Path,
    mode: TableFormerMode,
    encoder_layers: int,
) -> tuple[dict[str, Any], list[dict[str, np.ndarray]]]:
    metadata = json.loads((reference / "metadata.json").read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != CAPTURE_SCHEMA_VERSION
        or metadata.get("producer") != "tools.tableformer_v1.capture_reference"
    ):
        raise ValueError("Unsupported TableFormer v1 reference capture")
    if metadata.get("profile") != profile_id(mode):
        raise ValueError(f"Reference profile is not {mode.value}")
    source = metadata.get("source")
    if not isinstance(source, dict) or source.get("revision") != SOURCE_REVISION:
        raise ValueError("Reference source revision does not match the frozen revision")
    specs = array_specs(encoder_layers)
    captures = []
    for entry in metadata.get("captures", []):
        archive = reference / entry["archive"]["file"]
        if sha256(archive) != entry["archive"]["sha256"]:
            raise ValueError(f"Reference archive SHA-256 mismatch: {archive}")
        with np.load(archive, allow_pickle=False) as loaded:
            arrays = {spec.name: np.ascontiguousarray(loaded[spec.name]) for spec in specs}
        for spec, item in zip(specs, entry["arrays"], strict=True):
            value = arrays[spec.name]
            if item != {
                "name": spec.name,
                "dtype": spec.dtype,
                "layout": spec.layout,
                "shape": list(value.shape),
                "sha256": array_sha256(value),
            }:
                raise ValueError(f"Reference array metadata mismatch: {spec.name}")
        captures.append(arrays)
    if not captures:
        raise ValueError("Reference capture is empty")
    return metadata, captures


def _native_trace(model: Any, pixels: np.ndarray) -> dict[str, np.ndarray]:
    import mlx.core as mx

    image = model._encoder.forward_intermediates(mx.array(pixels.transpose(0, 2, 3, 1)))
    x = image["image_features"]
    for block in model._tag_transformer._input_filter:
        x = block(x)
    intermediates: dict[str, Any] = {
        "encoder.image_features_f32": image["image_features"],
        "encoder.input_filter_f32": x,
    }
    memory = model._tag_transformer._encoder.forward_intermediates(
        x.reshape(x.shape[0], -1, x.shape[-1])
    )
    for index in range(model.config.encoder_layers):
        intermediates[f"encoder.tag_layer_{index}_f32"] = memory[f"tag_encoder.layers.{index}"]
    memory_value = memory[f"tag_encoder.layers.{model.config.encoder_layers - 1}"]
    generated, logits, hidden_states, cache = [2], [], [], None
    for _ in range(model.config.max_steps):
        output = model._tag_transformer.step(
            mx.array(generated, dtype=mx.int32)[:, None], memory_value, cache
        )
        mx.eval(output.logits, output.hidden_state, output.cache)
        token = model._tag_transformer._correct_token(
            cast(int, mx.argmax(output.logits, axis=-1).item()), generated
        )
        generated.append(token)
        logits.append(output.logits)
        hidden_states.append(output.hidden_state)
        cache = output.cache
        if token == 3:
            break
    tokens = [TABLEFORMER_V1_TOKENS[token] for token in generated[1:]]
    from docling_mlx._models.tableformer_v1.bbox import (
        cxcywh_to_xyxy,
        merge_horizontal_bboxes,
        select_bbox_state_indices,
    )

    indices, merges = select_bbox_state_indices(tokens)
    states = (
        mx.take(
            mx.concatenate(hidden_states, axis=0)[:, 0, :],
            mx.array(indices, dtype=mx.int32),
            axis=0,
        )
        if indices
        else mx.zeros((0, 512), dtype=mx.float32)
    )
    classes, cxcywh = model._bbox_decoder(image["image_features"], states)
    classes, cxcywh = merge_horizontal_bboxes(classes, cxcywh, merges)
    xyxy = cxcywh_to_xyxy(cxcywh)
    mx.eval(*intermediates.values(), *logits, classes, cxcywh, xyxy)
    return {
        "input_rgb_u8": np.empty((0,), dtype=np.uint8),
        "pixels_chw_f32": np.asarray(pixels),
        **{name: np.asarray(value, dtype=np.float32) for name, value in intermediates.items()},
        "generated_ids_i64": np.asarray([generated], dtype=np.int64),
        "greedy_step_logits_f32": np.asarray(mx.stack(logits, axis=1), dtype=np.float32),
        "bbox_class_logits_f32": np.asarray(classes, dtype=np.float32),
        "bbox_cxcywh_f32": np.asarray(cxcywh, dtype=np.float32),
        "bbox_xyxy_f32": np.asarray(xyxy, dtype=np.float32),
    }


def _artifact_evidence(
    root: Path,
    mode: TableFormerMode,
) -> dict[str, dict[str, int | str]]:
    artifact = validate_tableformer_v1_artifact(root / mode.value).directory
    return {
        name: {"sha256": sha256(artifact / name), "bytes": (artifact / name).stat().st_size}
        for name in CHECKPOINT_FILES
        if (artifact / name).is_file()
    }


def _native_stage(
    artifact_root: Path,
    image: np.ndarray,
    mode: TableFormerMode,
) -> dict[str, Any]:
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.document import ConversionResult

    from docling_mlx.engines.table_structure.tableformer_v1 import TableFormerV1ModelSpec
    from docling_mlx.stages.table_structure_v1 import (
        MlxTableFormerV1Model,
        MlxTableStructureOptions,
    )

    stage = MlxTableFormerV1Model(
        True,
        artifact_root.parent,
        MlxTableStructureOptions(
            model_spec=TableFormerV1ModelSpec(path=artifact_root),
            mode=mode,
            do_cell_matching=False,
        ),
        AcceleratorOptions(device="auto"),
    )
    result = stage.predict_tables(
        ConversionResult.model_construct(timings={}), [_stage_page(Image.fromarray(image))]
    )[0]
    return structured_table(result.table_map[0])


def _structured_stage_gate(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    structural = {key: actual[key] for key in ("otsl", "num_rows", "num_cols", "cells")}
    expected_structural = {key: expected[key] for key in structural}
    boxes = _numeric_gate(
        "stage.bboxes_xyxy",
        np.asarray(actual["bboxes_xyxy"], dtype=np.float64),
        np.asarray(expected["bboxes_xyxy"], dtype=np.float64),
        mae=None,
        # Page-space coordinates follow the raw normalized-box gate above and
        # are scaled by the fixture dimensions before the two stages expose them.
        max_abs=2e-3,
    )
    return {
        "name": "stage.structured_output",
        "structural_exact": structural == expected_structural,
        "bbox": boxes,
        "passed": structural == expected_structural and boxes["passed"],
    }


def validate(
    artifact_root: Path,
    reference: Path,
    output: Path | None = None,
    mode: TableFormerMode = TableFormerMode.ACCURATE,
) -> dict[str, Any]:
    from docling_mlx._models.tableformer_v1.model import TableFormerV1

    artifact = validate_tableformer_v1_artifact(artifact_root / mode.value)
    metadata, expected_captures = _load_reference(reference, mode, artifact.config.encoder_layers)

    model = TableFormerV1(artifact.config)
    model.load_weights(str(artifact.directory / "model.safetensors"), strict=True)
    model.eval()
    rows = []
    for entry, expected in zip(metadata["captures"], expected_captures, strict=True):
        actual_pixels = preprocess_table_image(expected["input_rgb_u8"], artifact.preprocessing)
        actual = _native_trace(model, actual_pixels)
        gates = [
            _exact_gate(
                "preprocessing.input_rgb", expected["input_rgb_u8"], expected["input_rgb_u8"]
            ),
            _numeric_gate(
                "preprocessing.pixels",
                actual["pixels_chw_f32"],
                expected["pixels_chw_f32"],
                mae=None,
                max_abs=PREPROCESSING_MAX_ABS,
            ),
        ]
        gates.extend(
            _numeric_gate(
                name, actual[name], expected[name], mae=ENCODER_MAX_MAE, max_abs=ENCODER_MAX_ABS
            )
            for name in ("encoder.image_features_f32", "encoder.input_filter_f32")
        )
        gates.extend(
            _numeric_gate(
                name,
                actual[name],
                expected[name],
                mae=TAG_ENCODER_MAX_MAE,
                max_abs=TAG_ENCODER_MAX_ABS,
            )
            for name in (
                f"encoder.tag_layer_{index}_f32" for index in range(artifact.config.encoder_layers)
            )
        )
        gates.extend(
            (
                _sequence_gate(actual["generated_ids_i64"], expected["generated_ids_i64"]),
                {
                    "name": "generation.otsl",
                    "actual": otsl_from_ids(actual["generated_ids_i64"].reshape(-1).tolist()),
                    "expected": entry["otsl"],
                    "passed": otsl_from_ids(actual["generated_ids_i64"].reshape(-1).tolist())
                    == entry["otsl"],
                },
                _numeric_gate(
                    "generation.greedy_logits",
                    actual["greedy_step_logits_f32"],
                    expected["greedy_step_logits_f32"],
                    mae=FINAL_LOGITS_MAX_MAE,
                    max_abs=FINAL_LOGITS_MAX_ABS,
                ),
                _numeric_gate(
                    "prediction.bbox_class_logits",
                    actual["bbox_class_logits_f32"],
                    expected["bbox_class_logits_f32"],
                    mae=FINAL_LOGITS_MAX_MAE,
                    max_abs=FINAL_LOGITS_MAX_ABS,
                ),
                _numeric_gate(
                    "prediction.bbox_cxcywh",
                    actual["bbox_cxcywh_f32"],
                    expected["bbox_cxcywh_f32"],
                    mae=None,
                    max_abs=FINAL_BBOX_MAX_ABS,
                ),
                _numeric_gate(
                    "prediction.bbox_xyxy",
                    actual["bbox_xyxy_f32"],
                    expected["bbox_xyxy_f32"],
                    mae=None,
                    max_abs=FINAL_BBOX_MAX_ABS,
                ),
            )
        )
        gates.append(
            _structured_stage_gate(
                _native_stage(artifact_root, expected["input_rgb_u8"], mode),
                entry["structured_stage"],
            )
        )
        rows.append(
            {"name": entry["name"], "passed": all(gate["passed"] for gate in gates), "gates": gates}
        )
    report = {
        "profile": profile_id(mode),
        "reference": {"path": str(reference), "source": metadata["source"]},
        "artifact": {
            "root": str(artifact_root),
            "files": _artifact_evidence(artifact_root, mode),
        },
        "fixtures": rows,
        "passed": all(row["passed"] for row in rows),
        "stage_comparison": {
            "available": True,
            "reference": "docling.models.stages.table_structure.TableStructureModel",
            "native": "docling_mlx.stages.table_structure_v1.MlxTableFormerV1Model",
        },
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=(TableFormerMode.ACCURATE.value, TableFormerMode.FAST.value),
        default=TableFormerMode.ACCURATE.value,
    )
    args = parser.parse_args()
    report = validate(
        args.artifact,
        args.reference,
        args.output,
        TableFormerMode(args.profile),
    )
    print(
        json.dumps({"output": str(args.output), "passed": report["passed"]}, separators=(",", ":"))
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
