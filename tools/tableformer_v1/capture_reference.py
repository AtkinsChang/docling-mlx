# SPDX-License-Identifier: Apache-2.0

"""Capture a pinned Torch CPU TableFormer v1 oracle outside Git."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from docling.datamodel.pipeline_options import TableFormerMode
from PIL import Image

from docling_mlx._models.tableformer_v1.config import TABLEFORMER_V1_TOKENS
from tools.pinned_versions import require_locked_versions
from tools.tableformer_v1.source import SOURCE_REVISION, profile_id, sha256, verify_source

CAPTURE_SCHEMA_VERSION = 1
REFERENCE_PACKAGES = (
    "docling",
    "docling-ibm-models",
    "numpy",
    "opencv-python",
    "pillow",
    "torch",
    "torchvision",
)


@dataclass(frozen=True, slots=True)
class ArraySpec:
    name: str
    dtype: str
    layout: str


def array_specs(encoder_layers: int) -> tuple[ArraySpec, ...]:
    return (
        ArraySpec("input_rgb_u8", "uint8", "HWC"),
        ArraySpec("pixels_chw_f32", "float32", "BCHW"),
        ArraySpec("encoder.image_features_f32", "float32", "BHWC"),
        ArraySpec("encoder.input_filter_f32", "float32", "BHWC"),
        *(
            ArraySpec(f"encoder.tag_layer_{index}_f32", "float32", "SBH")
            for index in range(encoder_layers)
        ),
        ArraySpec("generated_ids_i64", "int64", "BL"),
        ArraySpec("greedy_step_logits_f32", "float32", "B(L-1)V"),
        ArraySpec("bbox_class_logits_f32", "float32", "C3"),
        ArraySpec("bbox_cxcywh_f32", "float32", "C4"),
        ArraySpec("bbox_xyxy_f32", "float32", "C4"),
    )


ARRAY_SPECS = array_specs(6)


def array_sha256(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def write_compressed_npz(
    path: Path,
    arrays: dict[str, np.ndarray],
    specs: Sequence[ArraySpec] = ARRAY_SPECS,
) -> None:
    if list(arrays) != [spec.name for spec in specs]:
        raise ValueError("TableFormer v1 capture arrays do not match the stable schema")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for spec in specs:
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer, np.ascontiguousarray(arrays[spec.name]), allow_pickle=False
            )
            archive.writestr(
                zipfile.ZipInfo(f"{spec.name}.npy", date_time=(1980, 1, 1, 0, 0, 0)),
                buffer.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def otsl_from_ids(ids: list[int]) -> list[str]:
    return [
        TABLEFORMER_V1_TOKENS[token]
        for token in ids
        if TABLEFORMER_V1_TOKENS[token] not in {"<pad>", "<unk>", "<start>", "<end>"}
    ]


def _versions() -> dict[str, str]:
    return require_locked_versions(REFERENCE_PACKAGES, context="TableFormer v1 capture")


def configure_torch(torch: Any, cpu_threads: int) -> dict[str, Any]:
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    return {"device": "cpu", "eval": True, "inference_mode": True, "num_threads": cpu_threads}


def preprocess_exact(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """The upstream OpenCV normalization, resize, transpose, and scale sequence."""

    import cv2

    normalization = config["dataset"]["image_normalization"]
    normalized = (image.astype(np.float32) - 255.0 * np.array(normalization["mean"])) / np.array(
        normalization["std"]
    )
    resized = cv2.resize(
        normalized,
        (config["dataset"]["resized_image"], config["dataset"]["resized_image"]),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.ascontiguousarray(resized.transpose(2, 1, 0)[None] / 255.0, dtype=np.float32)


def load_reference(
    source: Path,
    torch: Any,
    mode: TableFormerMode = TableFormerMode.ACCURATE,
) -> tuple[Any, dict[str, Any]]:
    from docling_ibm_models.tableformer.models.table04_rs.tablemodel04_rs import TableModel04_rs
    from safetensors.torch import load_model

    profile = source / "model_artifacts/tableformer" / mode.value
    config = json.loads((profile / "tm_config.json").read_text(encoding="utf-8"))
    config["model"]["save_dir"] = str(profile)
    model = TableModel04_rs(config, {"word_map": config["dataset_wordmap"]}, "cpu").cpu().eval()
    missing, unexpected = load_model(model, profile / f"tableformer_{mode.value}.safetensors")
    if missing or unexpected:
        raise RuntimeError(
            f"Official {mode.value} checkpoint did not strict-load: {missing}, {unexpected}"
        )
    return model, config


def reference_trace(
    model: Any, config: dict[str, Any], pixels: np.ndarray, torch: Any
) -> dict[str, np.ndarray]:
    """Run source-equivalent cached decoding and retain each release boundary."""

    captured: dict[str, np.ndarray] = {}
    handles = [
        model._encoder.register_forward_hook(
            lambda _m, _a, output: captured.__setitem__(
                "encoder.image_features_f32", output.cpu().numpy()
            )
        ),
        model._tag_transformer._input_filter.register_forward_hook(
            lambda _m, _a, output: captured.__setitem__(
                "encoder.input_filter_f32", output.permute(0, 2, 3, 1).cpu().numpy()
            )
        ),
    ]
    handles.extend(
        layer.register_forward_hook(
            lambda _m, _a, output, index=index: captured.__setitem__(
                f"encoder.tag_layer_{index}_f32", output.cpu().numpy()
            )
        )
        for index, layer in enumerate(model._tag_transformer._encoder.layers)
    )
    try:
        with torch.inference_mode():
            image_features = model._encoder(torch.from_numpy(pixels))
            filtered = model._tag_transformer._input_filter(image_features.permute(0, 3, 1, 2))
            inputs = filtered.permute(0, 2, 3, 1).reshape(1, -1, 512).permute(1, 0, 2)
            positions = inputs.shape[0]
            mask = torch.zeros(
                (model._tag_transformer._n_heads, positions, positions)
            ) == torch.ones((model._tag_transformer._n_heads, positions, positions))
            memory = model._tag_transformer._encoder(inputs, mask=mask)
            word_map = config["dataset_wordmap"]["word_map_tag"]
            decoded = torch.tensor([[word_map["<start>"]]], dtype=torch.long)
            cache = None
            tags: list[int] = []
            logits: list[Any] = []
            states: list[Any] = []
            skip, previous_ucel, first_lcel, line_num, current, bbox_index = (
                True,
                False,
                True,
                0,
                -1,
                0,
            )
            merges: dict[int, int] = {}
            for _ in range(config["predict"]["max_steps"]):
                output, cache = model._tag_transformer._decoder(
                    model._tag_transformer._positional_encoding(
                        model._tag_transformer._embedding(decoded)
                    ),
                    memory,
                    cache,
                    memory_key_padding_mask=mask,
                )
                step_logits = model._tag_transformer._fc(output[-1])
                logits.append(step_logits)
                tag = step_logits.argmax(1).item()
                if line_num == 0 and tag == word_map["xcel"]:
                    tag = word_map["lcel"]
                if previous_ucel and tag == word_map["lcel"]:
                    tag = word_map["fcel"]
                tags.append(tag)
                if tag == word_map["<end>"]:
                    decoded = torch.cat((decoded, torch.tensor([[tag]], dtype=torch.long)))
                    break
                if not skip and tag in {
                    word_map[value]
                    for value in ("fcel", "ecel", "ched", "rhed", "srow", "nl", "ucel")
                }:
                    states.append(output[-1])
                    if not first_lcel:
                        merges[current] = bbox_index
                    bbox_index += 1
                if tag != word_map["lcel"]:
                    first_lcel = True
                elif first_lcel:
                    states.append(output[-1])
                    first_lcel, current = False, bbox_index
                    merges[current] = -1
                    bbox_index += 1
                skip = tag in {word_map[value] for value in ("nl", "ucel", "xcel")}
                previous_ucel = tag == word_map["ucel"]
                decoded = torch.cat((decoded, torch.tensor([[tag]], dtype=torch.long)))
            classes, cxcywh = model._bbox_decoder.inference(image_features, states)
            merged_classes, merged_boxes, skipped = [], [], set()
            for index in range(len(cxcywh)):
                if index in merges:
                    merged_boxes.append(model.mergebboxes(cxcywh[index], cxcywh[merges[index]]))
                    merged_classes.append(classes[index])
                    skipped.add(merges[index])
                elif index not in skipped:
                    merged_boxes.append(cxcywh[index])
                    merged_classes.append(classes[index])
            classes = torch.stack(merged_classes) if merged_classes else torch.empty((0, 3))
            cxcywh = torch.stack(merged_boxes) if merged_boxes else torch.empty((0, 4))
    finally:
        for handle in handles:
            handle.remove()
    xyxy = torch.cat((cxcywh[:, :2] - cxcywh[:, 2:] / 2, cxcywh[:, :2] + cxcywh[:, 2:] / 2), dim=1)
    captured.update(
        {
            "generated_ids_i64": decoded.transpose(0, 1).cpu().numpy().astype(np.int64),
            "greedy_step_logits_f32": torch.stack(logits, dim=1).cpu().numpy().astype(np.float32),
            "bbox_class_logits_f32": classes.cpu().numpy().astype(np.float32),
            "bbox_cxcywh_f32": cxcywh.cpu().numpy().astype(np.float32),
            "bbox_xyxy_f32": xyxy.cpu().numpy().astype(np.float32),
        }
    )
    return captured


def _stage_page(image: Image.Image) -> Any:
    from typing import cast

    from docling.datamodel.base_models import Cluster, LayoutPrediction, Page
    from docling_core.types.doc import BoundingBox, DocItemLabel, Size

    class Backend:
        def is_valid(self) -> bool:
            return True

        def get_page_image(self, scale: float, cropbox: Any = None) -> Image.Image:
            if scale != 2.0 or cropbox is not None:
                raise AssertionError("release stage requires one full 2x page render")
            return image

        def get_segmented_page(self) -> None:
            return None

        def get_text_in_rect(self, bbox: Any) -> str:
            del bbox
            return ""

    width, height = image.width / 2, image.height / 2
    page = Page(page_no=0, size=Size(width=width, height=height))
    page.predictions.layout = LayoutPrediction(
        clusters=[
            Cluster(
                id=0,
                label=DocItemLabel.TABLE,
                bbox=BoundingBox(l=0, t=0, r=width, b=height),
                cells=[],
            )
        ]
    )
    page._backend = cast(Any, Backend())
    return page


def structured_table(table: Any) -> dict[str, Any]:
    """Keep the comparable stage contract compact; raw boxes stay in the NPZ oracle."""

    cells = []
    boxes = []
    for cell in table.table_cells:
        cells.append(
            {
                name: getattr(cell, name)
                for name in (
                    "row_span",
                    "col_span",
                    "start_row_offset_idx",
                    "end_row_offset_idx",
                    "start_col_offset_idx",
                    "end_col_offset_idx",
                    "column_header",
                    "row_header",
                    "row_section",
                )
            }
        )
        if cell.bbox is None:
            raise RuntimeError("TableFormer v1 stage emitted a cell without a bbox")
        boxes.append(list(cell.bbox.as_tuple()))
    return {
        "otsl": list(table.otsl_seq),
        "num_rows": table.num_rows,
        "num_cols": table.num_cols,
        "cells": cells,
        "bboxes_xyxy": boxes,
    }


def reference_stage(
    source: Path,
    image: Image.Image,
    cpu_threads: int,
    mode: TableFormerMode = TableFormerMode.ACCURATE,
) -> dict[str, Any]:
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.document import ConversionResult
    from docling.datamodel.pipeline_options import TableStructureOptions
    from docling.models.stages.table_structure.table_structure_model import TableStructureModel

    stage = TableStructureModel(
        True,
        source,
        TableStructureOptions(mode=mode, do_cell_matching=False),
        AcceleratorOptions(device="cpu", num_threads=cpu_threads),
    )
    result = stage.predict_tables(
        ConversionResult.model_construct(timings={}), [_stage_page(image)]
    )[0]
    return structured_table(result.table_map[0])


def capture(
    source: Path,
    images: list[Path],
    output: Path,
    cpu_threads: int = 1,
    mode: TableFormerMode = TableFormerMode.ACCURATE,
) -> dict[str, Any]:
    if not images or len({path.resolve() for path in images}) != len(images):
        raise ValueError("capture needs one or more unique images")
    source = source.resolve()
    verify_source(source)
    versions = _versions()
    import torch

    torch_settings = configure_torch(torch, cpu_threads)
    model, config = load_reference(source, torch, mode)
    specs = array_specs(config["model"]["enc_layers"])
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Reference output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.capture-", dir=output.parent))
    try:
        captures = []
        for index, path in enumerate(images):
            with Image.open(path) as image:
                input_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            arrays = {
                "input_rgb_u8": input_rgb,
                "pixels_chw_f32": preprocess_exact(input_rgb, config),
            }
            arrays.update(reference_trace(model, config, arrays["pixels_chw_f32"], torch))
            arrays = {spec.name: np.ascontiguousarray(arrays[spec.name]) for spec in specs}
            archive = temporary / f"{index:03d}-{path.stem}.npz"
            write_compressed_npz(archive, arrays, specs)
            ids = arrays["generated_ids_i64"].reshape(-1).tolist()
            captures.append(
                {
                    "name": path.stem,
                    "source_file_sha256": sha256(path),
                    "otsl": otsl_from_ids(ids),
                    "structured_stage": reference_stage(
                        source, Image.fromarray(input_rgb), cpu_threads, mode
                    ),
                    "archive": {"file": archive.name, "sha256": sha256(archive)},
                    "arrays": [
                        {
                            "name": spec.name,
                            "dtype": spec.dtype,
                            "layout": spec.layout,
                            "shape": list(arrays[spec.name].shape),
                            "sha256": array_sha256(arrays[spec.name]),
                        }
                        for spec in specs
                    ],
                }
            )
        metadata = {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "producer": "tools.tableformer_v1.capture_reference",
            "profile": profile_id(mode),
            "source": {
                "revision": SOURCE_REVISION,
            },
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "dependencies": versions,
                "torch": torch_settings,
                "environment": {
                    key: os.environ[key] for key in sorted(os.environ) if key.startswith("OMP_")
                },
            },
            "captures": captures,
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.rename(temporary, output)
        return metadata
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument(
        "--profile",
        choices=(TableFormerMode.ACCURATE.value, TableFormerMode.FAST.value),
        default=TableFormerMode.ACCURATE.value,
    )
    args = parser.parse_args()
    metadata = capture(
        args.source,
        args.images,
        args.output,
        args.cpu_threads,
        TableFormerMode(args.profile),
    )
    print(
        json.dumps(
            {"output": str(args.output), "captures": len(metadata["captures"])},
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
