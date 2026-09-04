# SPDX-License-Identifier: Apache-2.0

"""Capture pinned PyTorch CPU TableFormerV2 full-generation references.

The capture is model scoped and writes only to a caller-selected directory.
Production code must not import this Torch oracle or its generated values.
"""

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from docling_mlx._models.tableformer_v2.config import TABLEFORMER_V2_TOKENS
from tools.pinned_versions import require_locked_versions
from tools.tableformer_v2.source import SOURCE_REPO, SOURCE_REVISION, sha256, verify_source

CAPTURE_SCHEMA_VERSION = 1
MAX_GENERATION_STEPS = 512
REFERENCE_PACKAGES = (
    "docling-ibm-models",
    "numpy",
    "pillow",
    "torch",
    "torchvision",
    "transformers",
)


@dataclass(frozen=True, slots=True)
class ArraySpec:
    """Stable name, dtype, and layout for one per-fixture capture array."""

    name: str
    dtype: str
    layout: str


ARRAY_SPECS = (
    ArraySpec("input_rgb_u8", "uint8", "HWC"),
    ArraySpec("resized_rgb_u8", "uint8", "HWC"),
    ArraySpec("pixels_nhwc_f32", "float32", "BHWC"),
    ArraySpec("encoder_last_hidden_state_f32", "float32", "BSE"),
    ArraySpec("generated_ids_i64", "int64", "BL"),
    ArraySpec("greedy_step_logits_f32", "float32", "B(L-1)V"),
    ArraySpec("final_logits_f32", "float32", "BLV"),
    ArraySpec("normalized_bboxes_f32", "float32", "C4"),
)


def array_sha256(array: np.ndarray) -> str:
    """Hash dtype, shape, and C-order content for one captured array."""

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def write_compressed_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a reproducible compressed NPZ in stable schema order."""

    expected = [spec.name for spec in ARRAY_SPECS]
    if list(arrays) != expected:
        raise ValueError("TableFormerV2 capture arrays do not match the stable schema order")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for spec in ARRAY_SPECS:
            array = np.ascontiguousarray(arrays[spec.name])
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            member = zipfile.ZipInfo(
                filename=f"{spec.name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            archive.writestr(
                member,
                buffer.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def otsl_from_ids(token_ids: list[int] | tuple[int, ...]) -> list[str]:
    """Decode the closed token map into Docling's tag-only OTSL sequence."""

    result: list[str] = []
    for token_id in token_ids:
        if not 0 <= token_id < len(TABLEFORMER_V2_TOKENS):
            raise ValueError(f"Unsupported TableFormerV2 token ID: {token_id}")
        token = TABLEFORMER_V2_TOKENS[token_id]
        if token in {"<pad>", "[UNK]", "<start>", "<end>"}:
            continue
        result.append(token[1:-1])
    return result


def _verify_reference_versions() -> dict[str, str]:
    return require_locked_versions(REFERENCE_PACKAGES, context="TableFormerV2 capture")


def _configure_torch(torch: Any, cpu_threads: int) -> dict[str, Any]:
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        "device": "cpu",
        "eval": True,
        "inference_mode": True,
        "deterministic_algorithms": True,
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
    }


def _verify_tokenizer(tokenizer: Any) -> None:
    actual_tokens = tokenizer.convert_ids_to_tokens(list(range(len(TABLEFORMER_V2_TOKENS))))
    if actual_tokens != list(TABLEFORMER_V2_TOKENS) or len(tokenizer) != len(TABLEFORMER_V2_TOKENS):
        raise ValueError("Pinned tokenizer does not match the closed TableFormerV2 token map")
    if (
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
    ) != (0, 1, 2, 3):
        raise ValueError("Pinned tokenizer special-token IDs do not match the closed profile")


def _torch_generation_trace(
    model: Any,
    encoder_outputs: dict[str, Any],
    tokenizer: Any,
    torch: Any,
) -> tuple[Any, Any]:
    generated_ids = torch.full((1, 1), tokenizer.bos_token_id, dtype=torch.long)
    current_input = generated_ids
    past_key_values = None
    greedy_logits: list[Any] = []
    for _ in range(MAX_GENERATION_STEPS):
        output = model(
            input_ids=current_input,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        step_logits = output.logits[:, -1, :]
        greedy_logits.append(step_logits)
        next_token = step_logits.argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat((generated_ids, next_token), dim=1)
        current_input = next_token
        past_key_values = output.past_key_values
        if torch.all(next_token == tokenizer.eos_token_id):
            break
    return generated_ids, torch.stack(greedy_logits, dim=1)


def _capture_fixture(
    image_path: Path,
    model: Any,
    tokenizer: Any,
    torch: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from tools.tableformer_v2.reference_preprocessing import preprocess_with_torchvision

    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
    input_rgb = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    resized_rgb, pixels_nhwc = preprocess_with_torchvision(image)
    pixels_nhwc = np.ascontiguousarray(pixels_nhwc, dtype=np.float32)
    torch_pixels = torch.from_numpy(pixels_nhwc.transpose(2, 0, 1)[None].copy())
    with torch.inference_mode():
        encoder_outputs = model.encode_images(torch_pixels)
        generated_ids, greedy_logits = _torch_generation_trace(
            model, encoder_outputs, tokenizer, torch
        )
        final_output = model(
            input_ids=generated_ids,
            attention_mask=torch.ones_like(generated_ids),
            encoder_outputs=encoder_outputs,
            past_key_values=None,
            use_cache=False,
            return_dict=True,
        )
    if final_output.logits is None or final_output.predicted_bboxes is None:
        raise RuntimeError("Pinned TableFormerV2 omitted required final outputs")
    expected_next_ids = final_output.logits[:, :-1].argmax(dim=-1)
    if not torch.equal(expected_next_ids, generated_ids[:, 1:]):
        raise RuntimeError("Pinned full-sequence logits disagree with cached greedy generation")

    arrays = {
        "input_rgb_u8": input_rgb,
        "resized_rgb_u8": np.ascontiguousarray(resized_rgb, dtype=np.uint8),
        "pixels_nhwc_f32": pixels_nhwc[None],
        "encoder_last_hidden_state_f32": np.ascontiguousarray(
            encoder_outputs["last_hidden_state"].cpu().numpy(), dtype=np.float32
        ),
        "generated_ids_i64": np.ascontiguousarray(generated_ids.cpu().numpy(), dtype=np.int64),
        "greedy_step_logits_f32": np.ascontiguousarray(
            greedy_logits.cpu().numpy(), dtype=np.float32
        ),
        "final_logits_f32": np.ascontiguousarray(
            final_output.logits.cpu().numpy(), dtype=np.float32
        ),
        "normalized_bboxes_f32": np.ascontiguousarray(
            final_output.predicted_bboxes.cpu().numpy(), dtype=np.float32
        ),
    }
    for spec in ARRAY_SPECS:
        array = arrays[spec.name]
        if str(array.dtype) != spec.dtype:
            raise RuntimeError(f"Reference dtype mismatch for {spec.name}: {array.dtype}")
        if not np.isfinite(array).all():
            raise RuntimeError(f"Reference output is non-finite: {spec.name}")
    token_ids = arrays["generated_ids_i64"][0].tolist()
    details = {
        "source_path": str(image_path),
        "source_file_sha256": sha256(image_path),
        "input_rgb_sha256": array_sha256(input_rgb),
        "native_size": list(image.size),
        "encoder_spatial_size": list(encoder_outputs["spatial_size"]),
        "generated_token_count": len(token_ids),
        "otsl": otsl_from_ids(token_ids),
    }
    return arrays, details


def capture(
    *,
    source: Path,
    images: list[Path],
    output_directory: Path,
    cpu_threads: int = 1,
) -> dict[str, Any]:
    """Capture one or more immutable fixtures into a new local directory."""

    if cpu_threads < 1:
        raise ValueError("cpu_threads must be at least 1")
    if not images:
        raise ValueError("At least one TableFormerV2 capture image is required")
    resolved_images = [path.expanduser().resolve() for path in images]
    if len(resolved_images) != len(set(resolved_images)):
        raise ValueError("TableFormerV2 capture images must be unique")
    for path in resolved_images:
        if not path.is_file():
            raise FileNotFoundError(f"Missing capture image: {path}")

    source = source.expanduser().resolve()
    verify_source(source)
    versions = _verify_reference_versions()
    import torch
    from docling_ibm_models.tableformer_v2.model import TableFormerV2
    from transformers import AutoTokenizer

    torch_settings = _configure_torch(torch, cpu_threads)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    _verify_tokenizer(tokenizer)
    model: Any = TableFormerV2.from_pretrained(
        source,
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.float32,
    )
    model = model.to("cpu")
    model.eval()

    output_directory = output_directory.expanduser().resolve()
    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError(f"Reference output already exists: {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_directory.name}.capture-", dir=output_directory.parent)
    )
    try:
        captures: list[dict[str, Any]] = []
        for index, image_path in enumerate(resolved_images):
            arrays, details = _capture_fixture(image_path, model, tokenizer, torch)
            archive_name = f"{index:03d}-{image_path.stem}.npz"
            archive = temporary / archive_name
            write_compressed_npz(archive, arrays)
            captures.append(
                {
                    "name": image_path.stem,
                    **details,
                    "arrays": [
                        {
                            "name": spec.name,
                            "dtype": spec.dtype,
                            "layout": spec.layout,
                            "shape": list(arrays[spec.name].shape),
                            "sha256": array_sha256(arrays[spec.name]),
                        }
                        for spec in ARRAY_SPECS
                    ],
                    "archive": {
                        "file": archive_name,
                        "sha256": sha256(archive),
                    },
                }
            )
        metadata = {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "producer": "tools.tableformer_v2.capture_reference",
            "profile": "tableformer_v2",
            "source": {
                "repo_id": SOURCE_REPO,
                "revision": SOURCE_REVISION,
            },
            "token_map": list(TABLEFORMER_V2_TOKENS),
            "generation": {"max_generation_steps": MAX_GENERATION_STEPS},
            "archive": {
                "format": "NPZ",
                "compression": "deflate",
                "normalized_zip_timestamp": True,
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
        os.rename(temporary, output_directory)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Explicit pinned Hugging Face snapshot"
    )
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New local reference capture directory"
    )
    parser.add_argument("--cpu-threads", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = capture(
        source=args.source,
        images=args.images,
        output_directory=args.output,
        cpu_threads=args.cpu_threads,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "captures": len(metadata["captures"]),
                "source_revision": metadata["source"]["revision"],
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
