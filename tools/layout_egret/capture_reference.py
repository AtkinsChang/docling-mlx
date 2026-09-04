# SPDX-License-Identifier: Apache-2.0

"""Capture a pinned D-FINE Torch CPU oracle outside Git."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from tools.layout_egret.convert_weights import (
    sha256,
    verify_source,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPOSITORY_ROOT / "tests/fixtures/layout_heron/benchmark_gradient.png"
CAPTURE_SCHEMA_VERSION = 1
CAPTURE_ARCHIVE = "outputs.npz"
REFERENCE_VERSIONS = {
    "numpy": "2.5.2",
    "pillow": "12.3.0",
    "torch": "2.14.0",
    "torchvision": "0.29.0",
    "transformers": "5.8.1",
}


@dataclass(frozen=True, slots=True)
class ArraySpec:
    name: str
    layout: str


ARRAY_SPECS = (
    ArraySpec("pixel_values_nchw_f32", "NCHW"),
    ArraySpec("logits_f32", "BQC"),
    ArraySpec("pred_boxes_cxcywh_f32", "BQ4"),
)


def array_sha256(array: np.ndarray) -> str:
    """Hash an array's dtype, shape, and C-order bytes."""

    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def write_compressed_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write the three arrays with stable order and ZIP timestamps."""

    if list(arrays) != [spec.name for spec in ARRAY_SPECS]:
        raise ValueError("Egret capture arrays do not match the stable schema")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for spec in ARRAY_SPECS:
            value = np.ascontiguousarray(arrays[spec.name])
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, value, allow_pickle=False)
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


def _reference_versions() -> dict[str, str]:
    actual = {name: importlib.metadata.version(name) for name in REFERENCE_VERSIONS}
    if actual != REFERENCE_VERSIONS or sys.version_info[:2] != (3, 13):
        raise RuntimeError(
            "Egret capture requires Python 3.13 and the pinned reference versions: "
            f"expected {REFERENCE_VERSIONS}, got {actual}"
        )
    return actual


def _configure_torch(torch: Any, cpu_threads: int) -> dict[str, Any]:
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be at least 1")
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    return {
        "device": "cpu",
        "eval": True,
        "inference_mode": True,
        "deterministic_algorithms": True,
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
    }


def _as_float32(tensor: Any, name: str) -> np.ndarray:
    array = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise RuntimeError(f"Torch produced non-finite {name}")
    return np.ascontiguousarray(array)


def capture(
    source: Path,
    output: Path,
    *,
    repo_id: str,
    revision: str,
    cpu_threads: int = 1,
) -> dict[str, Any]:
    """Capture one checkpoint into a new caller-selected directory."""

    source = source.expanduser().resolve()
    verify_source(source)
    versions = _reference_versions()

    import torch
    from transformers import DFineForObjectDetection, RTDetrImageProcessor
    from transformers.image_processing_backends import TorchvisionBackend

    torch_settings = _configure_torch(torch, cpu_threads)
    processor = RTDetrImageProcessor.from_pretrained(source, local_files_only=True)
    if (
        type(processor) is not RTDetrImageProcessor
        or TorchvisionBackend not in type(processor).__mro__
    ):
        raise RuntimeError("Egret capture requires RTDetrImageProcessor with TorchvisionBackend")

    with Image.open(FIXTURE_PATH) as image:
        original_size = image.size
        processed = processor(images=image.convert("RGB"), return_tensors="pt")
    if set(processed) != {"pixel_values"}:
        raise RuntimeError(f"Unexpected RTDetrImageProcessor outputs: {sorted(processed)}")
    pixel_values = processed["pixel_values"].to(device="cpu", dtype=torch.float32)

    model: Any = DFineForObjectDetection.from_pretrained(
        source,
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.float32,
    )
    model = model.to("cpu")
    model.eval()
    with torch.inference_mode():
        result = model(pixel_values=pixel_values, return_dict=True)
    arrays = {
        "pixel_values_nchw_f32": _as_float32(pixel_values, "pixel_values"),
        "logits_f32": _as_float32(result.logits, "logits"),
        "pred_boxes_cxcywh_f32": _as_float32(result.pred_boxes, "pred_boxes"),
    }

    output = output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Reference output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.capture-", dir=output.parent))
    try:
        archive = temporary / CAPTURE_ARCHIVE
        write_compressed_npz(archive, arrays)
        metadata = {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "producer": "tools.layout_egret.capture_reference",
            "source": {
                "repo_id": repo_id,
                "revision": revision,
            },
            "fixture": {
                "path": str(FIXTURE_PATH.relative_to(REPOSITORY_ROOT)),
                "sha256": sha256(FIXTURE_PATH),
                "size": list(original_size),
            },
            "processor": {
                "class": type(processor).__name__,
                "backend": "TorchvisionBackend",
            },
            "arrays": [
                {
                    **asdict(spec),
                    "dtype": str(arrays[spec.name].dtype),
                    "shape": list(arrays[spec.name].shape),
                    "sha256": array_sha256(arrays[spec.name]),
                }
                for spec in ARRAY_SPECS
            ],
            "archive": {"file": CAPTURE_ARCHIVE, "sha256": sha256(archive)},
            "runtime": {
                "python": platform.python_version(),
                "dependencies": versions,
                "torch": torch_settings,
                "environment": {
                    key: os.environ[key] for key in sorted(os.environ) if key.startswith("OMP_")
                },
            },
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    metadata = capture(
        args.source,
        args.output,
        repo_id=args.repo_id,
        revision=args.revision,
        cpu_threads=args.cpu_threads,
    )
    print(
        json.dumps(
            {
                "status": "captured",
                "output": str(args.output.expanduser().resolve()),
                "arrays": len(metadata["arrays"]),
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
