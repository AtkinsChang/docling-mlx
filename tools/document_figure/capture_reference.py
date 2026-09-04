# SPDX-License-Identifier: Apache-2.0

"""Capture an independent PyTorch oracle for the document figure classifier.

This module is deliberately a reference-only tool.  Production MLX code must
not import it or derive expected values from it.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import sys
import zipfile
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pypdfium2
import pypdfium2.raw as pdfium_raw
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from tools.document_figure.source import SOURCE_FILES, SOURCE_REPO, SOURCE_REVISION

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = REPOSITORY_ROOT / "tests/fixtures/document_figure/picture_classification.pdf"
DEFAULT_IMAGES = REPOSITORY_ROOT / "tests/fixtures/document_figure/reference_images"

SOURCE_PDF_SHA256 = "dc38947ee802b7bcf82915804ccb8d04c4611e1b17491c415972821a329352fb"
SOURCE_IMAGES = (
    ("bar_chart", 0, 0),
    ("geographical_map", 1, 0),
)
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class ExtractedImage:
    """One native-resolution image XObject extracted from the source PDF."""

    name: str
    page_index: int
    object_index: int
    bounds: tuple[float, float, float, float]
    image: Image.Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_relative_path(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()


def _fixture_metadata(
    extracted: ExtractedImage, image_path: Path, *, path_is_repository_relative: bool
) -> dict[str, Any]:
    return {
        "name": extracted.name,
        "path": (
            repository_relative_path(image_path) if path_is_repository_relative else str(image_path)
        ),
        "sha256": sha256_file(image_path),
        "page_index": extracted.page_index,
        "object_index": extracted.object_index,
        "bounds_pdf_points": extracted.bounds,
        "native_size": list(extracted.image.size),
        "method": "pypdfium2 image XObject render=True scale_to_original=True RGB",
    }


def write_compressed_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write a byte-reproducible compressed NPZ archive.

    ``numpy.savez_compressed`` stores the current ZIP timestamp, making the
    same oracle arrays produce a different file hash on every capture.
    """

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asanyarray(array), allow_pickle=False)
            member = zipfile.ZipInfo(filename=f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            archive.writestr(
                member,
                buffer.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def extract_source_images(pdf_path: Path) -> list[ExtractedImage]:
    """Extract the two figure XObjects without rasterizing or cropping a page.

    ``render=True`` applies an XObject's own transform and alpha mask, while
    ``scale_to_original=True`` retains its native pixel resolution.  This is
    intentionally different from a rendered-page crop.
    """

    if sha256_file(pdf_path) != SOURCE_PDF_SHA256:
        raise ValueError(f"Unexpected source PDF digest: {pdf_path}")

    extracted: list[ExtractedImage] = []
    with pypdfium2.PdfDocument(pdf_path) as document:
        for name, page_index, object_index in SOURCE_IMAGES:
            page = document[page_index]
            images = list(page.get_objects(filter=[pdfium_raw.FPDF_PAGEOBJ_IMAGE]))
            try:
                image_object = images[object_index]
            except IndexError as error:
                raise ValueError(
                    f"Expected image object {object_index} on source PDF page {page_index}"
                ) from error
            bitmap = image_object.get_bitmap(render=True, scale_to_original=True)
            image = bitmap.to_pil().convert("RGB")
            bounds = tuple(float(value) for value in image_object.get_bounds())
            if len(bounds) != 4:
                raise ValueError(f"Expected four image bounds, got {bounds}")
            extracted.append(
                ExtractedImage(
                    name=name,
                    page_index=page_index,
                    object_index=object_index,
                    bounds=(bounds[0], bounds[1], bounds[2], bounds[3]),
                    image=image,
                )
            )
    return extracted


def source_images(
    pdf_path: Path,
    image_directory: Path,
    mode: Literal["write", "validate", "committed"],
) -> list[dict[str, Any]]:
    """Load source images, optionally writing or validating extracted fixtures."""

    if mode == "committed":
        details = []
        for name, _, _ in SOURCE_IMAGES:
            image_path = image_directory / f"{name}.png"
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing committed fixture: {image_path}")
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            details.append(
                {
                    "name": name,
                    "path": repository_relative_path(image_path)
                    if image_path.resolve().is_relative_to(REPOSITORY_ROOT)
                    else str(image_path),
                    "sha256": sha256_file(image_path),
                    "native_size": list(image.size),
                }
            )
        return details

    if mode not in ("write", "validate"):
        raise ValueError(f"Unknown source image mode: {mode}")
    if mode == "write":
        image_directory.mkdir(parents=True, exist_ok=True)
    details = []
    for extracted in extract_source_images(pdf_path):
        image_path = image_directory / f"{extracted.name}.png"
        if mode == "write":
            extracted.image.save(image_path, format="PNG", optimize=True)
        else:
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Missing extracted fixture {image_path}; rerun with --write-fixtures"
                )
            with Image.open(image_path) as fixture:
                fixture_rgb = fixture.convert("RGB")
            if not np.array_equal(np.asarray(fixture_rgb), np.asarray(extracted.image)):
                raise ValueError(
                    f"Fixture {image_path} does not match source PDF image XObject {extracted.name}"
                )
        details.append(
            _fixture_metadata(
                extracted,
                image_path,
                path_is_repository_relative=image_path.resolve().is_relative_to(REPOSITORY_ROOT),
            )
        )
    return details


def _resolve_model_source(source: Path | None) -> Path:
    if source is not None:
        resolved = source.expanduser().resolve()
    else:
        resolved = Path(
            snapshot_download(
                repo_id=SOURCE_REPO,
                revision=SOURCE_REVISION,
                allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"],
            )
        )
    if not resolved.is_dir():
        raise ValueError(f"Model source is not a directory: {resolved}")
    for filename in SOURCE_FILES:
        file_path = resolved / filename
        if not file_path.is_file():
            raise ValueError(f"Missing source file: {file_path}")
    return resolved


def configure_torch(cpu_threads: int) -> dict[str, Any]:
    """Make CPU reference execution explicit and repeatable."""

    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # It can only be set before parallel work; record the effective value below.
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


def _processor_pixels(processor: Any, image: Image.Image) -> np.ndarray:
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]
    return pixel_values.squeeze(0).permute(1, 2, 0).numpy().astype(np.float32, copy=False)


def capture(
    output_directory: Path,
    pdf_path: Path,
    source: Path | None,
    cpu_threads: int,
    write_fixtures: bool,
    release_fixture_check: bool = False,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    torch_settings = configure_torch(cpu_threads)
    fixture_mode: Literal["write", "validate", "committed"] = (
        "write" if write_fixtures else "validate" if release_fixture_check else "committed"
    )
    source_fixture_details = source_images(pdf_path, DEFAULT_IMAGES, fixture_mode)
    model_source = _resolve_model_source(source)
    processor = AutoImageProcessor.from_pretrained(model_source)
    if processor.backend != "torchvision":
        raise RuntimeError(
            f"Expected AutoImageProcessor backend 'torchvision', got {processor.backend!r}"
        )
    model = AutoModelForImageClassification.from_pretrained(model_source).to("cpu").eval()
    labels = {int(index): label for index, label in model.config.id2label.items()}

    captures: list[dict[str, Any]] = []
    for source_image in source_fixture_details:
        name = source_image["name"]
        image_path = Path(source_image["path"])
        if not image_path.is_absolute():
            image_path = REPOSITORY_ROOT / image_path
        with Image.open(image_path) as opened_image:
            rgb_image = opened_image.convert("RGB")
        input_rgb_u8 = np.asarray(rgb_image, dtype=np.uint8)
        processor_pixels = _processor_pixels(processor, rgb_image)
        torch_pixels = torch.from_numpy(processor_pixels).permute(2, 0, 1).unsqueeze(0)
        with torch.inference_mode():
            logits_tensor = model(pixel_values=torch_pixels).logits.squeeze(0)
            probabilities_tensor = torch.softmax(logits_tensor, dim=0)
        logits = logits_tensor.cpu().numpy().astype(np.float32)
        probabilities = probabilities_tensor.cpu().numpy().astype(np.float32)
        top_label_id = int(np.argmax(probabilities))
        npz_path = output_directory / f"{name}.npz"
        write_compressed_npz(
            npz_path,
            input_rgb_u8=input_rgb_u8,
            processor_pixel_values_nhwc_f32=processor_pixels,
            logits_f32=logits,
            probabilities_f32=probabilities,
        )
        captures.append(
            {
                **source_image,
                "oracle_file": npz_path.name,
                "oracle_sha256": sha256_file(npz_path),
                "top_label": labels[top_label_id],
            }
        )

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "producer": "tools.document_figure.capture_reference",
        "model": {
            "repository": SOURCE_REPO,
            "revision": SOURCE_REVISION,
            "processor_backend": processor.backend,
            "id2label": labels,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "dependencies": {
                package: version(package)
                for package in (
                    "transformers",
                    "torch",
                    "torchvision",
                    "numpy",
                )
            },
            "torch": torch_settings,
            "environment": {
                key: os.environ[key] for key in sorted(os.environ) if key.startswith("OMP_")
            },
        },
        "arrays": {
            "input_rgb_u8": "H,W,3 uint8 extracted RGB image",
            "processor_pixel_values_nhwc_f32": "H,W,3 float32 AutoImageProcessor output",
            "logits_f32": "C float32 PyTorch CPU model logits",
            "probabilities_f32": "C float32 softmax(logits)",
        },
        "archive": {"format": "NPZ", "compression": "deflate", "normalized_zip_timestamp": True},
        "captures": captures,
    }
    metadata_path = output_directory / "metadata.json"
    metadata_path.write_text(json.dumps(_jsonable(metadata), indent=2, sort_keys=True) + "\n")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="Directory for NPZ oracle artifacts"
    )
    parser.add_argument("--source", type=Path, help="Optional local Hugging Face model snapshot")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="Vendored source PDF")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument(
        "--write-fixtures",
        action="store_true",
        help="Release-only: re-extract the committed PNG fixtures from the vendored source PDF",
    )
    parser.add_argument(
        "--release-fixture-check",
        action="store_true",
        help="Release-only: validate committed PNGs against the vendored source PDF",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be at least 1")
    if args.write_fixtures and not args.release_fixture_check:
        raise ValueError("--write-fixtures requires --release-fixture-check")
    capture(
        output_directory=args.output,
        pdf_path=args.pdf,
        source=args.source,
        cpu_threads=args.cpu_threads,
        write_fixtures=args.write_fixtures,
        release_fixture_check=args.release_fixture_check,
    )


if __name__ == "__main__":
    main()
