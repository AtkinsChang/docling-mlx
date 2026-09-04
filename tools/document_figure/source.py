# SPDX-License-Identifier: Apache-2.0

"""Pinned upstream DocumentFigure source metadata."""

from pathlib import Path

SOURCE_REPO = "docling-project/DocumentFigureClassifier-v2.5"
SOURCE_REVISION = "f859dfbff5c9916cd996942d4b0db7fa25808220"
SOURCE_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")


def verify_source(source: Path) -> None:
    """Require the files consumed by the converter."""
    for name in SOURCE_FILES:
        path = source / name
        if not path.is_file():
            raise ValueError(f"Missing source file: {path}")
