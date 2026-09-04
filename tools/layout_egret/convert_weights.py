# SPDX-License-Identifier: Apache-2.0

"""Convert an offline D-FINE checkpoint into an MLX artifact.

The source must be an already-downloaded, pinned snapshot. Conversion never
downloads or publishes weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, cast

from docling_mlx._models.dfine.weights import rename_key, sanitize_state_dict
from tools._common.atomic import _atomic_output
from tools._common.model_card import lookup_model_license, render_model_card

convert_state_dict = sanitize_state_dict

__all__ = ["convert", "convert_state_dict", "rename_key"]


_SOURCE_FILES = ("config.json", "model.safetensors", "preprocessor_config.json")


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one source or generated artifact file."""

    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_source(source: Path) -> None:
    """Require the files consumed by the converter."""

    for name in _SOURCE_FILES:
        path = source / name
        if not path.is_file():
            raise ValueError(f"Missing source file: {path}")


def _target_model(config_raw: dict[str, object]) -> tuple[Any, dict[str, tuple[int, ...]]]:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from docling_mlx._models.dfine.config import DFineConfig
    from docling_mlx._models.dfine.model import DFine

    config = DFineConfig.from_dict(config_raw)
    model = DFine(config)
    flattened = cast(list[tuple[str, Any]], tree_flatten(model.parameters()))
    if not flattened or any(value.dtype != mx.float32 for _, value in flattened):
        raise ValueError("Native D-FINE target parameters must be nonempty float32 tensors")
    return model, {key: tuple(value.shape) for key, value in flattened}


def convert(source: Path, output: Path, repo_id: str, revision: str) -> None:
    """Convert an offline frozen source snapshot into a new artifact directory."""

    verify_source(source)
    license = lookup_model_license(source, repo_id, revision)
    config_bytes = (source / "config.json").read_bytes()
    config_raw = json.loads(config_bytes)
    if not isinstance(config_raw, dict):
        raise ValueError("Expected config.json to contain a JSON object")

    from safetensors.numpy import load_file, save_file

    model, target_shapes = _target_model(config_raw)
    source_state = load_file(str(source / "model.safetensors"))
    converted, _, _ = convert_state_dict(source_state, target_shapes)

    import mlx.core as mx

    model.load_weights([(key, mx.array(value)) for key, value in converted.items()], strict=True)
    mx.eval(model.parameters())

    with _atomic_output(output) as temporary:
        save_file(converted, str(temporary / "model.safetensors"), metadata={"format": "mlx"})
        for name in ("config.json", "preprocessor_config.json"):
            shutil.copyfile(source / name, temporary / name)
        (temporary / "README.md").write_text(
            render_model_card(repo_id, revision, license), encoding="utf-8"
        )
        if (temporary / "config.json").read_bytes() != config_bytes:
            raise ValueError("Converted config.json did not preserve source bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    convert(args.source, args.output, args.repo_id, args.revision)
    print(
        json.dumps(
            {"output": str(args.output), "repo_id": args.repo_id, "revision": args.revision},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
