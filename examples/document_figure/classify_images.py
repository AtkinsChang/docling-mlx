# SPDX-License-Identifier: Apache-2.0

"""Classify images with the generic MLX EfficientNet engine."""

import argparse
from pathlib import Path

from PIL import Image

from docling_mlx.engines.image_classification.efficientnet import (
    EfficientNetEngine,
    EfficientNetEngineOptions,
    EfficientNetModelSpec,
)
from docling_mlx.presets import resolve_preset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        "--artifact",
        dest="checkpoint",
        type=Path,
        help="local EfficientNet checkpoint directory (default: pinned preset)",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("images", type=Path, nargs="+")
    args = parser.parse_args()
    if args.checkpoint is None:
        preset = resolve_preset("document_figure_classifier_v2")
        model_spec = EfficientNetModelSpec(repo_id=preset.repo_id, revision=preset.revision)
    else:
        model_spec = EfficientNetModelSpec(path=args.checkpoint)
    engine = EfficientNetEngine(model_spec, EfficientNetEngineOptions(top_k=args.top_k))
    for path in args.images:
        with Image.open(path) as image:
            prediction = engine.predict([image.convert("RGB")])[0]
        print(path)
        for label_id, probability in zip(
            prediction.label_ids, prediction.probabilities, strict=True
        ):
            print(f"  {prediction.id2label[label_id]}: {probability:.8f}")


if __name__ == "__main__":
    main()
