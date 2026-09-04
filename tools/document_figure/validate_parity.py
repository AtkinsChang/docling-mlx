# SPDX-License-Identifier: Apache-2.0

"""Compare the MLX artifact to independently captured, pinned CPU reference data."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from docling_mlx.engines.image_classification.efficientnet import (
    EfficientNetEngine,
    EfficientNetEngineOptions,
    EfficientNetModelSpec,
)
from tools.document_figure.source import SOURCE_REVISION


def validate(artifact: Path, reference: Path) -> dict:
    import numpy as np

    metadata = json.loads((reference / "metadata.json").read_text())
    if (
        metadata["model"]["revision"] != SOURCE_REVISION
        or metadata["model"]["processor_backend"] != "torchvision"
        or metadata["runtime"]["torch"]["device"] != "cpu"
    ):
        raise ValueError("Reference does not match the pinned torchvision/CPU oracle")
    if not metadata.get("captures"):
        raise ValueError("Reference must contain at least one capture")
    engine = EfficientNetEngine(
        EfficientNetModelSpec(path=artifact), EfficientNetEngineOptions(top_k=5)
    )
    engine.initialize()
    results = []
    for capture in metadata["captures"]:
        path = reference / capture["oracle_file"]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != capture["oracle_sha256"]:
            raise ValueError(f"Reference checksum mismatch: {path.name}")
        with np.load(path, allow_pickle=False) as expected:
            image = Image.fromarray(expected["input_rgb_u8"])
            prediction = engine.predict([image])[0]
            expected_ids = np.argsort(-expected["probabilities_f32"], kind="stable")[:5].tolist()
            top1_exact = prediction.label_ids[0] == expected_ids[0]
            top5_set_equal = set(prediction.label_ids) == set(expected_ids)
            if not top1_exact or not top5_set_equal:
                raise AssertionError(f"{capture['name']}: top-1 label mismatch")
            results.append(
                {
                    "fixture": capture["name"],
                    "top1_exact": top1_exact,
                    "top5_set_equal": top5_set_equal,
                    "top_label": engine.get_label_mapping()[prediction.label_ids[0]],
                }
            )
    return {"status": "passed", "source_revision": SOURCE_REVISION, "fixtures": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("tests/golden/document_figure"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.artifact, args.reference)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
