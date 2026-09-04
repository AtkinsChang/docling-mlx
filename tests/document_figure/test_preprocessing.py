# SPDX-License-Identifier: Apache-2.0

"""Portable validation for DocumentFigure preprocessing contracts."""

import ast
from pathlib import Path
from typing import Any

import pytest

from docling_mlx.engines.image_classification.efficientnet.preprocessing import (
    parse_preprocessing_config,
)

# preprocessor_config.json from DocumentFigureClassifier-v2.5 at
# f859dfbff5c9916cd996942d4b0db7fa25808220.
CONFIG: dict[str, Any] = {
    "do_convert_rgb": None,
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.485, 0.456, 0.406],
    "image_processor_type": "ViTImageProcessor",
    "image_std": [0.47853944, 0.4732864, 0.47434163],
    "resample": 2,
    "rescale_factor": 0.00392156862745098,
    "size": {"height": 224, "width": 224},
}


def test_dependency_specific_preprocessing_lanes_are_safe_to_collect() -> None:
    """Marker filtering imports every test module, including deselected lanes."""

    forbidden = {"mlx", "torch", "torchvision", "transformers"}
    for path in Path(__file__).parent.glob("test_preprocessing_*.py"):
        tree = ast.parse(path.read_text())
        top_level_imports = [
            alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names
        ]
        top_level_imports.extend(
            node.module or "" for node in tree.body if isinstance(node, ast.ImportFrom)
        )
        assert not any(name.split(".", maxsplit=1)[0] in forbidden for name in top_level_imports), (
            path.name
        )


def _unsupported_processor_contract() -> None:
    parse_preprocessing_config(CONFIG | {"resample": 3})


def _invalid_normalization() -> None:
    parse_preprocessing_config(CONFIG | {"image_std": [1, 0, 1]})


@pytest.mark.parametrize(
    ("equivalence_class", "operation"),
    [
        pytest.param("invalid_normalization", _invalid_normalization, id="normalization"),
    ],
)
def test_preprocessing_rejects_invalid_equivalence_classes(
    equivalence_class: str, operation: Any
) -> None:
    del equivalence_class
    with pytest.raises((TypeError, ValueError)):
        operation()


def test_non_runtime_processor_metadata_is_ignored() -> None:
    _unsupported_processor_contract()
