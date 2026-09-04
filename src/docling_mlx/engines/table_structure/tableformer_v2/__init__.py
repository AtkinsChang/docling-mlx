# SPDX-License-Identifier: Apache-2.0

"""Framework-free TableFormerV2 inference contracts."""

from .engine import (
    TableFormerV2Engine,
    TableFormerV2EngineOptions,
    TableFormerV2Prediction,
)
from .model_spec import TableFormerV2ModelSpec

__all__ = [
    "TableFormerV2Engine",
    "TableFormerV2EngineOptions",
    "TableFormerV2ModelSpec",
    "TableFormerV2Prediction",
]
