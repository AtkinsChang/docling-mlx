# SPDX-License-Identifier: Apache-2.0

"""Generic TableFormerV1 checkpoint identity."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TableFormerV1ModelSpec:
    """An immutable Hub model identity or a plain local checkpoint directory."""

    repo_id: str | None = None
    revision: str | None = None
    path: Path | str | None = None

    def __post_init__(self) -> None:
        if (self.path is None) == (self.repo_id is None):
            raise ValueError("Specify exactly one of model path or repo_id")


__all__ = ["TableFormerV1ModelSpec"]
