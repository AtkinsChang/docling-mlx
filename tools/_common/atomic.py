# SPDX-License-Identifier: Apache-2.0

"""Shared atomic output helpers for repository-only converters."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def _atomic_output(output: Path) -> Iterator[Path]:
    """Yield a private staging directory and atomically publish it on success."""

    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.convert-", dir=output.parent))
        yield temporary
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Output appeared during conversion: {output}")
        os.rename(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
