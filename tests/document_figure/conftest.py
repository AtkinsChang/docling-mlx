# SPDX-License-Identifier: Apache-2.0

"""One verified offline source snapshot for all live numerical oracles."""

import os
from pathlib import Path

import pytest

from tools.document_figure.source import (
    SOURCE_FILES,
    SOURCE_REPO,
    SOURCE_REVISION,
    verify_source,
)


@pytest.fixture(scope="session")
def pinned_source() -> Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import IncompleteSnapshotError, LocalEntryNotFoundError

    source = os.environ.get("DOCLING_MLX_SOURCE")
    if source is None:
        try:
            source = snapshot_download(
                SOURCE_REPO,
                revision=SOURCE_REVISION,
                local_files_only=True,
                allow_patterns=list(SOURCE_FILES),
            )
        except IncompleteSnapshotError as error:
            pytest.fail(f"Cached reference is missing required model files: {error}")
        except LocalEntryNotFoundError:
            pytest.fail("selected parity lane requires the pinned reference snapshot")
    path = Path(source).expanduser()
    verify_source(path)
    return path
