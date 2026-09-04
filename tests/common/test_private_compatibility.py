# SPDX-License-Identifier: Apache-2.0

"""Default-lane contracts for Docling private seams used by docling-mlx."""

from __future__ import annotations

import pytest
from docling.datamodel.base_models import Page
from docling.models.inference_engines.vlm._utils import resolve_model_artifacts_path

from docling_mlx._compat import docling


def test_docling_private_compatibility_contract() -> None:
    assert docling.resolve_model_artifacts_path is resolve_model_artifacts_path
    assert "_backend" in Page.__private_attributes__

    with pytest.raises(RuntimeError, match="test stage requires an initialized page backend"):
        docling.require_page_backend(Page(page_no=1), "test stage")
