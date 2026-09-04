# SPDX-License-Identifier: Apache-2.0

"""Portable boundaries between generic engines and Docling adaptors."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from docling_mlx.engines.image_classification.efficientnet import (
    EfficientNetEngine,
    EfficientNetModelSpec,
)
from docling_mlx.engines.object_detection.dfine import DFineEngine, DFineModelSpec
from docling_mlx.engines.object_detection.rt_detr_v2 import RtDetrV2Engine, RtDetrV2ModelSpec
from docling_mlx.engines.table_structure.tableformer_v1 import (
    TableFormerV1Engine,
    TableFormerV1ModelSpec,
)
from docling_mlx.engines.table_structure.tableformer_v2 import (
    TableFormerV2Engine,
    TableFormerV2ModelSpec,
)

_COLD_IMPORT_CASES = (
    pytest.param("docling_mlx.engines.object_detection.rt_detr_v2", id="rt-detr-v2"),
    pytest.param("docling_mlx.engines.object_detection.dfine", id="dfine"),
    pytest.param("docling_mlx.engines.image_classification.efficientnet", id="efficientnet"),
    pytest.param("docling_mlx.engines.table_structure.tableformer_v1", id="tableformer-v1"),
    pytest.param("docling_mlx.engines.table_structure.tableformer_v2", id="tableformer-v2"),
)
ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("module", _COLD_IMPORT_CASES)
def test_generic_engine_imports_do_not_load_docling(module: str) -> None:
    script = f"""
import importlib.abc
import sys

class BlockDocling(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'docling' or fullname.startswith('docling.'):
            raise RuntimeError(fullname)
        return None

sys.meta_path.insert(0, BlockDocling())
__import__({module!r})
assert not any(name == 'docling' or name.startswith('docling.') for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], text=True, capture_output=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


_MODEL_SPEC_TYPES = (
    RtDetrV2ModelSpec,
    DFineModelSpec,
    EfficientNetModelSpec,
    TableFormerV1ModelSpec,
    TableFormerV2ModelSpec,
)


@pytest.mark.parametrize("spec_type", _MODEL_SPEC_TYPES)
def test_generic_model_specs_require_exactly_one_checkpoint_source(spec_type: type) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        spec_type()
    with pytest.raises(ValueError, match="exactly one"):
        spec_type(repo_id="example/model", path=Path("/checkpoint"))
    assert spec_type(path=Path("/checkpoint")).path == Path("/checkpoint")


_EMPTY_PREDICTORS = (
    pytest.param(
        RtDetrV2Engine(RtDetrV2ModelSpec(path="/checkpoint")),
        lambda engine: engine.predict([]),
        id="rt-detr-v2",
    ),
    pytest.param(
        DFineEngine(DFineModelSpec(path="/checkpoint")),
        lambda engine: engine.predict([]),
        id="dfine",
    ),
    pytest.param(
        EfficientNetEngine(EfficientNetModelSpec(path="/checkpoint")),
        lambda engine: engine.predict([]),
        id="efficientnet",
    ),
    pytest.param(
        TableFormerV1Engine(TableFormerV1ModelSpec(path="/checkpoint")),
        lambda engine: engine.predict([]),
        id="tableformer-v1",
    ),
    pytest.param(
        TableFormerV2Engine(TableFormerV2ModelSpec(path="/checkpoint")),
        lambda engine: engine.predict([]),
        id="tableformer-v2",
    ),
)


@pytest.mark.parametrize(("engine", "predict"), _EMPTY_PREDICTORS)
def test_empty_generic_prediction_is_lazy(engine: object, predict: object) -> None:
    with patch.object(engine, "initialize", side_effect=AssertionError("empty input initialized")):
        assert predict(engine) == []  # type: ignore[operator]


_PLUGIN_CALLBACK_CASES = (
    pytest.param(
        "layout_engines",
        ("MlxLayoutObjectDetectionModel",),
        ("mlx", "torch", "transformers"),
        id="layout",
    ),
    pytest.param(
        "table_structure_engines",
        (
            "MlxGraniteVisionTableStructureModel",
            "MlxTableFormerV2Model",
            "MlxTableFormerV1Model",
        ),
        ("cv2", "mlx", "mlx_vlm", "torch", "torchvision", "transformers", "docling_ibm_models"),
        id="table",
    ),
)


@pytest.mark.parametrize(("callback", "stages", "blocked"), _PLUGIN_CALLBACK_CASES)
def test_plugin_callbacks_are_lazy_and_register_ordered_stages(
    callback: str, stages: tuple[str, ...], blocked: tuple[str, ...]
) -> None:
    script = f"""
import importlib.abc
import sys
import docling_mlx.plugins

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {set(blocked)!r}:
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, Block())
registered = getattr(docling_mlx.plugins, {callback!r})()
assert list(registered) == [{callback!r}]
assert [stage.__name__ for stage in registered[{callback!r}]] == {list(stages)!r}
for prefix in {blocked!r}:
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
