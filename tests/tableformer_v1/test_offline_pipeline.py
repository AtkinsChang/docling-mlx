# SPDX-License-Identifier: Apache-2.0

"""Offline Docling factory smoke for TableFormerV1 profile selection."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.parity


def test_offline_factory_constructs_disabled_v1_without_artifact_or_runtime_access() -> None:
    import torch  # noqa: F401

    script = """
import json
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import TableFormerMode
from docling.models.factories.table_factory import TableStructureFactory
from docling_mlx.engines.table_structure.tableformer_v1 import TableFormerV1ModelSpec
from docling_mlx.stages.table_structure_v1 import MlxTableStructureOptions

factory = TableStructureFactory()
factory.load_from_plugins(allow_external_plugins=True)
models = []
for mode in (TableFormerMode.ACCURATE, TableFormerMode.FAST):
    model = factory.create_instance(
        MlxTableStructureOptions(
            model_spec=TableFormerV1ModelSpec(
                repo_id='offline/tableformer-v1',
                revision='local',
            ),
            mode=mode,
        ),
        enabled=False,
        artifacts_path=Path('/does-not-exist'),
        accelerator_options=AcceleratorOptions(device='auto'),
        enable_remote_services=False,
    )
    models.append({
        'model': type(model).__name__,
        'options': type(model.options).__name__,
        'mode': model.options.mode.value,
        'enabled': model.enabled,
        'engine': model.engine,
    })
print('TFV1_OFFLINE=' + json.dumps(models, separators=(',', ':')))
"""
    environment = os.environ | {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[2],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    marker = next(
        line.removeprefix("TFV1_OFFLINE=")
        for line in completed.stdout.splitlines()
        if line.startswith("TFV1_OFFLINE=")
    )
    assert marker == (
        '[{"model":"MlxTableFormerV1Model","options":"MlxTableStructureOptions",'
        '"mode":"accurate","enabled":false,"engine":null},'
        '{"model":"MlxTableFormerV1Model","options":"MlxTableStructureOptions",'
        '"mode":"fast","enabled":false,"engine":null}]'
    )
