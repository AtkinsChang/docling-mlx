# SPDX-License-Identifier: Apache-2.0

"""D-FINE selection at the unified Docling layout-stage boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from docling.datamodel.accelerator_options import AcceleratorOptions

from docling_mlx.stages.layout import (
    MlxLayoutObjectDetectionModel,
    MlxLayoutObjectDetectionOptions,
    MlxObjectDetectionEngineOptions,
)


def test_egret_checkpoint_selects_dfine_and_initializes_eagerly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "d_fine"}), encoding="utf-8")
    created: list[object] = []

    class FakeDFineEngine:
        def __init__(self, model_spec, options) -> None:
            self.model_spec = model_spec
            self.options = options
            self.directory = None
            self.initialize_calls: list[bool] = []
            created.append(self)

        def initialize(self, warmup: bool = False) -> None:
            self.initialize_calls.append(warmup)
            self.directory = tmp_path

        def get_label_mapping(self) -> dict[int, str]:
            return {0: "text"}

        def predict(self, images):
            raise AssertionError(f"unexpected prediction for {images!r}")

    import docling_mlx.stages.layout as layout_module

    monkeypatch.setattr(
        layout_module,
        "resolve_artifact_checkpoint",
        lambda *_args, **_kwargs: tmp_path,
    )
    monkeypatch.setattr(layout_module, "DFineEngine", FakeDFineEngine)
    monkeypatch.setattr(
        layout_module,
        "RtDetrV2Engine",
        lambda *_args, **_kwargs: pytest.fail("D-FINE selected RT-DETR-v2"),
    )
    model = MlxLayoutObjectDetectionModel(
        artifacts_path=None,
        accelerator_options=AcceleratorOptions(device="mps"),
        options=MlxLayoutObjectDetectionOptions.from_preset(
            "layout_egret_medium",
            engine_options=MlxObjectDetectionEngineOptions(score_threshold=0.47),
        ),
    )

    engine = created[0]
    assert engine.options.score_threshold == pytest.approx(0.47)  # type: ignore[union-attr]
    assert engine.initialize_calls == [False]  # type: ignore[union-attr]
    assert model.engine.artifact_path == tmp_path


def test_unknown_checkpoint_family_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "not_a_detector"}), encoding="utf-8"
    )
    import docling_mlx.stages.layout as layout_module

    monkeypatch.setattr(
        layout_module,
        "resolve_artifact_checkpoint",
        lambda *_args, **_kwargs: tmp_path,
    )
    with pytest.raises(ValueError, match="not_a_detector"):
        MlxLayoutObjectDetectionModel(
            artifacts_path=None,
            accelerator_options=AcceleratorOptions(device="auto"),
            options=MlxLayoutObjectDetectionOptions.from_preset(
                "layout_egret_medium",
                engine_options=MlxObjectDetectionEngineOptions(),
            ),
        )
