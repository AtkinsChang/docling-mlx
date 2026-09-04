# SPDX-License-Identifier: Apache-2.0

"""Docling layout-stage integration without loading an MLX artifact."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.object_detection_engine_options import (
    TransformersObjectDetectionEngineOptions,
)
from docling.datamodel.pipeline_options import LayoutObjectDetectionOptions
from docling.datamodel.stage_model_specs import EngineModelConfig, ObjectDetectionModelSpec
from docling.models.inference_engines.object_detection.base import ObjectDetectionEngineType
from docling.models.stages.layout.layout_object_detection_model import (
    LayoutObjectDetectionModel,
)
from docling_core.types.doc import DocItemLabel

from docling_mlx.presets import PRESETS, resolve_preset
from docling_mlx.stages.layout import (
    MlxLayoutObjectDetectionModel,
    MlxLayoutObjectDetectionOptions,
    MlxObjectDetectionEngineOptions,
)


@pytest.mark.parametrize(
    "preset_id",
    [
        name
        for name, preset in PRESETS.items()
        if preset.engine_kind.startswith("object_detection/")
    ],
)
def test_official_preset_ids_select_published_mirrors(preset_id: str) -> None:
    preset = resolve_preset(preset_id)
    options = MlxLayoutObjectDetectionOptions.from_preset(preset_id)

    assert options.model_spec.repo_id == preset.repo_id
    assert options.model_spec.revision == preset.revision
    assert options.model_spec.engine_overrides == {}
    assert options.engine_options == MlxObjectDetectionEngineOptions()


def test_options_match_the_official_shape_and_own_their_registry() -> None:
    options = MlxLayoutObjectDetectionOptions()

    assert MlxLayoutObjectDetectionOptions.kind == "mlx_layout_object_detection"
    assert MlxLayoutObjectDetectionOptions.list_preset_ids() == [
        "layout_heron_default",
        "layout_heron_101",
        "layout_egret_medium",
        "layout_egret_large",
        "layout_egret_xlarge",
    ]
    assert MlxLayoutObjectDetectionOptions._presets is not LayoutObjectDetectionOptions._presets
    assert (
        options.model_spec
        == MlxLayoutObjectDetectionOptions.get_preset("layout_heron_default").model_spec
    )
    assert options.engine_options == MlxObjectDetectionEngineOptions()
    assert "preset" not in type(options).model_fields


def test_layout_preset_rejects_non_mlx_engine_options() -> None:
    with pytest.raises(TypeError, match="MlxObjectDetectionEngineOptions"):
        MlxLayoutObjectDetectionOptions.from_preset(
            "layout_heron_default",
            engine_options=TransformersObjectDetectionEngineOptions(),
        )


def test_stage_uses_checkpoint_model_type_and_initializes_eagerly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "rt_detr_v2"}), encoding="utf-8"
    )
    resolved: list[tuple[str, str, Path | str | None]] = []
    created: list[object] = []

    def resolve(repo_id, revision, artifacts_path, *, files):
        del files
        resolved.append((repo_id, revision, artifacts_path))
        return tmp_path

    class FakeRtDetrV2Engine:
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

    monkeypatch.setattr(layout_module, "resolve_artifact_checkpoint", resolve)
    monkeypatch.setattr(layout_module, "RtDetrV2Engine", FakeRtDetrV2Engine)
    monkeypatch.setattr(
        layout_module,
        "DFineEngine",
        lambda *_args, **_kwargs: pytest.fail("RT-DETR-v2 selected D-FINE"),
    )
    options = MlxLayoutObjectDetectionOptions(
        model_spec=ObjectDetectionModelSpec(
            name="upstream heron",
            repo_id="docling-project/docling-layout-heron",
            revision="upstream-revision",
            engine_overrides={
                ObjectDetectionEngineType.TRANSFORMERS: EngineModelConfig(
                    repo_id="ignored/transformers-export",
                    revision="ignored-revision",
                )
            },
        ),
        engine_options=MlxObjectDetectionEngineOptions(
            score_threshold=0.61,
            warmup=True,
        ),
    )
    model = MlxLayoutObjectDetectionModel(
        artifacts_path=Path("/artifact-root"),
        accelerator_options=AcceleratorOptions(device="auto"),
        options=options,
    )
    options.engine_options.score_threshold = 0.9

    engine = created[0]
    assert resolved == [
        (
            "docling-project/docling-layout-heron",
            "upstream-revision",
            Path("/artifact-root"),
        )
    ]
    assert engine.options.score_threshold == pytest.approx(0.61)  # type: ignore[union-attr]
    assert engine.initialize_calls == [True]  # type: ignore[union-attr]
    assert model._label_map == {0: DocItemLabel.TEXT}
    assert model.engine.predict_batch([]) == []
    assert MlxLayoutObjectDetectionModel.get_options_type() is MlxLayoutObjectDetectionOptions


def test_stage_constructor_matches_official_parameter_contract() -> None:
    expected = inspect.signature(LayoutObjectDetectionModel.__init__).parameters
    actual = inspect.signature(MlxLayoutObjectDetectionModel.__init__).parameters

    assert tuple(actual) == tuple(expected)
    assert actual["enable_remote_services"].default is False


def test_inherited_cluster_conversion_clips_detector_pixels_to_page_bounds() -> None:
    model = object.__new__(MlxLayoutObjectDetectionModel)
    model._label_map = {0: DocItemLabel.TEXT}
    model._unmapped_label_ids = set()
    page = SimpleNamespace(size=SimpleNamespace(width=20.0, height=10.0))
    image = SimpleNamespace(width=10, height=5)
    output = SimpleNamespace(label_ids=[0], scores=[0.7], bboxes=[[-2.0, -1.0, 12.0, 6.0]])

    clusters = model._predictions_to_clusters(page, image, output)

    assert clusters[0].bbox.as_tuple() == (0.0, 0.0, 20.0, 10.0)
