# SPDX-License-Identifier: Apache-2.0

"""Portable contracts for the framework-free TableFormerV1 engine."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from docling_mlx.engines.table_structure.tableformer_v1.engine import (
    TableFormerV1Engine,
    TableFormerV1EngineOptions,
    TableFormerV1Prediction,
)
from docling_mlx.engines.table_structure.tableformer_v1.model_spec import (
    TableFormerV1ModelSpec,
)

ENGINE_MODULE = "docling_mlx.engines.table_structure.tableformer_v1.engine"


def _engine(profile: str = "accurate") -> TableFormerV1Engine:
    return TableFormerV1Engine(
        TableFormerV1ModelSpec(repo_id="example/tableformer-v1", revision="test-revision"),
        TableFormerV1EngineOptions(checkpoint_subdirectory=profile),
    )


def test_prediction_validation_decoding_and_crop_pixel_boxes() -> None:
    prediction = TableFormerV1Engine._materialize_prediction(
        np.array([[2, 5, 9, 3]], dtype=np.int32),
        np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
        np.array([[0.1, 0.9, 0.0]], dtype=np.float32),
        Image.new("RGB", (100, 50)),
    )
    assert prediction == TableFormerV1Prediction(
        token_ids=(2, 5, 9, 3),
        otsl_tokens=("fcel", "nl"),
        cell_bboxes=((pytest.approx(10), pytest.approx(10), pytest.approx(30), pytest.approx(20)),),  # type: ignore[arg-type]
        bbox_classes=(1,),
    )


def test_decoding_without_eos_still_discards_the_final_generated_tag() -> None:
    assert TableFormerV1Engine._decode_token_ids((2, 5, 9)) == ("fcel",)


@pytest.mark.parametrize("profile", ["accurate", "fast"])
def test_upstream_source_root_resolves_the_selected_profile(tmp_path: Path, profile: str) -> None:
    upstream = tmp_path / "model_artifacts" / "tableformer" / profile
    upstream.mkdir(parents=True)
    (upstream / "tm_config.json").write_text("{}")

    engine = TableFormerV1Engine(
        TableFormerV1ModelSpec(path=tmp_path),
        TableFormerV1EngineOptions(checkpoint_subdirectory=profile),
    )

    assert engine._resolve_directory() == upstream


def test_strict_weight_loading_failure_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine()

    class Model:
        def __init__(self, config: object) -> None:
            pass

        def load_weights(self, weights: object, *, strict: bool) -> None:
            assert strict is True
            raise ValueError("strict weights")

    class Mx(ModuleType):
        float32 = np.float32

        def load(self, path: str) -> dict[str, object]:
            return {"weight": SimpleNamespace(dtype=np.float32)}

    core = Mx("mlx.core")
    mlx = ModuleType("mlx")
    mlx.core = core  # type: ignore[attr-defined]
    model_module = ModuleType("docling_mlx._models.tableformer_v1.model")
    model_module.TableFormerV1 = Model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)
    artifact = SimpleNamespace(
        upstream_weights=False,
        weights_path=Path("/resolved/model.safetensors"),
        config=object(),
    )
    with (
        patch(f"{ENGINE_MODULE}.require_apple_silicon"),
        patch(f"{ENGINE_MODULE}.resolve_checkpoint", return_value=Path("/resolved")),
        patch(f"{ENGINE_MODULE}.validate_tableformer_v1_artifact", return_value=artifact),
        pytest.raises(ValueError, match="strict weights"),
    ):
        engine.initialize()
    assert engine._model is None


@pytest.mark.parametrize(
    ("tokens", "boxes", "classes", "message"),
    [
        (
            np.array([5, 3], np.int32),
            np.empty((0, 4), np.float32),
            np.empty(0, np.int32),
            "BOS",
        ),
    ],
)
def test_prediction_validation_rejects_invalid_model_outputs(
    tokens: np.ndarray,
    boxes: np.ndarray,
    classes: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        TableFormerV1Engine._materialize_prediction(
            tokens, boxes, classes, Image.new("RGB", (1, 1))
        )


@pytest.mark.parametrize("profile", ["accurate", "fast"])
def test_fake_inference_is_ordered_and_resolves_only_the_selected_profile(
    monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    engine = _engine(profile)
    seen: list[int] = []
    models: list[object] = []

    class Model:
        def __init__(self, config: object) -> None:
            self.compile_calls = 0
            models.append(self)

        def load_weights(self, weights: object, *, strict: bool) -> None:
            assert strict is True

        def eval(self) -> None:
            pass

        def parameters(self) -> tuple[()]:
            return ()

        def compile_image_backbone(self) -> None:
            self.compile_calls += 1

        def generate(
            self, pixels: np.ndarray, *, max_generation_steps: int
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            assert max_generation_steps == 1024
            seen.append(int(pixels.item()))
            return (
                np.array([2, 5, 3], np.int32),
                np.array([1], np.int32),
                np.array([[0.1, 0.2, 0.3, 0.4]], np.float32),
            )

    class Mx(ModuleType):
        float32 = np.float32

        def array(self, value: object) -> np.ndarray:
            return np.asarray(value)

        def load(self, path: str) -> dict[str, object]:
            return {"weight": SimpleNamespace(dtype=np.float32)}

        def eval(self, *values: object) -> None:
            pass

    core = Mx("mlx.core")
    mlx = ModuleType("mlx")
    mlx.core = core  # type: ignore[attr-defined]
    model_module = ModuleType("docling_mlx._models.tableformer_v1.model")
    model_module.TableFormerV1 = Model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)
    artifact = SimpleNamespace(
        upstream_weights=False,
        weights_path=Path("/resolved/model.safetensors"),
        config=object(),
        preprocessing=object(),
        generation=SimpleNamespace(max_generation_steps=1024),
    )
    with (
        patch(f"{ENGINE_MODULE}.require_apple_silicon"),
        patch(f"{ENGINE_MODULE}.resolve_checkpoint", return_value=Path("/resolved")) as resolve,
        patch(
            f"{ENGINE_MODULE}.validate_tableformer_v1_artifact", return_value=artifact
        ) as validate,
        patch(
            f"{ENGINE_MODULE}.preprocess_table_image",
            side_effect=lambda image, spec: np.array(int(image[0, 0, 0])),
        ),
    ):
        outputs = engine.predict(
            [Image.new("RGB", (10, 10), (1, 0, 0)), Image.new("L", (10, 10), 2)]
        )

    assert seen == [1, 2]
    assert len(outputs) == 2
    assert engine.directory == Path("/resolved") / profile
    assert models[0].compile_calls == 1  # type: ignore[attr-defined]
    resolve.assert_called_once()
    validate.assert_called_once_with(Path("/resolved") / profile)
