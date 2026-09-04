# SPDX-License-Identifier: Apache-2.0

"""Portable contracts for the framework-free TableFormerV2 engine."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from docling_mlx.engines.table_structure.tableformer_v2.engine import (
    TableFormerV2Engine,
    TableFormerV2Prediction,
)
from docling_mlx.engines.table_structure.tableformer_v2.model_spec import (
    TableFormerV2ModelSpec,
)

ENGINE_MODULE = "docling_mlx.engines.table_structure.tableformer_v2.engine"


def _engine() -> TableFormerV2Engine:
    return TableFormerV2Engine(
        TableFormerV2ModelSpec(repo_id="example/tableformer-v2", revision="test-revision")
    )


def _artifact() -> SimpleNamespace:
    return SimpleNamespace(
        upstream_weights=False,
        weights_path=Path("/resolved/model.safetensors"),
        config=object(),
        preprocessing=object(),
        generation=SimpleNamespace(max_generation_steps=512),
        token_map=SimpleNamespace(
            tokens=(
                "<pad>",
                "<unk>",
                "<s>",
                "</s>",
                "<ecel>",
                "<fcel>",
                "<lcel>",
                "<ucel>",
                "<xcel>",
                "<nl>",
                "<ched>",
                "<rhed>",
                "<srow>",
            ),
            pad_token_id=0,
            bos_token_id=2,
            eos_token_id=3,
            data_cell_token_ids=frozenset({4, 5, 10, 11, 12}),
        ),
    )


class _FakeMx(ModuleType):
    float32 = np.float32

    def __init__(self, weights: object | None = None) -> None:
        super().__init__("mlx.core")
        self.weights = {"weight": SimpleNamespace(dtype=np.float32)} if weights is None else weights
        self.evaluations: list[tuple[object, ...]] = []

    def load(self, path: str) -> object:
        return self.weights

    def eval(self, *values: object) -> None:
        self.evaluations.append(values)


def _install_fake_runtime(monkeypatch: pytest.MonkeyPatch, model_type: type) -> _FakeMx:
    core = _FakeMx()
    mlx = ModuleType("mlx")
    mlx.core = core  # type: ignore[attr-defined]
    model_module = ModuleType("docling_mlx._models.tableformer_v2.model")
    model_module.TableFormerV2 = model_type  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)
    return core


def test_empty_batch_is_lazy_and_does_not_import_mlx() -> None:
    engine = _engine()
    with patch.object(engine, "initialize", side_effect=AssertionError("must stay lazy")):
        assert engine.predict([]) == []
    assert engine._model is None


def test_materialized_prediction_uses_crop_pixel_coordinates() -> None:
    prediction = TableFormerV2Engine._materialize_prediction(
        np.array([[2, 5, 3]], dtype=np.int32),
        np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
        _artifact(),
        Image.new("RGB", (100, 50)),
    )
    assert prediction == TableFormerV2Prediction(
        token_ids=(2, 5, 3),
        otsl_tokens=("fcel",),
        cell_bboxes=((pytest.approx(10), pytest.approx(10), pytest.approx(30), pytest.approx(20)),),  # type: ignore[arg-type]
    )


def test_materialized_prediction_rejects_invalid_outputs() -> None:
    with pytest.raises(RuntimeError, match="BOS"):
        TableFormerV2Engine._materialize_prediction(
            np.array([[1, 3]], dtype=np.int32),
            np.empty((0, 4), np.float32),
            _artifact(),
            Image.new("RGB", (1, 1)),
        )


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({"weight": SimpleNamespace(dtype=np.float16)}, "float32"),
        ({"weight": SimpleNamespace(dtype=np.float32)}, "strict weights"),
    ],
)
def test_initialization_rejects_invalid_or_unloadable_weights(
    monkeypatch: pytest.MonkeyPatch, weights: dict[str, object], message: str
) -> None:
    engine = _engine()

    class Model:
        def __init__(self, config: object) -> None:
            pass

        def load_weights(self, values: object, *, strict: bool) -> None:
            assert strict is True
            raise ValueError("strict weights")

    core = _install_fake_runtime(monkeypatch, Model)
    core.weights = weights
    with (
        patch(f"{ENGINE_MODULE}.require_apple_silicon"),
        patch(f"{ENGINE_MODULE}.resolve_checkpoint", return_value=Path("/resolved")),
        patch(f"{ENGINE_MODULE}.validate_tableformer_v2_artifact", return_value=_artifact()),
        pytest.raises(ValueError, match=message),
    ):
        engine.initialize()
    assert engine._model is None
    with pytest.raises(RuntimeError, match="number of cell boxes"):
        TableFormerV2Engine._materialize_prediction(
            np.array([[2, 5, 3]], dtype=np.int32),
            np.empty((0, 4), np.float32),
            _artifact(),
            Image.new("RGB", (1, 1)),
        )


def test_fake_inference_preserves_order(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine()
    seen: list[int] = []

    class Model:
        def __init__(self, config: object) -> None:
            pass

        def load_weights(self, weights: object, *, strict: bool) -> None:
            assert strict is True

        def eval(self) -> None:
            pass

        def parameters(self) -> tuple[()]:
            return ()

        def compile_image_backbone(self) -> None:
            pass

        def generate(self, pixels: int, *, max_generation_steps: int) -> SimpleNamespace:
            assert max_generation_steps == 512
            seen.append(pixels)
            return SimpleNamespace(
                generated_ids=np.array([[2, 5, 3]], dtype=np.int32),
                predicted_bboxes=np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
            )

    core = _install_fake_runtime(monkeypatch, Model)
    with (
        patch(f"{ENGINE_MODULE}.require_apple_silicon"),
        patch(f"{ENGINE_MODULE}.resolve_checkpoint", return_value=Path("/resolved")) as resolve,
        patch(f"{ENGINE_MODULE}.validate_tableformer_v2_artifact", return_value=_artifact()),
        patch(
            f"{ENGINE_MODULE}.preprocess_images",
            side_effect=lambda images, spec: int(np.asarray(images[0])[0, 0, 0]),
        ),
    ):
        outputs = engine.predict(
            [Image.new("RGB", (10, 10), (1, 0, 0)), Image.new("RGB", (10, 10), (2, 0, 0))]
        )

    assert seen == [1, 2]
    assert len(outputs) == 2
    assert any(len(values) == 2 for values in core.evaluations)
    resolve.assert_called_once()


def test_same_engine_concurrent_initialization_constructs_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine()
    barrier = Barrier(4)
    constructions: list[object] = []

    class Model:
        def __init__(self, config: object) -> None:
            constructions.append(config)

        def load_weights(self, weights: object, *, strict: bool) -> None:
            assert strict is True

        def eval(self) -> None:
            pass

        def parameters(self) -> tuple[()]:
            return ()

        def compile_image_backbone(self) -> None:
            pass

    _install_fake_runtime(monkeypatch, Model)
    with (
        patch(f"{ENGINE_MODULE}.require_apple_silicon") as platform_check,
        patch(f"{ENGINE_MODULE}.resolve_checkpoint", return_value=Path("/resolved")),
        patch(f"{ENGINE_MODULE}.validate_tableformer_v2_artifact", return_value=_artifact()),
    ):
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(lambda: (barrier.wait(timeout=10), engine.initialize()))
                for _ in range(4)
            ]
            for future in futures:
                future.result(timeout=10)

    assert len(constructions) == 1
    platform_check.assert_called_once_with()
