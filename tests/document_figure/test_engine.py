# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.stage_model_specs import EngineModelConfig
from docling.models.inference_engines.image_classification.base import (
    ImageClassificationEngineInput,
)
from PIL import Image

from docling_mlx.engines.image_classification.efficientnet.artifact import (
    CHECKPOINT_FILES,
)
from docling_mlx.engines.image_classification.efficientnet.artifact import (
    _validate_artifact as validate_artifact,
)
from docling_mlx.stages.picture_classification import (
    MlxImageClassificationEngineOptions,
    _MlxDocumentFigureClassificationEngine,
)

DOCUMENT_FIGURE_LABELS = tuple(f"label_{index}" for index in range(26))


@pytest.fixture
def artifact_dir() -> Path:
    path = Path(__file__).resolve().parents[2] / ".artifacts/document-figure-classifier"
    if not path.is_dir():
        pytest.fail("selected MLX or release lane requires the converted artifact")
    return path


@pytest.fixture
def artifact_contract_dir(tmp_path: Path) -> Path:
    path = tmp_path / "source-artifact"
    path.mkdir()
    (path / "model.safetensors").write_bytes(b"contract-only")
    labels = {str(index): label for index, label in enumerate(DOCUMENT_FIGURE_LABELS)}
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["EfficientNetForImageClassification"],
                "model_type": "efficientnet",
                "num_channels": 3,
                "image_size": 224,
                "width_coefficient": 1.0,
                "depth_coefficient": 1.0,
                "depth_divisor": 8,
                "in_channels": [32, 16, 24, 40, 80, 112, 192],
                "out_channels": [16, 24, 40, 80, 112, 192, 320],
                "kernel_sizes": [3, 3, 5, 3, 5, 5, 3],
                "strides": [1, 2, 2, 2, 1, 2, 1],
                "expand_ratios": [1, 6, 6, 6, 6, 6, 6],
                "num_block_repeats": [1, 2, 2, 3, 3, 4, 1],
                "depthwise_padding": [],
                "hidden_dim": 1280,
                "hidden_act": "swish",
                "batch_norm_eps": 0.001,
                "batch_norm_momentum": 0.99,
                "squeeze_expansion_ratio": 0.25,
                "pooling_type": "mean",
                "dtype": "float32",
                "id2label": labels,
                "label2id": {label: index for index, label in labels.items()},
            }
        )
    )
    (path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "do_normalize": True,
                "do_rescale": True,
                "do_resize": True,
                "image_mean": [0.485, 0.456, 0.406],
                "image_std": [0.47853944, 0.4732864, 0.47434163],
                "resample": 2,
                "rescale_factor": 1 / 255,
                "size": {"height": 224, "width": 224},
            }
        )
    )
    return path


@pytest.mark.release
def test_pinned_source_accepts_string_label_ids(artifact_dir: Path) -> None:
    config, _ = validate_artifact(artifact_dir)
    assert config.labels[5] == "bar_chart"


ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests/golden/document_figure"
REFERENCE_IMAGES = ROOT / "tests/fixtures/document_figure/reference_images"


def _engine(
    artifact: Path = Path("/not-resolved-for-cpu-contracts"),
    *,
    options: MlxImageClassificationEngineOptions | None = None,
    model_config: EngineModelConfig | None = None,
    accelerator_options: AcceleratorOptions | None = None,
) -> _MlxDocumentFigureClassificationEngine:
    return _MlxDocumentFigureClassificationEngine(
        options or MlxImageClassificationEngineOptions(),
        model_config=model_config
        or EngineModelConfig(repo_id=artifact.name, revision="local-test-revision"),
        accelerator_options=accelerator_options or AcceleratorOptions(device="auto"),
        artifacts_path=artifact.parent if artifact.is_dir() else None,
    )


def inputs():
    result = []
    for name in ("bar_chart", "geographical_map"):
        with Image.open(REFERENCE_IMAGES / f"{name}.png") as image:
            result.append(
                ImageClassificationEngineInput(
                    image=image.convert("RGB"), metadata={"fixture": name, "nested": {"x": 1}}
                )
            )
    return result


def synthetic_images() -> list[tuple[str, Image.Image]]:
    """Cover the preprocessing boundary with deterministic, awkward image inputs."""
    generator = np.random.default_rng(814)
    gradient_x = np.linspace(0, 255, 17, dtype=np.uint8)
    gradient_y = np.linspace(0, 255, 19, dtype=np.uint8)[:, None]
    gradient = np.stack(
        [
            np.broadcast_to(gradient_x, (19, 17)),
            np.broadcast_to(gradient_y, (19, 17)),
            (gradient_x[None, :] // 2 + gradient_y // 2).astype(np.uint8),
        ],
        axis=-1,
    )
    thin_lines = np.zeros((23, 17, 3), dtype=np.uint8)
    thin_lines[::2] = 255
    thin_lines[:, 1::3, 1] = 255
    checker = (np.indices((31, 29)).sum(axis=0) % 2 * 255).astype(np.uint8)
    checkerboard = np.stack([checker, 255 - checker, checker], axis=-1)
    grayscale_tall = np.arange(4097, dtype=np.uint8)[:, None]
    rgba_wide = np.empty((1, 4097, 4), dtype=np.uint8)
    rgba_wide[..., 0] = np.arange(4097, dtype=np.uint8)
    rgba_wide[..., 1] = 127
    rgba_wide[..., 2] = 255 - rgba_wide[..., 0]
    rgba_wide[..., 3] = np.arange(4097, dtype=np.uint8)
    return [
        ("black-rgb-1x1", Image.new("RGB", (1, 1))),
        ("white-rgb-251x317", Image.new("RGB", (251, 317), "white")),
        ("gradient-rgb-17x19", Image.fromarray(gradient, "RGB")),
        ("thin-lines-rgb-17x23", Image.fromarray(thin_lines, "RGB")),
        ("checkerboard-rgb-29x31", Image.fromarray(checkerboard, "RGB")),
        (
            "random-rgb-37x29",
            Image.fromarray(generator.integers(0, 256, (29, 37, 3), dtype=np.uint8), "RGB"),
        ),
        ("grayscale-l-1x4097", Image.fromarray(grayscale_tall, "L")),
        ("rgba-4097x1", Image.fromarray(rgba_wide, "RGBA")),
    ]


@pytest.fixture(scope="module")
def engine():
    path = ROOT / ".artifacts/document-figure-classifier"
    if not path.is_dir():
        pytest.fail("selected MLX lane requires the converted DocumentFigure artifact")
    return _engine(path)


@pytest.mark.mlx
def test_synthetic_images_preserve_output_contract(engine) -> None:
    cases = synthetic_images()
    images = [image for _, image in cases]
    try:
        actual_logits = engine.predict_logits(images)
        actual_probabilities = np.exp(actual_logits - actual_logits.max(axis=1, keepdims=True))
        actual_probabilities /= actual_probabilities.sum(axis=1, keepdims=True)
        assert actual_logits.shape == (len(images), 26)
        assert actual_logits.dtype == np.float32
        assert np.isfinite(actual_logits).all()

        outputs = engine(
            [
                ImageClassificationEngineInput(image=image, metadata={"case": name})
                for name, image in cases
            ]
        )
        for index, ((name, _), output) in enumerate(zip(cases, outputs, strict=True)):
            scores = np.asarray(output.scores, dtype=np.float32)
            assert output.metadata == {"case": name}
            assert np.isfinite(scores).all()
            assert scores.sum() == pytest.approx(1, abs=1e-6)
            assert np.all(scores[:-1] >= scores[1:])
            np.testing.assert_allclose(
                scores, actual_probabilities[index, output.label_ids], rtol=0, atol=1e-6
            )
    finally:
        for image in images:
            image.close()


@pytest.mark.mlx
@pytest.mark.parity
def test_real_image_golden_classification_contract(engine) -> None:
    items = inputs()
    for item, output in zip(items, engine(items), strict=True):
        with np.load(GOLDEN / f"{item.metadata['fixture']}.npz") as golden:
            expected_ids = np.argsort(-golden["probabilities_f32"], kind="stable")
            assert output.label_ids[0] == expected_ids[0]
            assert set(output.label_ids[:5]) == set(expected_ids[:5])
        assert output.metadata == item.metadata
        assert np.isfinite(output.scores).all()
        assert sum(output.scores) == pytest.approx(1, abs=1e-6)


@pytest.mark.mlx
def test_real_batch_four_preserves_order_and_matches_single(engine) -> None:
    base = inputs()
    items = [base[index % 2] for index in range(4)]
    expected = [engine.predict(item) for item in items]
    actual = engine(items)
    for single, batched in zip(expected, actual, strict=True):
        assert single.label_ids == batched.label_ids
        assert single.metadata == batched.metadata
        np.testing.assert_allclose(single.scores, batched.scores, rtol=0, atol=1e-6)
    for forward, backward in zip(actual, reversed(engine(list(reversed(items)))), strict=True):
        assert forward.label_ids == backward.label_ids
        np.testing.assert_allclose(forward.scores, backward.scores, rtol=0, atol=1e-6)


@pytest.mark.parametrize("top_k", [1, 5, None, 50])
def test_top_k_and_caller_options_copy_use_synthetic_logits(top_k) -> None:
    options = MlxImageClassificationEngineOptions(top_k=top_k)
    selected = _engine(options=options)
    options.top_k = 2
    count = 26 if top_k is None else min(top_k, 26)
    logits = np.linspace(13, -12, 26, dtype=np.float32)[None, :]
    item = ImageClassificationEngineInput(
        image=Image.new("RGB", (1, 1)), metadata={"case": "synthetic"}
    )
    with patch.object(selected, "predict_logits", return_value=logits):
        actual = selected.predict(item)
    assert actual.label_ids == list(range(count))
    assert actual.metadata == item.metadata
    assert selected._mlx_options.top_k == top_k


def test_engine_copies_model_settings_and_preserves_base_cache_identity() -> None:
    options = MlxImageClassificationEngineOptions(top_k=5)
    model_config = EngineModelConfig(
        repo_id="example/document-figure",
        revision="branch-or-tag-is-allowed",
        extra_config={"cache_identity": "custom"},
    )
    engine = _engine(options=options, model_config=model_config)

    options.top_k = 1
    model_config.repo_id = "mutated/repo"
    model_config.revision = "mutated"
    model_config.extra_config["cache_identity"] = "mutated"

    assert engine._mlx_options.top_k == 5
    assert engine._model_config == EngineModelConfig(
        repo_id="example/document-figure",
        revision="branch-or-tag-is-allowed",
        extra_config={"cache_identity": "custom"},
    )
    assert engine.model_config is engine._model_config


def test_engine_resolves_docling_artifact_with_model_settings() -> None:
    class StopValidation(Exception):
        pass

    model_config = EngineModelConfig(
        repo_id="example/document-figure",
        revision="refs/pr/123",
    )
    artifacts_path = Path("/prefetched-model-root")
    with (
        patch(
            "docling_mlx.stages.picture_classification.resolve_artifact_checkpoint",
            return_value=Path("/resolved"),
        ) as resolve,
        patch(
            "docling_mlx.stages.picture_classification.EfficientNetEngine.initialize",
            side_effect=StopValidation,
        ),
        pytest.raises(StopValidation),
    ):
        engine = _MlxDocumentFigureClassificationEngine(
            MlxImageClassificationEngineOptions(),
            model_config=model_config,
            accelerator_options=AcceleratorOptions(device="auto"),
            artifacts_path=artifacts_path,
        )
        engine.initialize()

    resolve.assert_called_once_with(
        "example/document-figure",
        "refs/pr/123",
        artifacts_path,
        files=CHECKPOINT_FILES,
    )


@pytest.mark.mlx
def test_preprocessing_runs_before_logit_materialization(engine) -> None:
    engine.initialize()
    assert engine._engine is not None

    with patch.object(
        engine._engine, "predict_logits", wraps=engine._engine.predict_logits
    ) as checked:
        result = engine.predict_logits([item.image for item in inputs()])
    checked.assert_called_once()
    assert result.shape == (2, 26)


@pytest.mark.mlx
def test_two_threads_initialize_once_and_predict_consistently(artifact_dir) -> None:
    from docling_mlx._models.efficientnet.model import EfficientNet

    shared = _engine(artifact_dir)
    barrier = Barrier(2)
    items = inputs()

    def run():
        barrier.wait(timeout=15)
        shared.initialize()
        first = shared(items)
        assert first == shared(items)
        return first

    with patch(
        "docling_mlx._models.efficientnet.model.EfficientNet", wraps=EfficientNet
    ) as constructor:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run) for _ in range(2)]
            outputs = [future.result(timeout=60) for future in futures]
        assert constructor.call_count == 1
    assert outputs[0] == outputs[1] == shared(items)


def test_empty_batch_and_exact_tie_policy() -> None:
    empty = _engine(options=MlxImageClassificationEngineOptions(top_k=5))
    assert empty([]) == []
    assert empty.predict_logits([]).shape == (0, 26)
    # Only postprocessing is isolated; inference tests above run the real model.
    with patch.object(empty, "predict_logits", return_value=np.zeros((1, 26), np.float32)):
        output = empty.predict(ImageClassificationEngineInput(image=Image.new("RGB", (1, 1))))
    assert output.label_ids == [0, 1, 2, 3, 4]
    np.testing.assert_allclose(output.scores, np.full(5, 1 / 26), rtol=0, atol=1e-7)


@pytest.mark.parametrize(
    ("file_name", "expected_exception"),
    [
        ("model.safetensors", FileNotFoundError),
        ("config.json", FileNotFoundError),
        ("preprocessor_config.json", FileNotFoundError),
    ],
)
def test_loader_rejects_missing_artifact_files(
    artifact_contract_dir: Path,
    tmp_path: Path,
    file_name: str,
    expected_exception: type[Exception],
) -> None:
    target = tmp_path / "artifact"
    shutil.copytree(artifact_contract_dir, target)
    (target / file_name).unlink()

    with pytest.raises(expected_exception):
        validate_artifact(target)


@pytest.mark.parametrize("file_name", ["config.json", "preprocessor_config.json"])
def test_loader_rejects_invalid_json(
    artifact_contract_dir: Path, tmp_path: Path, file_name: str
) -> None:
    target = tmp_path / "artifact"
    shutil.copytree(artifact_contract_dir, target)
    (target / file_name).write_text("not json")

    with pytest.raises(ValueError):
        validate_artifact(target)


@pytest.mark.parametrize(
    "change",
    [
        "model_type",
        "missing_label",
        "inconsistent_labels",
    ],
)
def test_loader_rejects_invalid_semantic_contract(
    artifact_contract_dir: Path, tmp_path: Path, change: str
) -> None:
    target = tmp_path / "artifact"
    shutil.copytree(artifact_contract_dir, target)
    path = target / "config.json"
    document = json.loads(path.read_text())
    if change == "model_type":
        document["model_type"] = "vit"
    elif change == "missing_label":
        document["id2label"].pop("25")
    else:
        document["label2id"][document["id2label"]["0"]] = "25"
    path.write_text(json.dumps(document))

    with pytest.raises((TypeError, ValueError)):
        validate_artifact(target)


def test_loader_accepts_non_runtime_processor_metadata(
    artifact_contract_dir: Path, tmp_path: Path
) -> None:
    target = tmp_path / "artifact"
    shutil.copytree(artifact_contract_dir, target)
    path = target / "preprocessor_config.json"
    document = json.loads(path.read_text())
    document["resample"] = 3
    path.write_text(json.dumps(document))
    validate_artifact(target)


def test_loader_accepts_architecture_metadata_drift(
    artifact_contract_dir: Path, tmp_path: Path
) -> None:
    target = tmp_path / "artifact"
    shutil.copytree(artifact_contract_dir, target)
    path = target / "config.json"
    document = json.loads(path.read_text())
    document["architectures"] = ["FutureEfficientNetForImageClassification"]
    document["future_metadata"] = {"producer_version": "next"}
    path.write_text(json.dumps(document))

    validate_artifact(target)


@pytest.mark.mlx
def test_loader_rejects_invalid_safetensors(artifact_dir: Path, tmp_path: Path) -> None:
    target = tmp_path / "artifact"
    shutil.copytree(artifact_dir, target)
    (target / "model.safetensors").write_bytes(b"not a safetensors file")
    engine = _engine(target)

    with pytest.raises((RuntimeError, ValueError)):
        engine.initialize()


@pytest.mark.mlx
def test_loader_accepts_float16_weight(artifact_dir: Path, tmp_path: Path) -> None:
    from safetensors.numpy import load_file, save_file

    target = tmp_path / "artifact"
    shutil.copytree(artifact_dir, target)
    weights = load_file(target / "model.safetensors")
    key = next(iter(weights))
    weights[key] = weights[key].astype(np.float16)
    save_file(weights, target / "model.safetensors")
    engine = _engine(target)

    engine.initialize()


@pytest.mark.mlx
@pytest.mark.parametrize("change", ["missing", "extra", "wrong_shape"])
def test_strict_weight_loading_rejects_incomplete_state(
    artifact_dir: Path, tmp_path: Path, change: str
) -> None:
    from safetensors.numpy import load_file, save_file

    target = tmp_path / "artifact"
    shutil.copytree(artifact_dir, target)
    weights = load_file(target / "model.safetensors")
    key = next(iter(weights))
    if change == "missing":
        weights.pop(key)
    elif change == "extra":
        weights["unexpected.weight"] = np.zeros((1,), dtype=np.float32)
    else:
        weights[key] = np.zeros((1,), dtype=np.float32)
    save_file(weights, target / "model.safetensors")
    engine = _engine(target)

    with pytest.raises(ValueError):
        engine.initialize()


@pytest.mark.mlx
@pytest.mark.parametrize("invalid", ["shape", "nonfinite"])
def test_predict_logits_rejects_invalid_model_output(invalid: str) -> None:
    import mlx.core as mx

    class Model:
        def __call__(self, pixels):
            if invalid == "shape":
                return mx.zeros((pixels.shape[0], 25), dtype=mx.float32)
            return mx.full((pixels.shape[0], 26), mx.nan, dtype=mx.float32)

    from docling_mlx.engines.image_classification.efficientnet import (
        EfficientNetEngine,
        EfficientNetModelSpec,
    )

    engine = EfficientNetEngine(EfficientNetModelSpec(path="/checkpoint"))
    engine._model = Model()
    engine._config = SimpleNamespace(num_labels=26)
    engine._preprocessing = SimpleNamespace()
    engine._dtype = mx.float32

    with (
        patch(
            "docling_mlx.engines.image_classification.efficientnet.engine.preprocess_images",
            return_value=mx.zeros((1, 224, 224, 3), dtype=mx.float32),
        ),
        pytest.raises(RuntimeError, match="invalid logits"),
    ):
        engine.predict_logits([Image.new("RGB", (1, 1))])


def test_unsupported_platform(monkeypatch) -> None:
    empty = _engine()
    monkeypatch.setattr(
        "docling_mlx.stages.picture_classification.EfficientNetEngine.initialize",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("Apple Silicon is required")),
    )
    with pytest.raises(RuntimeError, match="Apple Silicon"):
        empty.initialize()
