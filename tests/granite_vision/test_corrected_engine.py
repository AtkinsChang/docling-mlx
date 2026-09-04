# SPDX-License-Identifier: Apache-2.0

"""Download-free regressions for the corrected Granite Vision 4.1 engine."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from docling.datamodel.stage_model_specs import EngineModelConfig
from docling.datamodel.vlm_engine_options import MlxVlmEngineOptions

import docling_mlx.stages.granite_vision_engine as engine_module
from docling_mlx._compat.mlx_vlm import (
    correct_loaded_granite_vision_activations,
    replace_granite_vision_image_processor,
)
from docling_mlx.stages.granite_vision_engine import MlxGraniteVision41Engine


class FakeGelu:
    def __init__(self, *, approx: str) -> None:
        self.approx = approx


class FakeMlp:
    def __init__(self, activation: object | None = None) -> None:
        self.activation_fn = activation if activation is not None else object()


class FakeEncoderLayer:
    def __init__(self) -> None:
        self.mlp = FakeMlp()


class FakeEncoder:
    def __init__(self, layer_count: int) -> None:
        self.layers = [FakeEncoderLayer() for _ in range(layer_count)]


class FakeHead:
    def __init__(self, mlp: object | None = None) -> None:
        self.mlp = mlp or FakeMlp()


class FakeVisionModel:
    def __init__(self, *, layer_count: int, head_mlp: object | None) -> None:
        self.encoder = FakeEncoder(layer_count)
        self.head = FakeHead(head_mlp)


class FakeVisionTower:
    def __init__(self, *, layer_count: int, head_mlp: object | None) -> None:
        self.vision_model = FakeVisionModel(
            layer_count=layer_count,
            head_mlp=head_mlp,
        )


class FakeModel:
    def __init__(self, *, layer_count: int = 27, head_mlp: object | None = None) -> None:
        self.vision_tower = FakeVisionTower(
            layer_count=layer_count,
            head_mlp=head_mlp,
        )
        self.language_model = SimpleNamespace(mlp=FakeMlp("language-sentinel"))
        self.projector = SimpleNamespace(mlp=FakeMlp("projector-sentinel"))


class FakeProcessor:
    def __init__(self) -> None:
        self.image_processor = object()
        self.tokenizer = SimpleNamespace(stopping_criteria=object())
        self.detokenizer = object()


def _valid_config() -> dict[str, Any]:
    return {
        "deepstack_layer_map": [[-19, 9], [-13, 6], [-7, 3], [-1, 0]],
        "downsample_rate": "4/8",
        "dtype": "bfloat16",
        "eos_token_id": 100257,
        "generation_config": {"eos_token_id": 100257},
        "image_grid_pinpoints": [
            [384, 384],
            [384, 768],
            [384, 1152],
            [384, 1536],
            [384, 1920],
            [384, 2304],
            [384, 2688],
            [384, 3072],
            [384, 3456],
            [384, 3840],
            [768, 384],
            [768, 768],
            [768, 1152],
            [768, 1536],
            [768, 1920],
            [1152, 384],
            [1152, 768],
            [1152, 1152],
            [1536, 384],
            [1536, 768],
            [1920, 384],
            [1920, 768],
            [2304, 384],
            [2688, 384],
            [3072, 384],
            [3456, 384],
            [3840, 384],
        ],
        "image_token_index": 100352,
        "model_type": "granite4_vision",
        "spatial_target_layers": [12, 15, 18, 21],
        "spatial_vision_layer": -1,
        "text_config": {
            "attention_bias": False,
            "attention_multiplier": 0.015625,
            "embedding_multiplier": 12.0,
            "hidden_size": 2560,
            "intermediate_size": 8192,
            "logits_scaling": 10.0,
            "mlp_bias": False,
            "num_attention_heads": 40,
            "num_hidden_layers": 40,
            "num_key_value_heads": 8,
            "residual_multiplier": 0.22,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000000,
            "vocab_size": 100353,
        },
        "use_image_newline_parameter": True,
        "use_spatial_sampling": True,
        "vision_config": {
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 1152,
            "image_size": 384,
            "intermediate_size": 4304,
            "layer_norm_eps": 1e-6,
            "model_type": "siglip_vision_model",
            "num_attention_heads": 16,
            "num_channels": 3,
            "num_hidden_layers": 27,
            "patch_size": 16,
        },
        "vision_feature_select_strategy": "full",
    }


def _correct_model(model: object) -> None:
    correct_loaded_granite_vision_activations(model, gelu_type=FakeGelu)


def test_corrects_exactly_the_27_encoder_and_one_head_vision_mlps() -> None:
    model = FakeModel()
    language_activation = model.language_model.mlp.activation_fn
    projector_activation = model.projector.mlp.activation_fn

    _correct_model(model)

    mlps = [layer.mlp for layer in model.vision_tower.vision_model.encoder.layers] + [
        model.vision_tower.vision_model.head.mlp
    ]
    assert len(mlps) == 28
    assert all(isinstance(mlp.activation_fn, FakeGelu) for mlp in mlps)
    assert all(mlp.activation_fn.approx == "tanh" for mlp in mlps)
    assert model.language_model.mlp.activation_fn is language_activation
    assert model.projector.mlp.activation_fn is projector_activation


def test_rejects_unexpected_granite_vision_layer_count() -> None:
    with pytest.raises(ValueError, match="27 encoder"):
        _correct_model(FakeModel(layer_count=26))


def test_replaces_only_the_image_processor_and_requires_torchvision() -> None:
    artifact = Path("/models/granite")
    processor = FakeProcessor()
    original_tokenizer = processor.tokenizer
    original_detokenizer = processor.detokenizer
    original_stopper = processor.tokenizer.stopping_criteria  # type: ignore[attr-defined]
    replacement = SimpleNamespace(backend="torchvision")

    class FakeAutoImageProcessor:
        calls: list[tuple[Path, dict[str, object]]] = []

        @classmethod
        def from_pretrained(cls, path: Path, **kwargs: object) -> object:
            cls.calls.append((path, kwargs))
            return replacement

    replace_granite_vision_image_processor(
        processor,
        artifact,
        processor_type=FakeProcessor,
        auto_image_processor=FakeAutoImageProcessor,
    )

    assert FakeAutoImageProcessor.calls == [(artifact, {})]
    assert processor.image_processor is replacement
    assert processor.tokenizer is original_tokenizer
    assert processor.detokenizer is original_detokenizer
    assert processor.tokenizer.stopping_criteria is original_stopper  # type: ignore[attr-defined]

    with pytest.raises(TypeError, match="processor type"):
        replace_granite_vision_image_processor(
            object(),
            artifact,
            processor_type=FakeProcessor,
            auto_image_processor=FakeAutoImageProcessor,
        )

    rejected = FakeProcessor()
    original_image_processor = rejected.image_processor
    FakeAutoImageProcessor.from_pretrained = classmethod(  # type: ignore[method-assign]
        lambda cls, path, **kwargs: SimpleNamespace(backend="pil")
    )
    with pytest.raises(RuntimeError, match="torchvision"):
        replace_granite_vision_image_processor(
            rejected,
            artifact,
            processor_type=FakeProcessor,
            auto_image_processor=FakeAutoImageProcessor,
        )
    assert rejected.image_processor is original_image_processor


@pytest.mark.mlx
def test_granite_chat_template_is_image_first_without_global_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_utils = _load_installed_prompt_utils(monkeypatch)
    processor = object()
    original_mapping = dict(prompt_utils.MODEL_CONFIG)
    calls: list[tuple[object, list[dict[str, object]], bool, dict[str, object]]] = []

    def get_chat_template(
        processor_arg: object,
        messages: list[dict[str, object]],
        add_generation_prompt: bool,
        **kwargs: object,
    ) -> str:
        calls.append((processor_arg, messages, add_generation_prompt, kwargs))
        return "rendered"

    monkeypatch.setattr(prompt_utils, "get_chat_template", get_chat_template)

    result = engine_module._apply_granite_vision_chat_template(
        processor,
        {"model_type": "granite4_vision"},
        "<tables_otsl>",
        num_images=2,
        tokenize=True,
    )

    assert result == "rendered"
    assert calls == [
        (
            processor,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": "<tables_otsl>"},
                    ],
                }
            ],
            True,
            {"tokenize": True},
        )
    ]
    assert prompt_utils.MODEL_CONFIG == original_mapping
    assert (
        prompt_utils.MODEL_CONFIG["granite4_vision"] is prompt_utils.MessageFormat.LIST_WITH_IMAGE
    )


@pytest.mark.parametrize(
    ("config", "prompt", "kwargs", "match"),
    [
        ({"model_type": "granite4_vision"}, "prompt", {"num_images": 0}, "image"),
        (
            {"model_type": "granite4_vision"},
            "prompt",
            {"num_images": 1, "num_audios": 1},
            "audio",
        ),
        (
            {"model_type": "granite4_vision"},
            "prompt",
            {"num_images": 1, "return_messages": True},
            "return_messages",
        ),
    ],
)
def test_granite_chat_template_rejects_unsupported_inputs(
    config: object,
    prompt: object,
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        engine_module._apply_granite_vision_chat_template(
            object(),
            config,
            prompt,
            **kwargs,
        )


def _install_fake_module(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    **attributes: object,
) -> ModuleType:
    module = ModuleType(name)
    module.__dict__.update(attributes)
    if "." not in name or name.endswith(("models", "granite4_vision")):
        module.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_installed_prompt_utils(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    distribution = importlib.metadata.distribution("mlx-vlm")
    module_path = Path(distribution.locate_file("mlx_vlm/prompt_utils.py"))
    spec = importlib.util.spec_from_file_location("mlx_vlm.prompt_utils", module_path)
    assert spec is not None and spec.loader is not None
    package = ModuleType("mlx_vlm")
    package.__path__ = []  # type: ignore[attr-defined]
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "mlx_vlm", package)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", module)
    spec.loader.exec_module(module)
    return module


def _install_load_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model: FakeModel,
    processor: FakeProcessor,
    image_backend: str,
    calls: dict[str, object],
) -> None:
    upstream_chat_template = object()
    stream_generate = object()

    def load(path: Path, *, strict: bool) -> tuple[FakeModel, FakeProcessor]:
        calls["load"] = (path, strict)
        return model, processor

    class FakeAutoImageProcessor:
        @classmethod
        def from_pretrained(cls, path: Path, **kwargs: object) -> object:
            calls["image_processor"] = (path, kwargs)
            return SimpleNamespace(backend=image_backend)

    def load_config(path: Path) -> dict[str, Any]:
        calls["config_path"] = path
        return _valid_config()

    _install_fake_module(monkeypatch, "mlx", nn=SimpleNamespace(GELU=FakeGelu))
    _install_fake_module(
        monkeypatch,
        "mlx_vlm",
        load=load,
        stream_generate=stream_generate,
    )
    _install_fake_module(
        monkeypatch,
        "mlx_vlm.prompt_utils",
        apply_chat_template=upstream_chat_template,
    )
    _install_fake_module(
        monkeypatch,
        "mlx_vlm.models.granite4_vision.processing_granite4_vision",
        Granite4VisionProcessor=FakeProcessor,
    )
    _install_fake_module(
        monkeypatch,
        "mlx_vlm.utils",
        load_config=load_config,
    )
    _install_fake_module(
        monkeypatch,
        "transformers",
        AutoImageProcessor=FakeAutoImageProcessor,
    )
    calls["upstream_chat_template"] = upstream_chat_template
    calls["stream_generate"] = stream_generate


def test_initialize_replaces_base_hook_only_after_successful_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel()
    processor = FakeProcessor()
    calls: dict[str, object] = {}
    _install_load_dependencies(
        monkeypatch,
        model=model,
        processor=processor,
        image_backend="torchvision",
        calls=calls,
    )
    monkeypatch.setattr(
        "docling_mlx.stages.granite_vision_engine.resolve_model_artifacts_path",
        lambda *args: Path("/resolved/granite"),
    )

    engine = MlxGraniteVision41Engine(
        MlxVlmEngineOptions(),
        artifacts_path=None,
        model_config=EngineModelConfig(repo_id="ibm/model", revision="commit"),
    )

    assert engine.apply_chat_template is engine_module._apply_granite_vision_chat_template
    assert engine.stream_generate is calls["stream_generate"]
    assert engine.vlm_model is model
    assert engine.processor is processor
    assert engine._initialized is True


def test_load_uses_docling_resolution_strict_load_and_atomic_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = Path("/resolved/granite")
    model = FakeModel()
    processor = FakeProcessor()
    calls: dict[str, object] = {}
    _install_load_dependencies(
        monkeypatch,
        model=model,
        processor=processor,
        image_backend="torchvision",
        calls=calls,
    )

    def resolve(
        repo_id: str,
        revision: str,
        artifacts_path: Path,
        download_fn: object,
    ) -> Path:
        calls["resolve"] = (repo_id, revision, artifacts_path, download_fn)
        return artifact

    monkeypatch.setattr(
        "docling_mlx.stages.granite_vision_engine.resolve_model_artifacts_path",
        resolve,
    )
    engine = MlxGraniteVision41Engine(
        MlxVlmEngineOptions(),
        artifacts_path=Path("/models"),
        model_config=None,
    )
    upstream_hook = object()
    engine.apply_chat_template = upstream_hook

    engine._load_model_for_repo("ibm/model", revision="commit")

    assert calls["resolve"][:3] == ("ibm/model", "commit", Path("/models"))  # type: ignore[index]
    assert calls["config_path"] == artifact
    assert calls["load"] == (str(artifact), True)
    assert calls["image_processor"] == (artifact, {})
    assert engine.vlm_model is model
    assert engine.processor is processor
    assert engine.config == _valid_config()
    assert engine.apply_chat_template is engine_module._apply_granite_vision_chat_template


def test_load_failure_does_not_publish_partial_engine_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel()
    processor = FakeProcessor()
    calls: dict[str, object] = {}
    _install_load_dependencies(
        monkeypatch,
        model=model,
        processor=processor,
        image_backend="pil",
        calls=calls,
    )
    monkeypatch.setattr(
        "docling_mlx.stages.granite_vision_engine.resolve_model_artifacts_path",
        lambda *args: Path("/resolved/granite"),
    )
    engine = MlxGraniteVision41Engine(
        MlxVlmEngineOptions(),
        artifacts_path=None,
        model_config=None,
    )
    upstream_hook = object()
    engine.apply_chat_template = upstream_hook

    with pytest.raises(RuntimeError, match="torchvision"):
        engine._load_model_for_repo("ibm/model", revision="commit")

    assert engine.vlm_model is None
    assert engine.processor is None
    assert engine.config is None
    assert engine.apply_chat_template is upstream_hook
