# EfficientNet image classification

Docling MLX exposes a generic native MLX EfficientNet engine and keeps
DocumentFigureClassifier as a thin Docling stage adaptor. Weights are never
packaged in the wheel.

## Architecture and provenance

The engine implements EfficientNet (Tan and Le, 2019, arXiv:1905.11946): compound-scaled mobile
inverted bottleneck blocks with squeeze-and-excitation. The MLX modules were written after the
Hugging Face Transformers `efficientnet` reference implementation; module structure, parameter
names, and forward-pass order follow it so upstream and converted checkpoints load unchanged.

## Generic engine

```python
from docling_mlx.engines.image_classification.efficientnet import (
    EfficientNetEngine,
    EfficientNetEngineOptions,
    EfficientNetModelSpec,
)

engine = EfficientNetEngine(
    EfficientNetModelSpec(path="/path/to/checkpoint"),
    EfficientNetEngineOptions(top_k=5),
)
results = engine.predict(images)  # list[Classification]
```

`EfficientNetModelSpec` accepts either a local directory (`path`) or a Hub
repository (`repo_id`, with an optional `revision`). A checkpoint directory
contains `config.json`, `preprocessor_config.json`, and `model.safetensors`.
The loader accepts native MLX layouts and upstream Hugging Face safetensors;
the latter are renamed, convolution kernels are transposed from OIHW to OHWI,
and BatchNorm `num_batches_tracked` counters are discarded before
`load_weights(strict=True)`.

The model is built from the Hugging Face EfficientNet configuration rather
than a model profile. Width/depth coefficients, divisors, stage arrays,
padding indices, hidden dimensions, pooling, activation, BatchNorm settings,
image size/channels, and label mappings are read from the file. Unknown keys
are ignored, so B0 through B7 and compatible custom configurations share the
same implementation. `EfficientNetEngineOptions.dtype` controls inference
inputs; half-precision BatchNorm weights remain FP32 for numerical stability.

## Preprocessing

`EfficientNetImageProcessor` settings are read from `preprocessor_config.json`:
resize, resampling, center crop, rescale (including `rescale_offset`),
normalization, and `include_top`. The generic preprocessor matches the
Transformers Torchvision backend's fused float32 normalization order.

The upstream DocumentFigure snapshot identifies its processor as
`ViTImageProcessor`, and Docling's production predictor uses bilinear
`Resize(224)` followed by one `ToTensor`/`Normalize` pass. The DocumentFigure
preset intentionally keeps that behavior; applying EfficientNet's optional
`include_top` normalization would normalize twice and change predictions.

## DocumentFigure stage

`MlxDocumentPictureClassifierOptions` resolves the official
`document_figure_classifier_v2` preset unless an explicit `model_spec` overrides it. Its
`MlxImageClassificationEngineOptions` preserves the existing all-label Docling output.
`MlxDocumentPictureClassifier` inherits Docling's image preparation,
crop, metadata, provenance, and annotation behavior, and adapts generic
classification results to Docling prediction objects. Use `MlxStandardPdfPipeline` because Docling
has no picture-classifier plugin; call `configure()` directly only after constructing a standard
pipeline with picture classification disabled. Disabled stages do not import MLX or resolve a model.

## Conversion and validation

Conversion is source/repository driven:

```text
uv run --no-sync --group reference python -m tools.document_figure.convert_weights \
  --source /path/to/checkpoint --repo-id org/model --revision REV \
  --output .artifacts/org--model/REV
```

The converter writes a complete checkpoint directory. Keep it at the revision-qualified path above
when using `.artifacts` as an `artifacts_path` cache root; `DOCLING_MLX_ARTIFACTS` instead points
directly to the checkpoint directory for the parity lane. The converter uses the same sanitizer as
runtime, verifies a strict load and a finite forward, and publishes atomically without overwriting
an existing directory. See [`validation.md`](validation.md) for the recorded fixture and
official-checkpoint gates.
