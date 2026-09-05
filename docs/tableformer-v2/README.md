# TableFormerV2 native MLX component

This component runs the pinned four-decoder-layer
[`docling-project/TableFormerV2`](https://huggingface.co/docling-project/TableFormerV2)
checkpoint on a project-owned FP32 MLX runtime. It provides a standalone engine and a formal
Docling table-structure plugin. It does not depend on Torch, Torchvision, Transformers, or
`docling-ibm-models` at runtime.

## Architecture and provenance

The TableFormerV2 architecture was derived by reading the docling-ibm-models `tableformer_v2`
reference implementation, the only specification of the pinned checkpoint, and re-implemented
in MLX: an EfficientNetV2-S image encoder (Tan and Le, 2021, arXiv:2104.00298), a transformer
OTSL tag decoder in the TableFormer design (Nassar et al., 2022, arXiv:2203.01017; OTSL from
Lysak et al., 2023, arXiv:2305.03393), and an attention-based cell bounding-box head. Module
structure, parameter names, and forward-pass order follow the reference so the published
checkpoint loads unchanged.

## Runtime flow

```text
Docling table factory
  → MlxTableFormerV2Model (Docling adaptor)
  → TableFormerV2Engine
  → native MLX TableFormerV2
      ├── EfficientNetV2-S encoder
      ├── four-layer autoregressive token decoder
      └── two-layer bbox decoder
```

The stage preserves Docling 2.126.0's table-selection, page-render, round-then-scale crop, OTSL,
span/header, text-matching, and coordinate contracts. It writes matched text to `TableCell.text`
directly rather than relying on the upstream V2 adaptor's legacy `bbox.token` compatibility path.
Debug table visualization is not implemented.

The engine converts each input to RGB, uses Pillow bilinear resize to 448×448, divides uint8 values
by FP32 `255.0`, and applies ImageNet normalization in NHWC layout. This matches the pinned
TableFormerV2 Torchvision PIL resize path; Figure and Heron use their own Pillow/NumPy preprocessing.

The image is encoded once, with only the EfficientNetV2-S feature extractor compiled. Greedy
generation starts at BOS and keeps a fixed-capacity projected K/V cache for each decoder layer.
Encoder-memory K/V is projected once per layer, while each token step projects only the current
Q/K/V and uses MLX fast scaled-dot-product attention. The loop is double buffered: every step
submits the next step with `mx.async_eval` before reading the current token back to the host, so
one step past the stop condition is computed and discarded. Generation stops at EOS or after 512
generated tokens, so a no-EOS result contains 513 IDs including BOS. The final full-sequence
decoder states feed the bbox head once; every generated data-cell token must have exactly one
normalized `xyxy` box. Its public result is pure data: token IDs, OTSL tokens, and crop-pixel
boxes. The stage alone maps those values to Docling cells, coordinates, and text.

## Convert the pinned checkpoint

Use the development reference group so Torch and the source implementation do not enter the runtime
dependency graph:

```bash
uv run --no-sync --group reference \
  python -m tools.tableformer_v2.convert_weights \
  --output .artifacts/org--model/REV
```

Use `--source /path/to/pinned-snapshot` for an already downloaded source. Conversion verifies the
fixed source revision and hashes, maps the FP32 inference tensors, explicitly accounts for ignored
BatchNorm training counters and two unused Torchvision classifier tensors, strictly loads the
converted weights in memory, runs finite nonzero forward/generation smokes, and publishes
atomically without overwriting an existing directory.

The converted repository root contains:

```text
model.safetensors
config.json
preprocessor_config.json
generation_config.json
special_tokens_map.json
tokenizer.json
tokenizer_config.json
README.md
```

The seven non-README files are required at runtime. The tokenizer files are copied verbatim from
the pinned source; `README.md` is the generated model card; runtime derives the closed token
contract from the tokenizer files and uses strict key/shape loading.

## Select the Docling plugin

The custom options default to the published converted repository:

```python
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
from docling_mlx.stages.table_structure_v2 import MlxTableStructureV2Options

options = ThreadedPdfPipelineOptions(
    artifacts_path=None,
    allow_external_plugins=True,
    do_table_structure=True,
    table_structure_options=MlxTableStructureV2Options(do_cell_matching=True),
)
```

With `artifacts_path=None`, the repo ID and the default revision pinned in
`src/docling_mlx/presets.py` (recorded in [`validation.md`](validation.md)) are passed to Hugging
Face. For offline use, set `artifacts_path=Path(".artifacts")` and cache the mirror at
`.artifacts/org--model/REV`. Runtime
accepts branch, tag, or commit revisions as Docling does; override the default in the model spec
when needed. `DOCLING_MLX_TABLEFORMER_V2_ARTIFACT` instead points directly to the complete
checkpoint directory.

The plugin callback returns Granite Vision first and TableFormerV2 second. Each custom options type
dispatches only to its own stage; Docling's built-in table stages remain registered. Plugin
discovery and callback execution do not import MLX, Torch, Torchvision, Transformers, or
`docling_ibm_models` and do not resolve an artifact.

## Standalone engine

```python
from PIL import Image
from docling_mlx.engines.table_structure.tableformer_v2 import (
    TableFormerV2Engine,
    TableFormerV2ModelSpec,
)
from docling_mlx.presets import resolve_preset

preset = resolve_preset("tableformer_v2")
engine = TableFormerV2Engine(
    TableFormerV2ModelSpec(repo_id=preset.repo_id, revision=preset.revision)
)
with Image.open("table.png") as image:
    prediction = engine.predict([image.convert("RGB")])[0]

print(prediction.token_ids)
print(prediction.cell_bboxes)
```

The initial release generates each image independently. `predict()` preserves input order but
does not claim batched autoregressive decoding. Empty input returns without initialization. Lazy
initialization is guarded once per engine; prediction calls are not serialized, and separate engines
may run concurrently.

See [validation.md](validation.md) for exact parity, real-fixture, offline pipeline, packaging,
concurrency, and performance evidence.
