# Granite Vision 4.1 adaptors

Docling MLX uses the unchanged official `ibm-granite/granite-vision-4.1-4b` artifact through its
internal corrected `MlxGraniteVision41Engine`, a thin subclass of Docling's `MlxVlmEngine`. The
subclass corrects the Granite tanh GELU activation, torchvision processor backend, and image-first
chat-template boundary while inheriting Docling's generation, global coordination lock, and
lifecycle. It does not port the Granite architecture or convert or republish weights. Docling's
official Transformers Granite stages pass `tokenizer.model_max_length` (a `1e30` sentinel,
effectively unbounded), while the MLX stages use Docling's `VlmModelSpec` default of 4096 new
tokens, adjustable through `model_spec.max_new_tokens`.

## Model provenance

The architecture, weights, and evaluation are the vendor's. See the
[model card](https://huggingface.co/ibm-granite/granite-vision-4.1-4b), which cites ChartNet
(Kondic et al., 2026, arXiv:2603.27064) for the methodology and data behind this release.

## Why not mlx-vlm's Granite Vision as-is

mlx-vlm has `granite4_vision` and Docling wraps it in `MlxVlmEngine`; both are reused unchanged
for loading, generation, and locking. A thin subclass fixes three places where mlx-vlm diverges
from the official Transformers path, each verified against it:

- **Vision tower activation.** The config declares `gelu_pytorch_tanh`; mlx-vlm uses plain GELU.
  All 27 SigLIP encoder layers and the head MLP are switched to tanh GELU.
- **Image processor.** mlx-vlm uses its PIL processor; the official path uses Transformers'
  torchvision backend. The torchvision processor is installed instead, after which image tensors
  match the official ones exactly.
- **Chat template.** Messages are built image-first, matching Docling's prompt byte-for-byte.

Without these the model is not the same model. Docling's own Granite table and chart stages are
Transformers-only and CPU-only on macOS, which is why MLX stages exist here.

The default model specs pin revision
`dd48e97503de471803850df70843cf9eb5da8712`. The official artifact is BF16 and remains unchanged.
The wheel contains neither weights nor model configuration files.

Speed: on the recorded M4 Pro run, MLX was 4.58x faster per table crop and 4.77x faster per chart
crop than official Torch CPU; see [validation.md](validation.md).

## Responsibility boundary

```text
TableStructureFactory                    application enrichment pipe
  → MlxGraniteVisionTableStructureModel    → MlxGraniteVisionChartExtractionModel
                    └──────────────┬──────────────┘
                                   ↓
                    MlxGraniteVision41Engine
                    (Docling MlxVlmEngine subclass)
                                   ↓
              ibm-granite/granite-vision-4.1-4b
```

The adaptors own the Docling-facing work:

- select and crop table or chart items;
- build task-specific prompts and preserve batch order;
- parse OTSL, CSV, and Python responses;
- update `TableStructurePrediction` and `PictureMeta`;
- preserve stage enablement, accelerator, and engine-ownership contracts.

Enabled table stages, and chart stages with active prompts, create and initialize the corrected
engine during construction. Docling continues to provide model resolution, MLX model loading, token
generation, global coordination locking, and engine cleanup for internally created stages; the
internal subclass supplies only the three Granite boundary corrections above.

## Table structure plugin

`MlxGraniteVisionTableStructureModel` implements Docling's table factory contract and registers a
unique `MlxGraniteVisionTableStructureOptions` type. It does not replace Docling's built-in Granite
table options.

```python
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
from docling_mlx.stages.table_structure import (
    MlxGraniteVisionTableStructureOptions,
)

pipeline_options = ThreadedPdfPipelineOptions(
    allow_external_plugins=True,
    do_table_structure=True,
    table_structure_options=MlxGraniteVisionTableStructureOptions(),
)
```

The stage selects `TABLE` and `DOCUMENT_INDEX` layout clusters, crops at scale 1, sends one
`<tables_otsl>` request per crop, and parses closed, open, and self-closing OTSL tokens into table
cells and row/column spans. Invalid pages and missing table clusters produce empty predictions
without a model call. Malformed model output is isolated to that table.

Plugin discovery and the `table_structure_engines()` callback do not import MLX, Torch, or
Transformers and do not resolve an artifact. Model loading begins only when Docling constructs an
enabled selected stage.

## Chart extraction component

Docling (as of 2.124.0) has no chart factory. `MlxStandardPdfPipeline` places
`MlxGraniteVisionChartExtractionModel` after picture classification without application subclassing;
call `configure()` directly only after constructing a standard pipeline with the MLX-owned picture
and chart stages disabled.

```python
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling_mlx.stages.chart_extraction import (
    MlxGraniteVisionChartExtractionModel,
    MlxChartExtractionModelOptions,
)

stage = MlxGraniteVisionChartExtractionModel(
    enabled=True,
    artifacts_path=Path(".artifacts/granite-vlm"),
    options=MlxChartExtractionModelOptions(
        chart2csv=True,
        chart2summary=True,
        chart2code=True,
    ),
    accelerator_options=AcceleratorOptions(device="auto"),
)
```

With no injected engine, the enabled chart stage creates and initializes its corrected
`MlxGraniteVision41Engine` during construction when prompts are active.

The inherited enrichment preparation contract supplies scale-2 images. The component accepts only
`PictureItem` instances whose main classification is `bar_chart`, `pie_chart`, or `line_chart`.
For each image it sends active prompts in CSV, summary, then code order and writes:

- `<chart2csv>` to `TabularChartMetaField`;
- `<chart2summary>` to `DescriptionMetaField`;
- `<chart2code>` to `CodeMetaField(language=PYTHON)`.

One parser failure skips only that result. Unparsable Granite responses are caught broadly and
logged; they yield an empty table or leave chart metadata unchanged, matching Docling 2.124.0, and
there is no option to make them raise. The Python parser accepts Docling's fenced format and
syntactically valid bare Python because the pinned official model can emit bare code. Arbitrary
prose and non-Python fences remain rejected.

## Artifact resolution

Install the VLM extra:

```bash
uv run --no-sync --extra vlm python -c 'import docling'
```

With `artifacts_path=None`, Docling downloads the pinned official repository. Local offline use
follows Docling's repository-folder convention:

```text
<artifacts_path>/ibm-granite--granite-vision-4.1-4b/
```

This is a resolver root layout, not a direct checkpoint path; set
`DOCLING_MLX_GRANITE_VISION_ARTIFACTS_ROOT` to the root containing that directory.

The base wheel and `[standard]` extra do not add a Granite or `mlx-vlm` dependency. The `[vlm]`
extra depends only on `docling[vlm]>=2.124`; current transitive versions are recorded in
`uv.lock`, not duplicated as production pins.

## Sharing one loaded model

A plugin-created table stage owns its corrected engine. To share one model between table and chart
work, the application must construct the corrected engine explicitly and inject it into both stages:

```python
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.models.inference_engines.vlm.base import VlmEngineType
from docling_mlx.stages.granite_vision_engine import MlxGraniteVision41Engine
from docling_mlx.stages.chart_extraction import (
    MlxGraniteVisionChartExtractionModel,
    MlxChartExtractionModelOptions,
)
from docling_mlx.stages.table_structure import (
    MlxGraniteVisionTableStructureModel,
    MlxGraniteVisionTableStructureOptions,
)

accelerator = AcceleratorOptions(device="auto")
table_options = MlxGraniteVisionTableStructureOptions()
chart_options = MlxChartExtractionModelOptions()
engine = MlxGraniteVision41Engine(
    table_options.engine_options,
    artifacts_path=None,
    model_config=table_options.model_spec.get_engine_config(VlmEngineType.MLX),
)
table_stage = MlxGraniteVisionTableStructureModel(
    True, None, table_options, accelerator, engine=engine
)
chart_stage = MlxGraniteVisionChartExtractionModel(
    True, None, chart_options, accelerator, engine=engine
)
try:
    ...
finally:
    table_stage.cleanup()
    chart_stage.cleanup()
    engine.cleanup()
```

The stages validate the injected engine's type, options, repository, and revision. Neither stage
cleans up an injected engine. The application owns final cleanup.

See [validation.md](validation.md) for the task, packaging, and performance evidence.
