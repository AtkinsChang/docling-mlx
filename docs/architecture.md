# Architecture

## Boundaries

Docling MLX separates native inference from Docling integration:

```text
stages/      Docling options, lifecycle, page/document adaptation
    ↓
presets.py   immutable mirror identities
    ↓
engines/     Docling-independent PIL-to-data inference
    ↓
_models/     native MLX model code
```

Nothing below `stages/` imports Docling. Generic engines accept a local checkpoint or a repository
identity and return pure detections, classifications, or table data. Their scope is the paper-level
model family expressed by the Hugging Face config (RT-DETR-v2, D-FINE, EfficientNet variants). A
stage owns Docling labels, page state, artifact resolution, and accelerator validation.

An engine is constructed once and its `predict()` runs on whichever pipeline thread Docling
schedules the stage on. MLX 0.32.2 already hands every thread its own default stream, so those
threads do not queue behind one another and no extra stream is needed. Sharing a compiled function
across them is still unsafe: MLX traces a compiled function on its first call for each input shape,
and a second thread inside the same compiled function observes tracer arrays and fails to evaluate
them with `[eval] Attempting to eval an array without a primitive`. Every compiled backbone is
therefore built through `_models/_compile.py`, which admits one thread at a time. The lock covers
one compiled call, not a prediction, so preprocessing, postprocessing, and TableFormer's
autoregressive decode loop still overlap; the trace state is per compiled function, so distinct
engines never contend. Warming a single shape at initialization would not do, because a later batch
of a new shape traces again. Serializing the compiled call changed no measured latency.

## Official Docling hooks

The package entry point exposes only the factories Docling provides (observed at 2.126.0):

```text
docling entry point
  layout_engines
    → MlxLayoutObjectDetectionModel
  table_structure_engines
    → MlxGraniteVisionTableStructureModel
    → MlxTableFormerV2Model
    → MlxTableFormerV1Model
```

Docling dispatches by the exact options type, so each plugin stage publishes a distinct options
class and `kind`: `mlx_layout_object_detection`, `mlx_tableformer`, `mlx_tableformer_v2`, and
`mlx_granite_vision_table`. Applications select those stages with
`allow_external_plugins=True`. The package never extends Docling's closed object-detection or
image-classification engine enums.

`MlxLayoutObjectDetectionOptions` has the official `model_spec` shape plus
`MlxObjectDetectionEngineOptions(score_threshold, dtype, warmup)`. It registers the official
layout preset IDs. The selected checkpoint's `config.json` chooses RT-DETR-v2 or D-FINE, so there
is one layout stage rather than one stage per model family. Engine overrides for non-MLX Docling
engines are ignored.

Table options inherit their corresponding official options: the official `mode` on
`MlxTableStructureOptions` selects the TableFormer v1 Accurate or Fast preset, and both v1 and v2
options carry `model_spec` and `engine_options` (including warmup). `MlxGraniteVisionTableStructureOptions`
uses Docling's VLM engine-options mixin. All table stages use `enable_remote_services: Literal[False]`.

## Hooks Docling does not provide

Docling (as of 2.126.0) directly constructs `DocumentPictureClassifier` and its chart stages; it
has no factory for either. `configure(pipeline: StandardPdfPipeline)` replaces those initialized
entries only when `picture_classification_options` is `MlxDocumentPictureClassifierOptions` or
`chart_extraction_options` is `MlxChartExtractionModelOptions`.

Use `MlxStandardPdfPipeline` in a `PdfFormatOption` rather than subclassing a Docling pipeline in
an application. It calls the official initialization and then `configure()`. Because Docling creates
the picture classifier before `StandardPdfPipeline._init_models`, the package temporarily disables
the MLX picture/chart pair for that bootstrap, restores the original options, and performs the
single replacement. Call `configure()` directly only on a standard pipeline constructed with the
MLX-owned picture and chart stages disabled; enabling those options during plain construction fails
inside Docling before `configure()` can run. After bootstrap, `configure()` only recomputes
Docling's five-flag `keep_backend` expression; nothing else from Docling's pipeline is copied.

## Lifecycle

Enabled native layout, picture, and table stages validate `auto`/`mps` at construction and call
`engine.initialize()`. `warmup=True` additionally uses the engine warmup path. Disabled stages do
not resolve an artifact or initialize MLX. Direct generic engine `predict([])` is still lazy.

Artifact lookup follows Docling's revision directory first and flat-cache fallback second. Presets
are the sole catalog of converted mirror IDs and immutable revisions; explicit `model_spec` values
can point at compatible upstream checkpoints.
