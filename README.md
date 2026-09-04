# Docling MLX

Docling MLX provides native MLX engines and Docling stage adapters for Apple Silicon; it requires
Docling 2.124 or newer.
It is a community project, not an official Docling or IBM project.

## Docling integration

| Component                             | Public option class                     | Selection                                                              |
| ------------------------------------- | --------------------------------------- | ---------------------------------------------------------------------- |
| RT-DETR-v2 and D-FINE layout          | `MlxLayoutObjectDetectionOptions`       | Docling layout plugin                                                  |
| TableFormer v1                        | `MlxTableStructureOptions`              | Docling table plugin                                                   |
| TableFormer v2                        | `MlxTableStructureV2Options`            | Docling table plugin                                                   |
| Granite Vision table                  | `MlxGraniteVisionTableStructureOptions` | Docling table plugin                                                   |
| DocumentFigure picture classification | `MlxDocumentPictureClassifierOptions`   | `MlxStandardPdfPipeline` (or `configure()` after a disabled bootstrap) |
| Granite Vision chart extraction       | `MlxChartExtractionModelOptions`        | `MlxStandardPdfPipeline` (or `configure()` after a disabled bootstrap) |

Set `allow_external_plugins=True` for the plugin-selected layout and table stages. Docling has no
picture-classification or chart-extraction factory, so `MlxStandardPdfPipeline` is the normal path;
it is the only pipeline subclass in this package and installs those two stages after Docling's
normal initialization. Call `configure()` directly only after constructing a standard pipeline
with the MLX-owned picture and chart stages disabled. The generic engines remain usable directly
and do not import Docling.

Enabled native stages validate only `auto` and `mps` accelerator selections and initialize their
engine in the constructor. `warmup=True` additionally performs the engine's warmup path. Generic
engine `predict()` remains lazy for direct users.

## Quick start

Run from a provisioned environment with `uv run --no-sync`:

```python
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from docling_mlx.pipeline import MlxStandardPdfPipeline
from docling_mlx.stages.layout import (
    MlxLayoutObjectDetectionOptions,
    MlxObjectDetectionEngineOptions,
)
from docling_mlx.stages.picture_classification import MlxDocumentPictureClassifierOptions
from docling_mlx.stages.table_structure_v2 import MlxTableStructureV2Options

options = ThreadedPdfPipelineOptions(
    artifacts_path=Path(".artifacts"),
    allow_external_plugins=True,
    do_ocr=False,
    layout_options=MlxLayoutObjectDetectionOptions.from_preset(
        "layout_heron_default", engine_options=MlxObjectDetectionEngineOptions()
    ),
    do_table_structure=True,
    table_structure_options=MlxTableStructureV2Options(),
    do_picture_classification=True,
    picture_classification_options=MlxDocumentPictureClassifierOptions(),
)
result = DocumentConverter(
    allowed_formats=[InputFormat.PDF],
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=MlxStandardPdfPipeline, pipeline_options=options
        )
    },
).convert("input.pdf")
print(result.document.export_to_markdown())
```

`examples/std_pdf_pipeline_all_mlx.py` and `examples/mlx_pipeline/pipeline.py` use this same
shape. The latter adds OCRMac and a small JSON summary; it does not copy or subclass a Docling
pipeline.

## Presets and artifacts

| Preset ID                                         | Published MLX mirror                            |
| ------------------------------------------------- | ----------------------------------------------- |
| `layout_heron_default`                            | `atkinschang/docling-layout-heron-mlx`          |
| `layout_heron_101`                                | `atkinschang/docling-layout-heron-101-mlx`      |
| `layout_egret_medium`                             | `atkinschang/docling-layout-egret-medium-mlx`   |
| `layout_egret_large`                              | `atkinschang/docling-layout-egret-large-mlx`    |
| `layout_egret_xlarge`                             | `atkinschang/docling-layout-egret-xlarge-mlx`   |
| `document_figure_classifier_v2`                   | `atkinschang/DocumentFigureClassifier-v2.5-MLX` |
| `tableformer_v1_accurate` / `tableformer_v1_fast` | `atkinschang/TableFormer-MLX`                   |
| `tableformer_v2`                                  | `atkinschang/TableFormerV2-MLX`                 |

The layout and figure rows use Docling-official IDs; the three table rows are project IDs.
`src/docling_mlx/presets.py` pins each mirror to an immutable commit; each component's
`validation.md` records that commit and the upstream source revision it was converted from.

For an offline stage, `artifacts_path` is a Docling cache root. Resolution first uses
`<repo-id-with-slashes-replaced-by-->/<revision>/`, then the legacy flat repository directory.
The model spec may instead name a compatible upstream Hugging Face checkpoint. The
`DOCLING_MLX_*_ARTIFACT` lane variables point directly to complete checkpoint directories, while
the `DOCLING_MLX_*_SOURCE` variables name source snapshots for conversion.

Converted weights are separate artifacts. Mirror provenance records the immutable upstream source
revision; it is not a license claim. Follow the source repository's terms.

## Development

See [DEVELOPMENT.md](https://github.com/AtkinsChang/docling-mlx/blob/main/DEVELOPMENT.md) for development commands, source and artifact gates, and
[docs/architecture.md](https://github.com/AtkinsChang/docling-mlx/blob/main/docs/architecture.md) for the boundary between plugins and `configure()`.

## Performance

One fresh batch-size-one process per implementation performed one
construction-plus-inference warm-up and three timed rounds over all 200
DPBench PDFs through the reduced standard pipeline (layout, table structure,
and picture classification only). MLX used Metal and official Docling used
Torch MPS with equivalent Heron-default layout and TableFormer settings.

Machine: Apple M4 Pro, 48 GiB unified memory, macOS 26.5.2; Python 3.13.13;
Docling 2.124.0, docling-ibm-models 4.0.1, MLX 0.32.2, mlx-vlm 0.6.4, Torch
2.14.0, Transformers 5.8.1; measured on 2026-09-03 with `tools/compare_backends.py` schema 2.

| implementation | device    | warm ms/item (median) | first-call ms | peak RSS | markdown identity | layout cluster agreement at IoU >= 0.5 | table structure exact | unmatched layout MLX / official | unmatched tables MLX / official |
| -------------- | --------- | --------------------: | ------------: | -------: | ----------------: | -------------------------------------: | --------------------: | ------------------------------- | ------------------------------- |
| mlx            | mlx-metal |               107.476 |      3459.828 | 1.79 GiB |          1.000000 |                               0.999494 |              1.000000 | 1 / 1                           | 0 / 0                           |
| official       | torch-mps |               148.836 |      3557.298 | 2.25 GiB |         reference |                              reference |             reference | reference                       | reference                       |

Regenerate this matrix with [`tools/compare_backends.py`](https://github.com/AtkinsChang/docling-mlx/blob/main/DEVELOPMENT.md#qualification-and-tools).
