# Docling MLX

Native MLX engines and Docling stage adaptors for Apple Silicon. Docling MLX is a community
project, not an official Docling or IBM project. It requires Python 3.13 or newer and Docling
2.124 or newer.

## Install

```bash
uv add "docling-mlx[standard]"
```

MLX itself is installed only on macOS arm64; elsewhere the package imports but its engines cannot
run. The extras select how much of Docling comes with it:

| Extra            | Adds                                 | Needed for                                    |
| ---------------- | ------------------------------------ | --------------------------------------------- |
| (none)           | `docling-slim[convert-core]`         | the layout, TableFormerV2, and picture stages |
| `standard`       | the full `docling` distribution      | Docling's default OCR and enrichment stacks   |
| `vlm`            | `docling[vlm]`, which brings mlx-vlm | the Granite Vision table and chart stages     |
| `tableformer-v1` | `opencv-python`                      | TableFormer v1 preprocessing                  |

## What it provides

| Stage                                      | Option class                            | Enabled through          |
| ------------------------------------------ | --------------------------------------- | ------------------------ |
| Layout: RT-DETR-v2 (Heron), D-FINE (Egret) | `MlxLayoutObjectDetectionOptions`       | Docling layout plugin    |
| Table structure: TableFormer v1            | `MlxTableStructureOptions`              | Docling table plugin     |
| Table structure: TableFormerV2             | `MlxTableStructureV2Options`            | Docling table plugin     |
| Table structure: Granite Vision            | `MlxGraniteVisionTableStructureOptions` | Docling table plugin     |
| Picture classification: DocumentFigure     | `MlxDocumentPictureClassifierOptions`   | `MlxStandardPdfPipeline` |
| Chart extraction: Granite Vision           | `MlxChartExtractionModelOptions`        | `MlxStandardPdfPipeline` |

- The plugin stages need `allow_external_plugins=True` in the pipeline options.
- Docling has no picture-classification or chart-extraction factory. `MlxStandardPdfPipeline`, the
  only pipeline subclass in this package, installs those two stages after Docling's normal
  initialization. `configure()` does the same for a standard pipeline that was constructed with the
  MLX-owned picture and chart stages disabled.
- Enabled stages accept only the `auto` and `mps` accelerator selections and initialize their
  engine in the constructor; `warmup=True` additionally runs the engine's warmup path.
- The engines under `docling_mlx.engines` do not import Docling, and their `predict()` stays lazy
  for direct users.

## Quick start

Convert a PDF with the Heron layout, TableFormerV2, and picture classification stages:

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

[`examples/std_pdf_pipeline_all_mlx.py`](https://github.com/AtkinsChang/docling-mlx/blob/main/examples/std_pdf_pipeline_all_mlx.py) and
[`examples/mlx_pipeline/pipeline.py`](https://github.com/AtkinsChang/docling-mlx/blob/main/examples/mlx_pipeline/pipeline.py) use this same shape.
The latter adds OCRMac and a small JSON summary; it does not copy or subclass a Docling pipeline.

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

- The layout and figure rows use Docling-official preset IDs; the three table rows are project IDs.
- `src/docling_mlx/presets.py` pins each mirror to an immutable commit, and each component's
  `validation.md` records that commit and the upstream source revision it was converted from.
- For an offline stage, `artifacts_path` is a Docling cache root. Resolution first uses
  `<repo-id-with-slashes-replaced-by-->/<revision>/`, then the legacy flat repository directory.
  The model spec may instead name a compatible upstream Hugging Face checkpoint. The
  `DOCLING_MLX_*_ARTIFACT` lane variables point directly to complete checkpoint directories, while
  the `DOCLING_MLX_*_SOURCE` variables name source snapshots for conversion.
- Converted weights are separate artifacts. Mirror provenance records the immutable upstream source
  revision; it is not a license claim. Follow the source repository's terms.

## Performance

Warm median latency per item on DPBench, one fresh batch-size-one process per implementation,
MLX on Metal against the official Docling stage. The layout, figure, and TableFormer outputs
match the official implementations on the same inputs (identical labels, top-1 classes, and OTSL
sequences); the Granite outputs differ within BF16 kernel noise. Each component's `validation.md`
records the full comparison.

| Component                     | Item       | MLX (Metal) | Official | Official device |
| ----------------------------- | ---------- | ----------: | -------: | --------------- |
| Heron R50 layout              | page       |     42.1 ms |  69.3 ms | Torch MPS       |
| Heron R101 layout             | page       |     65.6 ms |  97.6 ms | Torch MPS       |
| Egret medium layout           | page       |     30.1 ms |  46.1 ms | Torch MPS       |
| Egret large layout            | page       |     40.0 ms |  59.3 ms | Torch MPS       |
| Egret xlarge layout           | page       |     61.8 ms |  86.3 ms | Torch MPS       |
| DocumentFigure classification | picture    |      3.4 ms |  12.2 ms | Torch MPS       |
| TableFormer v1 accurate       | table      |     81.1 ms | 154.1 ms | Torch MPS       |
| TableFormer v1 fast           | table      |     40.3 ms |  78.3 ms | Torch MPS       |
| TableFormerV2                 | table      |     36.5 ms | 146.0 ms | Torch MPS       |
| Granite Vision table          | table crop |      15.4 s |   70.9 s | Torch CPU       |
| Granite Vision chart          | chart crop |      18.7 s |   73.5 s | Torch CPU       |

The pipeline tables report the mean per page and the timed-round total rather than the median:
only pages with tables or charts run the table and Granite stages, so the median page carries
none of that work.

The reduced standard pipeline (layout, table structure, and picture classification only) over all
200 DPBench PDFs, one construction-plus-inference warm-up and three timed rounds, with equivalent
Heron-default layout and TableFormer settings:

| implementation | device    | warm ms/page (mean) | timed round s | first-call ms | peak RSS | markdown identity | layout cluster agreement at IoU >= 0.5 | table structure exact |
| -------------- | --------- | ------------------: | ------------: | ------------: | -------: | ----------------: | -------------------------------------: | --------------------: |
| mlx            | mlx-metal |             112.364 |          22.5 |      2602.021 | 1.72 GiB |          1.000000 |                               0.999494 |              1.000000 |
| official       | torch-mps |             203.526 |          40.7 |      3191.657 | 2.18 GiB |         reference |                              reference |             reference |

The full standard pipeline with Granite Vision table structure and chart extraction over the first
50 DPBench PDFs, one construction-plus-inference warm-up and one timed round because Docling's
official Granite stages run on the CPU on macOS; each side decoded 6 table crops and 14 chart
crops, and markdown identity is below 1.0 because Granite differs within BF16 noise:

| implementation | device        | warm ms/page (mean) | timed round s | first-call ms |  peak RSS | markdown identity | layout cluster agreement at IoU >= 0.5 | table structure exact |
| -------------- | ------------- | ------------------: | ------------: | ------------: | --------: | ----------------: | -------------------------------------: | --------------------: |
| mlx            | mlx-metal     |            2769.909 |         138.5 |      3855.916 | 16.03 GiB |          0.980000 |                               1.000000 |              1.000000 |
| official       | torch-mps+cpu |           14331.059 |         716.6 |      5237.575 | 17.80 GiB |         reference |                              reference |             reference |

Machine: Apple M4 Pro, 48 GiB unified memory, macOS 26.5.2; Python 3.13.13;
Docling 2.126.0, docling-ibm-models 4.0.2, MLX 0.32.2, mlx-vlm 0.6.17, Torch
2.14.0, Transformers 5.16.1; measured on 2026-09-05 with `tools/compare_backends.py` schema 2 and
`MLX_ENABLE_TF32=0`.

Regenerate the tables with [`tools/compare_backends.py`](https://github.com/AtkinsChang/docling-mlx/blob/main/DEVELOPMENT.md#qualification-and-tools).

## Documentation

- [DEVELOPMENT.md](https://github.com/AtkinsChang/docling-mlx/blob/main/DEVELOPMENT.md): development commands, qualification lanes, and artifact
  staging.
- [docs/architecture.md](https://github.com/AtkinsChang/docling-mlx/blob/main/docs/architecture.md): the boundary between engines, stages, plugins,
  and `configure()`.
- Component guides and validation records: [layout Heron](https://github.com/AtkinsChang/docling-mlx/blob/main/docs/layout-heron/README.md),
  [layout Egret](https://github.com/AtkinsChang/docling-mlx/blob/main/docs/layout-egret/README.md),
  [DocumentFigure](https://github.com/AtkinsChang/docling-mlx/blob/main/docs/document-figure/README.md),
  [TableFormer v1](https://github.com/AtkinsChang/docling-mlx/blob/main/docs/tableformer-v1/README.md),
  [TableFormerV2](https://github.com/AtkinsChang/docling-mlx/blob/main/docs/tableformer-v2/README.md), and
  [Granite Vision](https://github.com/AtkinsChang/docling-mlx/blob/main/docs/granite-vision/README.md).
- [CONTRIBUTING.md](https://github.com/AtkinsChang/docling-mlx/blob/main/CONTRIBUTING.md) and the commit scopes in
  [docs/commits.md](https://github.com/AtkinsChang/docling-mlx/blob/main/docs/commits.md).
