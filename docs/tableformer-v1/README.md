# TableFormerV1

Docling MLX separates TableFormerV1 into a generic engine, pure preset data, and a Docling
adaptor. Accurate and Fast are two checkpoint presets for the same engine:

- `TableFormerV1ModelSpec`, `TableFormerV1EngineOptions`, and `TableFormerV1Engine` are
  Docling-independent and accept Pillow table crops.
- `tableformer_v1_accurate` and `tableformer_v1_fast` live in `docling_mlx.presets`.
- `MlxTableStructureOptions` and `MlxTableFormerV1Model` are the Docling adaptor.

Install the OpenCV dependency only when using this component:

```bash
uv run --no-sync --extra tableformer-v1 python -c 'import cv2'
```

The extra pins `opencv-python==5.0.0.93`. The base wheel remains OpenCV-free, and installing the
extra does not download a model.

## Architecture and provenance

TableFormer pairs a convolutional image encoder with a transformer structure decoder and an
attention-based cell bounding-box head (Nassar et al., 2022, arXiv:2203.01017). These profiles use
a ResNet-18 encoder (He et al., 2015, arXiv:1512.03385) and emit OTSL tags (Lysak et al., 2023,
arXiv:2305.03393). The MLX modules were written after the docling-ibm-models
`tableformer/models/table04_rs` reference implementation and Torchvision's ResNet; module
structure, parameter names, and forward-pass order follow them so the published checkpoints load
unchanged.

## Local artifact and publication boundary

Convert from a local pinned source snapshot:

```bash
uv run --no-sync --group reference --extra standard --extra tableformer-v1 \
  python -m tools.tableformer_v1.convert_weights \
  --source /path/to/pinned-tableformer-v1-snapshot \
  --output .artifacts/org--model/REV
```

The artifact directory must contain:

```text
├── accurate/
│   ├── model.safetensors
│   ├── config.json
│   ├── generation_config.json
│   └── preprocessor_config.json
└── fast/
    ├── model.safetensors
    ├── config.json
    ├── generation_config.json
    └── preprocessor_config.json
```

For offline use, set `artifacts_path=Path(".artifacts")` and keep the mirror at
`.artifacts/org--model/REV`. `src/docling_mlx/presets.py` pins the published mirror and its
default revision, and [`validation.md`](validation.md) records them. Set `artifacts_path=None` to
resolve it remotely, or override the revision in the model spec when needed.
`DOCLING_MLX_TABLEFORMER_V1_ARTIFACT` instead
points directly to this complete v1 artifact directory. The stage resolves the Accurate preset by
default; `TableFormerMode.FAST` selects the Fast preset.

## Docling plugin

```python
from docling.datamodel.pipeline_options import TableFormerMode, ThreadedPdfPipelineOptions
from docling_mlx.stages.table_structure_v1 import MlxTableStructureOptions

options = ThreadedPdfPipelineOptions(
    artifacts_path=None,
    allow_external_plugins=True,
    do_table_structure=True,
    table_structure_options=MlxTableStructureOptions(
        mode=TableFormerMode.FAST,
        do_cell_matching=True,
    ),
)
```

Omit `mode` or set `TableFormerMode.ACCURATE` for the official Accurate default. The
selected profile's `config.json` carries its topology: Accurate is encoder/decoder 6/6 and Fast is
4/2. The table callback preserves Docling built-ins and returns project stages in
this order: Granite Vision, TableFormerV2, then TableFormerV1. Plugin discovery is lazy: it does
not import OpenCV, MLX, Torch, Torchvision, Transformers, or `docling-ibm-models`; the engine is
created only after Docling selects `MlxTableStructureOptions`.

For application-owned crops, construct the generic engine directly:

```python
from docling_mlx.engines.table_structure.tableformer_v1 import (
    TableFormerV1Engine,
    TableFormerV1EngineOptions,
    TableFormerV1ModelSpec,
)

engine = TableFormerV1Engine(
    TableFormerV1ModelSpec(path="/path/to/tableformer-v1"),
    TableFormerV1EngineOptions(checkpoint_subdirectory="accurate"),
)
predictions = engine.predict(table_crops)
```

Each prediction contains OTSL tokens and crop-pixel cell boxes. Only the stage accesses a page
backend or converts those values to `TableCell` objects. The engine requires Apple Silicon when
initialized; the stage also validates Docling accelerator options.
Runtime initialization compiles only the ResNet-18 image backbone; tag encoding, autoregressive
decoding, and bbox decoding remain eager. The greedy loop is double buffered: the structural
`xcel`/`lcel`/`fcel` correction runs on device so every step can submit the next step with
`mx.async_eval` before reading the current token back to the host, and one step past the stop
condition is computed and discarded.

## Exact upstream preprocessing and structure compatibility

The V1 path keeps the source OpenCV semantics exactly. The stage first obtains Docling's scale-2
page render, converts it to an array, and resizes it to 1024 pixels high with `cv2.INTER_AREA`; the
new width is `int(width * 1024 / height)`. It rounds table coordinates, scales them by two, applies
the page-resize scale, rounds again, and crops the resized page as RGB.

For each crop, it converts RGB values to FP32, subtracts `255 * mean`
(`0.94247851`, `0.94254675`, `0.94292611`), divides by std (`0.17910956`, `0.17940403`,
`0.17931663`), resizes to 448×448 with `cv2.INTER_LINEAR`, transposes HWC to CWH, divides by 255,
and adds the batch dimension. CWH is intentional and differs from conventional CHW.

The source-compatible HTML round trip recognizes spans only through 20. When decoded OTSL would
produce a colspan or rowspan above 20, postprocessing uses 1, matching
`docling-ibm-models==4.0.2` behavior.

## Qualification

See [validation.md](validation.md). Raw captures, generated qualification JSON, and benchmark
samples remain local and ignored.
