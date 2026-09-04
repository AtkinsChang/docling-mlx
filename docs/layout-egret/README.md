# Egret layout and generic D-FINE

The D-FINE implementation is a framework-free MLX object-detection engine. It
loads a local directory or Hugging Face model identity and exposes
`DFineEngine.predict(images) -> list[Detections]`; each result has pixel
`xyxy` boxes, scores, label IDs, and the checkpoint's `id2label` mapping.
The Docling Egret layout class is only an adapter around that API.

| Topic                       | Authoritative location                   |
| --------------------------- | ---------------------------------------- |
| Plugin use and installation | [`README.md`](../../README.md)           |
| Developer commands          | [`DEVELOPMENT.md`](../../DEVELOPMENT.md) |
| Qualification evidence      | [`validation.md`](validation.md)         |

## Architecture and provenance

D-FINE recasts DETR box regression as fine-grained distribution refinement (Peng et al., 2024,
arXiv:2410.13842); its hybrid encoder and query decoder descend from RT-DETR (Zhao et al., 2023,
arXiv:2304.08069), and the HGNetV2 backbone comes from PaddleClas PP-HGNetV2. The MLX modules
were written after the Hugging Face Transformers `d_fine` and `hgnet_v2` reference implementations:
module structure, parameter names, and forward-pass order follow them so upstream and converted
checkpoints load unchanged.

## Checkpoints and configuration

`config.json` constructs the model. Supported inference configuration is the
Transformers D-FINE HGNetV2 family: its stem/stage channel and block fields,
selected backbone outputs, encoder dimensions and AIFI settings, decoder
layers/queries/feature levels, scalar or per-level sampling points, both
`default` and `discrete` decoder methods, labels, focal or softmax output
postprocessing, and preprocessing declared by `preprocessor_config.json`.
Unknown Transformer keys are ignored.

The runtime accepts either an MLX-layout safetensor artifact or an upstream HF
safetensor. Upstream state is sanitized without Torch: keys are renamed,
convolution weights are transposed from OIHW to OHWI, and only paired BatchNorm
counters plus the evaluation-unreachable denoising embedding are dropped.
Every remaining tensor must match the constructed model exactly. `dtype` is a
runtime option; FP16 is supported but has no numerical-parity claim.

The three Egret presets, `layout_egret_medium`, `layout_egret_large`, and `layout_egret_xlarge`,
remain application data, not engine topology profiles; `src/docling_mlx/presets.py` pins each to
its published mirror commit, and [`validation.md`](validation.md) records those commits with the
upstream source revisions they were converted from.

Use these IDs with `MlxLayoutObjectDetectionOptions.from_preset(...,
engine_options=MlxObjectDetectionEngineOptions())`; no legacy aliases are provided.

## Conversion

The converter is source-driven and never downloads or publishes weights:

```bash
uv run --no-sync python -m tools.layout_egret.convert_weights \
  --source /path/to/snapshot \
  --repo-id org/model \
  --revision REV \
  --output .artifacts/org--model/REV
```

Keep this directory at the revision-qualified location when using `.artifacts` as an
`artifacts_path` cache root. `DOCLING_MLX_EGRET_MEDIUM_ARTIFACT`,
`DOCLING_MLX_EGRET_LARGE_ARTIFACT`, and `DOCLING_MLX_EGRET_XLARGE_ARTIFACT` instead point directly
to complete checkpoint directories. The converter writes `model.safetensors`, preserves
`config.json` and `preprocessor_config.json`, and generates a provenance README.
