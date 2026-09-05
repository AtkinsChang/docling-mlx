# RT-DETR-v2 validation

Phase A preserved the stored DPBench output baselines: Heron R50 and R101
produced identical Markdown and text on both 10-page slices (20/20 pages per
model). Loading each pinned upstream Heron safetensor checkpoint and its MLX
artifact produced bit-identical detections.

The presets resolve to these published artifacts, converted from the listed upstream revisions:

| Preset                 | Published MLX artifact                                                              | Converted from                                                                      |
| ---------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `layout_heron_default` | `atkinschang/docling-layout-heron-mlx@0e868578f271f8f6a6c907fb4f3aa723143d85f4`     | `docling-project/docling-layout-heron@8f39ad3c0b4c58e9c2d2c84a38465abf757272d8`     |
| `layout_heron_101`     | `atkinschang/docling-layout-heron-101-mlx@42b86329d788e310562a608190a1a5e54ef79bda` | `docling-project/docling-layout-heron-101@2e4993cf6bb211112084a2f80938f26138008917` |

**Adaptation source versus pinned reference.** The native RT-DETR-v2 modules and the shared detector
primitives were adapted from mlx-vlm 0.6.4 (commit
`2b909c68a2735914163ea7cb1bac690ec868484d`, `mlx_vlm/models/rt_detr_v2`), a tree unchanged in
currently locked mlx-vlm 0.6.17. Parity is qualified against
`transformers.RTDetrV2ForObjectDetection` from the pinned reference group.

Phase B validates `PekingU/rtdetr_v2_r18vd`, `r34vd`, `r50vd`, and `r101vd`
against `transformers.RTDetrV2ForObjectDetection` on CPU. The parity lane uses
the repository image fixture and three DPBench pages; every checkpoint retains
the same labels at threshold 0.3, scores within `1e-2`, matched-box IoU at
least `0.99`, and no unmatched detections.

Run the cached checkpoint checks with:

```console
uv run --no-sync pytest -m parity -q -rs tests/rt_detr_v2
```

The ordinary test lanes also cover strict artifact loading, conversion
sanitization, preprocessing, discrete sampling, and Docling's Heron adaptor.

## Backend comparison

Each row used a fresh batch-size-one process: one construction-plus-inference
warm-up call, then three timed rounds over all 200 DPBench page images. MLX
used Metal and the official Docling implementation used Torch MPS.

Machine: Apple M4 Pro, 48 GiB unified memory, macOS 26.5.2; Python 3.13.13;
Docling 2.126.0, docling-ibm-models 4.0.2, MLX 0.32.2, mlx-vlm 0.6.17, Torch
2.14.0, Transformers 5.16.1; measured on 2026-09-05 with `tools/compare_backends.py` schema 2 and
`MLX_ENABLE_TF32=0`.

Same-box label swaps are differently labelled matched detections for which MLX
also emitted the official label at IoU >= 0.9; they are excluded from label
agreement because score-order ties can pair duplicate query-by-label outputs
crosswise.

Threshold-boundary unmatched detections are unmatched boxes within 0.02 of the
recorded 0.3 score threshold; all other unmatched counts were zero.

The reduced pipeline's 200-page Markdown identity rate was 1.000000, so
neither class changed a document output.

### Heron R50

| implementation | device    | warm ms/item (median) | first-call ms | peak RSS | label agreement at IoU >= 0.5, excluding same-box swaps | same-box label swaps | matched-box mean IoU | mean abs score delta | threshold-boundary unmatched MLX / official | other unmatched MLX / official |
| -------------- | --------- | --------------------: | ------------: | -------: | ------------------------------------------------------: | -------------------: | -------------------: | -------------------: | ------------------------------------------- | ------------------------------ |
| mlx            | mlx-metal |                42.052 |       153.357 | 1.80 GiB |                                                1.000000 |                    0 |             0.999929 |             0.000203 | 1 / 0                                       | 0 / 0                          |
| official       | torch-mps |                69.296 |      2855.645 | 2.30 GiB |                                               reference |            reference |            reference |            reference | reference                                   | reference                      |

### Heron R101

| implementation | device    | warm ms/item (median) | first-call ms | peak RSS | label agreement at IoU >= 0.5, excluding same-box swaps | same-box label swaps | matched-box mean IoU | mean abs score delta | threshold-boundary unmatched MLX / official | other unmatched MLX / official |
| -------------- | --------- | --------------------: | ------------: | -------: | ------------------------------------------------------: | -------------------: | -------------------: | -------------------: | ------------------------------------------- | ------------------------------ |
| mlx            | mlx-metal |                65.633 |       572.849 | 2.06 GiB |                                                1.000000 |                    4 |             0.999926 |             0.000217 | 1 / 3                                       | 0 / 0                          |
| official       | torch-mps |                97.623 |      2859.845 | 2.31 GiB |                                               reference |            reference |            reference |            reference | reference                                   | reference                      |
