# D-FINE validation

## Egret regression evidence

The presets resolve to these published artifacts, converted from the listed upstream revisions:

| Preset                | Published MLX artifact                                                                 | Converted from                                                                         |
| --------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `layout_egret_medium` | `atkinschang/docling-layout-egret-medium-mlx@a500e62df89b586b716111709ec8626aa28072c0` | `docling-project/docling-layout-egret-medium@77ede7cc7bed96d853c58f319734803d6ea2ea5c` |
| `layout_egret_large`  | `atkinschang/docling-layout-egret-large-mlx@3f75c0befdd32a4ba4c1f42720cfcf95e3be04db`  | `docling-project/docling-layout-egret-large@fff417c78abd6bab338c87706c95a8d79dc68f1e`  |
| `layout_egret_xlarge` | `atkinschang/docling-layout-egret-xlarge-mlx@0df67dc77ff5794e5ebc75adc2bff8b75c08e2b2` | `docling-project/docling-layout-egret-xlarge@23857d16596e0106716b3162d132212d733769e7` |

**Reference implementation versus pinned reference.** The native D-FINE and HGNetV2 modules were
written after Hugging Face Transformers 5.8.1 (commit
`cc832f9055ba11c8c55f918ab4bda9472b910d48`, `transformers/models/d_fine`). The `d_fine` changes
through currently pinned 5.16.1, the parity oracle, are neutral for these configurations: an
MPS-only position-embedding dtype, a feature-level branch that three-level configs never take, and
cache and transpose refactors.

Phase A exercised the generic engine against the stored Egret-medium DPBench
baselines using the no-op native-path harness:

| Run            | Pages | Identical markdown | Identical text | Unmatched layout / tables |
| -------------- | ----: | -----------------: | -------------: | ------------------------: |
| `run-10`       |    10 |            10 / 10 |        10 / 10 |                     0 / 0 |
| `run-15-table` |    10 |            10 / 10 |        10 / 10 |                     0 / 0 |

The converted MLX artifacts and upstream snapshots produced bit-identical
`Detections` for Egret medium, large, and xlarge on the layout fixture. The
test also loads Egret medium with `dtype=mx.float16`; this is a loader smoke
test, not a parity assertion.

## Official COCO qualification

`tests/layout_egret/test_hf_parity.py` compares final native detections with
CPU `transformers.DFineForObjectDetection` on one repository fixture and three
DPBench pages. At threshold 0.3 it requires identical labels, score error at
most `1e-2`, matched box IoU at least `0.99`, and no unmatched detections.

The tests deliberately fail rather than skip while the local snapshots lack
`config.json`. Hydrate each cached snapshot with:

```bash
uv run --no-sync hf download ustc-community/dfine-nano-coco config.json README.md
uv run --no-sync hf download ustc-community/dfine-small-coco config.json README.md
uv run --no-sync hf download ustc-community/dfine-medium-coco config.json README.md
uv run --no-sync hf download ustc-community/dfine-large-coco config.json README.md
uv run --no-sync hf download ustc-community/dfine-xlarge-coco config.json README.md
```

Run the repository qualification lanes from [`DEVELOPMENT.md`](../../DEVELOPMENT.md)
after the snapshots are complete. Artifact-dependent failures are qualification
gates, not evidence of a portable regression.

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

| model  | implementation | device    | warm ms/item (median) | first-call ms | peak RSS | label agreement at IoU >= 0.5, excluding same-box swaps | same-box label swaps | matched-box mean IoU | mean abs score delta | threshold-boundary unmatched MLX / official | other unmatched MLX / official |
| ------ | -------------- | --------- | --------------------: | ------------: | -------: | ------------------------------------------------------: | -------------------: | -------------------: | -------------------: | ------------------------------------------- | ------------------------------ |
| medium | mlx            | mlx-metal |                30.066 |       219.154 | 1.69 GiB |                                                1.000000 |                    1 |             0.999938 |             0.000240 | 3 / 0                                       | 0 / 0                          |
| medium | official       | torch-mps |                46.116 |      2747.635 | 2.32 GiB |                                               reference |            reference |            reference |            reference | reference                                   | reference                      |
| large  | mlx            | mlx-metal |                39.990 |       297.620 | 1.77 GiB |                                                1.000000 |                    8 |             0.999934 |             0.000284 | 1 / 1                                       | 0 / 0                          |
| large  | official       | torch-mps |                59.279 |      2856.302 | 2.33 GiB |                                               reference |            reference |            reference |            reference | reference                                   | reference                      |
| xlarge | mlx            | mlx-metal |                61.758 |       506.425 | 2.08 GiB |                                                1.000000 |                    0 |             0.999938 |             0.000186 | 0 / 2                                       | 0 / 0                          |
| xlarge | official       | torch-mps |                86.303 |      2992.342 | 2.34 GiB |                                               reference |            reference |            reference |            reference | reference                                   | reference                      |
