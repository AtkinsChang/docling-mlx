# EfficientNet validation

Requalified on 2026-09-05 on an Apple M4 Pro (macOS 26.5.2 arm64), with MLX 0.32.2,
Transformers 5.16.1, Torch 2.14.0, NumPy 2.5.2, and Pillow 12.3.0. The
runtime remains weight-free; model files are local or Hub artifacts.

The preset resolves to this published artifact, converted from the listed upstream revision:

| Preset                          | Published MLX artifact                                                                   | Converted from                                                                           |
| ------------------------------- | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `document_figure_classifier_v2` | `atkinschang/DocumentFigureClassifier-v2.5-MLX@673c86192056a6f0e6c6c295647ac3232fde5f34` | `docling-project/DocumentFigureClassifier-v2.5@f859dfbff5c9916cd996942d4b0db7fa25808220` |

**Reference implementation versus pinned reference.** The native EfficientNet module was written
after Hugging Face Transformers 5.8.1
(`transformers/models/efficientnet/modeling_efficientnet.py`). That module is unchanged apart from
lint through currently pinned 5.16.1, the parity oracle.

## Phase A: DocumentFigure identity

The converted artifact and the pinned upstream
`docling-project/DocumentFigureClassifier-v2.5@f859dfbff5c9916cd996942d4b0db7fa25808220` snapshot
produced bit-identical logits on the two committed reference crops. The existing Docling preset
kept its bilinear resize and single fused normalization path.

| Crop               | Top-1     | Top-5 set | Maximum probability delta |
| ------------------ | --------- | --------- | ------------------------: |
| `bar_chart`        | identical | identical |                   4.24e-6 |
| `geographical_map` | identical | identical |                   2.28e-6 |

The layer-level native-vs-Transformers check passed with mean absolute error at
most `1e-4` and maximum absolute error at most `1e-3` for every captured layer
and logits. The artifact-vs-upstream logits comparison was byte-for-byte equal.

## Phase B: official checkpoints

Temporary safetensor copies of the official Google checkpoints were compared
against `EfficientNetForImageClassification` on CPU using the same fixture
crops. Top-1 and top-5 sets matched for every image; maximum softmax deltas
were below the required `1e-2`.

| Checkpoint               | Maximum logit delta | Maximum probability delta | Top-1     | Top-5 set |
| ------------------------ | ------------------: | ------------------------: | --------- | --------- |
| `google/efficientnet-b0` |             9.86e-2 |                   4.13e-3 | identical | identical |
| `google/efficientnet-b3` |             2.12e-2 |                   2.04e-3 | identical | identical |
| `google/efficientnet-b7` |             4.01e-2 |                   8.75e-3 | identical | identical |

The repository parity tests look for `config.json`,
`preprocessor_config.json`, and `model.safetensors` in the corresponding
Hugging Face cache snapshot and fail explicitly when a snapshot is absent.

## Phases

- Phase A: generic config-driven engine, shared loader, and processor semantics.
- Phase B: official B0/B3/B7 parity gate.
- Phase C: generic artifact boundary, source-driven
  converter, and documentation.

## Reproduction

Use `uv run --no-sync` for every lane. The standard repository lanes are the
default tests, `-m "mlx and not release"`, `-m parity`, and `-m release`, plus
ruff, mypy, cold-import, and REUSE checks. The caller must provide the lane's
artifact environment; no images or model weights are committed.

## Backend comparison

Each row used a fresh batch-size-one process: one construction-plus-inference
warm-up call, then three timed rounds over all 168 DPBench ground-truth picture
crops. MLX used Metal and the official Docling implementation used Torch MPS.

Machine: Apple M4 Pro, 48 GiB unified memory, macOS 26.5.2; Python 3.13.13;
Docling 2.124.0, docling-ibm-models 4.0.1, MLX 0.32.2, mlx-vlm 0.6.4, Torch
2.14.0, Transformers 5.8.1; measured on 2026-09-03 with `tools/compare_backends.py` schema 2.

| implementation | device    | warm ms/item (median) | first-call ms | peak RSS | top-1 agreement | mean abs probability delta | max abs probability delta |
| -------------- | --------- | --------------------: | ------------: | -------: | --------------: | -------------------------: | ------------------------: |
| mlx            | mlx-metal |                 3.838 |        84.400 | 0.24 GiB |        1.000000 |                   0.000008 |                  0.000979 |
| official       | torch-mps |                13.478 |      2645.975 | 0.80 GiB |       reference |                  reference |                 reference |
