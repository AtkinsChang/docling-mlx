# TableFormerV1 validation

The recorded qualification covers the frozen Accurate and Fast conversions on an Apple M4 Pro with
Docling 2.126.0. It does not establish general table quality or other-hardware behavior.

## Frozen conversions

Accurate and Fast both originate from `docling-project/docling-models` revision
`fc0f2d45e2218ea24bce5045f58a389aed16dc23` (tag `v2.3.0`). Accurate is encoder/decoder 6/6; Fast
is 4/2.
The converted profiles are published together at `atkinschang/TableFormer-MLX`, whose default
revision is `28bb5171682eed2a7d3c0a2f29f80f32dcccc18e`. Each profile contains
`model.safetensors`, `config.json`, `generation_config.json`, and `preprocessor_config.json`.

**Reference implementation versus pinned reference.** The native modules were written after
docling-ibm-models 3.14.0 (commit `313d0790d688aedefbfdf79092a3f1287150ed97`,
`docling_ibm_models/tableformer`) and Torchvision v0.28.0 (`torchvision/models/resnet.py`). The
referenced docling-ibm-models tableformer files changed in currently pinned 4.0.2
(`matching_post_processor.py`, `tf_cell_matcher.py`, `tf_predictor.py`,
`models/common/base_model.py`, and the three `table04_rs` model files); this implementation
remains based on 3.14.0 and does not follow those newer changes. The referenced torchvision
`resnet.py` is unchanged in currently pinned v0.29.0, and this implementation is validated against
that version.

## Parity

Every fixture had exact token IDs, OTSL, and structured-stage output. Caps are logits MAE ≤ `1e-4`,
logits max absolute error ≤ `1e-3`, and normalized box max absolute error ≤ `1e-4`.

| Profile  | Fixture | Logits MAE | Logits max abs | Normalized box max abs |
| -------- | ------- | ---------: | -------------: | ---------------------: |
| Accurate | basin 1 |  3.0383e-6 |      1.8120e-5 |              5.1260e-6 |
| Accurate | basin 2 |  1.9735e-6 |      2.1935e-5 |              2.3842e-7 |
| Accurate | basin 3 |  2.1036e-6 |      2.8610e-5 |              3.5763e-7 |
| Fast     | basin 1 |  3.3994e-6 |      2.0027e-5 |              3.1888e-6 |
| Fast     | basin 2 |  2.4774e-6 |      1.8120e-5 |              7.0781e-7 |
| Fast     | basin 3 |  5.1420e-6 |      1.1039e-4 |              4.7684e-6 |

The default Fast offline-PDF oracle was independently captured with pinned Torch CPU Heron R50 and
upstream TableFormer Fast; it is separate from native output and has SHA-256
`a97524aaba39732b0750533b93976ecc1be5ecaa65bfa860b92c57d62ac24273`.

## Gates and observations

V1 excluding offline: 125 passed, 3 deselected (15.89 s); release parity: 2 passed, 6 deselected
(8.39 s); offline PDF default oracle: 3 passed (8.09 s); concurrency pytest: 1 passed (23.14 s);
9 fresh processes passed; clean wheel: 9 passed (193.93 s).

Thirty-sample MLX TableFormer v1 p50/p95 timings (ms) were Engine Accurate 273.2589/279.3067 and Fast
106.4566/108.6819; Stage Accurate 260.9589/265.1952 and Fast 143.0580/145.7010. These are
environment-specific observations, not a general speed claim. Raw timing, RSS, MLX-peak, and
concurrency reports remain under ignored `reports/`.

Earlier Accurate-only Torch CPU comparisons remain historical evidence only; they are not the
current two-profile benchmark. The pinned mirror revision above has the upstream checkpoint layout;
runtime supports that layout as well as converted artifacts.

## Async-eval double buffering

The greedy loop applies the `xcel`/`lcel`/`fcel` structural correction on device, so the next decode
step is built from the previous token without a host round trip and can be submitted with
`mx.async_eval` before the current token is read back, following the mlx-lm `generate_step` pattern.
The stop decision lands one step late, so one step past it is computed and discarded. The loop does
not build that lookahead on the final permitted step, so the profile step budget still caps the
sequence and the decoder never sees a sequence past its positional-encoding limit.

Measured on an Apple M4 Pro, 48 GiB, macOS 26.5.2, Python 3.13.13, MLX 0.32.2, Transformers 5.16.1,
docling-ibm-models 4.0.2, with `MLX_ENABLE_TF32=0`, on 2026-09-05. One fresh process per profile and
side ran the engine boundary over the three basin crops in one batch of three, with 5 warmups and 30
measured rounds; the values below divide each batch measurement by three. These crops generate
longer sequences than the inputs behind the p50/p95 timings recorded above, so the two records
measure different work and are not comparable.

| Profile  | IDs per crop | p50 before | p50 after | Mean before | Mean after | p50 improvement |
| -------- | ------------ | ---------: | --------: | ----------: | ---------: | --------------: |
| Accurate | 101/242/272  |  542.78 ms | 429.67 ms |   541.12 ms |  429.77 ms |           20.8% |
| Fast     | 72/212/210   |  223.85 ms | 144.22 ms |   223.00 ms |  144.62 ms |           35.6% |

The 30-sample batch ranges do not overlap: Accurate 1577–1646 ms before against 1276–1312 ms after,
and Fast 653–676 ms before against 424–447 ms after.

Token IDs, OTSL tokens, cell boxes, and bbox classes were captured over the same three crops before
and after, and the serialized captures are byte-identical. `tools/tableformer_v1/validate_parity.py`
was run from both sides against fresh pinned Torch CPU captures: all 51 Accurate and 45 Fast gates
match field for field, and both sides reproduce the parity table above exactly. Worst observed values
were normalized box max absolute error `5.13e-6` for Accurate and `5.42e-6` for Fast against the
`1e-4` cap, greedy-step logits max absolute error `2.86e-5` and `1.10e-4` and bbox class-logit max
absolute error `1.53e-4` and `1.01e-4` against the `1e-3` cap, and page-space stage box max absolute
error `9.77e-4` and `1.01e-3` against that gate's `2e-3` cap. That validator traces its own step loop
for the generation and prediction gates, so the changed loop is covered there by the
`stage.structured_output` gate, which runs the Docling stage end to end.

## Compatibility behavior

The qualified stage preserves the exact OpenCV preprocessing documented in [README.md](README.md).
Its legacy structural fallback matches `docling-ibm-models==4.0.2`: colspan or rowspan values above
20 become 1 during the source-compatible HTML round trip.

## Backend comparison

Each row used a fresh batch-size-one process: one construction-plus-inference
warm-up call, then three timed rounds over all 63 DPBench ground-truth table
crops. MLX used Metal and the official Docling implementation used Torch MPS.

Machine: Apple M4 Pro, 48 GiB unified memory, macOS 26.5.2; Python 3.13.13;
Docling 2.124.0, docling-ibm-models 4.0.1, MLX 0.32.2, mlx-vlm 0.6.4, Torch
2.14.0, Transformers 5.8.1; measured on 2026-09-03 with `tools/compare_backends.py` schema 2.

| mode     | implementation | device    | warm ms/item (median) | first-call ms | peak RSS | exact OTSL sequence | tree TEDS | cell-bbox mean IoU | unmatched MLX / official |
| -------- | -------------- | --------- | --------------------: | ------------: | -------: | ------------------: | --------: | -----------------: | ------------------------ |
| accurate | mlx            | mlx-metal |                98.970 |       388.194 | 0.64 GiB |            1.000000 |  1.000000 |           0.999997 | 0 / 0                    |
| accurate | official       | torch-mps |               166.989 |      1795.021 | 0.99 GiB |           reference | reference |          reference | reference                |
| fast     | mlx            | mlx-metal |                57.346 |       242.457 | 0.59 GiB |            1.000000 |  1.000000 |           0.999997 | 0 / 0                    |
| fast     | official       | torch-mps |                79.396 |      1483.307 | 0.87 GiB |           reference | reference |          reference | reference                |
