# TableFormerV2 validation

Requalified on 2026-09-05 on an Apple M4 Pro against Docling 2.126.0. These results qualify this
exact FP32 implementation and plugin boundary. They do not establish general table-recognition
accuracy or performance on other hardware. The converted checkpoint is published at
`atkinschang/TableFormerV2-MLX`, whose default revision is
`79a9ab108f1bf6882c64226b5794886ffd972c18`. It is converted from
`docling-project/TableFormerV2` at source revision `51559fad3946873e26a6f9b8e912f948e8745bef`.

## Qualified inputs

- Source checkpoint: `docling-project/TableFormerV2` revision
  `51559fad3946873e26a6f9b8e912f948e8745bef`.
- Reference: `docling-ibm-models==4.0.2`, Torch 2.14.0, Torchvision 0.29.0,
  Transformers 5.16.1, Pillow 12.3.0, and NumPy 2.5.2 on deterministic Torch CPU.
- Native runtime: MLX 0.32.2, FP32, NHWC, four decoder layers.
- Converted weight SHA-256:
  `1bed301e719b583b71ca4c278499282c9701b63ddf7339ae6ef0093adf9477c8`.

**Reference implementation versus pinned reference.** The native modules were written after
docling-ibm-models 3.14.0 (commit `313d0790d688aedefbfdf79092a3f1287150ed97`,
`docling_ibm_models/tableformer_v2`) and Torchvision v0.28.0
(`torchvision/models/efficientnet.py`). The scoped docling-ibm-models `tableformer_v2` source tree
is unchanged in currently pinned 4.0.2, and this implementation remains validated against that
version. The referenced torchvision `efficientnet.py` is unchanged in currently pinned v0.29.0, and
this implementation is validated against that version.

## Numeric parity

All three public-domain USGS real-table crops produced exact token IDs and OTSL between the pinned
Torch CPU oracle and MLX. Resize bytes and normalized pixels were exact. The worst observed encoder
max error was `6.68e-6`, greedy-step logits max error was `6.58e-5`, final-logit MAE/max were
`6.30e-6`/`1.30e-4`, and normalized bbox max error was `1.55e-6`; all stayed below the fixed release
caps.

The dense scanned Table 2 reaches the source model's 512-step limit without EOS under both Torch
and MLX, producing 513 IDs including BOS. This proves exact no-EOS control flow and is also an
upstream model-quality limitation on that input. It is not presented as a semantically useful table
prediction. Table 1 and Table 3 terminate with EOS and exercise regular cells plus column headers.

Component promotion tests also cover the 695-tensor encoder, projected self/cross-attention caches,
manual-versus-fast SDPA, full/cached token decoding, position wrap at 512, the bbox decoder,
complete 821-key strict state load, and source accounting of all 936 checkpoint entries. Large
activation captures and generated JSON reports remain local.

## Docling and packaging contracts

The offline standard PDF smoke selected `MlxLayoutObjectDetectionModel` through the layout factory and
`MlxTableFormerV2Model` through the table factory. On the deterministic one-page USGS fixture it
completed successfully with one 6×8 table and 48 cells. External plugins disabled retain only
Docling built-ins; enabling them registers the custom options types without replacing those
built-ins.

A clean base-wheel environment imported both plugin callbacks while blocking MLX and all reference
framework imports, then ran native TableFormer inference offline. The wheel contains no weights,
tools, tests, goldens, or reference environments. Base, `[standard]`, and `[vlm]` dependency checks
passed; the latter remains the separate Granite qualification path.

Five fresh processes raced cold initialization and ten synchronized inference rounds for
Figure/Figure, Heron R50/R101, Figure/Heron R50, Figure/TableFormer, and TableFormer/TableFormer. All
outputs matched their sequential baselines without crash, deadlock, timeout, nonfinite value, shape
drift, or ordering error. The report records timing and memory but does not infer concurrent
speedup.

## Generation optimization

The original native runtime cached decoder states before attention projection. The promoted runtime
instead preallocates 512 projected K/V positions per decoder layer, projects encoder memory K/V once,
and uses MLX fast scaled-dot-product attention for token self-attention, cross-attention, and final
full-sequence decoding. Bbox attention remains manual because its focused A/B was 12–14% slower with
the fast primitive.

Each candidate and fixture ran in a fresh process for three rounds. Every round used 20 warmups and
30 measured batch-1 samples. The table reports the median of the three p50 values. PDF parsing is
excluded; the stage boundary starts from a prepared 2× page render.

|  IDs | Boundary   |  Original | Projected cache | Cache + fused SDPA | Total improvement |
| ---: | ---------- | --------: | --------------: | -----------------: | ----------------: |
|   56 | token loop |  83.49 ms |        59.42 ms |           56.04 ms |             32.9% |
|   56 | engine     | 104.58 ms |        83.67 ms |           78.48 ms |             25.0% |
|   56 | stage      | 105.45 ms |        85.06 ms |           80.23 ms |             23.9% |
|  227 | token loop | 350.63 ms |       249.32 ms |          231.59 ms |             33.9% |
|  227 | engine     | 377.95 ms |       275.94 ms |          260.47 ms |             31.1% |
|  227 | stage      | 381.35 ms |       280.04 ms |          263.12 ms |             31.0% |
|  513 | token loop | 918.73 ms |       773.45 ms |          534.91 ms |             41.8% |
|  513 | engine     | 915.52 ms |       606.32 ms |          566.18 ms |             38.2% |
|  513 | stage      | 923.75 ms |       610.64 ms |          571.13 ms |             38.2% |

Projected cache and fused SDPA each cleared the 5% p50 promotion threshold at every sequence length
for token loop, engine, and stage. No candidate showed three-round p95 regression. Some raw
token-loop rounds were affected by unrelated host load; the complete engine/stage matrix remained
consistent and is the stronger end-to-end result. Raw samples remain in ignored `reports/`.

Final engine MLX peak memory was 366,038,220 bytes at 56 IDs, 792,737,212 bytes at 227 IDs, and
1,589,984,956 bytes at 513 IDs. The 56-ID final engine cold total was 209.50 ms versus 159.63 ms for
the original because the first fast-attention shapes pay kernel-selection cost; 227- and 513-ID cold
totals improved. Cold latency was recorded but was not a promotion gate for cache or SDPA.

The compiled token-step spike was rejected. Its 56-ID warm p50 was 65.95 ms versus 56.04 ms for the
promoted eager fused-SDPA path, about 17.7% slower, so the autoregressive decoder remains eager.
Only the separately qualified image feature extractor is compiled at runtime.

The previously qualified four-thread Torch CPU `basin_table_1` observations remain 297.16 ms engine
p50 and 299.99 ms stage p50. Against those pinned observations, the final MLX p50 values are 78.48
ms and 80.23 ms. These are descriptive M4 Pro measurements, not a general speedup claim.

## Async-eval double buffering

The greedy loop submits the next decode step with `mx.async_eval` before reading the current token
back to the host, following the mlx-lm `generate_step` pattern, and tests EOS on the token integers
the host already holds instead of building a second `mx.all` graph. The EOS decision lands one step
late, so one step past the stop condition is computed and discarded. The loop does not build that
lookahead on the final permitted step, so a no-EOS result still contains 513 IDs including BOS.

Measured on an Apple M4 Pro, 48 GiB, macOS 26.5.2, Python 3.13.13, MLX 0.32.2, Transformers 5.16.1,
docling-ibm-models 4.0.2, with `MLX_ENABLE_TF32=0`, on 2026-09-05. One fresh process per side ran the
engine boundary over the three qualified crops in one batch of three, with 5 warmups and 30 measured
rounds; the values below divide each batch measurement by three.

| Profile | IDs per crop | p50 before | p50 after | Mean before | Mean after | p50 improvement |
| ------- | ------------ | ---------: | --------: | ----------: | ---------: | --------------: |
| V2      | 56/513/227   |  298.37 ms | 266.52 ms |   298.43 ms |  265.65 ms |           10.7% |

The 30-sample batch ranges do not overlap: 886–905 ms before against 783–807 ms after.

Token IDs, OTSL tokens, and cell boxes were captured over the same three crops before and after, and
the serialized captures are byte-identical. `tools/tableformer_v2/validate_parity.py`, which calls
`TableFormerV2.generate` directly, was run from both sides against a fresh pinned Torch CPU capture:
all 36 gates match field for field, and both sides reproduce the numeric-parity figures above
exactly, worst observed encoder max `6.68e-6` against the `1e-4` cap, greedy-step logits max
`6.58e-5` and final-logit max `1.30e-4` against the `1e-3` cap, and normalized bbox max `1.55e-6`
against the `1e-4` cap.

## Repository qualification

The three pytest lanes, Ruff, formatting, Mypy, root/reference locks, and wheel checks passed for
the recorded qualification. Current commands and required artifacts are maintained in
[`DEVELOPMENT.md`](../../DEVELOPMENT.md); generated JSON reports remain under ignored `reports/` or
CI artifacts.

## Backend comparison

Each row used a fresh batch-size-one process: one construction-plus-inference
warm-up call, then three timed rounds over all 63 DPBench ground-truth table
crops. MLX used Metal and the official Docling implementation used Torch MPS.

Machine: Apple M4 Pro, 48 GiB unified memory, macOS 26.5.2; Python 3.13.13;
Docling 2.124.0, docling-ibm-models 4.0.1, MLX 0.32.2, mlx-vlm 0.6.4, Torch
2.14.0, Transformers 5.8.1; measured on 2026-09-03 with `tools/compare_backends.py` schema 2.

| implementation | device    | warm ms/item (median) | first-call ms | peak RSS | exact OTSL sequence | tree TEDS | cell-bbox mean IoU | unmatched MLX / official |
| -------------- | --------- | --------------------: | ------------: | -------: | ------------------: | --------: | -----------------: | ------------------------ |
| mlx            | mlx-metal |                48.770 |       450.812 | 0.39 GiB |            1.000000 |  1.000000 |           0.999996 | 0 / 0                    |
| official       | torch-mps |               149.611 |      4303.018 | 0.74 GiB |           reference | reference |          reference | reference                |
