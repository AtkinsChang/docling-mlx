# RT-DETR-v2 layout engine

`RtDetrV2Engine` is a Docling-independent MLX detector. Give it either a local
checkpoint directory or `RtDetrV2ModelSpec(repo_id=..., revision=...)`, then
call `predict(list[PIL.Image.Image])`. Each result supplies pixel `xyxy` boxes,
scores, label IDs, and the checkpoint's `id2label` mapping.

## Architecture and provenance

RT-DETR-v2 is a real-time detection transformer: the RT-DETR hybrid encoder and query decoder
(Zhao et al., 2023, arXiv:2304.08069) with the v2 bag-of-freebies refinements, including the
discrete deformable sampling this engine supports (Lv et al., 2024, arXiv:2407.17140), over a
ResNet backbone (He et al., 2015, arXiv:1512.03385) in its ResNet-vd variant, which applies the
ResNet-D stem and downsampling tweaks of He et al., 2018, arXiv:1812.01187.

## Why not mlx-vlm's RT-DETR-v2

mlx-vlm ships an RT-DETR-v2 port; this engine is adapted from it rather than imported, for
three reasons.

- **grid_sample.** mlx-vlm's Metal kernel reads out-of-range memory before masking, and its
  argument passing fails on mlx 0.32.2. This engine uses a bounds-checked kernel that returns the
  same values: on identical weights the logits and boxes are otherwise bit-identical, and the
  grid_sample difference is about 2.4e-6.
- **Shared primitives.** D-FINE (Egret) needs the same anchor generation, top-k query selection,
  and deformable attention, so one in-repo `detector_primitives` module serves both detectors.
- **Post-processing.** RT-DETR-v2 scores every query × label pair with a sigmoid and keeps the top
  `num_queries` pairs above the threshold. mlx-vlm sorts with `argsort` (tie order is
  implementation-dependent), keeps `>= threshold`, and clips boxes to the image. This engine sorts
  with a deterministic lexsort (score, then index), keeps `> threshold`, and returns unclipped
  boxes, matching Docling's Transformers post-processor so downstream clusters are identical.

The engine reads Hugging Face `config.json` and `preprocessor_config.json`.
It supports the RT-DETR ResNet basic and bottleneck backbones, configured
depths/channels/stages, hybrid-encoder dimensions and attention, extra feature
levels, scalar or per-level decoder sampling points, and `default` or
`discrete` decoder sampling. Unknown Hugging Face keys are ignored.

Both an MLX-layout artifact and an upstream Hugging Face safetensor checkpoint
load strictly. For example, an upstream checkpoint needs no conversion:

```python
from docling_mlx.engines.object_detection.rt_detr_v2.engine import (
    RtDetrV2Engine,
    RtDetrV2ModelSpec,
)

engine = RtDetrV2Engine(
    RtDetrV2ModelSpec(
        repo_id="PekingU/rtdetr_v2_r18vd",
        revision="REV",
    )
)
detections = engine.predict(images)
```

`MlxLayoutObjectDetectionModel` is the single Docling adaptor for RT-DETR-v2 and D-FINE. Select
the published Heron mirrors with the official `layout_heron_default` or `layout_heron_101` preset
IDs on `MlxLayoutObjectDetectionOptions`; checkpoint `config.json` selects the native family.

To create an MLX-layout artifact while retaining a provenance README:

```console
uv run --no-sync --group reference python -m tools.layout_heron.convert_weights \
  --source /path/to/checkpoint --repo-id PekingU/rtdetr_v2_r18vd \
  --revision REV \
  --output .artifacts/PekingU--rtdetr_v2_r18vd/REV
```

Keep this directory at the revision-qualified location when using `.artifacts` as an
`artifacts_path` cache root. `DOCLING_MLX_HERON_R50_ARTIFACT` and
`DOCLING_MLX_HERON_R101_ARTIFACT` instead point directly to their complete checkpoint directories.

Speed: on the recorded M4 Pro run, MLX was 1.67x faster per page for Heron R50 and 1.46x for
R101 than official Torch MPS; see [validation.md](validation.md).
