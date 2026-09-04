# Examples

- [`mlx_pipeline/`](mlx_pipeline/README.md) is the Torch-free MLX PDF pipeline
  (a small, dependency-isolated profile).
- [`document_figure/classify_images.py`](document_figure/classify_images.py)
  runs the standalone DocumentFigure image classifier.
- [`std_pdf_pipeline_all_mlx.py`](std_pdf_pipeline_all_mlx.py) configures the
  default Heron layout, TableFormerV2 table structure, and DocumentFigure picture
  classification through `MlxStandardPdfPipeline`. OCR, chart extraction,
  code/formula enrichment, and picture description are disabled.

Evaluate the pinned Docling DPBench snapshot with [`../tools/compare_dpbench.py`](../tools/compare_dpbench.py):

```bash
uv run --no-sync --extra standard --extra vlm --with 'docling-eval==1.4.2' \
  python tools/compare_dpbench.py --output-dir reports/pipeline-comparison
```
