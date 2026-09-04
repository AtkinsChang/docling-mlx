# Torch-free MLX PDF pipeline

This standalone project runs Docling's standard PDF flow with:

- Heron R101 native-mode layout through the `docling-mlx` plugin;
- OCRMac;
- TableFormerV2 through its public `docling-mlx` stage;
- DocumentFigure through `MlxStandardPdfPipeline`.

The selected environment contains no Torch, Torchvision, Transformers, or `docling-ibm-models`.
Granite Vision, CodeFormulaV2, and picture description are therefore excluded from this specific
example. They are enabled on both sides by
[`../../tools/compare_dpbench.py`](../../tools/compare_dpbench.py), whose full comparison environment
includes their dependencies.

Heron and TableFormerV2 use Docling's external factories. DocumentFigure has no Docling factory, so
the package-owned `MlxStandardPdfPipeline` installs it after the official pipeline initialization.

```bash
uv run --no-sync --project examples/mlx_pipeline python examples/mlx_pipeline/pipeline.py \
  input.pdf --output-dir output
```

For a host smoke run in a full Docling environment, retain the same pipeline selection but
explicitly bypass the environment assertion with `--allow-torch`. Add `--no-ocr` only when
that environment does not include OCRMac; the default profile still enables OCRMac.

Pass `--artifacts-path .artifacts` for Docling's local repository-folder layout. Without it, models
are resolved from their configured Hugging Face repositories.

The comparison lists the official and full MLX pipeline definitions separately so their stage
choices can be read side by side:

```bash
uv run --no-sync --extra standard --extra vlm python tools/compare_dpbench.py \
  --output-dir reports/pipeline-comparison --limit 10
```
