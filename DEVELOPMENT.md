# Development

This guide describes the current artifact and qualification workflow. Use Python 3.13 or newer on Apple
Silicon macOS for MLX work, and use `uv run --no-sync` so an already provisioned environment is
not rewritten.

Provision the development environment with the `dev` group; add `reference` for parity and
converter lanes. Add `standard`, `vlm`, or `tableformer-v1` only where the selected lane requires
that integration.

```bash
uv sync --group dev
uv sync --group dev --group reference
```

## Dev environment

The optional Nix flake devshell provides the formatters, linters, `uv`, `gitleaks`, and `committed`;
`direnv allow` (via nix-direnv) or `nix develop` provides it and installs the git hooks (pre-commit:
treefmt + gitleaks; commit-msg: committed). The `uvx` commands in "Before every commit" are the
canonical checks and are the checks CI runs. Nix style: group dotted keys (`foo = { a = …; b = …; }`,
not `foo.a = …; foo.b = …;`).

## Before every commit

Run the repository-wide native checks:

```bash
uvx rumdl check .
uvx typos .
uvx taplo fmt --check .
uvx taplo lint
uvx --from actionlint-py actionlint
uvx ruff check .
uvx ruff format --check .
uvx --with 'reuse[charset-normalizer]' reuse lint
```

## Checks and lanes

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy
uv run --no-sync pytest -q -rs
uv lock --check
```

The default pytest configuration is `not (mlx or parity or release)` and is self-contained. The
explicit lanes are:

```bash
uv run --no-sync pytest -q -rs -m "mlx and not parity and not release"
uv run --no-sync pytest -q -rs -m parity
uv run --no-sync pytest -q -rs -m release
```

Selected lanes require Apple Silicon, Metal, the reference group where applicable, and staged
artifacts. Every selected lane must finish with zero skips; a missing artifact is a failed setup,
not a qualified result. Source-dependent tests require these variables:

| Variable                                     | Required directory                                                                       |
| -------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `DOCLING_MLX_TABLEFORMER_V1_SOURCE`          | Pinned `docling-project/docling-models` snapshot containing both TableFormer v1 profiles |
| `DOCLING_MLX_TABLEFORMER_V2_SOURCE`          | Pinned `docling-project/TableFormerV2` snapshot                                          |
| `DOCLING_MLX_EGRET_MEDIUM_SOURCE`            | Pinned official Egret medium source snapshot                                             |
| `DOCLING_MLX_EGRET_LARGE_SOURCE`             | Pinned official Egret large source snapshot                                              |
| `DOCLING_MLX_EGRET_XLARGE_SOURCE`            | Pinned official Egret xlarge source snapshot                                             |
| `DOCLING_MLX_OBJECT_DETECTION_PARITY_INPUTS` | First three pinned DPBench PDFs, named `0000.pdf` through `0002.pdf`                     |

Keep machine-specific lane variables in the ignored `.envrc.local` file.

The TableFormer download helpers print the cache directory to export:

```bash
uv run --no-sync python -c \
  'from tools.tableformer_v1.source import download_source; print(download_source())'
uv run --no-sync python -c \
  'from tools.tableformer_v2.source import download_source; print(download_source())'
export DOCLING_MLX_TABLEFORMER_V1_SOURCE=/path/printed/by/the/first/command
export DOCLING_MLX_TABLEFORMER_V2_SOURCE=/path/printed/by/the/second/command
```

Download each official Egret source at the immutable revision recorded in
[`docs/layout-egret/validation.md`](docs/layout-egret/validation.md):

```bash
export DOCLING_MLX_EGRET_MEDIUM_SOURCE=/path/to/egret-medium-source
export DOCLING_MLX_EGRET_LARGE_SOURCE=/path/to/egret-large-source
export DOCLING_MLX_EGRET_XLARGE_SOURCE=/path/to/egret-xlarge-source
hf download docling-project/docling-layout-egret-medium \
  --revision REV \
  --local-dir "$DOCLING_MLX_EGRET_MEDIUM_SOURCE"
hf download docling-project/docling-layout-egret-large \
  --revision REV \
  --local-dir "$DOCLING_MLX_EGRET_LARGE_SOURCE"
hf download docling-project/docling-layout-egret-xlarge \
  --revision REV \
  --local-dir "$DOCLING_MLX_EGRET_XLARGE_SOURCE"
```

Produce the detector parity inputs from the same pinned DPBench revision used by
`tools/compare_dpbench.py`. The target directory must end in `pdfs` and must not already exist:

```bash
export DOCLING_MLX_OBJECT_DETECTION_PARITY_INPUTS=/path/to/dpbench-workspace/pdfs
uv run --no-sync --with 'docling-eval==1.4.2' python - <<'PY'
import os
from pathlib import Path

from tools.compare_dpbench import _prepare_benchmark, _snapshot_path

inputs = Path(os.environ["DOCLING_MLX_OBJECT_DETECTION_PARITY_INPUTS"])
_prepare_benchmark(_snapshot_path(), inputs.parent, 3)
PY
```

CI runs the portable checks on Ubuntu: frozen dev sync, mypy, default pytest, wheel build, and an
installed-wheel import smoke, plus a separate uvx-based lint job. It does not qualify MLX, parity,
or release artifacts.

## Stage artifacts

Converters consume cached upstream snapshots and never upload or overwrite a
published model. Their `--output` argument is the complete checkpoint directory. The examples use
Docling's revision-qualified cache-root layout: `<artifacts_path>/<repo-id with '/' replaced by
'--'>/<revision>/`. Use the source repository and immutable revision that the resulting model card
records.

The generic converter shape is:

```bash
uv run --no-sync --group reference python -m tools.document_figure.convert_weights \
  --source /path/to/snapshot --repo-id org/model --revision REV \
  --output .artifacts/org--model/REV
uv run --no-sync --group reference python -m tools.layout_heron.convert_weights \
  --source /path/to/snapshot --repo-id org/model --revision REV \
  --output .artifacts/org--model/REV
uv run --no-sync --group reference python -m tools.layout_egret.convert_weights \
  --source /path/to/snapshot --repo-id org/model --revision REV \
  --output .artifacts/org--model/REV
```

TableFormer converters use the frozen source loaders (or `--source` explicitly):

```bash
uv run --no-sync --group reference --extra standard --extra tableformer-v1 \
  python -m tools.tableformer_v1.convert_weights \
  --source "$DOCLING_MLX_TABLEFORMER_V1_SOURCE" \
  --output .artifacts/org--model/REV
uv run --no-sync --group reference python -m tools.tableformer_v2.convert_weights \
  --source "$DOCLING_MLX_TABLEFORMER_V2_SOURCE" \
  --output .artifacts/org--model/REV
```

The release environment normally sets these artifact overrides when paths differ from the
defaults:

```text
DOCLING_MLX_SOURCE
DOCLING_MLX_ARTIFACTS
HF_HOME
DOCLING_MLX_HERON_R50_ARTIFACT
DOCLING_MLX_HERON_R101_ARTIFACT
DOCLING_MLX_EGRET_MEDIUM_ARTIFACT
DOCLING_MLX_EGRET_LARGE_ARTIFACT
DOCLING_MLX_EGRET_XLARGE_ARTIFACT
DOCLING_MLX_TABLEFORMER_V1_ARTIFACT
DOCLING_MLX_TABLEFORMER_V1_REFERENCE
DOCLING_MLX_TABLEFORMER_V1_FAST_REFERENCE
DOCLING_MLX_TABLEFORMER_V1_FAST_PDF_ORACLE
DOCLING_MLX_TABLEFORMER_V2_ARTIFACT
DOCLING_MLX_GRANITE_VISION_ARTIFACTS_ROOT
HF_HUB_OFFLINE
TRANSFORMERS_OFFLINE
```

When `artifacts_path` is a cache root, the resolver checks
`<artifacts_path>/<repo-id-with-slashes-replaced-by-->/<revision>` first, then the legacy flat
`<artifacts_path>/<repo-id-with-slashes-replaced-by-->` directory. The `DOCLING_MLX_*_ARTIFACT`
variables above instead point directly to complete checkpoint directories and may use arbitrary
paths; the `DOCLING_MLX_*_SOURCE` variables point to source snapshots for conversion. The
`DOCLING_MLX_GRANITE_VISION_ARTIFACTS_ROOT` variable is a resolver root, not a direct checkpoint
path.

Granite Vision is an unchanged official artifact, not a converted project model:

```bash
hf download ibm-granite/granite-vision-4.1-4b \
  --revision dd48e97503de471803850df70843cf9eb5da8712 \
  --local-dir .artifacts/granite-vlm/ibm-granite--granite-vision-4.1-4b
```

Use `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` with
`DOCLING_MLX_GRANITE_VISION_ARTIFACTS_ROOT=.artifacts/granite-vlm` for the offline Granite release
test. Official parity captures use the corresponding pinned source snapshot and `hf download`;
never substitute an unpinned branch for a release record.

## Qualification and tools

After staging artifacts, run the lanes above and inspect `-rs`; zero skips is the policy. The
unified benchmark is one fresh process per component and boundary:

```bash
uv run --no-sync python -m tools.benchmark --component document-figure \
  --artifact "$DOCLING_MLX_ARTIFACTS" \
  --images tests/fixtures/document_figure/reference_images/bar_chart.png \
  --target engine --warmup 5 --rounds 30 --output reports/document-figure-engine.json
```

`tools/compare_dpbench.py` compares the official and MLX application pipelines.

`tools/compare_backends.py` reproduces the full DPBench backend comparison with one fresh process
per component and implementation. It materializes the pinned page, picture, and table inputs,
resumes completed raw JSON reports under `reports/backend-bench/`, and writes the paired quality
and latency tables used by the component validation documents. Its `pipeline` component measures the
reduced standard pipeline over all 200 pages with three timed rounds, and `pipeline_granite`
measures the full pipeline with the Granite Vision table structure and chart extraction stages over
the first 50 pages with one timed round, because Docling's official Granite stages are CPU-only on
macOS:

```bash
uv run --no-sync python -m tools.compare_backends all
```

The release concurrency tool is the authoritative fresh-process check and requires the Figure,
TableFormerV2, and TableFormerV1 artifacts:

```bash
uv run --no-sync python -m tools.release.qualify_concurrency \
  --figure-artifact "$DOCLING_MLX_ARTIFACTS" \
  --tableformer-artifact "$DOCLING_MLX_TABLEFORMER_V2_ARTIFACT" \
  --tableformer-v1-artifact "$DOCLING_MLX_TABLEFORMER_V1_ARTIFACT"
```

Component-specific preprocessing, parser, and acceptance details belong in
`docs/<component>/README.md` and `validation.md`.

## Releases

`release.yaml` runs release-please on every push to `main`. It keeps a release pull request open that
bumps `version` in `pyproject.toml` and the matching entry in `uv.lock`, and rewrites `CHANGELOG.md`
from the conventional commits since the last release (`feat`, `fix`, `perf`, and `revert` are
listed; other types are hidden). Merging that pull request creates the `v*` tag and the GitHub
release, then publishes the tag to PyPI through `publish.yaml`, which is a reusable workflow with
trusted publishing under the `pypi` environment. Below 1.0, breaking changes and features both bump
the minor version and fixes bump the patch version. The first release is pinned by a `Release-As:
0.1.0` footer in the initial history.

To rehearse publishing, run `publish.yaml` manually from the Actions tab with `testpypi` selected;
that uses the `testpypi` environment and TestPyPI's own trusted publisher, and skips files that
TestPyPI already holds. Both environments must exist in the repository settings, and GitHub Actions
must be allowed to create pull requests for release-please to open the release pull request.

## Adding a model

Read [`docs/architecture.md`](docs/architecture.md) first. A native family needs exactly three
layers: a Docling-independent engine with a plain-checkpoint test, a `ModelPreset` entry, and a
Docling stage adaptor. Keep model-independent inference data in the engine and Docling labels,
page state, batching, enablement, accelerator checks, and parse-error policy in the adaptor. Add a
plugin entry point only where Docling exposes the corresponding factory; otherwise keep the stage
directly constructible.
