# SPDX-License-Identifier: Apache-2.0

"""Download the pinned artifacts and sources the `mlx` lane reads.

Every download is revision-pinned. An already staged directory is left alone,
so re-running the tool costs nothing and never changes a staged input.
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

from docling_mlx.presets import PRESETS
from tools.document_figure.source import SOURCE_FILES, SOURCE_REPO, SOURCE_REVISION
from tools.layout_egret.source import SOURCES as EGRET_SOURCES

ROOT = Path(__file__).resolve().parents[1]

# Directory name under --artifacts for each preset the lane loads. The lane
# hardcodes this flat layout; `tableformer-v1` holds the whole mirror because
# its profiles live in checkpoint subdirectories.
ARTIFACTS: Mapping[str, str] = {
    "heron-r50": "layout_heron_default",
    "egret-medium": "layout_egret_medium",
    "egret-large": "layout_egret_large",
    "egret-xlarge": "layout_egret_xlarge",
    "document-figure-classifier": "document_figure_classifier_v2",
    "tableformer-v1": "tableformer_v1_accurate",
    "tableformer-v2": "tableformer_v2",
}

# Source snapshot directory name under --sources for each lane variable.
SOURCE_VARIABLES: Mapping[str, str] = {
    "DOCLING_MLX_EGRET_MEDIUM_SOURCE": "egret-medium",
    "DOCLING_MLX_EGRET_LARGE_SOURCE": "egret-large",
    "DOCLING_MLX_EGRET_XLARGE_SOURCE": "egret-xlarge",
    "DOCLING_MLX_SOURCE": "document-figure",
}


def stage(
    repo_id: str,
    revision: str,
    target: Path,
    *,
    allow_patterns: Sequence[str] | None = None,
) -> Path:
    """Download one pinned repository into `target` unless it already exists."""

    from huggingface_hub import snapshot_download

    if target.is_dir():
        print(f"cached {target}")
        return target
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=target,
        allow_patterns=None if allow_patterns is None else list(allow_patterns),
    )
    # snapshot_download leaves its own bookkeeping copy beside the files.
    shutil.rmtree(target / ".cache", ignore_errors=True)
    print(f"staged {target} from {repo_id}@{revision}")
    return target


def stage_all(artifacts: Path, sources: Path) -> None:
    """Stage every artifact and source snapshot the `mlx` lane reads."""

    for name, preset_name in ARTIFACTS.items():
        preset = PRESETS[preset_name]
        stage(preset.repo_id, preset.revision, artifacts / name)
    for profile, source in EGRET_SOURCES.items():
        stage(source.repo_id, source.revision, sources / f"egret-{profile}")
    stage(
        SOURCE_REPO,
        SOURCE_REVISION,
        sources / "document-figure",
        allow_patterns=SOURCE_FILES,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / ".artifacts")
    parser.add_argument("--sources", type=Path, default=ROOT / ".sources")
    args = parser.parse_args()
    stage_all(args.artifacts, args.sources)
    for variable, name in SOURCE_VARIABLES.items():
        print(f"export {variable}={(args.sources / name).resolve()}")


if __name__ == "__main__":
    main()
