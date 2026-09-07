# SPDX-License-Identifier: Apache-2.0

"""Backend summaries must compare complete reports over the same inputs."""

import argparse
import json
from pathlib import Path

import pytest

from tools.compare_backends import summarize


def _report(implementation: str) -> dict:
    return {
        "component": "figure",
        "implementation": implementation,
        "input_ids": ["picture-1", "picture-2"],
        "outputs": {
            "picture-1": {"probabilities": {"bar_chart": 1.0}},
            "picture-2": {"probabilities": {"line_chart": 1.0}},
        },
        "device": "mlx-metal" if implementation == "mlx" else "torch-mps",
        "warm_per_item_ms": {"median": 1.0},
        "first_call_ms": 2.0,
        "process_peak_rss_bytes": 1024,
        "metadata": {
            "chip": "Apple M4 Pro",
            "memory_bytes": 48 * 1024**3,
            "macos": "26.5.2",
            "python": "3.13.13",
            "versions": {},
            "git_commit": "test",
        },
    }


@pytest.mark.parametrize("mismatch", [None, "input_ids", "missing_output", "extra_output"])
def test_summary_requires_matching_complete_inputs(tmp_path: Path, mismatch: str | None) -> None:
    mlx, official = _report("mlx"), _report("official")
    if mismatch == "input_ids":
        official["input_ids"] = ["picture-1"]
        del official["outputs"]["picture-2"]
    elif mismatch == "missing_output":
        del official["outputs"]["picture-2"]
    elif mismatch == "extra_output":
        mlx["outputs"]["picture-3"] = {"probabilities": {"pie_chart": 1.0}}

    paths = [tmp_path / "mlx.json", tmp_path / "official.json"]
    for path, report in zip(paths, (mlx, official), strict=True):
        path.write_text(json.dumps(report), encoding="utf-8")
    output = tmp_path / "summary.md"
    args = argparse.Namespace(inputs=paths, output=output)
    if mismatch is None:
        summarize(args)
        assert '"top1_agreement": 1.0' in output.read_text(encoding="utf-8")
    else:
        with pytest.raises(ValueError, match="Mismatched or incomplete inputs for figure/"):
            summarize(args)
        assert not output.exists()
