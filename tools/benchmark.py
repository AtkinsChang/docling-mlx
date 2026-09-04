# SPDX-License-Identifier: Apache-2.0

"""Unified developer-facing benchmark entry point."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

from tools._common.benchmark import run, write_report

COMPONENTS = (
    "document-figure",
    "layout-heron",
    "layout-egret",
    "tableformer-v1",
    "tableformer-v2",
)
ADAPTERS = {
    "document-figure": "tools.document_figure.benchmark",
    "layout-heron": "tools.layout_heron.benchmark",
    "layout-egret": "tools.layout_egret.benchmark",
    "tableformer-v1": "tools.tableformer_v1.benchmark",
    "tableformer-v2": "tools.tableformer_v2.benchmark",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=COMPONENTS, required=True)
    parser.add_argument("--backend", choices=("mlx", "torch"), default="mlx")
    parser.add_argument("--target", choices=("preprocessing", "forward", "engine", "stage"))
    parser.add_argument("--artifact", "--artifacts-path", dest="artifact", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument(
        "--images", "--input", dest="image_paths", type=Path, nargs="+", required=True
    )
    parser.add_argument("--profile", dest="profile")
    parser.add_argument("--device", choices=("cpu", "mps"))
    parser.add_argument("--threads", "--cpu-threads", dest="cpu_threads", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--batch-sizes", type=int, nargs="+")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", "--repeats", dest="repeats", type=int, default=30)
    parser.add_argument("--output", type=Path)
    return parser


def _validate(parser: argparse.ArgumentParser, options: Any) -> None:
    if options.warmup < 0 or options.repeats < 1:
        parser.error("warmup must be nonnegative and rounds must be positive")
    if options.cpu_threads is not None and options.cpu_threads < 1:
        parser.error("threads must be positive")
    if options.batch_size is not None and options.batch_size < 1:
        parser.error("batch size must be positive")
    if options.batch_sizes is not None and any(size < 1 for size in options.batch_sizes):
        parser.error("batch sizes must be positive")


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    options = parser.parse_args(argv)
    _validate(parser, options)
    if options.batch_sizes is None and options.batch_size is not None:
        options.batch_sizes = [options.batch_size]
    adapter = importlib.import_module(ADAPTERS[options.component])
    for option in adapter.REQUIRES:
        if getattr(options, option) is None:
            parser.error(f"{options.component} requires --{option.replace('_', '-')}")
    if options.cpu_threads is None:
        options.cpu_threads = adapter.DEFAULT_THREADS
    try:
        report = run(adapter, options, Path(__file__).resolve().parents[1])
    except ValueError as error:
        parser.error(str(error))
    write_report(report, options.output)


if __name__ == "__main__":
    main()
