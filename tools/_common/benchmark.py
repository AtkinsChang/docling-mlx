# SPDX-License-Identifier: Apache-2.0

"""Shared, dependency-light benchmark reporting helpers."""

from __future__ import annotations

import json
import platform
import resource
import subprocess
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from tools._common.hashing import _package_versions, _sha256


def measure(function: Callable[[], Any], batch_size: int, warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        function()
        samples.append(perf_counter() - start)
    p50, p95 = np.percentile(samples, [50, 95])
    return {
        "p50_ms": float(p50 * 1000),
        "p95_ms": float(p95 * 1000),
        "images_per_second": float(batch_size / p50),
        "samples": repeats,
        "samples_ms": [sample * 1000 for sample in samples],
    }


def measure_first_call(function: Callable[[], Any]) -> float:
    start = perf_counter()
    function()
    return (perf_counter() - start) * 1000


def peak_rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def measure_fixed_input[Input](
    prepare: Callable[[], Input],
    invoke: Callable[[Input], Any],
    batch_size: int,
    warmup: int,
    repeats: int,
) -> tuple[float, dict]:
    """Prepare one input before timing repeated invocations against it."""

    fixed_input = prepare()

    def operation() -> Any:
        return invoke(fixed_input)

    return measure_first_call(operation), measure(operation, batch_size, warmup, repeats)


def git_provenance(repository_root: Path) -> dict[str, str | bool]:
    """Capture only revision state, never local source paths or file hashes."""

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def load_images(image_paths: list[Path]) -> list[Image.Image]:
    """Load RGB copies once, outside every timed operation."""

    if not image_paths:
        raise ValueError("at least one image is required")
    images: list[Image.Image] = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def _image_identity(image_paths: list[Path], images: list[Image.Image]) -> list[dict[str, Any]]:
    return [
        {"name": path.name, "sha256": _sha256(path), "size": list(image.size)}
        for path, image in zip(image_paths, images, strict=True)
    ]


def run(adapter: Any, options: Any, repository_root: Path) -> dict[str, Any]:
    """Run an adapter's operations and add shared evidence/reporting fields."""

    image_paths = list(options.image_paths)
    images = load_images(image_paths)
    started = perf_counter()
    state = adapter.load(options, images)
    state["initialization_ms"] = (perf_counter() - started) * 1000
    reset_memory = state.get("reset_memory")
    if reset_memory is not None:
        reset_memory()
    try:
        rows: list[dict[str, Any]] = []
        for size in state["batch_sizes"]:
            batch = [images[index % len(images)] for index in range(size)]
            row: dict[str, Any] = {"batch_size": size}
            calls: dict[str, Callable[[], Any]] = {}
            first_calls: dict[str, float] = {}
            for target, operation in state["operations"].items():

                def invoke(value: Any, target: str = target) -> Any:
                    return adapter.predict(state, target, value)

                prepare = operation.get("prepare")
                if prepare is None:

                    def operation_call(
                        batch: list[Any] = batch, invoke: Callable[[Any], Any] = invoke
                    ) -> Any:
                        return invoke(batch)

                    first_calls[target] = measure_first_call(operation_call)
                    calls[target] = operation_call
                else:
                    fixed_input = prepare(batch)

                    def fixed_call(
                        fixed_input: Any = fixed_input, invoke: Callable[[Any], Any] = invoke
                    ) -> Any:
                        return invoke(fixed_input)

                    first_calls[target] = measure_first_call(fixed_call)
                    calls[target] = fixed_call
            for target, operation in state["operations"].items():
                measurement = measure(calls[target], size, options.warmup, options.repeats)
                first_key = operation.get("first_key")
                if first_key is not None:
                    row[first_key] = first_calls[target]
                row[operation.get("measurement_key", target)] = measurement
            rows.append(row)
        report = adapter.normalize_result(state, rows, image_paths, images, options)
        report.update(
            {
                "repository": git_provenance(repository_root),
                "python": platform.python_version(),
                "versions": _package_versions(tuple(state["version_names"])),
                "warmup": options.warmup,
                "process_peak_rss_bytes": peak_rss_bytes(),
            }
        )
        return report
    finally:
        close = state.get("close")
        if close is not None:
            close()


def write_report(report: dict[str, Any], output: Path | None) -> str:
    text = json.dumps(report, indent=2) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return text
