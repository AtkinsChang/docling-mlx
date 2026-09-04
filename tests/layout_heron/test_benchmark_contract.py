# SPDX-License-Identifier: Apache-2.0

"""CPU-only contract tests for the Heron materialized benchmark report."""

from __future__ import annotations

from tools.layout_heron.benchmark import measure


def test_measure_reports_raw_samples_percentiles_and_throughput() -> None:
    calls = 0

    def operation() -> None:
        nonlocal calls
        calls += 1

    result = measure(operation, batch_size=4, warmup=2, repeats=3)

    assert calls == 5
    assert result["samples"] == 3
    assert len(result["samples_ms"]) == 3
    assert result["p50_ms"] >= 0
    assert result["p95_ms"] >= result["p50_ms"]
    assert result["images_per_second"] > 0
