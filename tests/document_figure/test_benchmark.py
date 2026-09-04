# SPDX-License-Identifier: Apache-2.0

from tools._common import benchmark as benchmark_common


def test_fixed_input_is_prepared_once_before_all_timed_invocations(monkeypatch) -> None:
    events: list[str] = []
    fixed_input = object()
    elapsed = 0.0

    def timer() -> float:
        nonlocal elapsed
        events.append("timer")
        elapsed += 0.001
        return elapsed

    monkeypatch.setattr(benchmark_common, "perf_counter", timer)

    def prepare() -> object:
        events.append("prepare")
        return fixed_input

    def invoke(value: object) -> None:
        assert value is fixed_input
        events.append("invoke")

    first_call_ms, measurement = benchmark_common.measure_fixed_input(
        prepare, invoke, batch_size=1, warmup=2, repeats=3
    )

    assert events[0] == "prepare"
    assert events.count("prepare") == 1
    assert events.count("invoke") == 6
    assert first_call_ms >= 0
    assert measurement["samples"] == 3
