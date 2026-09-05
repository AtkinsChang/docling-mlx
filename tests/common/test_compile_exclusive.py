# SPDX-License-Identifier: Apache-2.0

"""The shared compiled-call guard admits one thread at a time.

Marked ``mlx`` only because ``_models._compile`` imports ``mlx.core`` at module
scope; the guard itself is exercised against a fake compiled callable, so no
Metal work and no MLX array ever runs here.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

pytestmark = pytest.mark.mlx


def test_second_thread_cannot_enter_while_the_first_holds_the_compiled_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from docling_mlx._models import _compile

    entered = threading.Semaphore(0)
    release = threading.Event()
    entries: list[int] = []

    def fake_compile(function: Any, inputs: Any = None, **kwargs: Any) -> Any:
        del inputs, kwargs

        def fake(value: int) -> int:
            entries.append(value)
            entered.release()
            # Hold the compiled region until the test lets go.
            release.wait(timeout=10)
            return function(value)

        return fake

    monkeypatch.setattr(_compile.mx, "compile", fake_compile)
    guarded = _compile.compile_exclusive(lambda value: value * 2)

    results: dict[str, int] = {}

    def call(name: str, value: int) -> None:
        results[name] = guarded(value)

    first = threading.Thread(target=call, args=("first", 1))
    first.start()
    assert entered.acquire(timeout=5), "the first thread never entered"
    assert entries == [1]

    second = threading.Thread(target=call, args=("second", 3))
    second.start()
    # The guard must keep the second thread out while the first holds the region.
    assert not entered.acquire(timeout=0.5), "the second thread entered concurrently"
    assert entries == [1]

    release.set()
    first.join(timeout=10)
    assert entered.acquire(timeout=5), "the second thread never entered after release"
    second.join(timeout=10)

    assert not first.is_alive() and not second.is_alive()
    assert entries == [1, 3]
    assert results == {"first": 2, "second": 6}
