# SPDX-License-Identifier: Apache-2.0

"""Compilation shared by the native models."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Any

import mlx.core as mx


def compile_exclusive[**P, R](function: Callable[P, R], *, inputs: Any = None) -> Callable[P, R]:
    """Compile ``function`` and admit one thread at a time.

    MLX traces a compiled function on its first call for each input shape, and that
    trace is not thread safe: a second thread inside the same compiled function sees
    tracer arrays and fails to evaluate them. Engines are shared across Docling's
    pipeline threads, so the compiled region is mutually exclusive per model. The
    lock covers one compiled call, not a prediction: preprocessing, postprocessing,
    autoregressive decoding, and other engines all still run concurrently.
    """

    compiled = mx.compile(function, inputs=inputs)
    lock = Lock()

    def guarded(*args: P.args, **kwargs: P.kwargs) -> R:
        with lock:
            return compiled(*args, **kwargs)

    return guarded


__all__ = ["compile_exclusive"]
