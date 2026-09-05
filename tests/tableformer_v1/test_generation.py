# SPDX-License-Identifier: Apache-2.0

"""Greedy generation contracts for the TableFormer v1 decoder."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.tableformer_v1.conftest import accurate_config, fast_config

pytestmark = pytest.mark.mlx


def _decoder(*, fast: bool = False) -> Any:
    from docling_mlx._models.tableformer_v1.config import TableFormerV1Config
    from docling_mlx._models.tableformer_v1.decoder import TableFormerV1TokenDecoder

    return TableFormerV1TokenDecoder(
        TableFormerV1Config.from_dict(fast_config() if fast else accurate_config())
    )


def _install_schedule(
    monkeypatch: pytest.MonkeyPatch, decoder: Any, schedule: list[int]
) -> list[int]:
    import mlx.core as mx

    calls: list[int] = []

    def step(input_ids: Any, _memory: Any, _cache: Any = None) -> Any:
        from docling_mlx._models.tableformer_v1.decoder import DecoderStepOutput

        index = len(calls)
        calls.append(input_ids.shape[0])
        token_id = schedule[index] if index < len(schedule) else schedule[-1]
        logits = mx.full((1, 13), -1.0, dtype=mx.float32)
        logits[:, token_id] = 1.0
        return DecoderStepOutput(
            logits=logits,
            hidden_state=mx.full((1, 1, 512), index, dtype=mx.float32),
            cache=mx.zeros((decoder.num_decoder_layers, index + 1, 1, 1), dtype=mx.float32),
        )

    monkeypatch.setattr(decoder, "step", step)
    return calls


def test_generation_applies_structural_corrections_and_stops_on_eos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mlx.core as mx

    decoder = _decoder()
    calls = _install_schedule(monkeypatch, decoder, [8, 9, 8, 7, 6, 3])
    output = decoder.generate(mx.zeros((7, 1, 512), dtype=mx.float32))
    mx.eval(output.generated_ids, output.hidden_states, output.cache)

    np.testing.assert_array_equal(
        np.asarray(output.generated_ids),
        np.array([[2, 6, 9, 6, 7, 5, 3]], dtype=np.int32),
    )
    np.testing.assert_array_equal(np.asarray(output.hidden_states)[0, :, 0], np.arange(6))
    # Six consumed steps plus the double-buffered lookahead discarded at EOS.
    assert calls == [1, 2, 3, 4, 5, 6, 7]
    assert output.cache.shape == (6, 6, 1, 1)


def test_fast_generation_uses_the_two_layer_decoder_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mlx.core as mx

    decoder = _decoder(fast=True)
    _install_schedule(monkeypatch, decoder, [4, 3])
    output = decoder.generate(mx.zeros((7, 1, 512), dtype=mx.float32))
    mx.eval(output.cache)

    assert output.cache.shape == (2, 2, 1, 1)


def test_every_row_xcel_correction_preserves_bbox_span_state_selection() -> None:
    from docling_mlx._models.tableformer_v1.bbox import select_bbox_state_indices

    tokens = ["lcel", "nl", "lcel", "ucel", "fcel", "<end>"]
    selected, merge_endpoints = select_bbox_state_indices(tokens)

    assert selected == [0, 1, 2, 3]
    assert merge_endpoints == {0: 1, 2: 3}


def test_generation_runs_1024_steps_after_bos(monkeypatch: pytest.MonkeyPatch) -> None:
    import mlx.core as mx

    decoder = _decoder()
    calls = _install_schedule(monkeypatch, decoder, [4])
    output = decoder.generate(mx.zeros((1, 1, 512), dtype=mx.float32))
    mx.eval(output.generated_ids, output.hidden_states)

    assert output.generated_ids.shape == (1, 1025)
    assert output.hidden_states.shape == (1, 1024, 512)
    assert calls == list(range(1, 1025))


@pytest.mark.parametrize("steps", [0, 1025])
def test_generation_rejects_invalid_step_limits(steps: int) -> None:
    import mlx.core as mx

    with pytest.raises(ValueError, match="between 1"):
        _decoder().generate(mx.zeros((1, 1, 512), dtype=mx.float32), max_generation_steps=steps)
