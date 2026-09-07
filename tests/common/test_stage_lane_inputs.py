# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import huggingface_hub
import pytest

from tools.stage_lane_inputs import stage


def test_stage_only_caches_completed_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "artifact"
    calls = 0

    def snapshot_download(*, local_dir: Path, **_kwargs: object) -> str:
        nonlocal calls
        calls += 1
        local_dir.mkdir(exist_ok=True)
        (local_dir / "model.safetensors").write_bytes(b"complete")
        (local_dir / ".cache").mkdir()
        if calls == 1:
            raise RuntimeError("interrupted")
        return str(local_dir)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)

    with pytest.raises(RuntimeError, match="interrupted"):
        stage("owner/model", "0" * 40, target)
    assert not target.exists()

    assert stage("owner/model", "0" * 40, target) == target
    assert (target / "model.safetensors").read_bytes() == b"complete"
    assert not (target / ".cache").exists()
    assert calls == 2

    assert stage("owner/model", "0" * 40, target) == target
    assert calls == 2
