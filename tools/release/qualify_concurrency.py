# SPDX-License-Identifier: Apache-2.0

"""Qualify concurrent MLX engines in required fresh-process pairings."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from time import perf_counter
from typing import Any, Literal

import numpy as np
from docling.datamodel.pipeline_options import TableFormerMode

from tools._common.benchmark import peak_rss_bytes

ROOT = Path(__file__).resolve().parents[2]
Participant = Literal["figure", "tableformer", "tableformer_v1"]
CaseName = Literal["figure-tableformer", "tableformer-v1-shared"]
CASES: dict[CaseName, tuple[Participant, Participant]] = {
    "figure-tableformer": ("figure", "tableformer"),
    "tableformer-v1-shared": ("tableformer_v1", "tableformer_v1"),
}
REQUIRED_FILES = ("model.safetensors", "config.json", "preprocessor_config.json")
TABLEFORMER_REQUIRED_FILES = REQUIRED_FILES + (
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
ROUNDS = 3


def _tableformer_v1_required_files(mode: TableFormerMode) -> tuple[str, ...]:
    profile = mode.value
    return tuple(
        f"{profile}/{name}"
        for name in (
            "model.safetensors",
            "config.json",
            "preprocessor_config.json",
            "generation_config.json",
        )
    )


def _require_artifact(
    path: Path,
    label: str,
    *,
    tableformer: bool = False,
    tableformer_v1_mode: TableFormerMode | None = None,
) -> Path:
    resolved = path.expanduser().resolve()
    required = (
        _tableformer_v1_required_files(tableformer_v1_mode)
        if tableformer_v1_mode is not None
        else TABLEFORMER_REQUIRED_FILES
        if tableformer
        else REQUIRED_FILES
    )
    missing = [name for name in required if not (resolved / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{label} artifact {resolved} is missing {missing}")
    return resolved


def _make_engine(
    participant: Participant,
    artifact: Path,
    tableformer_v1_mode: TableFormerMode = TableFormerMode.ACCURATE,
) -> Any:
    if participant == "tableformer_v1":
        from docling_mlx.engines.table_structure.tableformer_v1 import (
            TableFormerV1Engine,
            TableFormerV1EngineOptions,
            TableFormerV1ModelSpec,
        )

        return TableFormerV1Engine(
            TableFormerV1ModelSpec(path=artifact),
            TableFormerV1EngineOptions(checkpoint_subdirectory=tableformer_v1_mode.value),
        )
    if participant == "tableformer":
        from docling_mlx.engines.table_structure.tableformer_v2 import (
            TableFormerV2Engine,
            TableFormerV2ModelSpec,
        )

        return TableFormerV2Engine(TableFormerV2ModelSpec(path=artifact))
    if participant == "figure":
        from docling_mlx.engines.image_classification.efficientnet import (
            EfficientNetEngine,
            EfficientNetEngineOptions,
            EfficientNetModelSpec,
        )

        return EfficientNetEngine(
            EfficientNetModelSpec(path=artifact), EfficientNetEngineOptions(top_k=None)
        )
    raise AssertionError(f"unsupported participant: {participant}")


def _run_one(
    participant: Participant,
    engine: Any,
    image_path: Path,
    *,
    lane: int,
    round_index: int,
) -> dict[str, Any]:
    from PIL import Image

    metadata = {"lane": lane, "round": round_index, "participant": participant}
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    if participant in {"tableformer", "tableformer_v1"}:
        output = engine.predict([image])[0]
        return {
            "token_ids": output.token_ids,
            "bboxes": output.cell_bboxes,
            "metadata": metadata,
        }
    if participant == "figure":
        output = engine.predict([image])[0]
        return {
            "label_ids": output.label_ids,
            "scores": output.scores,
            "metadata": metadata,
        }

    from docling.models.inference_engines.object_detection.base import (
        ObjectDetectionEngineInput,
    )

    output = engine.predict_batch([ObjectDetectionEngineInput(image=image, metadata=metadata)])[0]
    return {
        "label_ids": output.label_ids,
        "scores": output.scores,
        "bboxes": output.bboxes,
        "metadata": output.metadata,
    }


def _assert_equivalent(
    actual: dict[str, Any],
    expected: dict[str, Any],
    participant: Participant,
    *,
    lane: int,
    round_index: int,
) -> None:
    expected_metadata = {"lane": lane, "round": round_index, "participant": participant}
    if actual["metadata"] != expected_metadata:
        raise AssertionError(
            f"{participant} lane {lane} round {round_index} metadata/order mismatch"
        )
    if participant in {"tableformer", "tableformer_v1"}:
        if actual["token_ids"] != expected["token_ids"]:
            raise AssertionError(f"{participant} lane {lane} round {round_index} token mismatch")
        np.testing.assert_allclose(actual["bboxes"], expected["bboxes"], rtol=0, atol=1e-4)
        return
    if actual["label_ids"] != expected["label_ids"]:
        raise AssertionError(f"{participant} lane {lane} round {round_index} label mismatch")
    np.testing.assert_allclose(actual["scores"], expected["scores"], rtol=0, atol=1e-6)
    if participant != "figure":
        np.testing.assert_allclose(actual["bboxes"], expected["bboxes"], rtol=0, atol=5e-4)


def _worker(
    case: CaseName,
    artifacts: dict[Participant, Path],
    figure_image: Path,
    tableformer_image: Path,
    tableformer_v1_mode: TableFormerMode = TableFormerMode.ACCURATE,
) -> dict[str, Any]:
    import mlx.core as mx

    participants = CASES[case]
    images = {
        "figure": figure_image,
        "tableformer": tableformer_image,
        "tableformer_v1": tableformer_image,
    }
    if case == "tableformer-v1-shared":
        shared = _make_engine("tableformer_v1", artifacts["tableformer_v1"], tableformer_v1_mode)
        concurrent_engines = [shared, shared]
    else:
        concurrent_engines = [
            _make_engine(participant, artifacts[participant], tableformer_v1_mode)
            for participant in participants
        ]
    mx.reset_peak_memory()
    started = perf_counter()
    barrier = Barrier(2)

    def run_lane(lane: int) -> list[dict[str, Any]]:
        participant = participants[lane]
        outputs = []
        for round_index in range(ROUNDS):
            barrier.wait(timeout=180)
            outputs.append(
                _run_one(
                    participant,
                    concurrent_engines[lane],
                    images[participant],
                    lane=lane,
                    round_index=round_index,
                )
            )
        return outputs

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_lane, lane) for lane in range(2)]
        concurrent_outputs = [future.result(timeout=900) for future in futures]
    concurrent_wall_s = perf_counter() - started

    concurrent_engines.clear()
    gc.collect()
    clear_cache = getattr(mx, "clear_cache", None)
    if clear_cache is not None:
        clear_cache()

    baselines: list[dict[str, Any]] = []
    for lane, participant in enumerate(participants):
        engine = _make_engine(participant, artifacts[participant], tableformer_v1_mode)
        baselines.append(
            _run_one(participant, engine, images[participant], lane=lane, round_index=0)
        )
        del engine
        gc.collect()
        if clear_cache is not None:
            clear_cache()

    for lane, participant in enumerate(participants):
        for round_index, actual in enumerate(concurrent_outputs[lane]):
            _assert_equivalent(
                actual,
                baselines[lane],
                participant,
                lane=lane,
                round_index=round_index,
            )

    peak = mx.get_peak_memory()
    if not isinstance(peak, int):
        raise TypeError("mlx.get_peak_memory() must return an integer")
    return {
        "case": case,
        "participants": list(participants),
        "tableformer_v1_mode": (
            tableformer_v1_mode.value if "tableformer_v1" in participants else None
        ),
        "rounds": ROUNDS,
        "cold_initialization_raced": True,
        "status": "passed",
        "wall_time_s": perf_counter() - started,
        "concurrent_wall_time_s": concurrent_wall_s,
        "process_peak_rss_bytes": peak_rss_bytes(),
        "mlx_peak_memory_bytes": peak,
    }


def qualify(
    figure_artifact: Path,
    tableformer_artifact: Path,
    tableformer_v1_artifact: Path,
    figure_image: Path,
    tableformer_image: Path,
) -> dict[str, Any]:
    artifacts = {
        "figure": _require_artifact(figure_artifact, "Figure"),
        "tableformer": _require_artifact(
            tableformer_artifact,
            "TableFormerV2",
            tableformer=True,
        ),
        "tableformer_v1": _require_artifact(
            tableformer_v1_artifact,
            "TableFormerV1 accurate",
            tableformer_v1_mode=TableFormerMode.ACCURATE,
        ),
    }
    _require_artifact(
        artifacts["tableformer_v1"],
        "TableFormerV1 fast",
        tableformer_v1_mode=TableFormerMode.FAST,
    )
    reports = []
    runs = [("figure-tableformer", TableFormerMode.ACCURATE)] + [
        ("tableformer-v1-shared", mode) for mode in (TableFormerMode.ACCURATE, TableFormerMode.FAST)
    ]
    for case, mode in runs:
        command = [
            sys.executable,
            "-m",
            "tools.release.qualify_concurrency",
            "--case",
            case,
            "--figure-artifact",
            str(artifacts["figure"]),
            "--tableformer-artifact",
            str(artifacts["tableformer"]),
            "--tableformer-v1-artifact",
            str(artifacts["tableformer_v1"]),
            "--tableformer-v1-profile",
            mode.value,
            "--figure-image",
            str(figure_image),
            "--tableformer-image",
            str(tableformer_image),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=1_200,
        )
        if completed.returncode:
            raise RuntimeError(
                f"{case} fresh-process qualification failed\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        marker = next(
            (
                line.removeprefix("CONCURRENCY_RESULT=")
                for line in completed.stdout.splitlines()
                if line.startswith("CONCURRENCY_RESULT=")
            ),
            None,
        )
        if marker is None:
            raise RuntimeError(f"{case} did not emit a concurrency result")
        reports.append(json.loads(marker))
    return {
        "status": "passed",
        "fresh_processes": len(reports),
        "tableformer_v1_modes": [
            TableFormerMode.ACCURATE.value,
            TableFormerMode.FAST.value,
        ],
        "cases": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES))
    parser.add_argument(
        "--figure-artifact",
        type=Path,
        default=ROOT / ".artifacts/document-figure-classifier",
    )
    parser.add_argument(
        "--tableformer-artifact",
        type=Path,
        default=ROOT / ".artifacts/tableformer-v2",
    )
    parser.add_argument(
        "--tableformer-v1-artifact",
        type=Path,
        default=ROOT / ".artifacts/tableformer-v1",
    )
    parser.add_argument(
        "--tableformer-v1-profile",
        choices=(TableFormerMode.ACCURATE.value, TableFormerMode.FAST.value),
        default=TableFormerMode.ACCURATE.value,
    )
    parser.add_argument(
        "--figure-image",
        type=Path,
        default=ROOT / "tests/fixtures/document_figure/reference_images/bar_chart.png",
    )
    parser.add_argument(
        "--tableformer-image",
        type=Path,
        default=ROOT / "tests/fixtures/granite_vision/table-simple.png",
    )
    args = parser.parse_args()
    tableformer_v1_mode = TableFormerMode(args.tableformer_v1_profile)
    if args.case is not None:
        requested = set(CASES[args.case])
        artifacts: dict[Participant, Path] = {}
        if "figure" in requested:
            artifacts["figure"] = _require_artifact(args.figure_artifact, "Figure")
        if "tableformer" in requested:
            artifacts["tableformer"] = _require_artifact(
                args.tableformer_artifact,
                "TableFormerV2",
                tableformer=True,
            )
        if "tableformer_v1" in requested:
            artifacts["tableformer_v1"] = _require_artifact(
                args.tableformer_v1_artifact,
                f"TableFormerV1 {tableformer_v1_mode.value}",
                tableformer_v1_mode=tableformer_v1_mode,
            )
        result = _worker(
            args.case,
            artifacts,
            args.figure_image,
            args.tableformer_image,
            tableformer_v1_mode,
        )
        print("CONCURRENCY_RESULT=" + json.dumps(result, separators=(",", ":")))
        return
    print(
        json.dumps(
            qualify(
                args.figure_artifact,
                args.tableformer_artifact,
                args.tableformer_v1_artifact,
                args.figure_image,
                args.tableformer_image,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
