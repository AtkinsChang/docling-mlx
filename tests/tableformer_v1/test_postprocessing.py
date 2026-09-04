# SPDX-License-Identifier: Apache-2.0

"""Accurate TableFormer structural postprocessing contracts."""

from __future__ import annotations

import pytest

from docling_mlx.engines.table_structure.tableformer_v1.postprocessing import (
    _intersection_over_pdf,
    _repair_unmatched_columns,
    pad_invalid_rows,
    postprocess_prediction,
    repair_legacy_bbox_desync,
)


def _pinned_matching_output(
    otsl: list[str],
    boxes: list[list[float]],
    classes: list[int],
    pdf_cells: list[dict[str, object]],
) -> list[dict[str, object]]:
    from docling_ibm_models.tableformer.data_management.matching_post_processor import (
        MatchingPostProcessor,
    )
    from docling_ibm_models.tableformer.data_management.tf_cell_matcher import CellMatcher
    from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor
    from docling_ibm_models.tableformer.otsl import otsl_to_html

    config = {"predict": {"pdf_cell_iou_thres": 0.05}}
    prediction = {
        "bboxes": boxes,
        "classes": classes,
        "html_seq": otsl_to_html(otsl, False),
        "rs_seq": otsl,
    }
    details = CellMatcher(config).match_cells(
        {"tokens": pdf_cells, "height": 1, "width": 1},
        [0, 0, 1, 1],
        prediction,
    )
    details = MatchingPostProcessor(config).process(details)
    predictor = object.__new__(TFPredictor)
    source = predictor._generate_tf_response(details["table_cells"], details["matches"])
    source.sort(key=lambda item: item["cell_id"])
    return predictor._merge_tf_output(source, details["pdf_cells"])


def test_invalid_rows_are_padded_with_horizontal_continuations() -> None:
    assert pad_invalid_rows(["fcel", "fcel", "nl", "fcel", "nl"]) == [
        "fcel",
        "fcel",
        "nl",
        "fcel",
        "lcel",
        "nl",
    ]


def test_legacy_html_round_trip_drops_spans_larger_than_twenty() -> None:
    within_limit, _, _ = postprocess_prediction(
        ["fcel", *("lcel" for _ in range(19)), "nl"],
        [[0.0, 0.0, 1.0, 1.0]],
        [1],
        do_cell_matching=False,
    )
    oversized_column, _, _ = postprocess_prediction(
        ["fcel", *("lcel" for _ in range(20)), "nl"],
        [[0.0, 0.0, 1.0, 1.0]],
        [1],
        do_cell_matching=False,
    )
    oversized_row, _, _ = postprocess_prediction(
        ["fcel", "nl", *(["ucel", "nl"] * 20)],
        [[0.0, 0.0, 1.0, 1.0]],
        [1],
        do_cell_matching=False,
    )

    assert within_limit[0]["col_span"] == 20
    assert oversized_column[0]["col_span"] == 1
    assert oversized_row[0]["row_span"] == 1


def test_dummy_output_compacts_sparse_raw_columns() -> None:
    otsl = [
        "ched",
        "ched",
        "ched",
        *(["lcel"] * 30),
        "nl",
        "ched",
        "ched",
        "ched",
        "ched",
        *(["lcel"] * 30),
        "nl",
        "fcel",
        "fcel",
        "fcel",
        *(["lcel"] * 30),
    ]
    cells, rows, columns = postprocess_prediction(
        otsl,
        [[0.0, 0.0, 1.0, 1.0]] * 10,
        [2] * 10,
        do_cell_matching=False,
    )

    assert (rows, columns) == (3, 4)
    assert [(cell["start_row_offset_idx"], cell["start_col_offset_idx"]) for cell in cells] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 0),
        (2, 1),
        (2, 2),
    ]


def test_xcel_contributes_to_both_anchor_spans_and_remains_a_source_cell() -> None:
    cells, rows, columns = postprocess_prediction(
        ["fcel", "xcel", "nl", "ucel", "xcel", "nl"],
        [[0.1, 0.2, 0.9, 0.8]],
        [1],
        do_cell_matching=False,
    )

    assert (rows, columns) == (2, 2)
    assert cells[0]["row_span"] == 2
    assert cells[0]["col_span"] == 2
    assert [cell["label"] for cell in cells] == ["fcel", "xcel", "xcel"]
    assert cells[1]["bbox"] == {"l": 0.0, "t": 0.0, "r": 0.0, "b": 0.0}


def test_legacy_horizontal_endpoint_bbox_is_removed() -> None:
    boxes, classes = repair_legacy_bbox_desync(
        ["fcel", "lcel", "nl", "fcel", "fcel", "nl"],
        [[0.0, 0.0, 0.1, 0.1], [0.1, 0.0, 0.2, 0.1], [0.2, 0.0, 0.3, 0.1], [0.3, 0.0, 0.4, 0.1]],
        [0, 1, 2, 0],
    )
    assert boxes == [
        [0.0, 0.0, 0.1, 0.1],
        [0.2, 0.0, 0.3, 0.1],
        [0.3, 0.0, 0.4, 0.1],
    ]
    # The 2.123.1 compatibility repair removes only the box; class logits keep
    # their legacy order even though that leaves their lengths desynchronized.
    assert classes == [0, 1, 2, 0]


def test_matching_and_dummy_modes_keep_their_distinct_output_contracts() -> None:
    kwargs = {
        "otsl_seq": ["ched", "nl"],
        "bboxes": [[0.1, 0.2, 0.9, 0.8]],
        "classes": [1],
        "table_bbox": [10.0, 20.0, 110.0, 220.0],
        "pdf_cells": [
            {"id": 4, "text": " heading ", "bbox": {"l": 15, "t": 50, "r": 100, "b": 170}}
        ],
    }

    dummy, _, _ = postprocess_prediction(**kwargs, do_cell_matching=False)
    assert dummy[0]["bbox"] == {"l": 20.0, "t": 60.0, "r": 100.0, "b": 180.0}
    assert dummy[0]["text_cell_bboxes"] == []

    matched, _, _ = postprocess_prediction(**kwargs, do_cell_matching=True)
    assert matched[0]["text_cell_bboxes"] == [
        {"b": 170.0, "l": 15.0, "r": 100.0, "t": 50.0, "token": " heading "}
    ]
    assert (
        postprocess_prediction(
            ["ched", "nl"],
            [[0.1, 0.2, 0.9, 0.8]],
            [1],
            do_cell_matching=True,
        )[0]
        == []
    )


def test_matching_assigns_each_pdf_cell_once_and_aligns_the_predicted_bbox() -> None:
    cells, rows, columns = postprocess_prediction(
        ["fcel", "fcel", "nl"],
        [[0.0, 0.0, 0.6, 1.0], [0.4, 0.0, 1.0, 1.0]],
        [2, 2],
        pdf_cells=[{"id": 7, "text": "shared", "bbox": {"l": 0.45, "t": 0.1, "r": 0.55, "b": 0.9}}],
    )

    assert (rows, columns) == (1, 1)
    assert len(cells) == 1
    assert cells[0]["start_col_offset_idx"] == 0
    assert cells[0]["bbox"] == {"l": 0.45, "t": 0.1, "r": 0.55, "b": 0.9}
    assert cells[0]["text_cell_bboxes"] == [
        {"b": 0.9, "l": 0.45, "r": 0.55, "t": 0.1, "token": "shared"}
    ]


def test_matching_recovers_an_orphan_pdf_cell_at_its_row_and_column() -> None:
    cells, rows, columns = postprocess_prediction(
        ["fcel", "fcel", "nl", "fcel", "fcel", "nl"],
        [
            [0.0, 0.0, 0.45, 0.45],
            [0.55, 0.0, 1.0, 0.45],
            [0.0, 0.55, 0.45, 1.0],
            [1.2, 1.2, 1.3, 1.3],
        ],
        [2, 2, 2, 2],
        pdf_cells=[
            {"id": 0, "text": "a", "bbox": {"l": 0.1, "t": 0.1, "r": 0.4, "b": 0.4}},
            {"id": 1, "text": "b", "bbox": {"l": 0.6, "t": 0.1, "r": 0.9, "b": 0.4}},
            {"id": 2, "text": "c", "bbox": {"l": 0.1, "t": 0.6, "r": 0.4, "b": 0.9}},
            {"id": 3, "text": "d", "bbox": {"l": 0.6, "t": 0.6, "r": 0.9, "b": 0.9}},
        ],
    )

    assert (rows, columns) == (2, 2)
    assert [(cell["start_row_offset_idx"], cell["start_col_offset_idx"]) for cell in cells] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert cells[-1]["bbox"] == {"l": 0.6, "t": 0.6, "r": 0.9, "b": 0.9}
    assert cells[-1]["text_cell_bboxes"] == [{"b": 0.9, "l": 0.6, "r": 0.9, "t": 0.6, "token": "d"}]


def test_matching_reindexes_surviving_columns_without_gaps() -> None:
    cells, rows, columns = postprocess_prediction(
        ["fcel", "fcel", "fcel", "nl"],
        [[0.0, 0.0, 0.3, 1.0], [0.35, 0.0, 0.65, 1.0], [0.7, 0.0, 1.0, 1.0]],
        [2, 2, 2],
        pdf_cells=[
            {"id": 0, "text": "left", "bbox": [0.05, 0.1, 0.25, 0.9]},
            {"id": 1, "text": "right", "bbox": [0.75, 0.1, 0.95, 0.9]},
        ],
    )

    assert (rows, columns) == (1, 2)
    assert [cell["start_col_offset_idx"] for cell in cells] == [0, 1]


@pytest.mark.release
def test_row_padding_and_legacy_repair_match_docling_2_123_1() -> None:
    from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor
    from docling_ibm_models.tableformer.otsl import otsl_pad_to_sqr, otsl_to_html

    invalid = ["fcel", "fcel", "nl", "fcel", "nl"]
    assert pad_invalid_rows(invalid) == otsl_pad_to_sqr(invalid, "lcel")

    otsl = ["fcel", "lcel", "nl", "fcel", "fcel", "nl"]
    boxes = [[0.0, 0.0, 0.1, 0.1], [0.1, 0.0, 0.2, 0.1], [0.2, 0.0, 0.3, 0.1], [0.3, 0.0, 0.4, 0.1]]
    source = object.__new__(TFPredictor)
    _, expected = source._check_bbox_sync({"html_seq": otsl_to_html(otsl, False), "bboxes": boxes})
    actual, _ = repair_legacy_bbox_desync(otsl, boxes, [0, 1, 2, 0])
    assert actual == expected


@pytest.mark.release
@pytest.mark.parametrize(
    "otsl",
    [
        ["fcel", "lcel", "nl", "fcel", "fcel", "nl"],
        ["fcel", "xcel", "nl", "ucel", "xcel", "nl"],
    ],
)
def test_structural_cells_match_docling_2_123_1(otsl: list[str]) -> None:
    from docling_ibm_models.tableformer.data_management.tf_cell_matcher import CellMatcher
    from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor
    from docling_ibm_models.tableformer.otsl import otsl_to_html

    boxes = [[index / 10, 0.0, (index + 1) / 10, 0.1] for index in range(4)]
    classes = [0, 1, 2, 0]
    html = otsl_to_html(otsl, False)
    _, source_boxes = object.__new__(TFPredictor)._check_bbox_sync(
        {"html_seq": html, "bboxes": boxes}
    )
    source = CellMatcher({"predict": {"pdf_cell_iou_thres": 0.05}})._build_table_cells(
        html, otsl, source_boxes, classes
    )
    actual, _, _ = postprocess_prediction(otsl, boxes, classes, do_cell_matching=False)

    assert len(actual) == len(source)
    for native, expected in zip(actual, source, strict=True):
        assert native["label"] == expected["label"]
        assert native["start_row_offset_idx"] == expected["row_id"]
        assert native["start_col_offset_idx"] == expected["column_id"]
        assert native["col_span"] == expected.get("colspan_val", 1)
        assert native["row_span"] == expected.get("rowspan_val", 1)
        assert list(native["bbox"].values()) == expected["bbox"]
        assert native["cell_class"] == expected["cell_class"]


@pytest.mark.release
def test_matching_repair_matches_docling_ibm_models_3_14_0() -> None:
    otsl = ["fcel", "fcel", "nl", "fcel", "fcel", "nl"]
    boxes = [
        [0.0, 0.0, 0.45, 0.45],
        [0.55, 0.0, 1.0, 0.45],
        [0.0, 0.55, 0.45, 1.0],
        [1.2, 1.2, 1.3, 1.3],
    ]
    classes = [2, 2, 2, 2]
    pdf_cells = [
        {"id": 0, "text": "a", "bbox": {"l": 0.1, "t": 0.1, "r": 0.4, "b": 0.4}},
        {"id": 1, "text": "b", "bbox": {"l": 0.6, "t": 0.1, "r": 0.9, "b": 0.4}},
        {"id": 2, "text": "c", "bbox": {"l": 0.1, "t": 0.6, "r": 0.4, "b": 0.9}},
        {"id": 3, "text": "d", "bbox": {"l": 0.6, "t": 0.6, "r": 0.9, "b": 0.9}},
    ]
    source = _pinned_matching_output(otsl, boxes, classes, pdf_cells)

    actual, _, _ = postprocess_prediction(
        otsl,
        boxes,
        classes,
        pdf_cells=pdf_cells,
    )
    source_keys = set(source[0])
    assert [{key: cell[key] for key in source_keys} for cell in actual] == source


@pytest.mark.release
def test_matching_repair_weights_a_cell_once_per_matching_pdf_word() -> None:
    otsl = ["fcel", "nl", "fcel", "nl", "fcel", "nl"]
    boxes = [
        [0.0, 0.0, 0.1, 0.2],
        [0.4, 0.4, 0.5, 0.6],
        [0.8, 0.8, 0.9, 1.0],
    ]
    classes = [2, 2, 2]
    pdf_cells = [
        {"id": 0, "text": "a1", "bbox": {"l": 0.0, "t": 0.02, "r": 0.04, "b": 0.08}},
        {"id": 1, "text": "a2", "bbox": {"l": 0.06, "t": 0.1, "r": 0.1, "b": 0.18}},
        {"id": 2, "text": "b", "bbox": {"l": 0.4, "t": 0.42, "r": 0.48, "b": 0.58}},
        {"id": 3, "text": "c", "bbox": {"l": 0.02, "t": 0.82, "r": 0.08, "b": 0.98}},
    ]
    table_cells = [
        {
            "cell_id": index,
            "bbox": bbox,
            "row_span": 1,
            "col_span": 1,
            "start_col_offset_idx": 0,
            "cell_class": 2,
        }
        for index, bbox in enumerate(boxes)
    ]
    repaired = _repair_unmatched_columns(
        table_cells,
        _intersection_over_pdf(table_cells, pdf_cells),
        1,
    )

    source = _pinned_matching_output(otsl, boxes, classes, pdf_cells)
    actual, rows, columns = postprocess_prediction(
        otsl,
        boxes,
        classes,
        pdf_cells=pdf_cells,
    )

    assert (rows, columns) == (3, 1)
    assert repaired[-1]["bbox"] == pytest.approx([0.0, 0.8, 0.1, 1.0])
    assert source[-1]["text_cell_bboxes"][0]["token"] == "c"
    source_keys = set(source[0])
    assert [{key: cell[key] for key in source_keys} for cell in actual] == source
