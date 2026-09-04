# SPDX-License-Identifier: Apache-2.0

"""Portable OTSL parser contracts copied from realistic Granite outputs."""

from __future__ import annotations

from docling_mlx.stages._otsl import parse_otsl_output


def test_parse_closed_and_open_otsl_forms() -> None:
    closed = "<otsl><ched>Name</ched><ched>Value</ched><nl><fcel>A</fcel><fcel>1</fcel><nl></otsl>"
    opened = "<ched>Name<ched>Value<nl><fcel>A<fcel>1<nl>"

    closed_result = parse_otsl_output(closed)
    open_result = parse_otsl_output(opened)

    assert closed_result[0] == ["ched", "ched", "nl", "fcel", "fcel", "nl"]
    assert open_result[0] == closed_result[0]
    assert [(cell.text, cell.column_header) for cell in closed_result[1]] == [
        ("Name", True),
        ("Value", True),
        ("A", False),
        ("1", False),
    ]
    assert [(cell.text, cell.column_header) for cell in open_result[1]] == [
        ("Name", True),
        ("Value", True),
        ("A", False),
        ("1", False),
    ]
    assert closed_result[2:] == (2, 2)
    assert open_result[2:] == (2, 2)


def test_parse_self_closing_span_tokens() -> None:
    text = (
        "<ched>Region</ched><lcel/><ched>Value</ched><nl/>"
        "<fcel>North</fcel><lcel/><fcel>1</fcel><nl/>"
        "<ucel/><xcel/><fcel>2</fcel><nl/>"
    )

    otsl, cells, rows, cols = parse_otsl_output(text)

    assert otsl == [
        "ched",
        "lcel",
        "ched",
        "nl",
        "fcel",
        "lcel",
        "fcel",
        "nl",
        "ucel",
        "xcel",
        "fcel",
        "nl",
    ]
    assert (rows, cols) == (3, 3)
    region, value, north, one, two = cells
    assert (region.col_span, region.row_span) == (2, 1)
    assert (value.start_col_offset_idx, value.end_col_offset_idx) == (2, 3)
    assert (north.col_span, north.row_span) == (2, 2)
    assert (one.start_row_offset_idx, one.start_col_offset_idx) == (1, 2)
    assert (two.start_row_offset_idx, two.start_col_offset_idx) == (2, 2)


def test_parse_empty_or_non_otsl_text_as_empty_table() -> None:
    for source in ["", "plain prose without tags"]:
        assert parse_otsl_output(source) == ([], [], 0, 0)
