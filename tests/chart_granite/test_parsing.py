# SPDX-License-Identifier: Apache-2.0

"""Granite Vision chart parser compatibility tests."""

from __future__ import annotations

import pandas as pd
import pytest

from docling_mlx.stages._chart_granite import (
    extract_csv_table,
    extract_python_code,
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("```csv\nYear,Sales\n2025,10\n```", ["Year", "Sales", "2025", "10"]),
        ("Year,Sales\n2025,10", ["Year", "Sales", "2025", "10"]),
        ("```\nYear,Sales\n2025,10\n```", ["Year", "Sales", "2025", "10"]),
        ('Category,Value\n"North, East",3.5', ["Category", "Value", "North, East", "3.5"]),
    ],
)
def test_csv_parser_accepts_docling_granite_output_shapes(
    source: str,
    expected: list[str],
) -> None:
    table = extract_csv_table(source)

    assert [cell.text for cell in table.table_cells] == expected
    assert table.num_rows == 2
    assert table.num_cols == 2


@pytest.mark.parametrize("source", ["", "```csv\n\n```", 'a,b\n"unterminated'])
def test_csv_parser_rejects_empty_or_invalid_tables(source: str) -> None:
    with pytest.raises((pd.errors.EmptyDataError, pd.errors.ParserError)):
        extract_csv_table(source)


def test_csv_parser_matches_docling_sparse_row_padding() -> None:
    table = extract_csv_table("a,b\n1")

    assert (table.num_rows, table.num_cols) == (2, 2)
    assert [cell.text for cell in table.table_cells] == ["a", "b", "1", ""]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("before\n```python\nprint(1)\n```\nafter", "print(1)"),
        (
            "import matplotlib.pyplot as plt\nplt.plot([1, 2])",
            "import matplotlib.pyplot as plt\nplt.plot([1, 2])",
        ),
        ("x =", None),
        ("```javascript\nconsole.log(1)\n```", None),
    ],
)
def test_python_parser_accepts_granite_output_shapes(
    source: str,
    expected: str | None,
) -> None:
    assert extract_python_code(source) == expected


def test_python_parser_returns_fenced_text_without_validation() -> None:
    assert extract_python_code("") is None
    assert extract_python_code("not a fenced Python block") is None
    assert extract_python_code("```python\n\n```") == ""
    assert extract_python_code("```python\nx=\n```") == "x="
