# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Granite Vision 4.1 chart-output parsing.

The parsing and table-construction rules are adapted from Docling's
Granite Vision chart stage (MIT).  Keeping them local avoids importing the
upstream Torch-backed stage from the MLX component.
"""

from __future__ import annotations

import ast
import re
from io import StringIO
from typing import Any

import pandas as pd
from docling_core.types.doc import TableCell, TableData


def extract_csv_table(decoded_text: str) -> TableData:
    """Parse a fenced or bare chart CSV response into Docling table data."""

    csv_match = re.search(r"```csv\s*\r?\n(.*?)\r?\n```", decoded_text, re.DOTALL)
    if csv_match:
        csv_content = csv_match.group(1).strip()
    else:
        csv_content = re.sub(r"^```+(?:csv)?\s*", "", decoded_text.strip())
        csv_content = re.sub(r"```+\s*$", "", csv_content).strip()

    dataframe = pd.read_csv(StringIO(csv_content), header=None)
    first_row_is_header = len(dataframe) > 0 and all(
        not _is_numeric(value) for value in dataframe.iloc[0]
    )
    cells: list[TableCell] = []

    if first_row_is_header:
        for col_idx, value in enumerate(dataframe.iloc[0]):
            cells.append(
                TableCell(
                    text=str(value),
                    start_row_offset_idx=0,
                    end_row_offset_idx=1,
                    start_col_offset_idx=col_idx,
                    end_col_offset_idx=col_idx + 1,
                    row_span=1,
                    col_span=1,
                    column_header=True,
                    row_header=False,
                    row_section=False,
                    fillable=False,
                )
            )

    data = dataframe.iloc[1:] if first_row_is_header else dataframe
    row_offset = 1 if first_row_is_header else 0
    for row_idx, (_, row) in enumerate(data.iterrows()):
        for col_idx, value in enumerate(row):
            cells.append(
                TableCell(
                    text="" if pd.isna(value) else str(value),
                    start_row_offset_idx=row_idx + row_offset,
                    end_row_offset_idx=row_idx + row_offset + 1,
                    start_col_offset_idx=col_idx,
                    end_col_offset_idx=col_idx + 1,
                    row_span=1,
                    col_span=1,
                    column_header=False,
                    row_header=not _is_numeric(value),
                    row_section=False,
                    fillable=False,
                )
            )

    return TableData(
        table_cells=cells,
        num_rows=len(dataframe),
        num_cols=len(dataframe.columns),
    )


def extract_python_code(decoded_text: str) -> str | None:
    """Extract fenced Python or syntactically-valid bare MLX output."""

    match = re.search(r"```python\s*\r?\n(.*?)\r?\n```", decoded_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    candidate = decoded_text.strip()
    if not candidate or "```" in candidate:
        return None
    try:
        ast.parse(candidate)
    except SyntaxError:
        return None
    return candidate


def _is_numeric(value: Any) -> bool:
    if pd.isna(value):
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


__all__ = ["extract_csv_table", "extract_python_code"]
