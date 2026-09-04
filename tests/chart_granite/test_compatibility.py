# SPDX-License-Identifier: Apache-2.0

"""Differential parser checks against Docling 2.123.1's Torch-backed stage."""

from __future__ import annotations

import pytest

from docling_mlx.stages._chart_granite import extract_csv_table

pytestmark = pytest.mark.release


@pytest.mark.parametrize(
    "source",
    [
        "Region,Revenue\nNorth,12.5\nSouth,9",
        "1,2\n3,4",
        "Category,Value\nUnknown,\nKnown,3.0",
        'Category,Value\n"North, East",3.5',
        "a,b\n1",
    ],
)
def test_csv_parser_matches_docling_2_123_1(source: str) -> None:
    from docling.models.stages.chart_extraction.granite_vision import (
        ChartExtractionModelGraniteVisionV4,
        _BaseChartExtractionModelGraniteVision,
    )

    class UpstreamParser:
        _is_numeric = _BaseChartExtractionModelGraniteVision._is_numeric
        _dataframe_to_tabledata = _BaseChartExtractionModelGraniteVision._dataframe_to_tabledata
        _extract_csv_to_dataframe = ChartExtractionModelGraniteVisionV4._extract_csv_to_dataframe

    oracle = UpstreamParser()
    dataframe = oracle._extract_csv_to_dataframe(source)
    expected = oracle._dataframe_to_tabledata(dataframe)

    assert extract_csv_table(source).model_dump(mode="json") == expected.model_dump(mode="json")
