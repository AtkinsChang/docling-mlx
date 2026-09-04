# Matching behavior follows docling-ibm-models (tableformer); implemented independently.
# SPDX-License-Identifier: Apache-2.0
"""Pure-Python structural postprocessing for TableFormer v1 output."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import groupby
from statistics import median
from typing import Any

_CELL_TOKENS = frozenset({"fcel", "ecel", "ched", "rhed", "srow"})
_STRUCTURAL_CELL_TOKENS = _CELL_TOKENS | {"xcel"}
_SPECIAL_TOKENS = frozenset({"<pad>", "<unk>", "<start>", "<end>"})
_MAX_LEGACY_SPAN = 20


def pad_invalid_rows(tokens: Sequence[str], pad_token: str = "lcel") -> list[str]:
    """Pad non-rectangular OTSL rows exactly as the source HTML converter does."""
    cleaned = [token for token in tokens if token not in _SPECIAL_TOKENS]
    rows = [
        list(row)
        for is_newline, row in groupby(cleaned, lambda token: token == "nl")
        if not is_newline
    ]
    if not rows:
        return []
    width = max(map(len, rows))
    return [token for row in rows for token in (*row, *([pad_token] * (width - len(row))), "nl")]


def repair_legacy_bbox_desync(
    tokens: Sequence[str],
    bboxes: Sequence[Sequence[float]],
    classes: Sequence[int],
) -> tuple[list[list[float]], list[int]]:
    """Drop the extra endpoint bbox emitted after legacy horizontal spans."""
    boxes = [list(box) for box in bboxes]
    cell_classes = list(classes)
    expected = sum(token in _CELL_TOKENS for token in tokens)
    if len(boxes) == expected:
        return boxes, cell_classes

    padded = pad_invalid_rows(tokens)
    rows = [
        list(row)
        for is_newline, row in groupby(padded, lambda token: token == "nl")
        if not is_newline
    ]
    delete_indices: list[int] = []
    cell_index = 0
    for row in rows:
        for column, token in enumerate(row):
            if token not in _CELL_TOKENS:
                continue
            if column + 1 < len(row) and row[column + 1] in {"lcel", "xcel"}:
                delete_indices.append(cell_index + 1)
            cell_index += 1

    for index in reversed(delete_indices):
        if len(boxes) <= expected:
            break
        if index < len(boxes):
            del boxes[index]
    return boxes, cell_classes


def _bbox_list(bbox: Mapping[str, Any] | Sequence[float]) -> list[float]:
    if isinstance(bbox, Mapping):
        return [float(bbox[key]) for key in ("l", "t", "r", "b")]
    if len(bbox) != 4:
        raise ValueError("Expected a four-coordinate bounding box")
    return [float(value) for value in bbox]


def _translate_bbox(bbox: Sequence[float], table_bbox: Sequence[float] | None) -> list[float]:
    if len(bbox) != 4:
        raise ValueError("Expected a four-coordinate bounding box")
    if table_bbox is None:
        return [float(value) for value in bbox]
    if len(table_bbox) != 4:
        raise ValueError("Expected a four-coordinate table bounding box")
    left, top, right, bottom = (float(value) for value in table_bbox)
    width, height = right - left, bottom - top
    return [
        left + float(bbox[0]) * width,
        top + float(bbox[1]) * height,
        left + float(bbox[2]) * width,
        top + float(bbox[3]) * height,
    ]


def _intersection_over_pdf(
    table_cells: Sequence[dict[str, Any]],
    pdf_cells: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, float | int]]]:
    matches: dict[str, list[dict[str, float | int]]] = {}
    for table_cell in table_cells:
        table_bbox = _bbox_list(table_cell["bbox"])  # type: ignore[arg-type]
        for pdf_cell in pdf_cells:
            pdf_bbox = _bbox_list(pdf_cell["bbox"])
            width = min(table_bbox[2], pdf_bbox[2]) - max(table_bbox[0], pdf_bbox[0])
            height = min(table_bbox[3], pdf_bbox[3]) - max(table_bbox[1], pdf_bbox[1])
            pdf_area = (pdf_bbox[2] - pdf_bbox[0]) * (pdf_bbox[3] - pdf_bbox[1])
            if width <= 0 or height <= 0 or pdf_area <= 0:
                continue
            matches.setdefault(str(pdf_cell["id"]), []).append(
                {
                    "table_cell_id": int(table_cell["cell_id"]),
                    "iopdf": width * height / pdf_area,
                }
            )
    return matches


def _repair_unmatched_columns(
    table_cells: Sequence[dict[str, Any]],
    matches: Mapping[str, Sequence[Mapping[str, float | int]]],
    num_cols: int,
) -> list[dict[str, Any]]:
    repaired: list[dict[str, Any]] = []
    for column in range(num_cols):
        column_cells = [cell for cell in table_cells if int(cell["start_col_offset_idx"]) == column]
        good = [
            cell
            for cell in column_cells
            if int(cell["cell_class"]) > 1
            for match_list in matches.values()
            for match in match_list
            if int(match["table_cell_id"]) == int(cell["cell_id"])
        ]
        lefts = [_bbox_list(cell["bbox"])[0] for cell in good]  # type: ignore[arg-type]
        middles = [
            (_bbox_list(cell["bbox"])[0] + _bbox_list(cell["bbox"])[2]) / 2  # type: ignore[arg-type]
            for cell in good
        ]
        rights = [_bbox_list(cell["bbox"])[2] for cell in good]  # type: ignore[arg-type]
        alignment = "left"
        if lefts:
            deltas = [
                max(lefts) - min(lefts),
                max(middles) - min(middles),
                max(rights) - min(rights),
            ]
            alignment = ("left", "middle", "right")[deltas.index(min(deltas))]

        median_cells = [
            cell for cell in good if int(cell["row_span"]) == 1 and int(cell["col_span"]) == 1
        ]
        coordinates = {
            "left": [_bbox_list(cell["bbox"])[0] for cell in median_cells],  # type: ignore[arg-type]
            "middle": [
                (_bbox_list(cell["bbox"])[0] + _bbox_list(cell["bbox"])[2]) / 2  # type: ignore[arg-type]
                for cell in median_cells
            ],
            "right": [_bbox_list(cell["bbox"])[2] for cell in median_cells],  # type: ignore[arg-type]
        }
        median_x = float(median(coordinates[alignment])) if coordinates[alignment] else 0.0

        for cell in column_cells:
            copied = dict(cell)
            if cell not in good:
                left, top, right, bottom = _bbox_list(cell["bbox"])  # type: ignore[arg-type]
                width = right - left
                if alignment == "middle":
                    left = median_x - width / 2
                    right = left + width
                elif alignment == "right":
                    left = median_x - width
                    right = median_x
                else:
                    left = median_x
                    right = median_x + width
                copied["bbox"] = [left, top, right, bottom]
            repaired.append(copied)
    return sorted(repaired, key=lambda cell: int(cell["cell_id"]))


def _deduplicate_columns(
    table_cells: Sequence[dict[str, Any]],
    initial_matches: Mapping[str, Sequence[Mapping[str, float | int]]],
    matches: Mapping[str, Sequence[Mapping[str, float | int]]],
    num_cols: int,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, float | int]]]]:
    pdf_ids_by_column: list[set[int]] = []
    scores_by_column: list[float] = []
    for column in range(num_cols):
        cell_ids = {
            int(cell["cell_id"])
            for cell in table_cells
            if int(cell["start_col_offset_idx"]) == column
        }
        pdf_ids: set[int] = set()
        score = 0.0
        for source_matches in (initial_matches, matches):
            for pdf_id, match_list in source_matches.items():
                for match in match_list:
                    if int(match["table_cell_id"]) in cell_ids:
                        pdf_ids.add(int(pdf_id))
                        score += float(match["iopdf"])
        pdf_ids_by_column.append(pdf_ids)
        scores_by_column.append(score)

    removed_columns: set[int] = set()
    for column in range(num_cols - 1):
        current = pdf_ids_by_column[column]
        overlap = len(current & pdf_ids_by_column[column + 1]) / len(current) if current else 0.0
        if overlap > 0.6:
            removed_columns.add(
                column + 1 if scores_by_column[column] >= scores_by_column[column + 1] else column
            )

    removed_ids = {
        int(cell["cell_id"])
        for cell in table_cells
        if int(cell["start_col_offset_idx"]) in removed_columns
    }
    kept_cells = [cell for cell in table_cells if int(cell["cell_id"]) not in removed_ids]
    kept_matches = {
        pdf_id: [
            dict(match) for match in match_list if int(match["table_cell_id"]) not in removed_ids
        ]
        for pdf_id, match_list in matches.items()
    }
    return kept_cells, {pdf_id: values for pdf_id, values in kept_matches.items() if values}


def _align_cells_to_pdf(
    table_cells: Sequence[dict[str, Any]],
    pdf_cells: Sequence[Mapping[str, Any]],
    matches: Mapping[str, Sequence[Mapping[str, float | int]]],
) -> list[dict[str, Any]]:
    cells_by_id = {int(cell["cell_id"]): cell for cell in table_cells}
    pdf_by_id = {str(cell["id"]): cell for cell in pdf_cells}
    bboxes_by_cell: dict[int, list[list[float]]] = {}
    for pdf_id, match_list in matches.items():
        if pdf_id not in pdf_by_id:
            continue
        pdf_bbox = _bbox_list(pdf_by_id[pdf_id]["bbox"])
        for match in match_list:
            cell_id = int(match["table_cell_id"])
            if cell_id in cells_by_id:
                bboxes_by_cell.setdefault(cell_id, []).append(pdf_bbox)

    aligned: list[dict[str, Any]] = []
    for cell_id, bboxes in bboxes_by_cell.items():
        cell = dict(cells_by_id[cell_id])
        cell["bbox"] = [
            min(bbox[0] for bbox in bboxes),
            min(bbox[1] for bbox in bboxes),
            max(bbox[2] for bbox in bboxes),
            max(bbox[3] for bbox in bboxes),
        ]
        aligned.append(cell)
    return aligned


def _band_for_cells(
    table_cells: Sequence[dict[str, Any]], index: int, *, rows: bool
) -> tuple[float, float] | None:
    cells = [
        cell
        for cell in table_cells
        if int(cell["start_row_offset_idx" if rows else "start_col_offset_idx"]) == index
        and int(cell["row_span" if rows else "col_span"]) == 1
        and int(cell["cell_class"]) > 1
    ]
    if not cells:
        return None
    bboxes = [_bbox_list(cell["bbox"]) for cell in cells]  # type: ignore[arg-type]
    start, end = (1, 3) if rows else (0, 2)
    return min(bbox[start] for bbox in bboxes), max(bbox[end] for bbox in bboxes)


def _closest_band(
    bbox: Sequence[float], bands: Sequence[tuple[float, float] | None], *, rows: bool
) -> int | None:
    start, end = (1, 3) if rows else (0, 2)
    candidates: list[tuple[int, int]] = []
    for index, band in enumerate(bands):
        if band is None:
            continue
        band_start, band_end = band
        if bbox[start] <= band_end and bbox[end] >= band_start:
            depth = round(abs((band_start + band_end) / 2 - (bbox[start] + bbox[end]) / 2))
            candidates.append((depth, index))
    return min(candidates)[1] if candidates else None


def _recover_orphans(
    table_cells: list[dict[str, Any]],
    pdf_cells: Sequence[Mapping[str, Any]],
    matches: dict[str, list[dict[str, float | int]]],
    num_rows: int,
    num_cols: int,
) -> None:
    row_bands = [_band_for_cells(table_cells, row, rows=True) for row in range(num_rows)]
    col_bands = [_band_for_cells(table_cells, column, rows=False) for column in range(num_cols)]
    next_cell_id = max((int(cell["cell_id"]) for cell in table_cells), default=-1) + 1

    for pdf_cell in sorted(pdf_cells, key=lambda cell: int(cell["id"])):
        pdf_id = str(pdf_cell["id"])
        if pdf_id in matches:
            continue
        bbox = _bbox_list(pdf_cell["bbox"])
        row = _closest_band(bbox, row_bands, rows=True)
        column = _closest_band(bbox, col_bands, rows=False)
        if row is None or column is None:
            continue
        existing = next(
            (
                cell
                for cell in table_cells
                if int(cell["start_row_offset_idx"]) == row
                and int(cell["start_col_offset_idx"]) == column
            ),
            None,
        )
        if existing is None:
            existing = {
                "cell_id": next_cell_id,
                "bbox": bbox,
                "row_span": 1,
                "col_span": 1,
                "start_row_offset_idx": row,
                "end_row_offset_idx": row + 1,
                "start_col_offset_idx": column,
                "end_col_offset_idx": column + 1,
                "indentation_level": 0,
                "text_cell_bboxes": [],
                "column_header": False,
                "row_header": False,
                "row_section": False,
                "cell_class": 2,
                "label": "body",
            }
            next_cell_id += 1
            table_cells.append(existing)
        else:
            current = _bbox_list(existing["bbox"])  # type: ignore[arg-type]
            existing["bbox"] = [
                min(current[0], bbox[0]),
                min(current[1], bbox[1]),
                max(current[2], bbox[2]),
                max(current[3], bbox[3]),
            ]
        matches[pdf_id] = [{"table_cell_id": int(existing["cell_id"]), "iopdf": 0.0}]


def _normalize_cell_indices(
    cells: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    row_ids = sorted({int(cell["start_row_offset_idx"]) for cell in cells})
    col_ids = sorted({int(cell["start_col_offset_idx"]) for cell in cells})
    for cell in cells:
        cell["start_row_offset_idx"] = row_ids.index(int(cell["start_row_offset_idx"]))
        cell["end_row_offset_idx"] = int(cell["start_row_offset_idx"]) + int(cell["row_span"])
        cell["start_col_offset_idx"] = col_ids.index(int(cell["start_col_offset_idx"]))
        cell["end_col_offset_idx"] = int(cell["start_col_offset_idx"]) + int(cell["col_span"])
    return (
        cells,
        max((int(cell["end_row_offset_idx"]) for cell in cells), default=0),
        max((int(cell["end_col_offset_idx"]) for cell in cells), default=0),
    )


def _matched_output(
    cells: Sequence[dict[str, Any]],
    pdf_cells: Sequence[Mapping[str, Any]],
    num_rows: int,
    num_cols: int,
) -> tuple[list[dict[str, Any]], int, int]:
    clean_pdf_cells = [cell for cell in pdf_cells if str(cell.get("text", "")) != ""]
    initial_matches = _intersection_over_pdf(cells, clean_pdf_cells)
    repaired = _repair_unmatched_columns(cells, initial_matches, num_cols)
    rematched = _intersection_over_pdf(repaired, clean_pdf_cells)
    deduplicated, rematched = _deduplicate_columns(repaired, initial_matches, rematched, num_cols)
    matches = {
        pdf_id: [max(match_list, key=lambda match: float(match["iopdf"]))]
        for pdf_id, match_list in rematched.items()
    }
    aligned = (
        deduplicated
        if len(clean_pdf_cells) > 300
        else _align_cells_to_pdf(deduplicated, clean_pdf_cells, matches)
    )
    _recover_orphans(aligned, clean_pdf_cells, matches, num_rows, num_cols)

    cells_by_id = {int(cell["cell_id"]): cell for cell in aligned}
    pdf_by_id = {str(cell["id"]): cell for cell in clean_pdf_cells}
    merged: dict[tuple[int, int], dict[str, Any]] = {}
    for pdf_id in sorted(matches, key=int):
        table_cell = cells_by_id.get(int(matches[pdf_id][0]["table_cell_id"]))
        pdf_cell = pdf_by_id.get(pdf_id)
        if table_cell is None or pdf_cell is None:
            continue
        row = int(table_cell["start_row_offset_idx"])
        column = int(table_cell["start_col_offset_idx"])
        key = (row, column)
        if key not in merged:
            output = {name: value for name, value in table_cell.items() if name != "cell_id"}
            bbox = _bbox_list(table_cell["bbox"])  # type: ignore[arg-type]
            output["bbox"] = {"l": bbox[0], "t": bbox[1], "r": bbox[2], "b": bbox[3]}
            output["text_cell_bboxes"] = []
            merged[key] = output
        pdf_bbox = _bbox_list(pdf_cell["bbox"])
        text_boxes = merged[key]["text_cell_bboxes"]
        if not isinstance(text_boxes, list):
            raise TypeError("TableFormerV1 matching returned invalid text boxes")
        text_boxes.append(
            {
                "b": pdf_bbox[3],
                "l": pdf_bbox[0],
                "r": pdf_bbox[2],
                "t": pdf_bbox[1],
                "token": str(pdf_cell.get("text", "")),
            }
        )

    return _normalize_cell_indices(list(merged.values()))


def postprocess_prediction(
    otsl_seq: Sequence[str],
    bboxes: Sequence[Sequence[float]],
    classes: Sequence[int],
    *,
    table_bbox: Sequence[float] | None = None,
    pdf_cells: Sequence[Mapping[str, Any]] = (),
    do_cell_matching: bool = True,
) -> tuple[list[dict[str, Any]], int, int]:
    """Build source-shaped cells from normalized model boxes and OTSL.

    Matching mode attaches intersecting source text boxes.  Dummy mode keeps
    the same structural cells and predicted boxes but deliberately ignores
    source text, matching Docling's ``do_cell_matching=False`` contract.
    """
    padded = pad_invalid_rows(otsl_seq)
    rows = [
        list(row)
        for is_newline, row in groupby(padded, lambda token: token == "nl")
        if not is_newline
    ]
    if not rows:
        return [], 0, 0

    boxes, cell_classes = repair_legacy_bbox_desync(otsl_seq, bboxes, classes)
    num_rows = len(rows)
    num_cols = len(rows[0])
    cells: list[dict[str, Any]] = []
    bbox_index = 0

    for row_index, row in enumerate(rows):
        for column_index, token in enumerate(row):
            if token not in _STRUCTURAL_CELL_TOKENS:
                continue
            normalized_bbox = boxes[bbox_index] if bbox_index < len(boxes) else [0.0] * 4
            page_bbox = _translate_bbox(normalized_bbox, table_bbox)
            cell_class = cell_classes[bbox_index] if bbox_index < len(cell_classes) else 2
            bbox_index += 1

            col_span = 1
            row_span = 1
            if token in _CELL_TOKENS:
                while column_index + col_span < num_cols and row[column_index + col_span] in {
                    "lcel",
                    "xcel",
                }:
                    col_span += 1
                while row_index + row_span < num_rows and rows[row_index + row_span][
                    column_index
                ] in {"ucel", "xcel"}:
                    row_span += 1
                # docling-ibm-models 3.14.0 round-trips OTSL through an HTML
                # parser whose recognized colspan/rowspan values stop at 20.
                if col_span > _MAX_LEGACY_SPAN:
                    col_span = 1
                if row_span > _MAX_LEGACY_SPAN:
                    row_span = 1

            cells.append(
                {
                    "cell_id": len(cells),
                    "bbox": {
                        "l": page_bbox[0],
                        "t": page_bbox[1],
                        "r": page_bbox[2],
                        "b": page_bbox[3],
                    },
                    "row_span": row_span,
                    "col_span": col_span,
                    "start_row_offset_idx": row_index,
                    "end_row_offset_idx": row_index + row_span,
                    "start_col_offset_idx": column_index,
                    "end_col_offset_idx": column_index + col_span,
                    "indentation_level": 0,
                    "text_cell_bboxes": [],
                    "column_header": token == "ched",
                    "row_header": token == "rhed",
                    "row_section": token == "srow",
                    "cell_class": cell_class,
                    "label": token,
                }
            )
    if do_cell_matching:
        return _matched_output(cells, pdf_cells, num_rows, num_cols)
    cells, num_rows, num_cols = _normalize_cell_indices(cells)
    for cell in cells:
        del cell["cell_id"]
    return cells, num_rows, num_cols


__all__ = ["pad_invalid_rows", "postprocess_prediction", "repair_legacy_bbox_desync"]
