# SPDX-License-Identifier: Apache-2.0

"""Generate minimal provenance cards for converted model repositories."""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
from huggingface_hub.errors import HfHubHTTPError, OfflineModeIsEnabled

License = str | list[str] | None


def _yaml_license(license: License) -> str:
    if license is None:
        return ""
    if isinstance(license, str):
        return f"license: {license}\n"
    return "license:\n" + "".join(f"  - {item}\n" for item in license)


def render_model_card(upstream_repo: str, revision: str, license: License) -> str:
    """Render the shared converted-model card."""

    name = upstream_repo.rsplit("/", 1)[-1]
    source = (
        f"MLX-converted weights of [`{upstream_repo}`](https://huggingface.co/{upstream_repo}) "
        f"at revision `{revision}`."
    )
    return f"""---
{_yaml_license(license)}library_name: docling-mlx
tags:
  - mlx
base_model:
  - {upstream_repo}
---

# {name} - MLX

{source}
"""


def _read_local_license(source: Path) -> License:
    readme = source / "README.md"
    if not readme.is_file():
        return None
    text = readme.read_text(encoding="utf-8")
    match = re.match(r"---\s*\n(.*?)\n---(?:\s*\n|$)", text, re.DOTALL)
    if match is None:
        return None
    lines = match.group(1).splitlines()
    for index, line in enumerate(lines):
        license_match = re.fullmatch(r"license:[ \t]*(.*)", line)
        if license_match is None:
            continue
        if value := license_match.group(1):
            return value.strip("'\"")
        licenses = []
        for item in lines[index + 1 :]:
            if not item.startswith((" ", "\t")):
                break
            item_match = re.fullmatch(r"\s*-\s+(.+?)\s*", item)
            if item_match is not None:
                licenses.append(item_match.group(1).strip("'\""))
        return licenses or None
    return None


def lookup_model_license(source: Path, upstream_repo: str, revision: str) -> License:
    """Resolve Hub card metadata, falling back to a supplied source card."""

    try:
        from huggingface_hub import model_info

        card_data: Any = model_info(upstream_repo, revision=revision).card_data
        license = (
            card_data.get("license")
            if isinstance(card_data, Mapping)
            else getattr(card_data, "license", None)
        )
        if isinstance(license, str) or (
            isinstance(license, list) and all(isinstance(item, str) for item in license)
        ):
            return license
    except (HfHubHTTPError, OfflineModeIsEnabled, httpx.HTTPError) as error:
        print(
            f"Hub license lookup failed for {upstream_repo} at {revision}: {error}", file=sys.stderr
        )
    if (license := _read_local_license(source)) is not None:
        return license
    print(
        f"No license metadata for {upstream_repo} at {revision}; omitting license from README.md.",
        file=sys.stderr,
    )
    return None
