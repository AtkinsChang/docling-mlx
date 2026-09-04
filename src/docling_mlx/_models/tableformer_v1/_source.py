# SPDX-License-Identifier: Apache-2.0

from typing import Any

import mlx.core as mx
import mlx.nn as nn


def source_parameter_filter(_module: nn.Module, _key: str, value: Any) -> bool:
    return isinstance(value, (dict, list, mx.array))
