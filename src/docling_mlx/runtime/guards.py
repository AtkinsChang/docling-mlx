# SPDX-License-Identifier: Apache-2.0

"""Platform guard for native MLX components."""

import platform
import sys
from typing import Any


def validate_mlx_accelerator(accelerator_options: Any) -> None:
    device = accelerator_options.device
    device = getattr(device, "value", device)
    if device not in {"auto", "mps"}:
        from docling.exceptions import AcceleratorDeviceNotAvailableError

        raise AcceleratorDeviceNotAvailableError(
            f"{str(device).upper()} is not supported by this model. "
            "Supported devices: ['auto', 'mps']"
        )


def require_apple_silicon() -> None:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("docling-mlx inference requires macOS on Apple Silicon (arm64).")
