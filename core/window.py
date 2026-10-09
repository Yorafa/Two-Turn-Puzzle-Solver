"""
Backwards-compatibility shim for core.device.
Direct ADB device management is now the default backend.
"""

from core.device import (
    ADBController,
    DeviceManager,
    WindowManager,
    detect_board_bbox,
    enable_dpi_awareness,
)

__all__ = [
    "ADBController",
    "DeviceManager",
    "WindowManager",
    "detect_board_bbox",
    "enable_dpi_awareness",
]
