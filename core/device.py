"""
Pure ADB device manager and resolution-adaptive board capture.
Provides direct, high-speed device control without desktop GUI or window dependencies.
"""

import json
import os
import subprocess
import time

import cv2
import numpy as np

from config import (
    AD_BUTTON_REL,
    APP_ACTIVITY,
    APP_PACKAGE,
    CLICK_DELAY,
    COL,
    CONFIG_FILE,
    DEADLOCK_OK_REL,
    DEFAULT_BOARD_BBOX,
    DIALOG_RETRY_REL,
    FARM_MINIGAME_BUTTON_REL,
    JEWEL_POPUP_CLOSE_REL,
    LANDSCAPE_RELATIVE_BBOX,
    PAIR_DELAY,
    PORTRAIT_RELATIVE_BBOX,
    REWARDS_CLOSE_REL,
    ROW,
    STAGE_CLOSE_REL,
    STAGE_QUIT_CONFIRM_REL,
    START_BUTTON_REL,
    TITLE_START_REL,
)


def enable_dpi_awareness() -> None:
    """No-op retained for backwards compatibility (pure ADB mode needs no desktop DPI scaling)."""


class ADBController:
    """Direct high-speed control of connected Android devices via ADB."""

    def __init__(self, serial: str | None = None):
        self.serial = serial
        self.device_id = self._find_device()

    def _find_device(self) -> str | None:
        try:
            res = subprocess.run(["adb", "devices"], capture_output=True, text=True, check=True)
            lines = res.stdout.strip().splitlines()[1:]
            devices = [line.split()[0] for line in lines if "device" in line]
            if self.serial and self.serial in devices:
                return self.serial
            return devices[0] if devices else None
        except (subprocess.SubprocessError, OSError):
            return None

    def is_available(self) -> bool:
        return self.device_id is not None

    def capture_screen(self) -> np.ndarray | None:
        """Capture pixel-perfect raw screen via ADB screencap."""
        if not self.device_id:
            return None
        cmd = ["adb", "-s", self.device_id, "exec-out", "screencap", "-p"]
        try:
            res = subprocess.run(cmd, capture_output=True, check=True)
            img_array = np.frombuffer(res.stdout, dtype=np.uint8)
            img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img_bgr
        except (subprocess.SubprocessError, OSError, cv2.error) as e:
            print(f"[!] ADB capture failed: {e}", flush=True)
            return None

    def tap(self, x: int, y: int) -> None:
        """Send a single tap via ADB."""
        if self.device_id:
            subprocess.run(["adb", "-s", self.device_id, "shell", "input", "tap", str(x), str(y)], check=True)

    def press_back(self) -> None:
        """Send Back key (KEYCODE_BACK) via ADB to dismiss dialogs or full-screen ads."""
        if self.device_id:
            subprocess.run(["adb", "-s", self.device_id, "shell", "input", "keyevent", "4"], check=True)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        """Send a swipe/drag gesture via ADB."""
        if self.device_id:
            cmd = [
                "adb",
                "-s",
                self.device_id,
                "shell",
                "input",
                "swipe",
                str(x1),
                str(y1),
                str(x2),
                str(y2),
                str(duration_ms),
            ]
            subprocess.run(cmd, check=True)

    def execute_tap_batch(
        self,
        solution_path: list[tuple[tuple[int, int], tuple[int, int]]],
        board_abs_bbox: tuple[int, int, int, int],
        row_count: int = ROW,
        col_count: int = COL,
        tap_delay: float = CLICK_DELAY,
        pair_delay: float = PAIR_DELAY,
    ) -> None:
        """
        Execute an entire sequence of elimination taps by uploading a single shell script.
        Eliminates sub-process overhead and accomplishes full board clears in < 1 second.
        """
        if not self.device_id:
            return
        dev_id: str = self.device_id

        commands = ["#!/bin/sh"]
        for p1, p2 in solution_path:
            x1, y1 = DeviceManager.get_cell_screen_coords(p1[0], p1[1], board_abs_bbox, row_count, col_count)
            x2, y2 = DeviceManager.get_cell_screen_coords(p2[0], p2[1], board_abs_bbox, row_count, col_count)

            commands.append(f"input tap {x1} {y1}")
            if tap_delay > 0:
                commands.append(f"sleep {tap_delay}")
            commands.append(f"input tap {x2} {y2}")
            if pair_delay > 0:
                commands.append(f"sleep {pair_delay}")

        script_bytes = ("\n".join(commands) + "\n").encode("utf-8")
        subprocess.run(
            ["adb", "-s", dev_id, "shell", "sh"],
            input=script_bytes,
            check=True,
        )

    def restart_game(self, package: str = APP_PACKAGE, activity: str = APP_ACTIVITY) -> None:
        """Force-stop and restart the game application via ADB."""
        if not self.device_id:
            return
        dev_id: str = self.device_id
        subprocess.run(["adb", "-s", dev_id, "shell", "am", "force-stop", package], check=True)
        subprocess.run(["adb", "-s", dev_id, "shell", "am", "start", "-n", f"{package}/{activity}"], check=True)


def detect_board_bbox(full_img_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Auto-detect board bounds from a full screen capture based on color profiles.
    The board area has a darker/translucent background surrounded by bright pastel colors.
    """
    h, w = full_img_bgr.shape[:2]
    if w <= h:
        return None
    col_profile = np.mean(full_img_bgr[int(h * 0.3) : int(h * 0.7), :], axis=(0, 2))
    row_profile = np.mean(full_img_bgr[:, int(w * 0.25) : int(w * 0.75)], axis=(1, 2))
    x_in = np.where(col_profile < 175)[0]
    y_in = np.where(row_profile < 175)[0]
    if len(x_in) > 100 and len(y_in) > 100:
        x1, x2 = int(x_in[0]), int(x_in[-1])
        y1, y2 = int(y_in[0]), int(y_in[-1])
        # The board is centered and never extends to the screen borders
        if (
            x1 > int(w * 0.04)
            and x2 < int(w * 0.96)
            and y1 > int(h * 0.04)
            and y2 < int(h * 0.96)
            and (x2 - x1) > w * 0.45
            and (y2 - y1) > h * 0.40
        ):
            return x1, y1, x2, y2
    return None


class DeviceManager:
    """Manages ADB device communication, screen capture, and adaptive board framing."""

    def __init__(self, serial: str | None = None):
        self.adb = ADBController(serial=serial)
        self.custom_bbox: tuple[float, float, float, float] | None = None
        self.load_calibration()

    def load_calibration(self) -> bool:
        """Load user calibration from board_config.json if available."""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, encoding="utf-8") as f:
                    data = json.load(f)
                    if "relative_bbox" in data:
                        self.custom_bbox = tuple(data["relative_bbox"])
                    return True
            except (json.JSONDecodeError, OSError) as e:
                print(f"[!] Warning: Failed to load {CONFIG_FILE}: {e}", flush=True)
        return False

    def save_calibration(self, relative_bbox: tuple[float, float, float, float]) -> None:
        """Save relative bounding box calibration to board_config.json."""
        self.custom_bbox = relative_bbox
        data = {
            "relative_bbox": list(relative_bbox),
        }
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[+] Calibration saved to {CONFIG_FILE}", flush=True)

    def get_relative_bbox(self, width: int, height: int) -> tuple[float, float, float, float]:
        """Automatically select landscape vs portrait ratio based on aspect ratio."""
        if self.custom_bbox:
            return self.custom_bbox
        if width > height:
            return LANDSCAPE_RELATIVE_BBOX
        return PORTRAIT_RELATIVE_BBOX

    def capture_screen(self) -> np.ndarray | None:
        """Capture the entire screen to inspect game state."""
        return self.adb.capture_screen()

    def capture_full_screen_or_window(self, use_adb: bool = True) -> np.ndarray | None:
        """Alias for capture_screen."""
        return self.capture_screen()

    def click_relative(self, rx: float, ry: float, use_adb: bool = True) -> None:
        """Click a position defined by relative fractions (0.0 to 1.0) of the screen."""
        screen = self.adb.capture_screen()
        if screen is not None:
            sh, sw = screen.shape[:2]
            self.adb.tap(int(rx * sw), int(ry * sh))

    def click_start(self, use_adb: bool = True) -> None:
        """Click 'Start Minigame' button."""
        self.click_relative(START_BUTTON_REL[0], START_BUTTON_REL[1])

    def click_close_rewards(self, use_adb: bool = True) -> None:
        """Click the close 'X' button on the Rewards dialog."""
        self.click_relative(REWARDS_CLOSE_REL[0], REWARDS_CLOSE_REL[1])

    def click_ad(self, use_adb: bool = True) -> None:
        """Click the 'Ad' button to skip cooldown for free."""
        self.click_relative(AD_BUTTON_REL[0], AD_BUTTON_REL[1])

    def click_close_jewel_popup(self, use_adb: bool = True) -> None:
        """Click the close 'X' button on the 'Spend Jewel(s)' popup."""
        self.click_relative(JEWEL_POPUP_CLOSE_REL[0], JEWEL_POPUP_CLOSE_REL[1])

    def press_back(self, use_adb: bool = True) -> None:
        """Send Back key via ADB to dismiss dialogs or ads."""
        self.adb.press_back()

    def click_stage_close(self, use_adb: bool = True) -> None:
        """Click the top-right 'X' button on an active game board and confirm quit dialog."""
        self.click_relative(STAGE_CLOSE_REL[0], STAGE_CLOSE_REL[1])
        time.sleep(0.8)
        self.click_relative(STAGE_QUIT_CONFIRM_REL[0], STAGE_QUIT_CONFIRM_REL[1])

    def click_deadlock_ok(self, use_adb: bool = True) -> None:
        """Click the center 'OK' button on the 'No more tiles can be eliminated' deadlock popup."""
        self.click_relative(DEADLOCK_OK_REL[0], DEADLOCK_OK_REL[1])

    def click_title_start(self, use_adb: bool = True) -> None:
        """Click center on Title screen to tap 'Tap to Start' ('點擊並開始遊戲')."""
        self.click_relative(TITLE_START_REL[0], TITLE_START_REL[1])

    def click_farm_minigame(self, use_adb: bool = True) -> None:
        """Click 'Minigame' (小遊戲) icon on the main Farm screen."""
        self.click_relative(FARM_MINIGAME_BUTTON_REL[0], FARM_MINIGAME_BUTTON_REL[1])

    def click_dialog_retry(self, use_adb: bool = True) -> None:
        """Click center button on network retry / reconnection dialog."""
        self.click_relative(DIALOG_RETRY_REL[0], DIALOG_RETRY_REL[1])

    def restart_game(self, use_adb: bool = True) -> None:
        """Restart the game application."""
        self.adb.restart_game()

    def capture_board(
        self,
        use_adb: bool = True,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """
        Capture the board area from ADB device dynamically with auto-framing.
        Returns:
          board_bgr: NumPy array of the cropped board (at native resolution).
          abs_bbox: (x1, y1, x2, y2) absolute coordinates for clicking.
        """
        screen = self.adb.capture_screen()
        if screen is not None:
            sh, sw = screen.shape[:2]
            auto_box = detect_board_bbox(screen)
            if auto_box is not None:
                x1, y1, x2, y2 = auto_box
            else:
                rx1, ry1, rx2, ry2 = self.get_relative_bbox(sw, sh)
                x1 = int(sw * rx1)
                y1 = int(sh * ry1)
                x2 = int(sw * rx2)
                y2 = int(sh * ry2)
            crop = screen[y1:y2, x1:x2]
            return crop, (x1, y1, x2, y2)

        return np.zeros((1, 1, 3), dtype=np.uint8), DEFAULT_BOARD_BBOX

    @staticmethod
    def get_cell_screen_coords(
        row: int,
        col: int,
        board_abs_bbox: tuple[int, int, int, int],
        row_count: int = ROW,
        col_count: int = COL,
    ) -> tuple[int, int]:
        """Calculate screen (x, y) coordinates for clicking the center of cell (row, col)."""
        x1, y1, x2, y2 = board_abs_bbox
        cell_w = (x2 - x1) / col_count
        cell_h = (y2 - y1) / row_count
        real_x = int(x1 + (col - 0.5) * cell_w)
        real_y = int(y1 + (row - 0.5) * cell_h)
        return real_x, real_y


# Backward compatibility alias
WindowManager = DeviceManager
