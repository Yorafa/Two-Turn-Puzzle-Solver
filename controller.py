"""
Game automation controller for high-speed ADB clicking, automation flow, and gestures.
Provides direct device interaction without desktop GUI or PyAutoGUI dependencies.
"""

import time

from config import (
    CLICK_DELAY,
    CLOSE_BUTTON,
    COL,
    CONFIRM_BUTTON,
    DEFAULT_BOARD_BBOX,
    EGG_POS,
    PAIR_DELAY,
    POT_INTERVAL,
    POT_POS,
    RETRY_BUTTON,
    ROW,
    SKIP_BUTTON,
)
from core.device import DeviceManager


class GameController:
    """Controls gameplay automation and tap sequences purely via ADB."""

    def __init__(
        self,
        board_bbox: tuple[int, int, int, int] = DEFAULT_BOARD_BBOX,
        click_delay: float = CLICK_DELAY,
        pair_delay: float = PAIR_DELAY,
        device_manager: DeviceManager | None = None,
    ):
        self.board_bbox = board_bbox
        self.click_delay = click_delay
        self.pair_delay = pair_delay
        self.device = device_manager or DeviceManager()

    def click_cell(
        self,
        row: int,
        col: int,
        board_bbox: tuple[int, int, int, int] | None = None,
    ) -> None:
        """Click the center of the cell at (row, col) [1-indexed] on the device screen."""
        bbox = board_bbox or self.board_bbox
        x, y = DeviceManager.get_cell_screen_coords(row, col, bbox, ROW, COL)
        self.device.adb.tap(x, y)

    def eliminate_pair(
        self,
        p1: tuple[int, int],
        p2: tuple[int, int],
        board_bbox: tuple[int, int, int, int] | None = None,
    ) -> None:
        """Click a pair of matching tiles to eliminate them."""
        bbox = board_bbox or self.board_bbox
        self.click_cell(p1[0], p1[1], bbox)
        if self.click_delay > 0:
            time.sleep(self.click_delay)
        self.click_cell(p2[0], p2[1], bbox)
        if self.pair_delay > 0:
            time.sleep(self.pair_delay)

    def batch_eliminate(
        self,
        solution_path: list[tuple[tuple[int, int], tuple[int, int]]],
        board_bbox: tuple[int, int, int, int] | None = None,
    ) -> int:
        """Execute a full sequence of elimination pairs via high-speed ADB batch script."""
        bbox = board_bbox or self.board_bbox
        self.device.adb.execute_tap_batch(
            solution_path,
            bbox,
            row_count=ROW,
            col_count=COL,
            tap_delay=self.click_delay,
            pair_delay=self.pair_delay,
        )
        return len(solution_path)

    def execute_solution(
        self,
        solution_path: list[tuple[tuple[int, int], tuple[int, int]]],
        board_bbox: tuple[int, int, int, int] | None = None,
        verbose: bool = True,
    ) -> int:
        """Execute a full sequence of elimination pairs."""
        return self.batch_eliminate(solution_path, board_bbox)

    # UI interaction helpers
    def click_skip(self) -> None:
        """Click the skip dialog/intro button."""
        self.device.adb.tap(SKIP_BUTTON[0], SKIP_BUTTON[1])

    def click_close(self) -> None:
        """Click the close window button."""
        self.device.adb.tap(CLOSE_BUTTON[0], CLOSE_BUTTON[1])

    def click_retry(self) -> None:
        """Click the retry/restart button."""
        self.device.adb.tap(RETRY_BUTTON[0], RETRY_BUTTON[1])

    def click_confirm(self) -> None:
        """Click confirm/OK button."""
        self.device.adb.tap(CONFIRM_BUTTON[0], CONFIRM_BUTTON[1])

    # Cooking minigame helper via pure ADB swipes and taps
    def fry(
        self,
        total: int,
        ingredient_pos: tuple[int, int] = EGG_POS,
        pot_base_pos: tuple[int, int] = POT_POS,
        cook_time: float = 4.3,
    ) -> None:
        """Automate frying/cooking sequence in 5-item batches via ADB gestures."""
        batches = (total // 5) + (1 if total % 5 != 0 else 0)
        last_batch = total % 5 if total % 5 != 0 else 5

        for b in range(batches):
            count = 5 if b != batches - 1 else last_batch
            # Drag to pots
            for j in range(count):
                pot_x = pot_base_pos[0] + POT_INTERVAL * j
                pot_y = pot_base_pos[1]
                self.device.adb.swipe(ingredient_pos[0], ingredient_pos[1], pot_x, pot_y, duration_ms=250)
                time.sleep(0.05)

            time.sleep(cook_time)

            # Collect from pots
            for j in range(count):
                pot_x = pot_base_pos[0] + POT_INTERVAL * j
                pot_y = pot_base_pos[1]
                self.device.adb.tap(pot_x, pot_y)
                time.sleep(0.05)

            if b != batches - 1:
                time.sleep(2.0)
