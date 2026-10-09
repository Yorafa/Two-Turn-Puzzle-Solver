"""
Unit tests and performance benchmarks for pathfinding and solver modules.
"""

import os
import time
import unittest

import cv2

from config import COL, IMG_DIR, ROW, TEMPLATE_DIR
from core.pathfinding import can_connect, get_connection_path, is_line_clear
from core.solver import solve_puzzle
from core.vision import TemplateLibrary, parse_board


class TestPathfinding(unittest.TestCase):
    def setUp(self):
        # 6x11 board with borders = 8x13
        self.board = [[-1] * (COL + 2) for _ in range(ROW + 2)]

    def test_0_turn_straight(self):
        self.board[2][2] = 1
        self.board[2][8] = 1
        self.assertTrue(can_connect(self.board, (2, 2), (2, 8)))

        self.board[2][5] = 99
        self.assertFalse(is_line_clear(self.board, 2, 2, 2, 8))

    def test_1_turn_corner(self):
        self.board[1][1] = 1
        self.board[3][5] = 1
        self.assertTrue(can_connect(self.board, (1, 1), (3, 5)))
        path = get_connection_path(self.board, (1, 1), (3, 5))
        self.assertIsNotNone(path)
        assert path is not None
        self.assertEqual(len(path), 3)

        self.board[1][5] = 99
        self.board[3][1] = 99
        self.assertTrue(can_connect(self.board, (1, 1), (3, 5)))

    def test_2_turn_perimeter_u_turn(self):
        for c in range(1, COL + 1):
            self.board[2][c] = 99
        self.board[1][2] = 5
        self.board[1][9] = 5
        self.assertTrue(can_connect(self.board, (1, 2), (1, 9)))


class TestRealBoardSolver(unittest.TestCase):
    def test_solve_actual_board(self):
        library = TemplateLibrary(TEMPLATE_DIR)
        fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "sample_initial_board.png")
        temp_path = os.path.join(TEMPLATE_DIR, "temp.png")
        board_path = os.path.join(IMG_DIR, "last_initial_board.png")

        if os.path.exists(fixture_path):
            board_img = cv2.imread(fixture_path)
        elif os.path.exists(temp_path):
            full_img = cv2.imread(temp_path)
            assert full_img is not None
            board_img = full_img[363:697, 115:725]
        elif os.path.exists(board_path):
            board_img = cv2.imread(board_path)
        else:
            self.skipTest("No test board image available.")

        self.assertIsNotNone(board_img)
        assert board_img is not None

        # Parse board
        board, cnter = parse_board(board_img, library)
        self.assertGreater(sum(len(v) for v in cnter.values()), 50, "Should detect game tiles.")

        # Benchmark solve
        t0 = time.perf_counter()
        success, path = solve_puzzle(board, cnter, timeout_seconds=1.0)
        dt = (time.perf_counter() - t0) * 1000

        self.assertTrue(success, "Solver should successfully clear the board without deadlock.")
        self.assertGreater(len(path), 20, "Should eliminate multiple pairs.")
        print(f"\n[BENCHMARK] Real board solved in {dt:.2f} ms with {len(path)} pairs eliminated!")
        self.assertLess(dt, 50.0, "Solver must complete in under 50ms.")


if __name__ == "__main__":
    unittest.main()
