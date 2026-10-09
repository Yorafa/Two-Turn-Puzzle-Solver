"""
Unit test for auto-harvesting tile templates from a raw game board.
"""

import os
import shutil
import tempfile
import unittest

import cv2

from config import STOP_TILE_ID
from core.solver import solve_puzzle
from core.vision import TemplateLibrary, harvest_templates_from_board, parse_board


class TestTemplateHarvest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_harvest_")
        stop_src = os.path.join("templates", "STOP.png")
        if os.path.exists(stop_src):
            shutil.copy(stop_src, os.path.join(self.temp_dir, "STOP.png"))

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_harvest_and_solve_from_scratch(self):
        fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "sample_initial_board.png")
        self.assertTrue(os.path.exists(fixture_path), "Sample board fixture must exist.")

        board_img = cv2.imread(fixture_path)
        self.assertIsNotNone(board_img)
        assert board_img is not None

        # Harvest templates with no pre-existing character templates
        harvested_count = harvest_templates_from_board(board_img, self.temp_dir)
        self.assertGreaterEqual(harvested_count, 12, "Should harvest at least 12 distinct tile types.")

        # Load harvested library
        library = TemplateLibrary(self.temp_dir)
        self.assertEqual(library.active_tile_count, harvested_count)

        # Parse board with freshly harvested templates
        board, cnter = parse_board(board_img, library)
        odd_tiles = {k: len(v) for k, v in cnter.items() if len(v) % 2 != 0 and k != STOP_TILE_ID}
        self.assertEqual(odd_tiles, {}, f"All harvested tile types must appear in even counts: {odd_tiles}")

        total_active = sum(len(v) for k, v in cnter.items() if k != STOP_TILE_ID)
        self.assertEqual(total_active, 64, "Should recognize 64 active playable tiles on sample board.")

        # Solve puzzle
        is_full, path = solve_puzzle(board, cnter, ignored_tiles={STOP_TILE_ID})
        self.assertTrue(is_full, "Harvested board state must be 100% solvable.")
        self.assertEqual(len(path), 32, "Should find 32 pairs to eliminate.")
