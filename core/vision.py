"""
Computer Vision module for LianLianKan tile detection and board parsing.
Optimized for high-speed template matching, blank cell filtering,
resolution-independent scale normalization, and game state detection.
"""

import os
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import cv2
import numpy as np

# Reusable high-speed worker pool for parallel cell template matching
_VISION_EXECUTOR = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4))

from config import (
    BLANK_MEAN_THRESHOLD,
    BLANK_STD_THRESHOLD,
    COL,
    MATCH_THRESHOLD,
    ROW,
    STOP_TILE_ID,
    TEMPLATE_DIR,
)

# Baseline reference dimensions for template matching
BASELINE_BOARD_WIDTH: int = 610
BASELINE_BOARD_HEIGHT: int = 334

_FARM_MINIGAME_TMPL: np.ndarray | None = None


def _get_farm_minigame_tmpl() -> np.ndarray | None:
    global _FARM_MINIGAME_TMPL
    if _FARM_MINIGAME_TMPL is None:
        p = os.path.join(TEMPLATE_DIR, "farm_minigame.png")
        if os.path.exists(p):
            _FARM_MINIGAME_TMPL = cv2.imread(p)
    return _FARM_MINIGAME_TMPL


class TemplateLibrary:
    """Manages pre-processed templates and color characteristics."""

    def __init__(self, template_dir: str):
        self.template_dir = template_dir
        self.templates: dict[int, np.ndarray] = {}
        self.hsv_means: dict[int, tuple[float, float, float]] = {}
        self.template_colors: dict[int, np.ndarray] = {}
        self.load_templates()

    @property
    def active_tile_count(self) -> int:
        return len([k for k in self.templates if k != STOP_TILE_ID])

    def load_templates(self) -> None:
        """Load and cache all templates as NumPy arrays and precompute color characteristics."""
        self.templates.clear()
        self.hsv_means.clear()
        self.template_colors.clear()
        if not os.path.exists(self.template_dir):
            return

        for filename in os.listdir(self.template_dir):
            if not filename.endswith(".png"):
                continue
            if filename in ("HINT.png", "space.png", "temp.png"):
                continue

            if filename == "STOP.png":
                tid = STOP_TILE_ID
            else:
                try:
                    tid = int(os.path.splitext(filename)[0])
                except ValueError:
                    continue

            path = os.path.join(self.template_dir, filename)
            img = cv2.imread(path)
            if img is not None:
                self.templates[tid] = img
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                mean_vals = cast(Sequence[float], cv2.mean(hsv))[:3]
                self.hsv_means[tid] = (float(mean_vals[0]), float(mean_vals[1]), float(mean_vals[2]))
                th, tw = img.shape[:2]
                center_roi = img[int(th * 0.3) : int(th * 0.7), int(tw * 0.3) : int(tw * 0.7)]
                mean_bgr = cast(Sequence[float], cv2.mean(center_roi))[:3]
                self.template_colors[tid] = np.array(mean_bgr, dtype=np.float32)


def detect_screen_state(full_img_bgr: np.ndarray | None) -> str:
    """
    Detect the current screen state:
      'SPEND_JEWEL_POPUP': 'Spend Jewel(s) and Play?' popup is visible (MUST DISMISS IMMEDIATELY)
      'REWARDS': Reward popup with red close button is visible
      'START': 'Start Minigame' button is visible
      'AD': 'Ad' button is visible to skip cooldown
      'NO_AD_COOLDOWN': Cooldown is active with green skip button, but no 'Ad' button
      'GAME': Game board is actively on screen
      'UNKNOWN': Unrecognized state
    """
    if full_img_bgr is None or full_img_bgr.size == 0:
        return "UNKNOWN"

    H, W = full_img_bgr.shape[:2]
    hsv = cv2.cvtColor(full_img_bgr, cv2.COLOR_BGR2HSV)

    # Red color range for dialog close buttons
    red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red1, red2)

    # 1. Check for Loading screen (polka-dot light cream background: very bright mean, low std)
    if full_img_bgr.mean() > 225.0 and full_img_bgr.std() < 30.0:
        return "LOADING"

    # 2. Check for Spend Jewel confirmation dialog (red close X at relative [0.65-0.72, 0.20-0.32])
    rx1, ry1 = int(W * 0.65), int(H * 0.20)
    rx2, ry2 = int(W * 0.72), int(H * 0.32)
    red_count = np.count_nonzero(red_mask[ry1:ry2, rx1:rx2])
    if red_count > 600:
        # Verify dialog card background at center (dialogs are light/cream colored)
        cx1, cy1 = int(W * 0.40), int(H * 0.35)
        cx2, cy2 = int(W * 0.60), int(H * 0.65)
        center_mean = float(np.mean(full_img_bgr[cy1:cy2, cx1:cx2]))
        if center_mean > 140:
            return "SPEND_JEWEL_POPUP"

    # 3. Check for Rewards popup (red close button at relative [0.70-0.75, 0.08-0.16])
    rx1, ry1 = int(W * 0.70), int(H * 0.08)
    rx2, ry2 = int(W * 0.75), int(H * 0.16)
    if np.count_nonzero(red_mask[ry1:ry2, rx1:rx2]) > 500:
        return "REWARDS"

    # 4. Check for Minigame menu buttons (only when minigame dialog is present)
    # The minigame popup has a prominent red circular close button at [0.83-0.89, 0.07-0.16]
    menu_close_red = np.count_nonzero(red_mask[int(H * 0.07) : int(H * 0.16), int(W * 0.83) : int(W * 0.89)])
    stage_roi = red_mask[int(H * 0.03) : int(H * 0.16), int(W * 0.85) : int(W * 0.94)]
    stage_red = np.count_nonzero(stage_roi)
    cyan_mask = cv2.inRange(hsv, np.array([85, 120, 150]), np.array([105, 255, 255]))

    if menu_close_red > 1000:
        # Check if wide Start Minigame button is at center (x: 45%-55%, y: 76%-88%)
        start_roi = cyan_mask[int(H * 0.76) : int(H * 0.88), int(W * 0.45) : int(W * 0.55)]
        if np.count_nonzero(start_roi) > (start_roi.size * 0.35):
            return "START"

        # Check if Ad button is present (cyan button at x: 52%-64%, y: 76%-88%)
        ad_roi = cyan_mask[int(H * 0.76) : int(H * 0.88), int(W * 0.52) : int(W * 0.64)]
        if np.count_nonzero(ad_roi) > (ad_roi.size * 0.35):
            return "AD"

        # Check if green Skip button is present without Ad button (cooldown active, no ad)
        green_mask = cv2.inRange(hsv, np.array([35, 120, 120]), np.array([75, 255, 255]))
        skip_roi = green_mask[int(H * 0.76) : int(H * 0.88), int(W * 0.36) : int(W * 0.48)]
        if np.count_nonzero(skip_roi) > (skip_roi.size * 0.35):
            return "NO_AD_COOLDOWN"

    # 5. Check for Main Farm screen (dock '小遊戲' boat icon, or clover+jewel counters)
    tmpl_boat = _get_farm_minigame_tmpl()
    dock_roi = full_img_bgr[int(H * 0.78) : int(H * 0.98), int(W * 0.58) : int(W * 0.74)]
    if tmpl_boat is not None and menu_close_red < 200 and stage_red < 200:
        target_w = max(10, int(tmpl_boat.shape[1] * W / 2280.0))
        target_h = max(10, int(tmpl_boat.shape[0] * H / 1080.0))
        scaled_tmpl = cv2.resize(tmpl_boat, (target_w, target_h), interpolation=cv2.INTER_AREA)
        if dock_roi.shape[0] >= scaled_tmpl.shape[0] and dock_roi.shape[1] >= scaled_tmpl.shape[1]:
            res = cv2.matchTemplate(dock_roi, scaled_tmpl, cv2.TM_CCOEFF_NORMED)
            _, mv, _, _ = cv2.minMaxLoc(res)
            if mv > 0.65:
                return "FARM"

    clover_roi = hsv[int(H * 0.02) : int(H * 0.12), int(W * 0.05) : int(W * 0.12)]
    green_clover = cv2.inRange(clover_roi, np.array([35, 100, 100]), np.array([75, 255, 255]))
    jewel_roi = hsv[int(H * 0.11) : int(H * 0.18), int(W * 0.75) : int(W * 0.83)]
    cyan_jewel = cv2.inRange(jewel_roi, np.array([85, 120, 150]), np.array([105, 255, 255]))
    if (
        np.count_nonzero(green_clover) > 200
        and np.count_nonzero(cyan_jewel) > 200
        and menu_close_red < 200
        and stage_red < 200
    ):
        return "FARM"

    # 6. Check for Title Screen ("點擊並開始遊戲" with cyan "設定" button in bottom-right)
    settings_roi = cyan_mask[int(H * 0.84) : int(H * 0.96), int(W * 0.83) : int(W * 0.95)]
    if np.count_nonzero(settings_roi) > 8000 and menu_close_red < 200 and stage_red < 200:
        return "TITLE_SCREEN"

    # 7. Check for Game board (top-right stage badge at x: 85%-94%, y: 3%-16%)
    if stage_red > 500:
        return "GAME"

    # 8. Check for 'CLEAR!!' victory animation (orange lettering on dark puzzle board)
    center_roi = full_img_bgr[int(H * 0.35) : int(H * 0.65), int(W * 0.30) : int(W * 0.70)]
    center_hsv = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)
    orange_mask = cv2.inRange(center_hsv, np.array([10, 150, 200]), np.array([25, 255, 255]))
    if np.count_nonzero(orange_mask) > 10000 and center_roi.mean() < 175.0:
        return "CLEAR"

    # 9. Check for generic popup dialogs (e.g. news/announcements/daily login bonus with red close button)
    popup_close_red = np.count_nonzero(red_mask[int(H * 0.05) : int(H * 0.28), int(W * 0.68) : int(W * 0.94)])
    if popup_close_red > 500 and menu_close_red < 200 and stage_red < 200:
        return "POPUP_DIALOG"

    return "UNKNOWN"


def is_blank_cell(
    cell: np.ndarray,
    mean_threshold: float = BLANK_MEAN_THRESHOLD,
    std_threshold: float = BLANK_STD_THRESHOLD,
) -> bool:
    """Quickly check if a cell is an empty/eliminated background cell."""
    if cell.size == 0:
        return True
    return bool(cell.mean() >= mean_threshold and cell.std() <= std_threshold)


def match_single_cell(
    cell: np.ndarray,
    library: TemplateLibrary,
    threshold: float = MATCH_THRESHOLD,
) -> tuple[int | None, float]:
    """
    Match a single cell against the template library.
    Prioritizes structural NCC score with secondary color verification to disambiguate
    shapes with identical outlines but different colors (e.g. 0, 12, 34).
    """
    if is_blank_cell(cell):
        return None, 0.0

    candidates = []

    for tid, tmpl in library.templates.items():
        if cell.shape[0] < tmpl.shape[0] or cell.shape[1] < tmpl.shape[1]:
            continue

        res = cv2.matchTemplate(cell, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val >= threshold:
            candidates.append((max_val, tid, max_loc, tmpl))

    if not candidates:
        return None, 0.0

    candidates.sort(key=lambda x: x[0], reverse=True)
    top_score = candidates[0][0]

    # Disambiguate structurally close matches (e.g. pink vs yellow bean) using body color distance
    close_candidates = [c for c in candidates if top_score - c[0] < 0.15]
    if len(close_candidates) == 1:
        return close_candidates[0][1], top_score

    best_tid = close_candidates[0][1]
    best_combined_score = -1e9

    for score, tid, max_loc, tmpl in close_candidates:
        mx, my = max_loc
        roi = cell[my : my + tmpl.shape[0], mx : mx + tmpl.shape[1]]
        th, tw = tmpl.shape[:2]
        roi_center = roi[int(th * 0.3) : int(th * 0.7), int(tw * 0.3) : int(tw * 0.7)]
        roi_color = np.mean(roi_center, axis=(0, 1))
        tmpl_color = library.template_colors[tid]
        color_dist = float(np.linalg.norm(roi_color - tmpl_color))
        # Color distance penalty: distance of 30 reduces score by 0.30
        adjusted_score = score - (color_dist / 100.0)
        if adjusted_score > best_combined_score:
            best_combined_score = adjusted_score
            best_tid = tid

    return best_tid, top_score


def normalize_board_image(board_bgr: np.ndarray) -> np.ndarray:
    """
    Universal resolution normalization:
    Resizes arbitrary board crops (e.g. 944x512, 1080x600, 1452x792) to baseline (610x334).
    Guarantees template matching works across all screen resolutions and DPIs.
    """
    H, W = board_bgr.shape[:2]
    if W == BASELINE_BOARD_WIDTH and H == BASELINE_BOARD_HEIGHT:
        return board_bgr

    interpolation = cv2.INTER_AREA if (W > BASELINE_BOARD_WIDTH) else cv2.INTER_CUBIC
    normalized = cv2.resize(board_bgr, (BASELINE_BOARD_WIDTH, BASELINE_BOARD_HEIGHT), interpolation=interpolation)
    return normalized


def parse_board(
    board_bgr: np.ndarray,
    library: TemplateLibrary,
    row_count: int = ROW,
    col_count: int = COL,
    threshold: float = MATCH_THRESHOLD,
) -> tuple[list[list[int]], dict[int, list[tuple[int, int]]]]:
    """
    Segment the board image into a grid and identify all tiles.
    Resolution-independent: automatically normalizes input to baseline dimensions.
    Returns:
      board: 2D array of dimensions (row_count+2, col_count+2) with -1 for empty space.
      cnter: Mapping from tile_id to list of (row, col) coordinates (1-indexed).
    """
    norm_board = normalize_board_image(board_bgr)
    H, W = norm_board.shape[:2]
    cell_w = W / col_count
    cell_h = H / row_count

    cells: list[tuple[int, int, np.ndarray]] = []
    for r in range(row_count):
        for c in range(col_count):
            x1 = int(c * cell_w)
            y1 = int(r * cell_h)
            x2 = int((c + 1) * cell_w)
            y2 = int((r + 1) * cell_h)
            cells.append((r + 1, c + 1, norm_board[y1:y2, x1:x2]))

    def _worker(item: tuple[int, int, np.ndarray]) -> tuple[int, int, int | None]:
        r_idx, c_idx, cell_crop = item
        tid, _ = match_single_cell(cell_crop, library, threshold)
        return r_idx, c_idx, tid

    results = list(_VISION_EXECUTOR.map(_worker, cells))

    board = [[-1] * (col_count + 2) for _ in range(row_count + 2)]
    cnter: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for r_idx, c_idx, tid in results:
        if tid is not None:
            board[r_idx][c_idx] = tid
            cnter[tid].append((r_idx, c_idx))

    # Odd-tile recovery: Active tile types must appear in pairs (even count).
    # If a tile has an odd count (e.g. 1), search unassigned cells with a relaxed threshold (0.64).
    for tid, positions in list(cnter.items()):
        if tid == STOP_TILE_ID or len(positions) % 2 == 0:
            continue
        tmpl = library.templates.get(tid)
        if tmpl is None:
            continue
        best_recov_score = 0.64
        best_recov_coord = None
        for r in range(row_count):
            for c in range(col_count):
                if board[r + 1][c + 1] != -1:
                    continue
                x1 = int(c * cell_w)
                y1 = int(r * cell_h)
                x2 = int((c + 1) * cell_w)
                y2 = int((r + 1) * cell_h)
                cell = norm_board[y1:y2, x1:x2]
                if cell.shape[0] < tmpl.shape[0] or cell.shape[1] < tmpl.shape[1]:
                    continue
                res = cv2.matchTemplate(cell, tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > best_recov_score:
                    best_recov_score = max_val
                    best_recov_coord = (r + 1, c + 1)
        if best_recov_coord:
            br, bc = best_recov_coord
            board[br][bc] = tid
            cnter[tid].append((br, bc))

    # Pass 2: Unassigned non-blank cell recovery:
    # If a cell is NOT an empty background cell but missed the primary MATCH_THRESHOLD (0.70),
    # retry with match_single_cell at relaxed threshold (0.63) to recover shaded or corner-vignetted tiles.
    for r in range(row_count):
        for c in range(col_count):
            if board[r + 1][c + 1] != -1:
                continue
            x1 = int(c * cell_w)
            y1 = int(r * cell_h)
            x2 = int((c + 1) * cell_w)
            y2 = int((r + 1) * cell_h)
            cell = norm_board[y1:y2, x1:x2]
            if is_blank_cell(cell):
                continue
            recov_tid, _ = match_single_cell(cell, library, threshold=0.63)
            if recov_tid is not None:
                board[r + 1][c + 1] = recov_tid
                cnter[recov_tid].append((r + 1, c + 1))

    return board, cnter


def annotate_board(
    board_bgr: np.ndarray,
    board: list[list[int]],
    path: list[tuple[tuple[int, int], tuple[int, int]]] | None = None,
    row_count: int = ROW,
    col_count: int = COL,
) -> np.ndarray:
    """Draw detection results and solution path lines onto a copy of the board image."""
    annotated = board_bgr.copy()
    H, W = annotated.shape[:2]
    cell_w = W / col_count
    cell_h = H / row_count

    font_scale = max(0.4, min(1.2, W / 800.0))
    line_thickness = max(1, int(W / 400.0))

    # Draw grid and tile IDs
    for r in range(1, row_count + 1):
        for c in range(1, col_count + 1):
            tid = board[r][c]
            x1 = int((c - 1) * cell_w)
            y1 = int((r - 1) * cell_h)
            x2 = int(c * cell_w)
            y2 = int(r * cell_h)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (80, 80, 80), 1)
            if tid != -1:
                label = "X" if tid == STOP_TILE_ID else str(tid)
                color = (0, 0, 255) if tid == STOP_TILE_ID else (0, 255, 0)
                cv2.putText(
                    annotated,
                    label,
                    (x1 + int(cell_w * 0.15), y1 + int(cell_h * 0.45)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    line_thickness,
                    cv2.LINE_AA,
                )

    # Draw solution steps
    if path:
        for idx, (p1, p2) in enumerate(path):
            c1_center = (int((p1[1] - 0.5) * cell_w), int((p1[0] - 0.5) * cell_h))
            c2_center = (int((p2[1] - 0.5) * cell_w), int((p2[0] - 0.5) * cell_h))
            cv2.line(annotated, c1_center, c2_center, (255, 200, 0), max(1, line_thickness))

    return annotated


def harvest_templates_from_board(
    board_bgr: np.ndarray,
    template_dir: str = TEMPLATE_DIR,
    existing_lib: TemplateLibrary | None = None,
    row_count: int = ROW,
    col_count: int = COL,
) -> int:
    """
    Automatically partition game board into cells, cluster identical tiles
    unsupervised using NCC and color distance, and save representative crops
    as template images in template_dir.
    Returns the number of new templates harvested.
    """
    os.makedirs(template_dir, exist_ok=True)
    norm = normalize_board_image(board_bgr)
    H, W = norm.shape[:2]
    cw, ch = W / col_count, H / row_count

    cells: list[np.ndarray] = []
    coords: list[tuple[int, int]] = []
    for r in range(row_count):
        for c in range(col_count):
            x1, y1 = int(c * cw), int(r * ch)
            x2, y2 = int((c + 1) * cw), int((r + 1) * ch)
            cell = norm[y1:y2, x1:x2]
            if is_blank_cell(cell):
                continue
            if existing_lib is not None and existing_lib.templates:
                tid, _ = match_single_cell(cell, existing_lib, threshold=0.68)
                if tid is not None:
                    continue  # Already known template
            cells.append(cell)
            coords.append((r + 1, c + 1))

    N = len(cells)
    if N == 0:
        return 0

    tmpls = []
    colors = []
    for cell in cells:
        h, w = cell.shape[:2]
        t = cell[h // 2 - 13 : h // 2 + 13, w // 2 - 13 : w // 2 + 13]
        tmpls.append(t)
        mean_c = cast(Sequence[float], cv2.cvtColor(t, cv2.COLOR_BGR2HSV).mean(axis=(0, 1)))[:3]
        colors.append(np.array(mean_c, dtype=np.float32))

    sim = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i, N):
            if i == j:
                sim[i][j] = 1.0
                continue
            res_ij = float(cv2.matchTemplate(cells[j], tmpls[i], cv2.TM_CCOEFF_NORMED).max())
            res_ji = float(cv2.matchTemplate(cells[i], tmpls[j], cv2.TM_CCOEFF_NORMED).max())
            c_dist = float(np.linalg.norm(colors[i] - colors[j]))
            s = (res_ij + res_ji) / 2.0 - (c_dist / 60.0)
            sim[i][j] = s
            sim[j][i] = s

    # Find optimal threshold minimizing odd-sized clusters
    best_clusters: list[list[int]] = []
    best_odd_count = 999
    for th in np.arange(0.58, 0.74, 0.02):
        adj = sim >= th
        visited = [False] * N
        clusters: list[list[int]] = []
        for i in range(N):
            if not visited[i]:
                comp: list[int] = []
                q = [i]
                visited[i] = True
                while q:
                    curr = q.pop()
                    comp.append(curr)
                    for nb in range(N):
                        if not visited[nb] and adj[curr][nb]:
                            visited[nb] = True
                            q.append(nb)
                clusters.append(comp)
        odd_count = sum(1 for c in clusters if len(c) % 2 != 0)
        if odd_count < best_odd_count:
            best_odd_count = odd_count
            best_clusters = clusters
            if odd_count == 0:
                break

    # Determine existing integer IDs
    existing_ids = set()
    if os.path.exists(template_dir):
        for fname in os.listdir(template_dir):
            stem = os.path.splitext(fname)[0]
            if stem.isdigit():
                existing_ids.add(int(stem))

    next_id = max(existing_ids) + 1 if existing_ids else 0

    stop_path = os.path.join(template_dir, "STOP.png")
    stop_tmpl = cv2.imread(stop_path) if os.path.exists(stop_path) else None

    saved_count = 0
    for cl in best_clusters:
        # Check if cluster is STOP obstacle
        is_obstacle = False
        if stop_tmpl is not None:
            rep_cell = cells[cl[0]]
            res = float(cv2.matchTemplate(rep_cell, stop_tmpl, cv2.TM_CCOEFF_NORMED).max())
            if res > 0.65:
                is_obstacle = True

        if is_obstacle:
            continue

        # Choose medoid (highest sum of similarity to other cluster members)
        best_intra = -999.0
        best_idx = cl[0]
        for member in cl:
            intra = float(sum(sim[member][other] for other in cl))
            if intra > best_intra:
                best_intra = intra
                best_idx = member

        medoid_cell = cells[best_idx]
        mh, mw = medoid_cell.shape[:2]
        crop = medoid_cell[mh // 2 - 13 : mh // 2 + 13, mw // 2 - 13 : mw // 2 + 13]
        out_file = os.path.join(template_dir, f"{next_id}.png")
        cv2.imwrite(out_file, crop)
        next_id += 1
        saved_count += 1

    return saved_count
