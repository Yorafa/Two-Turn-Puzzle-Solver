"""
Main entry point for Two-Turn-Puzzle-Solver.
Operates purely via Android ADB backend with zero desktop GUI dependencies.
Supports resolution-adaptive detection, multi-round automation, benchmarks, and rapid elimination.
"""

import argparse
import io
import os
import sys
import time

import cv2
import numpy as np

# Enable unbuffered output immediately
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(line_buffering=True)

from config import (
    COL,
    FAILURE_DIR,
    IMG_DIR,
    ROW,
    STOP_TILE_ID,
    TEMPLATE_DIR,
)
from core.device import DeviceManager, detect_board_bbox
from core.solver import solve_puzzle
from core.vision import (
    TemplateLibrary,
    annotate_board,
    detect_screen_state,
    harvest_templates_from_board,
    parse_board,
)


def run_benchmark(debug: bool = True) -> None:
    """Run benchmark across multiple resolutions to verify scale invariance."""
    print("=" * 65, flush=True)
    print("  Two-Turn-Puzzle-Solver: Universal Scale-Invariant Benchmark", flush=True)
    print("=" * 65, flush=True)

    library = TemplateLibrary(TEMPLATE_DIR)
    print(f"[*] Loaded {len(library.templates)} templates.", flush=True)

    test_png_path = (
        os.path.join(IMG_DIR, "test.png") if os.path.exists(os.path.join(IMG_DIR, "test.png")) else "test.png"
    )
    test_images = [
        (
            "temp.png (Original ~610x334 crop from 960x1080)",
            os.path.join(TEMPLATE_DIR, "temp.png"),
            (115, 363, 725, 697),
        ),
        ("test.png (High-Res 944x512 direct board crop)", test_png_path, None),
        ("last_initial_board.png (Live 1479x825 crop)", os.path.join(IMG_DIR, "last_initial_board.png"), None),
    ]

    for label, path, crop_box in test_images:
        if not os.path.exists(path):
            continue
        print(f"\n>> Testing on: {label}", flush=True)
        img = cv2.imread(path)
        if img is None:
            continue
        if crop_box:
            x1, y1, x2, y2 = crop_box
            board_img = img[y1:y2, x1:x2]
        else:
            board_img = img

        print(f"   Native input resolution: {board_img.shape[1]}x{board_img.shape[0]}", flush=True)
        t0 = time.perf_counter()
        board, cnter = parse_board(board_img, library)
        t_parse = (time.perf_counter() - t0) * 1000
        total_tiles = sum(len(v) for v in cnter.values())
        print(f"   [+] Recognized {total_tiles} active tiles in {t_parse:.2f} ms", flush=True)

        t0 = time.perf_counter()
        success, path_steps = solve_puzzle(board, cnter)
        t_solve = (time.perf_counter() - t0) * 1000

        if success:
            print(f"   [+] Solved in {t_solve:.2f} ms! Pairs to eliminate: {len(path_steps)}", flush=True)
        else:
            print(f"   [-] Unsolvable or deadlock in {t_solve:.2f} ms.", flush=True)

        if debug:
            annotated = annotate_board(board_img, board, path_steps)
            out_name = os.path.join(IMG_DIR, f"debug_{'temp' if crop_box else 'test'}.png")
            cv2.imwrite(out_name, annotated)
            print(f"   [+] Annotated result saved to: {out_name}", flush=True)

    print("\n" + "=" * 65, flush=True)


def run_calibrate(device: DeviceManager) -> None:
    """Capture raw screen via ADB, preview detected board region and grid alignment."""
    if not device.adb.is_available():
        print("[!] No ADB device found for calibration.", flush=True)
        return

    screen = device.capture_screen()
    if screen is None:
        print("[!] Failed to capture screen via ADB.", flush=True)
        return

    sh, sw = screen.shape[:2]
    auto_box = detect_board_bbox(screen)
    if auto_box:
        bx1, by1, bx2, by2 = auto_box
        print(f"[*] Auto-detected board bounds: ({bx1}, {by1}) -> ({bx2}, {by2})", flush=True)
    else:
        rx1, ry1, rx2, ry2 = device.get_relative_bbox(sw, sh)
        bx1 = int(rx1 * sw)
        by1 = int(ry1 * sh)
        bx2 = int(rx2 * sw)
        by2 = int(ry2 * sh)
        print(f"[*] Fallback relative bounds: ({bx1}, {by1}) -> ({bx2}, {by2})", flush=True)

    bw = bx2 - bx1
    bh = by2 - by1
    preview = screen.copy()
    cv2.rectangle(preview, (bx1, by1), (bx2, by2), (0, 0, 255), 3)

    # Draw grid cells
    for r in range(ROW + 1):
        y = int(by1 + r * (bh / ROW))
        cv2.line(preview, (bx1, y), (bx2, y), (0, 255, 0), 1)
    for c in range(COL + 1):
        x = int(bx1 + c * (bw / COL))
        cv2.line(preview, (x, by1), (x, by2), (0, 255, 0), 1)

    cv2.putText(
        preview,
        f"Board Region: {bw}x{bh}",
        (bx1, max(30, by1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    out_file = os.path.join(IMG_DIR, "calibration_preview.png")
    cv2.imwrite(out_file, preview)
    print(f"[+] Calibration preview saved to: {out_file}", flush=True)
    print(f"    Ratios: [rx1={bx1 / sw:.4f}, ry1={by1 / sh:.4f}, rx2={bx2 / sw:.4f}, ry2={by2 / sh:.4f}]", flush=True)


def save_failure_dump(
    device: DeviceManager,
    board_img: cv2.typing.MatLike,
    board: list[list[int]],
    cnter: dict[int, list[tuple[int, int]]],
    initial_data: (
        tuple[
            np.ndarray | None,
            cv2.typing.MatLike,
            list[list[int]],
            dict[int, list[tuple[int, int]]],
        ]
        | None
    ) = None,
    pass_idx: int = 1,
) -> str:
    """Save full screenshot, board crop, annotated detection grid, and diagnostic metadata on failure."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"fail_{timestamp}"

    # 1. Save Initial Opening Board (if captured at Pass 1)
    if initial_data is not None:
        init_full, init_board_img, init_board, init_cnter = initial_data
        if init_full is not None:
            cv2.imwrite(os.path.join(FAILURE_DIR, f"{prefix}_01_initial_fullscreen.png"), init_full)
        cv2.imwrite(os.path.join(FAILURE_DIR, f"{prefix}_01_initial_board.png"), init_board_img)
        cv2.imwrite(
            os.path.join(FAILURE_DIR, f"{prefix}_01_initial_annotated.png"),
            annotate_board(init_board_img, init_board),
        )
        with open(os.path.join(FAILURE_DIR, f"{prefix}_01_initial_info.txt"), "w", encoding="utf-8") as f:
            f.write(f"Timestamp: {timestamp}\nPhase: Initial Opening Board (Pre-elimination)\n")
            active = {k: len(v) for k, v in init_cnter.items() if k != STOP_TILE_ID}
            f.write(f"Active Tile Counts ({sum(active.values())} tiles): {active}\n")
            odd = {k: v for k, v in active.items() if v % 2 != 0}
            if odd:
                f.write(f"[WARNING] Odd-count tiles detected: {odd}\n")
            f.write(f"Obstacles (ID {STOP_TILE_ID}): {len(init_cnter.get(STOP_TILE_ID, []))}\n")
            f.write("\nBoard Grid (1-indexed, -1 is empty):\n")
            f.writelines(
                f"Row {r:2d}: " + " ".join(f"{init_board[r][c]:3d}" for c in range(1, COL + 1)) + "\n"
                for r in range(1, ROW + 1)
            )

    # 2. Save Current Failed Board State
    fail_prefix = f"{prefix}_02_fail_p{pass_idx}"
    curr_full = device.capture_screen()
    if curr_full is not None:
        cv2.imwrite(os.path.join(FAILURE_DIR, f"{fail_prefix}_fullscreen.png"), curr_full)
    cv2.imwrite(os.path.join(FAILURE_DIR, f"{fail_prefix}_board.png"), board_img)
    cv2.imwrite(
        os.path.join(FAILURE_DIR, f"{fail_prefix}_annotated.png"),
        annotate_board(board_img, board),
    )
    with open(os.path.join(FAILURE_DIR, f"{fail_prefix}_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"Timestamp: {timestamp}\nPass: {pass_idx}\n")
        active = {k: len(v) for k, v in cnter.items() if k != STOP_TILE_ID}
        f.write(f"Active Tile Counts ({sum(active.values())} tiles): {active}\n")
        odd = {k: v for k, v in active.items() if v % 2 != 0}
        if odd:
            f.write(f"[WARNING] Odd-count tiles detected: {odd}\n")
        f.write(f"Obstacles (ID {STOP_TILE_ID}): {len(cnter.get(STOP_TILE_ID, []))}\n")
        f.write("\nBoard Grid (1-indexed, -1 is empty):\n")
        f.writelines(
            f"Row {r:2d}: " + " ".join(f"{board[r][c]:3d}" for c in range(1, COL + 1)) + "\n" for r in range(1, ROW + 1)
        )

    print(f"[!] Failure diagnostic snapshot saved to: {FAILURE_DIR}/{prefix}_*.png", flush=True)
    return prefix


def run_once(
    library: TemplateLibrary,
    device: DeviceManager,
    max_wait_seconds: float = 10.0,
    max_passes: int = 5,
) -> bool:
    """Capture screen via ADB with animation wait, multi-pass solve, and execute rapid tap batch."""
    print(f"[*] Waiting for game board to settle from ADB ({device.adb.device_id})...", flush=True)

    start_time = time.perf_counter()
    board_img: cv2.typing.MatLike | None = None
    abs_bbox: tuple[int, int, int, int] | None = None
    board: list[list[int]] | None = None
    cnter: dict[int, list[tuple[int, int]]] | None = None
    prev_blocks = -1
    stable_count = 0

    # Wait for board tiles to land and stabilize (handling tile drop and 'GAME START!!' animation)
    while time.perf_counter() - start_time < max_wait_seconds:
        board_img, abs_bbox = device.capture_board()
        if library.active_tile_count == 0:
            print(
                f"[*] No tile templates in '{library.template_dir}'. Auto-harvesting from current game board...",
                flush=True,
            )
            time.sleep(1.0)
            board_img, abs_bbox = device.capture_board()
            harvested = harvest_templates_from_board(board_img, library.template_dir, existing_lib=library)
            print(f"[+] Harvested {harvested} tile templates! Reloading template library...", flush=True)
            library.load_templates()

        board, cnter = parse_board(board_img, library)
        total_blocks = sum(len(v) for k, v in cnter.items() if k != STOP_TILE_ID)
        has_odd_tiles = any(len(v) % 2 != 0 for k, v in cnter.items() if k != STOP_TILE_ID)

        # 1. Fast path: Full board has >= 60 active tiles and no odd-count tiles (no text occlusion)
        if total_blocks >= 60 and not has_odd_tiles:
            break

        # 2. Dynamic re-scan for boards < 60 tiles or occluded by 'GAME START!!' text
        if total_blocks >= 40 and not has_odd_tiles and total_blocks == prev_blocks:
            stable_count += 1
            if stable_count >= 2:
                break
        else:
            stable_count = 0

        print(
            f"[*] Board not ready ({total_blocks} tiles, odd_tiles={has_odd_tiles}). Re-scanning in 0.25s...",
            flush=True,
        )
        prev_blocks = total_blocks
        time.sleep(0.25)

    if board_img is None or abs_bbox is None or board is None or cnter is None:
        print("[!] Failed to capture board image.", flush=True)
        return False

    print(f"[*] Board resolution: {board_img.shape[1]}x{board_img.shape[0]} at bbox: {abs_bbox}", flush=True)

    # Multi-pass elimination loop
    total_eliminated = 0
    prev_pass_tiles = -1
    initial_data: (
        tuple[
            np.ndarray | None,
            cv2.typing.MatLike,
            list[list[int]],
            dict[int, list[tuple[int, int]]],
        ]
        | None
    ) = None

    initial_active_tiles = 0
    needed_pairs = 0
    curr_bbox = abs_bbox

    for pass_idx in range(1, max_passes + 1):
        if pass_idx > 1:
            board_img, curr_bbox = device.capture_board()
            board, cnter = parse_board(board_img, library)
        active_tiles = sum(len(v) for k, v in cnter.items() if k != STOP_TILE_ID)

        if initial_data is None:
            initial_data = (
                None,
                board_img.copy(),
                [r[:] for r in board],
                {k: list(v) for k, v in cnter.items()},
            )
            cv2.imwrite(os.path.join(IMG_DIR, "last_initial_board.png"), board_img)
            initial_active_tiles = active_tiles
            needed_pairs = initial_active_tiles // 2

        if active_tiles == 0:
            if initial_active_tiles > 0 and total_eliminated >= needed_pairs:
                print(f"[+] All tiles cleared from board! Total eliminated: {total_eliminated} pairs.", flush=True)
                return True
            print(
                f"[!] Deadlock / popup occlusion: 0 active tiles recognized, but only {total_eliminated}/{needed_pairs} pairs eliminated!",
                flush=True,
            )
            save_failure_dump(device, board_img, board, cnter, initial_data=initial_data, pass_idx=pass_idx)
            device.click_deadlock_ok()
            return False

        if pass_idx > 1 and active_tiles == prev_pass_tiles:
            print(
                f"[!] Board unchanged at {active_tiles} tiles after previous elimination batch. Aborting redundant passes.",
                flush=True,
            )
            save_failure_dump(device, board_img, board, cnter, initial_data=initial_data, pass_idx=pass_idx)
            return False

        prev_pass_tiles = active_tiles

        print(
            f"[*] Pass {pass_idx}: Recognized {active_tiles} active tiles ({sum(len(v) for v in cnter.values())} incl. obstacles).",
            flush=True,
        )
        is_full_solve, path = solve_puzzle(board, cnter)

        # Sanity check: An authentic opening board always has at least 20 active tiles
        if pass_idx == 1 and active_tiles < 20:
            print(
                f"[!] Pass 1: Impossibly low tile count ({active_tiles} tiles detected). Screen is not a valid game board!",
                flush=True,
            )
            save_failure_dump(device, board_img, board, cnter, initial_data=initial_data, pass_idx=pass_idx)
            return False

        # ONE-SHOT GATE FOR PASS 1:
        # On the opening board, strictly require a 100% complete solve before touching the phone screen!
        if pass_idx == 1 and not is_full_solve:
            needed_pairs = active_tiles // 2
            print(
                f"[!] Pass 1: Incomplete solution ({len(path)} pairs found, required {needed_pairs} pairs for full clear).",
                flush=True,
            )
            print("[*] Screen held static. Attempting extended DFS search (2.0s)...", flush=True)
            is_full_solve, path = solve_puzzle(board, cnter, timeout_seconds=2.0)

            if not is_full_solve:
                print(
                    f"[!] Pass 1: Board has recognition ambiguity or deadlock ({len(path)}/{needed_pairs} pairs).",
                    flush=True,
                )
                print(
                    "[!] One-Shot Gate ACTIVE: Refusing partial tap execution to prevent corrupting board state.",
                    flush=True,
                )
                save_failure_dump(device, board_img, board, cnter, initial_data=initial_data, pass_idx=pass_idx)
                return False

            print(f"[+] Extended search succeeded! 100% full solve verified ({len(path)} pairs).", flush=True)

        if not path:
            # Re-check once after a brief delay in case tiles were still fading or settling
            time.sleep(1.0)
            board_img, abs_bbox = device.capture_board()
            board, cnter = parse_board(board_img, library)
            active_tiles = sum(len(v) for k, v in cnter.items() if k != STOP_TILE_ID)
            if active_tiles == 0:
                if initial_active_tiles > 0 and total_eliminated >= needed_pairs:
                    print(f"[+] All tiles cleared from board! Total eliminated: {total_eliminated} pairs.", flush=True)
                    return True
                print(
                    f"[!] Deadlock / popup occlusion: 0 active tiles recognized, but only {total_eliminated}/{needed_pairs} pairs eliminated!",
                    flush=True,
                )
                save_failure_dump(device, board_img, board, cnter, initial_data=initial_data, pass_idx=pass_idx)
                device.click_deadlock_ok()
                return False
            is_full_solve, path = solve_puzzle(board, cnter)

        if not path:
            print(f"[!] No valid moves found with {active_tiles} tiles remaining.", flush=True)
            save_failure_dump(device, board_img, board, cnter, initial_data=initial_data, pass_idx=pass_idx)
            return False

        print(
            f"[+] Pass {pass_idx}: Found solution path ({len(path)} pairs). Executing rapid elimination...", flush=True
        )
        t0 = time.perf_counter()
        device.adb.execute_tap_batch(path, curr_bbox)
        eliminated = len(path)
        elapsed = time.perf_counter() - t0
        total_eliminated += eliminated
        print(
            f"[+] Pass {pass_idx} eliminated {eliminated} pairs in {elapsed:.2f} s ({elapsed / max(eliminated, 1):.3f} s/pair)!",
            flush=True,
        )

        # Early exit: If all needed pairs have been eliminated, board is completely cleared!
        if initial_active_tiles > 0 and total_eliminated >= needed_pairs:
            print(f"[+] All tiles cleared from board! Total eliminated: {total_eliminated} pairs.", flush=True)
            return True

        # Give tiles time to disappear
        time.sleep(0.5)
        # Selection Sanitization: Tap safe empty margin above board to cancel any dangling selection
        device.click_relative(0.50, 0.05)
        time.sleep(0.2)

    # Final check
    board_img, abs_bbox = device.capture_board()
    board, cnter = parse_board(board_img, library)
    remaining = sum(len(v) for k, v in cnter.items() if k != STOP_TILE_ID)
    if remaining == 0 and initial_active_tiles > 0 and total_eliminated >= needed_pairs:
        print(f"[+] Board completely cleared! Total: {total_eliminated} pairs.", flush=True)
        return True
    print(
        f"[!] Round incomplete: {total_eliminated}/{needed_pairs} pairs eliminated ({remaining} tiles remaining).",
        flush=True,
    )
    save_failure_dump(device, board_img, board, cnter, initial_data=initial_data, pass_idx=max_passes + 1)
    device.click_deadlock_ok()
    return False


def run_loop(
    rounds: int = 10,
    serial: str | None = None,
) -> None:
    """Automate complete game rounds via ADB using an intelligent state machine with zero-jewel spend policy."""
    device = DeviceManager(serial=serial)
    if not device.adb.is_available():
        print("[!] Error: No Android device detected via ADB! Please connect via USB or Wi-Fi.", flush=True)
        return

    library = TemplateLibrary(TEMPLATE_DIR)

    print(f"[*] Starting auto loop for {rounds} rounds via ADB ({device.adb.device_id}).", flush=True)
    print("[!] Zero-Jewel policy ACTIVE: Never spends jewels. Watches Ads or restarts.", flush=True)

    for r in range(rounds):
        print(f"\n==================== Round {r + 1}/{rounds} ====================", flush=True)

        # 1. State machine: Navigate from menu/rewards/ads to game board
        entered_game = False
        jewel_popup_consecutive = 0
        no_ad_consecutive = 0
        clear_consecutive = 0
        unknown_consecutive = 0

        for attempt in range(30):
            full_shot = device.capture_screen()
            if full_shot is None:
                time.sleep(1.0)
                continue

            state = detect_screen_state(full_shot)
            print(f"[*] Screen state: {state} (step {attempt + 1})", flush=True)

            if state != "UNKNOWN":
                unknown_consecutive = 0
            if state != "CLEAR":
                clear_consecutive = 0

            if state == "SPEND_JEWEL_POPUP":
                jewel_popup_consecutive += 1
                print(
                    f"[!] Spend Jewel popup detected (attempt {jewel_popup_consecutive})! Dismissing to protect jewels...",
                    flush=True,
                )
                if jewel_popup_consecutive >= 3:
                    # Save diagnostic screenshot so user can see what was on screen!
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    fail_path = os.path.join(FAILURE_DIR, f"fail_jewel_popup_{timestamp}.png")
                    cv2.imwrite(fail_path, full_shot)
                    print(
                        f"[!] Unable to dismiss jewel popup after 3 attempts. Diagnostic saved to: {fail_path}",
                        flush=True,
                    )
                    print("[!] Force-restarting game to strictly guarantee zero jewels spent...", flush=True)
                    device.restart_game()
                    time.sleep(8.0)
                    jewel_popup_consecutive = 0
                    continue

                device.click_close_jewel_popup()
                if jewel_popup_consecutive > 1:
                    device.press_back()
                time.sleep(1.5)

            else:
                jewel_popup_consecutive = 0

            if state == "TITLE_SCREEN":
                print("[*] Title Screen detected ('點擊並開始遊戲'). Tapping to enter farm...", flush=True)
                device.click_title_start()
                time.sleep(4.0)
                continue

            elif state == "LOADING":
                print("[*] Game loading screen active ('讀取中'). Waiting...", flush=True)
                time.sleep(2.5)
                continue

            elif state == "FARM":
                print("[*] Main Farm screen detected. Opening minigame entrance...", flush=True)
                device.click_farm_minigame()
                time.sleep(2.5)
                continue

            elif state == "POPUP_DIALOG":
                print("[*] Announcement or event popup detected. Dismissing to access farm...", flush=True)
                device.press_back()
                time.sleep(1.5)
                continue

            elif state == "CLEAR":
                clear_consecutive += 1
                if clear_consecutive <= 2:
                    print("[*] Victory animation 'CLEAR!!' detected. Waiting for rewards dialog...", flush=True)
                    time.sleep(1.5)
                else:
                    print("[*] Persistent 'CLEAR!!' or stuck screen. Tapping to dismiss/advance...", flush=True)
                    device.click_relative(0.50, 0.50)
                    time.sleep(1.5)
                continue

            elif state == "REWARDS":
                print("[*] Rewards dialog detected. Closing dialog...", flush=True)
                device.click_close_rewards()
                time.sleep(1.5)

            elif state == "START":
                no_ad_consecutive = 0
                print("[*] 'Start Minigame' is ready! Clicking start...", flush=True)
                device.click_start()
                time.sleep(2.0)  # Snappy wait for entry animation
                entered_game = True
                break

            elif state == "AD":
                no_ad_consecutive = 0
                print("[*] 'Ad' button detected. Watching ad to skip cooldown for free...", flush=True)
                device.click_ad()
                # Wait for ad to start/finish and enter game or board
                time.sleep(3.0)
                # Poll up to 35 seconds for ad completion / board entry
                for poll_idx in range(12):
                    shot = device.capture_screen()
                    st = detect_screen_state(shot)
                    if st == "GAME":
                        entered_game = True
                        break
                    # During early ad playback (<12s), ignore transient ad frame noise
                    if poll_idx >= 4 and st in ("START", "REWARDS", "SPEND_JEWEL_POPUP"):
                        break
                    time.sleep(2.5)
                if entered_game:
                    break

            elif state == "NO_AD_COOLDOWN":
                no_ad_consecutive += 1
                if no_ad_consecutive < 5:
                    print(
                        f"[*] Cooldown active. Waiting for free 'Ad' button to load from network ({no_ad_consecutive}/5)...",
                        flush=True,
                    )
                    time.sleep(2.0)
                    continue
                else:
                    print("[!] Cooldown active and no Ad loaded after 10s. Restarting game per policy...", flush=True)
                    device.restart_game()
                    time.sleep(6.0)
                    no_ad_consecutive = 0
                    continue

            elif state == "GAME":
                entered_game = True
                break

            elif state == "UNKNOWN":
                unknown_consecutive += 1
                if unknown_consecutive >= 4 and (unknown_consecutive % 2 == 0):
                    print(
                        f"[*] Screen remains UNKNOWN ({unknown_consecutive} steps). Sending Back key to dismiss any stuck modal...",
                        flush=True,
                    )
                    device.press_back()
                time.sleep(1.5)

        # 2. Gate: Never attempt board solving unless game board was actively entered
        if not entered_game:
            print("[!] Could not enter active game board within allowed attempts. Retrying next cycle...", flush=True)
            time.sleep(2.0)
            continue

        # 3. Capture and solve board with multi-pass elimination
        solved = run_once(library, device, max_wait_seconds=10.0)
        if not solved:
            print("[!] Round was not fully cleared. Checking screen state...", flush=True)
            time.sleep(1.5)
            chk_shot = device.capture_screen()
            if chk_shot is not None and detect_screen_state(chk_shot) == "GAME":
                print("[!] Stuck on unsolvable board state. Exiting current stage to avoid loop...", flush=True)
                device.click_stage_close()
                time.sleep(2.0)
                continue

        # 3. Post-game wait and rewards dismissal (poll for victory dialog)
        for _ in range(12):
            time.sleep(1.0)
            full_shot = device.capture_screen()
            if full_shot is not None:
                st = detect_screen_state(full_shot)
                if st == "CLEAR":
                    print("[*] Victory animation 'CLEAR!!' in progress, waiting...", flush=True)
                    continue
                elif st == "REWARDS":
                    print("[*] Dismissing completion rewards...", flush=True)
                    device.click_close_rewards()
                    time.sleep(1.5)
                    break
                elif st in ("START", "AD"):
                    break


def main():
    parser = argparse.ArgumentParser(description="Two-Turn-Puzzle-Solver CLI (Pure ADB Mode)")
    parser.add_argument("--test", action="store_true", help="Run benchmark across multiple sample resolutions")
    parser.add_argument("--once", action="store_true", help="Capture ADB screen, solve once, and execute")
    parser.add_argument("--loop", type=int, default=0, help="Run automated loop for N rounds via ADB")
    parser.add_argument("--device", type=str, default=None, help="Specific ADB device serial (default: auto-detect)")
    parser.add_argument("--fry", type=int, default=0, help="Run cooking minigame for N items via ADB gestures")
    parser.add_argument("--calibrate", action="store_true", help="Inspect and preview ADB board calibration")
    parser.add_argument(
        "--harvest",
        action="store_true",
        help="Capture ADB screen, partition board, and auto-harvest tile templates to templates/",
    )
    parser.add_argument(
        "--set-bbox",
        nargs=4,
        type=float,
        metavar=("RX1", "RY1", "RX2", "RY2"),
        help="Save custom relative bounding box ratios to board_config.json",
    )

    args = parser.parse_args()

    device = DeviceManager(serial=args.device)

    if device.adb.is_available():
        print(f"[*] Backend: Direct ADB Device ({device.adb.device_id})", flush=True)
    elif not args.test:
        print(
            "[!] Error: No Android device detected via ADB! Please connect device or enable USB debugging.", flush=True
        )
        return

    if args.set_bbox:
        device.save_calibration(tuple(args.set_bbox))
        return

    if args.harvest:
        library = TemplateLibrary(TEMPLATE_DIR)
        print("[*] Capturing board to auto-harvest tile templates...", flush=True)
        board_img, _ = device.capture_board()
        cnt = harvest_templates_from_board(board_img, TEMPLATE_DIR, existing_lib=library)
        print(f"[+] Successfully harvested {cnt} new templates into '{TEMPLATE_DIR}'!", flush=True)
    elif args.calibrate:
        run_calibrate(device)
    elif args.test or (not args.once and args.loop == 0 and args.fry == 0):
        run_benchmark(debug=True)
    elif args.once:
        library = TemplateLibrary(TEMPLATE_DIR)
        run_once(library, device)
    elif args.loop > 0:
        run_loop(args.loop, serial=args.device)
    elif args.fry > 0:
        from controller import GameController

        ctrl = GameController(device_manager=device)
        print(f"[*] Cooking {args.fry} items via ADB...", flush=True)
        ctrl.fry(args.fry)
        print("[+] Cooking complete.", flush=True)


if __name__ == "__main__":
    main()
