# Two-Turn-Puzzle-Solver

High-performance, automated solver and bot for $6 \times 11$ LianLianKan (Shisen-Sho / Onet Connect / 2-Turn Match) puzzle games. Powered by pure Android ADB control with zero desktop GUI or mouse-grabbing dependencies.

---

## 🌟 Key Features & Architecture

1. **Pure ADB Headless Automation (Zero GUI Dependency)**
   - Operates directly via native Android ADB shell commands. Runs silently in the background without requiring window focus, screencasting utilities, or mouse cursor interception.
   - **Strict Zero-Jewel Policy**: Built-in safety mechanisms guarantee zero accidental premium currency spending. Automatically dismisses gem confirmation modals, leverages free ad skips, or restarts cleanly.

2. **Bitmask State-Compressed DFS Solver (Deadlock-Free & Blazing Fast)**
   - Represents the 66-cell board state using a 66-bit integer bitmask, enabling $O(1)$ zero-allocation bitwise state transitions (`state_mask ^ (1 << p1 | 1 << p2)`).
   - Combines heuristic move ordering with lookahead backtracking search to guarantee 100% full-board clears in **7–8 milliseconds**.

3. **Scale-Invariant Computer Vision & Cold-Start Auto-Harvesting**
   - Automatically normalizes arbitrary device resolutions (e.g., 2280×1080, 1080×600) onto a canonical reference grid using normalized cross-correlation (NCC) and HSV color-space verification.
   - **Cold-Start Auto-Harvesting**: If launched without existing tile templates, the engine automatically partitions the active board, clusters identical tiles via even-parity graph connected components, and extracts representative template assets on the fly.

4. **Rapid Touch Batching & Selection Sanitization**
   - Compiles solution elimination paths into batched shell scripts (`sh`) for low-latency touch injection, completing an entire 32-pair (64-tile) board in **~7.9 seconds**.
   - Features inter-pass selection sanitization to reset any lingering Unity touch focus, eliminating input queue desynchronization.

---

## 📂 Project Structure

```text
Two-Turn-Puzzle-Solver/
├── config.py              # Global configuration (grid dimensions, relative ratios, delays)
├── core/
│   ├── __init__.py        # Core package exports
│   ├── device.py          # Pure ADB device manager, screenshot capture, and touch batching
│   ├── pathfinding.py     # O(ROW+COL) scanline line-of-sight & 2-turn path validation
│   ├── solver.py          # Bitmask-accelerated DFS lookahead puzzle solver
│   ├── vision.py          # Scale-invariant matching, state machine, and auto-harvesting
│   └── window.py          # Desktop window compatibility helpers
├── controller.py          # Auxiliary input controller (supports cooking minigame gestures)
├── main.py                # Unified CLI entry point (loop automation, benchmarks, calibration)
├── templates/             # Canonical tile templates (0~34.png, STOP.png, farm_minigame.png)
├── tests/                 # Unit test suite and benchmark fixtures
│   ├── fixtures/          # Benchmark board screenshots
│   ├── test_solver.py     # Pathfinding, solver logic, and benchmark tests
│   └── test_harvest.py    # Cold-start template auto-harvesting test
├── pyproject.toml         # Python project configuration (uv / pip)
└── README.md              # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+ (managed via [uv](https://github.com/astral-sh/uv) or standard Python virtual environment)
- Android device connected with **USB Debugging** enabled via ADB

```bash
# Clone the repository
git clone https://github.com/Yorafa/Two-Turn-Puzzle-Solver.git
cd Two-Turn-Puzzle-Solver

# Install dependencies with uv (or pip install -e .)
uv sync
```

---

## 🎮 Usage & CLI Commands

### 1. Run Unit Tests & Performance Benchmark

Verify connectivity, solver backtracking, and cold-start template harvesting:

```bash
uv run python -m unittest discover tests
```

### 2. Solve a Single Board Once

With the game board active on your connected phone screen:

```bash
uv run python main.py --once
```

### 3. Automated Multi-Round Loop

Runs fully autonomous continuous play—handles reward dismissals, cooldown timers, ad watching, and instant puzzle clearance:

```bash
# Run 10 consecutive automated rounds
uv run python main.py --loop 10
```

### 4. Template Auto-Harvesting & Board Calibration

```bash
# Auto-harvest tile templates from the current game board into templates/
uv run python main.py --harvest

# Preview adaptive bounding box detection
uv run python main.py --calibrate
```

### 5. Auxiliary Cooking Minigame

Automates dragging ingredients into the cooking pot and timing collections:

```bash
# Cook 30 batches (5 items per batch)
uv run python main.py --fry 30
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
