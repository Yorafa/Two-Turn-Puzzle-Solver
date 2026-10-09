"""
Game configuration parameters and defaults.
"""

import os

# Board geometry
ROW: int = 6
COL: int = 11
MAX_TURN: int = 2
STOP_TILE_ID: int = 36  # Obstacle block that cannot be eliminated

# Directory paths
TEMPLATE_DIR: str = "templates" if os.path.exists("templates") else "img2"
IMG_DIR: str = "img"
FAILURE_DIR: str = "failure"
CONFIG_FILE: str = "board_config.json"

# Ensure output directories exist
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(FAILURE_DIR, exist_ok=True)

# Window & screen coordinates
DEFAULT_WINDOW_TITLE: str = "SM-G973W"

# Reference baseline dimensions (for template matching scale invariance)
BASELINE_BOARD_WIDTH: int = 610
BASELINE_BOARD_HEIGHT: int = 334

# Landscape board bbox relative to full screen/window (Width > Height):
# From 2280x1080: x in [256, 1736] -> [0.1123, 0.7614], y in [128, 952] -> [0.1185, 0.8815]
LANDSCAPE_RELATIVE_BBOX: tuple[float, float, float, float] = (0.1123, 0.1185, 0.7614, 0.8815)

# Portrait board bbox relative to window (Height >= Width, e.g. old 974x1047 capture):
PORTRAIT_RELATIVE_BBOX: tuple[float, float, float, float] = (0.1181, 0.3467, 0.7444, 0.6657)

# Default fallback
DEFAULT_RELATIVE_BBOX: tuple[float, float, float, float] = LANDSCAPE_RELATIVE_BBOX
DEFAULT_BOARD_BBOX: tuple[int, int, int, int] = (256, 128, 1736, 952)

# Relative button locations (works on both phone screen and window):
START_BUTTON_REL: tuple[float, float] = (0.500, 0.808)  # Center of "Start Minigame"
AD_BUTTON_REL: tuple[float, float] = (0.577, 0.821)  # Center of "Ad" button
REWARDS_CLOSE_REL: tuple[float, float] = (0.725, 0.112)  # Top-right 'X' on Rewards popup
JEWEL_POPUP_CLOSE_REL: tuple[float, float] = (0.6855, 0.2509)  # Top-right 'X' on "Spend Jewel" popup
MENU_CLOSE_REL: tuple[float, float] = (0.880, 0.157)  # Top-right 'X' on Minigames menu
STAGE_CLOSE_REL: tuple[float, float] = (0.8882, 0.1056)  # Top-right 'X' on active stage badge
STAGE_QUIT_CONFIRM_REL: tuple[float, float] = (0.531, 0.514)  # Confirmation OK button on quit stage dialog
DEADLOCK_OK_REL: tuple[float, float] = (0.500, 0.630)  # Center 'OK' button on "No more moves" popup
TITLE_START_REL: tuple[float, float] = (0.500, 0.440)  # Center "Tap to Start" on Title screen
FARM_MINIGAME_BUTTON_REL: tuple[float, float] = (0.665, 0.884)  # "Minigame" (小遊戲) icon on Farm screen
DIALOG_RETRY_REL: tuple[float, float] = (0.500, 0.600)  # Center 'Retry/OK' on network reconnection dialogs

# Android app identity
APP_PACKAGE: str = "jp.co.imagineer.sumikkogurashi.farm"
APP_ACTIVITY: str = "com.unity3d.player.UnityPlayerActivity"

# Fallback absolute button coordinates
SKIP_BUTTON: tuple[int, int] = (1140, 800)
CLOSE_BUTTON: tuple[int, int] = (1653, 121)
RETRY_BUTTON: tuple[int, int] = (1020, 830)
CONFIRM_BUTTON: tuple[int, int] = (1210, 555)

# Optional cooking minigame positions
EGG_POS: tuple[int, int] = (750, 1000)
POTATO_POS: tuple[int, int] = (860, 1015)
POT_POS: tuple[int, int] = (355, 660)
POT_INTERVAL: int = 147

# Computer Vision parameters
BLANK_STD_THRESHOLD: float = 6.0
BLANK_MEAN_THRESHOLD: float = 245.0
MATCH_THRESHOLD: float = 0.70

# Automation timing
CLICK_DELAY: float = 0.025
PAIR_DELAY: float = 0.055
