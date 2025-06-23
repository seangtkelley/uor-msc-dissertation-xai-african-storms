from pathlib import Path

# file paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "figures"
SRC_DIR = REPO_ROOT / "src"

# location information
MAP_AREA_CENTER = (41.0, 9.0)  # lon, lat
MAP_AREA_EXTENT = (
    MAP_AREA_CENTER[0] - 12,
    MAP_AREA_CENTER[0] + 12,
    MAP_AREA_CENTER[1] - 9,
    MAP_AREA_CENTER[1] + 9,
)
