from pathlib import Path

# file paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "figures"
SRC_DIR = REPO_ROOT / "src"

# extent of filtered storms region
DATA_REGION_EXTENT = (
    34.0,  # lon min
    52.0,  # lon max
    3.0,  # lat min
    15.0,  # lat max
)
# extent of map region for plotting
MAP_REGION_EXTENT = (
    DATA_REGION_EXTENT[0] - 5.0,  # lon min
    DATA_REGION_EXTENT[1] + 5.0,  # lon max
    DATA_REGION_EXTENT[2] - 5.0,  # lat min
    DATA_REGION_EXTENT[3] + 5.0,  # lat max
)
