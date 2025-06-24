from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# file paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "figures"
SRC_DIR = REPO_ROOT / "src"

# extent of filtered storms region
STORM_DATA_EXTENT = (
    34.0,  # lon min
    52.0,  # lon max
    3.0,  # lat min
    15.0,  # lat max
)
# extent of map region for plotting
PADDED_DATA_EXTENT = (
    STORM_DATA_EXTENT[0] - 5.0,  # lon min
    STORM_DATA_EXTENT[1] + 5.0,  # lon max
    STORM_DATA_EXTENT[2] - 5.0,  # lat min
    STORM_DATA_EXTENT[3] + 5.0,  # lat max
)
# extent of ERA5 data
ERA5_DATA_EXTENT = (
    31.0,  # lon min
    53.0,  # lon max
    2.0,  # lat min
    16.0,  # lat max
)

# date range for data
DATA_START = "2014-01-01"
DATA_END = "2019-12-31"

# scope cmap terrain to start at green
TERRAIN_CMAP = ListedColormap(
    plt.get_cmap("terrain")(np.linspace(0.25, 1, plt.get_cmap("terrain").N))
)
