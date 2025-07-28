#!/usr/bin/env python

"""config.py: Constants and configuration for the project"""

__author__ = "Sean Kelley"
__copyright__ = "Copyright 2025, University of Reading"
__credits__ = ["Sean Kelley"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Sean Kelley"
__email__ = "s.g.t.kelley@student.reading.ac.uk"
__status__ = "Development"

from pathlib import Path

# file paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = REPO_ROOT / "figures"
SRC_DIR = REPO_ROOT / "src"
OUTPUT_MODEL_DIR = REPO_ROOT / "models"

RAW_STORM_DB_PATH = (
    DATA_DIR / "East_Africa_tracked_MCSs_2014_2019_longer_than_3_hours.csv"
)
PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "processed_dataset.csv"

HYPERPARAMETER_JSON_PATH = SRC_DIR / "json_config" / "hyperparameters.json"
TRAIN_PARAMETERS_JSON_PATH = SRC_DIR / "json_config" / "train_parameters.json"

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

COL_RENAME_MAP = {
    "Lon": "lon",
    "Lat": "lat",
    "Storm": "storm_id",
    "Area": "area",
    "u_ms": "zonal_speed",
    "v_ms": "meridional_speed",
    "duration_hr": "storm_total_duration",
    "Min BT": "min_bt",
    "Mean BT": "mean_bt",
}

# TODO: Uncomment when the features are ready
FEATURE_COL_NAMES = [
    "eat_hours",
    "storm_total_duration",
    "lon",
    "lat",
    "orography_height",
    "anor",
    "over_land",
    "acc_land_time",
    "storm_total_land_time",
    "mean_land_frac",
    "zonal_speed",
    "meridional_speed",
    "area",
    "storm_max_area",
    "bearing_from_prev",
    "distance_from_prev",
    "distance_traversed",
    "storm_bearing",
    "storm_distance_traversed",
    "storm_straight_line_distance",
    "mean_prcp_400",
    "min_bt",
    "dmin_bt_dt",
    "mean_bt",
    "dmean_bt_dt",
    "storm_min_bt",
    "storm_min_bt_reached",
    # "mean_land_skt",
    # "mean_dthetae_dp_900_750",
    # "mean_dthetae_dp_750_500",
    # "ushear_850",
    "mjo_phase",
    "mjo_amplitude",
]

TARGET_COL_NAMES = ["storm_total_duration", "mean_prcp_400", "storm_min_bt"]

DATASET_COL_NAMES = ["storm_id", "timestamp"] + FEATURE_COL_NAMES

# Weights & Biases configuration
WANDB_ENTITY = "uor-msc"
WANDB_PROJECT = "uor-msc-dissertation-xai-african-storms"
