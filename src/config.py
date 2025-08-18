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

# ==============================================================================
#                           PATH CONFIGURATION
# ==============================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = REPO_ROOT / "figures"
SRC_DIR = REPO_ROOT / "src"
MODEL_OUTPUT_DIR = REPO_ROOT / "models"

RAW_STORM_DB_PATH = (
    DATA_DIR / "East_Africa_tracked_MCSs_2014_2019_longer_than_3_hours.csv"
)
PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "processed_dataset.csv"


# ==============================================================================
#                           PLOTTING CONFIGURATION
# ==============================================================================
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


# ==============================================================================
#                           DATASET CONFIGURATION
# ==============================================================================
# date range for data
DATA_START = "2014-01-01"
DATA_END = "2019-12-31"

# degree Kelvin bounds for Earthly temperatures
EARTH_TEMP_BOUNDS = (180, 330)

# column renaming map from raw dataset to processed dataset
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

# meteorological features calculated from ERA5 data
ERA5_MET_FEATURE_NAMES = [
    "mean_skt",
    "mean_land_skt",
    "mean_sst",
    "mean_swvl1",
    "mean_swvl2",
    "mean_u850",
    "mean_u500",
    "mean_u200",
    "mean_v850",
    "mean_v500",
    "mean_v200",
    "domain_mean_u500",
    "mean_u_shear_850_500",
    "mean_v_shear_850_500",
    "mean_u_shear_850_200",
    "mean_v_shear_850_200",
    "wind_direction_850",
    "wind_angle_upslope",
    "mean_tcwv",
    "domain_mean_tcwv",
    "mean_q_850",
    "mean_q_500",
    "mean_q_200",
    "mean_cape",
    "domain_mean_cape",
    "olr_90",
    "olr_75",
    "olr_50",
    "mean_prcp_400",
]

FEATURE_COL_NAMES = (
    [
        "date_angle",
        "eat_hours",
        "storm_total_duration",
        "lon",
        "lat",
        "orography_height",
        "anor",
        "upslope_bearing",
        "slope_angle",
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
        "min_bt",
    ]
    + ERA5_MET_FEATURE_NAMES
    + [
        "dmin_bt_dt",
        "mean_bt",
        "dmean_bt_dt",
        "storm_min_bt",
        "storm_min_bt_reached",
        "mjo_phase",
        "mjo_amplitude",
    ]
)

TARGET_COL_NAMES = ["mean_prcp_400", "storm_min_bt"]

DATASET_COL_NAMES = ["storm_id", "timestamp"] + FEATURE_COL_NAMES

TARGET_EXCLUDE_COLS = {
    "mean_prcp_400": [],
    "storm_min_bt": [
        "min_bt",
        "dmin_bt_dt",
        "mean_bt",
        "dmean_bt_dt",
        "storm_min_bt_reached",
    ],
}


# ==============================================================================
#                       MODEL AND TRAINING CONFIGURATION
# ==============================================================================
RANDOM_STATE = 114

XGB_HYPERPARAMS = {
    "objective": "reg:squarederror",
    "colsample_bytree": 0.3,
    "learning_rate": 0.25,
    "max_depth": 6,
    "alpha": 10,
    "gamma": 0,
    "n_estimators": 120,
    "random_state": RANDOM_STATE,
}

# Weights & Biases configuration
WANDB_ENTITY = "uor-msc"
WANDB_PROJECT = "uor-msc-dissertation-xai-african-storms"

CV_PARAMS = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": RANDOM_STATE,
}

XGB_EARLY_STOPPING_PARAMS = {
    "rounds": 10,
    "metric_name": "rmse",
    "data_name": "validation_0",  # first validation set passed to fit function
    "maximize": False,
    "save_best": True,
}

# matching param config from https://github.com/kieranmrhunt/lps-xgboost/blob/main/testing-and-file-preparation/xgboost-bayesian-tuning.py
WANDB_SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {
        "name": "val-rmse",
        "goal": "minimize",
    },
    "parameters": {
        "gamma": {
            "distribution": "uniform",
            "min": 0,
            "max": 5,
        },
        "alpha": {
            "distribution": "uniform",
            "min": 0,
            "max": 20,
        },
        "learning_rate": {"distribution": "uniform", "min": 0, "max": 1.0},
        "max_depth": {"values": [6]},
        "n_estimators": {"values": [120]},
        "random_state": {"values": [RANDOM_STATE]},
    },
    "early_terminate": {
        "type": "hyperband",
        "max_iter": 20,
        "eta": 3,
        "s": 1,
    },
}

WANDB_DEFAULT_SWEEP_TRIALS = 15
