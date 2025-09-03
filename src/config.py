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

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# ==============================================================================
#                           PATH CONFIGURATION
# ==============================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = REPO_ROOT / "figures" / "generated"
EXPLORATION_FIGURES_DIR = REPO_ROOT / "figures" / "generated" / "exploration"
EXPERIMENT_FIGURES_DIR = REPO_ROOT / "figures" / "generated" / "experiments"
SRC_DIR = REPO_ROOT / "src"
WANDB_LOG_DIR = REPO_ROOT / "wandb"
SHAP_VALUES_DIR = REPO_ROOT / "shap_values"

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
# scope cmap terrain to start at green
TERRAIN_CMAP = ListedColormap(
    plt.get_cmap("terrain")(np.linspace(0.25, 1, plt.get_cmap("terrain").N))
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
ERA5_MET_FEATURE_COLS = [
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

ALL_FEATURE_COLS = (
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
        "bearing_to_next",
        "distance_from_prev",
        "distance_to_next",
        "distance_traversed",
        "storm_bearing",
        "storm_distance_traversed",
        "storm_straight_line_distance",
    ]
    + ERA5_MET_FEATURE_COLS
    + [
        "min_bt",
        "dmin_bt_dt",
        "mean_bt",
        "dmean_bt_dt",
        "storm_min_bt",
        "storm_min_bt_reached",
        "mjo_phase",
        "mjo_amplitude",
    ]
)

TARGET_COLS = [
    "mean_prcp_400",
    "storm_min_bt",
    "storm_bearing",
    "dmin_bt_dt",
    "bearing_to_next",
    "distance_to_next",
]

DATASET_COLS = ["storm_id", "storm_obs_idx", "timestamp"] + ALL_FEATURE_COLS

ALL_TARGET_EXCLUDE_COLS = [
    # storm aggregate features
    "storm_total_duration",
    "storm_total_land_time",
    "storm_max_area",
    "storm_bearing",
    "storm_distance_traversed",
    "storm_straight_line_distance",
    "storm_min_bt",
    "storm_min_bt_reached",
    # future info features
    "bearing_to_next",
    "distance_to_next",
    # features removed due to high correlation with others
    "mean_swvl2",
    "olr_75",  # need to justify these two
    "olr_50",
]

TARGET_EXCLUDE_COLS_MAP = {
    "mean_prcp_400": [],
    "storm_min_bt": [
        "min_bt",
        "dmin_bt_dt",
        "mean_bt",
        "dmean_bt_dt",
    ],
    "storm_bearing": ["zonal_speed", "meridional_speed", "bearing_from_prev"],
    "dmin_bt_dt": [
        "min_bt",
        "mean_bt",
        "dmean_bt_dt",
    ],
    "bearing_to_next": ["zonal_speed", "meridional_speed", "bearing_from_prev"],
    "distance_to_next": [
        "zonal_speed",
        "meridional_speed",
        "distance_from_prev",
    ],
}


# ==============================================================================
#                       MODEL AND TRAINING CONFIGURATION
# ==============================================================================
RANDOM_STATE = 114

VAL_SIZE, TEST_SIZE = 0.2, 0.2

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
        "reg_alpha": {
            "distribution": "uniform",
            "min": 0,
            "max": 5,
        },
        "reg_lambda": {
            "distribution": "uniform",
            "min": 0,
            "max": 5,
        },
        "colsample_bytree": {
            "distribution": "uniform",
            "min": 0,
            "max": 1.0,
        },
        "learning_rate": {"distribution": "uniform", "min": 0, "max": 1.0},
        "max_depth": {"values": [3, 6, 9, 12]},
        "n_estimators": {"values": [60, 120, 180]},
        "random_state": {"values": [RANDOM_STATE]},
    },
    "early_terminate": {
        "type": "hyperband",
        "max_iter": 20,
        "eta": 3,
        "s": 1,
    },
}

WANDB_MAX_SWEEP_TRIALS = 20


# ==============================================================================
#                       EXPERIMENT CONFIGURATION
# ==============================================================================
EXPERIMENT_CONFIG = {
    "storm_max_intensity_all": {
        "first_points_only": False,
        "target_col": "storm_min_bt",
        "feature_cols": "all",
        "target_units": "K",
    },
    "storm_max_intensity_all_first_points": {
        "first_points_only": True,
        "target_col": "storm_min_bt",
        "feature_cols": "all",
        "target_units": "K",
    },
    "storm_max_intensity_era5": {
        "first_points_only": False,
        "target_col": "storm_min_bt",
        "feature_cols": "era5",
        "target_units": "K",
    },
    "storm_max_intensity_era5_first_points": {
        "first_points_only": True,
        "target_col": "storm_min_bt",
        "feature_cols": "era5",
        "target_units": "K",
    },
    "storm_direction_all": {
        "first_points_only": False,
        "target_col": "storm_bearing",
        "feature_cols": "all",
        "target_units": "degrees",
    },
    "storm_direction_all_first_points": {
        "first_points_only": True,
        "target_col": "storm_bearing",
        "feature_cols": "all",
        "target_units": "degrees",
    },
    "storm_direction_era5": {
        "first_points_only": False,
        "target_col": "storm_bearing",
        "feature_cols": "era5",
        "target_units": "degrees",
    },
    "storm_direction_era5_first_points": {
        "first_points_only": True,
        "target_col": "storm_bearing",
        "feature_cols": "era5",
        "target_units": "degrees",
    },
    "obs_intensification_all": {
        "first_points_only": False,
        "target_col": "dmin_bt_dt",
        "feature_cols": "all",
        "target_units": "K/h",
    },
    "obs_intensification_era5": {
        "first_points_only": False,
        "target_col": "dmin_bt_dt",
        "feature_cols": "era5",
        "target_units": "K/h",
    },
    "obs_next_direction_all": {
        "first_points_only": False,
        "target_col": "bearing_to_next",
        "feature_cols": "all",
        "target_units": "degrees",
    },
    "obs_next_direction_era5": {
        "first_points_only": False,
        "target_col": "bearing_to_next",
        "feature_cols": "era5",
        "target_units": "degrees",
    },
    "obs_next_distance_all": {
        "first_points_only": False,
        "target_col": "distance_to_next",
        "feature_cols": "all",
        "target_units": "km",
    },
    "obs_next_distance_era5": {
        "first_points_only": False,
        "target_col": "distance_to_next",
        "feature_cols": "era5",
        "target_units": "km",
    },
    "obs_precipitation_all": {
        "first_points_only": False,
        "target_col": "mean_prcp_400",
        "feature_cols": "all",
        "target_units": "mm/h",
    },
    "obs_precipitation_era5": {
        "first_points_only": False,
        "target_col": "mean_prcp_400",
        "feature_cols": "era5",
        "target_units": "mm/h",
    },
}

EXPERIMENT_GROUPS = {
    "storm_max_intensity": [
        "storm_max_intensity_all",
        "storm_max_intensity_era5",
    ],
    "storm_max_intensity_first_points": [
        "storm_max_intensity_all_first_points",
        "storm_max_intensity_era5_first_points",
    ],
    "storm_direction": [
        "storm_direction_all",
        "storm_direction_era5",
    ],
    "storm_direction_first_points": [
        "storm_direction_all_first_points",
        "storm_direction_era5_first_points",
    ],
    "obs_intensification": [
        "obs_intensification_all",
        "obs_intensification_era5",
    ],
    "obs_next_direction": [
        "obs_next_direction_all",
        "obs_next_direction_era5",
    ],
    "obs_next_distance": [
        "obs_next_distance_all",
        "obs_next_distance_era5",
    ],
    "obs_precipitation": [
        "obs_precipitation_all",
        "obs_precipitation_era5",
    ],
}


# ==============================================================================
#                       EXPLAINABILITY CONFIGURATION
# ==============================================================================
R_SQUARED_THRESHOLD = 0.5
CORR_HEATMAP_CMAP = "coolwarm"
SHAP_MAP_CMAP = "seismic"
SHAP_VALUES_DESCRIPTION = {
    "storm_max_intensity": {
        "positive": "Less intense",
        "negative": "More intense",
    },
    "storm_direction": {"positive": "Northward", "negative": "Southward"},
    "obs_intensification": {
        "positive": "Intensifies slower",
        "negative": "Intensifies faster",
    },
    "obs_next_direction": {"positive": "Northward", "negative": "Southward"},
    "obs_next_distance": {"positive": "Farther", "negative": "Nearer"},
    "obs_precipitation": {
        "positive": "More rainfall",
        "negative": "Less rainfall",
    },
}
