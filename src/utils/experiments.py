#!/usr/bin/env python

"""experiments.py: Experiment definitions for model training."""

__author__ = "Sean Kelley"
__copyright__ = "Copyright 2025, University of Reading"
__credits__ = ["Sean Kelley"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Sean Kelley"
__email__ = "s.g.t.kelley@student.reading.ac.uk"
__status__ = "Development"

import pandas as pd

import config
from utils import modelling

# ==============================================================================
#  Experiment Group: Predict Storm Aggregate Features from First Point
# ==============================================================================


# =======================================
#  Experiment 1: Max Intensity
#  Target: storm_min_bt
#  Features: All columns except leakage
# =======================================
def run_exp1(processed_df: pd.DataFrame):
    target_col = "storm_min_bt"
    run_base_name = modelling.setup_run_metadata(target_col)

    # get only first points of each storm
    first_points_df = processed_df.groupby("storm_id").first()

    modelling.wandb_sweep(
        processed_df=first_points_df,
        target_col=target_col,
        feature_cols=config.ALL_FEATURE_COLS,
        run_base_name=run_base_name,
        wandb_mode="online",
    )


# =======================================
#  Experiment 2: Max Intensity
#  Target: storm_min_bt
#  Features: ERA5 meteorological cols only
# =======================================
def run_exp2(processed_df: pd.DataFrame):
    target_col = "storm_min_bt"
    run_base_name = modelling.setup_run_metadata(target_col)

    # get only first points of each storm
    first_points_df = processed_df.groupby("storm_id").first()

    modelling.wandb_sweep(
        processed_df=first_points_df,
        target_col=target_col,
        feature_cols=config.ERA5_MET_FEATURE_COLS,
        run_base_name=run_base_name,
        wandb_mode="online",
    )


# =======================================
#  Experiment 3: Propagation Direction
#  Target: storm_bearing
#  Features: All columns except leakage
# =======================================
def run_exp3(processed_df: pd.DataFrame):
    target_col = "storm_bearing"
    run_base_name = modelling.setup_run_metadata(target_col)

    # get only first points of each storm
    first_points_df = processed_df.groupby("storm_id").first()

    modelling.wandb_sweep(
        processed_df=first_points_df,
        target_col=target_col,
        feature_cols=config.ALL_FEATURE_COLS,
        run_base_name=run_base_name,
        wandb_mode="online",
    )


# =======================================
#  Experiment 4: Propagation Direction
#  Target: storm_bearing
#  Features: ERA5 meteorological cols only
# =======================================
def run_exp4(processed_df: pd.DataFrame):
    target_col = "storm_bearing"
    run_base_name = modelling.setup_run_metadata(target_col)

    # get only first points of each storm
    first_points_df = processed_df.groupby("storm_id").first()

    # start wandb sweep
    modelling.wandb_sweep(
        processed_df=first_points_df,
        target_col=target_col,
        feature_cols=config.ERA5_MET_FEATURE_COLS,
        run_base_name=run_base_name,
        wandb_mode="online",
    )


# ==============================================================================
#  Experiment Group: Predict Next Point Features
# ==============================================================================


# =======================================
#  Experiment 5: Intensification
#  Target: dmin_bt_dt
#  Features: All columns except leakage
# =======================================
def run_exp5(processed_df: pd.DataFrame):
    target_col = "dmin_bt_dt"
    run_base_name = modelling.setup_run_metadata(target_col)

    # start wandb sweep
    modelling.wandb_sweep(
        processed_df=processed_df,
        target_col=target_col,
        feature_cols=config.ALL_FEATURE_COLS,
        run_base_name=run_base_name,
        wandb_mode="online",
    )


# =======================================
#  Experiment 6: Intensification
#  Target: dmin_bt_dt
#  Features: ERA5 meteorological cols only
# =======================================
def run_exp6(processed_df: pd.DataFrame):
    target_col = "dmin_bt_dt"
    run_base_name = modelling.setup_run_metadata(target_col)

    # start wandb sweep
    modelling.wandb_sweep(
        processed_df=processed_df,
        target_col=target_col,
        feature_cols=config.ERA5_MET_FEATURE_COLS,
        run_base_name=run_base_name,
        wandb_mode="online",
    )


# =======================================
#  Experiment 7: Propagation Direction
#  Target: bearing_to_next
#  Features: All columns except leakage
# =======================================
def run_exp7(processed_df: pd.DataFrame):
    target_col = "bearing_to_next"
    run_base_name = modelling.setup_run_metadata(target_col)

    # start wandb sweep
    modelling.wandb_sweep(
        processed_df=processed_df,
        target_col=target_col,
        feature_cols=config.ALL_FEATURE_COLS,
        run_base_name=run_base_name,
        wandb_mode="online",
    )


# =======================================
#  Experiment 8: Propagation Direction
#  Target: bearing_to_next
#  Features: ERA5 meteorological cols only
# =======================================
def run_exp8(processed_df: pd.DataFrame):
    target_col = "bearing_to_next"
    run_base_name = modelling.setup_run_metadata(target_col)

    # start wandb sweep
    modelling.wandb_sweep(
        processed_df=processed_df,
        target_col=target_col,
        feature_cols=config.ERA5_MET_FEATURE_COLS,
        run_base_name=run_base_name,
        wandb_mode="online",
    )


# =======================================
#  Experiment 9: Propagation Distance
#  Target: distance_to_next
#  Features: All columns except leakage
# =======================================
def run_exp9(processed_df: pd.DataFrame):
    target_col = "distance_to_next"
    run_base_name = modelling.setup_run_metadata(target_col)

    # start wandb sweep
    modelling.wandb_sweep(
        processed_df=processed_df,
        target_col=target_col,
        feature_cols=config.ERA5_MET_FEATURE_COLS,
        run_base_name=run_base_name,
        wandb_mode="online",
    )


# =======================================
#  Experiment 10: Propagation Distance
#  Target: distance_to_next
#  Features: ERA5 meteorological cols only
# =======================================
def run_exp10(processed_df: pd.DataFrame):
    target_col = "distance_to_next"
    run_base_name = modelling.setup_run_metadata(target_col)

    # start wandb sweep
    modelling.wandb_sweep(
        processed_df=processed_df,
        target_col=target_col,
        feature_cols=config.ERA5_MET_FEATURE_COLS,
        run_base_name=run_base_name,
        wandb_mode="online",
    )


# =======================================
#  Experiment 11: Precipitation
#  Target: mean_prcp_400
#  Features: All columns except leakage
# =======================================
def run_exp11(processed_df: pd.DataFrame):
    target_col = "mean_prcp_400"
    run_base_name = modelling.setup_run_metadata(target_col)

    # start wandb sweep
    modelling.wandb_sweep(
        processed_df=processed_df,
        target_col=target_col,
        feature_cols=config.ALL_FEATURE_COLS,
        run_base_name=run_base_name,
        wandb_mode="online",
    )


# =======================================
#  Experiment 12: Precipitation
#  Target: mean_prcp_400
#  Features: ERA5 meteorological cols only
# =======================================
def run_exp12(processed_df: pd.DataFrame):
    target_col = "mean_prcp_400"
    run_base_name = modelling.setup_run_metadata(target_col)

    # start wandb sweep
    modelling.wandb_sweep(
        processed_df=processed_df,
        target_col=target_col,
        feature_cols=config.ERA5_MET_FEATURE_COLS,
        run_base_name=run_base_name,
        wandb_mode="online",
    )


all_experiments = [
    run_exp1,
    run_exp2,
    run_exp3,
    run_exp4,
    run_exp5,
    run_exp6,
    run_exp7,
    run_exp8,
    run_exp9,
    run_exp10,
    run_exp11,
]
