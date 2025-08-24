#!/usr/bin/env python

"""train_model_sweep.py: Modified training script for W&B sweeps"""

__author__ = "Sean Kelley"
__copyright__ = "Copyright 2025, University of Reading"
__credits__ = ["Sean Kelley"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Sean Kelley"
__email__ = "s.g.t.kelley@student.reading.ac.uk"
__status__ = "Development"

import argparse
from typing import List

import pandas as pd
from dotenv import load_dotenv

import config
from utils import modelling

load_dotenv()

# parse cli arguments
parser = argparse.ArgumentParser(
    description="Train model on processed storm dataset given specified parameters"
)
parser.add_argument(
    "--target_cols",
    type=str,
    help="Comma-separated list of the target columns in the dataset",
)
parser.add_argument(
    "--wandb_mode",
    type=str,
    choices=["online", "offline", "disabled"],
    default="disabled",
    help="Mode for W&B logging",
)
parser.add_argument(
    "--wandb_sweep_count",
    type=int,
    default=config.WANDB_DEFAULT_SWEEP_TRIALS,
    help="Number of runs for the W&B sweep",
)
args = parser.parse_args()

# load the processed dataset
print("Loading processed dataset...")
processed_df = pd.read_csv(
    config.PROCESSED_DATASET_PATH, parse_dates=["timestamp"]
)

# define target columns
target_cols: List[str] = (
    config.TARGET_COLS
    if args.target_cols is None
    else args.target_cols.split(",")
)

# find best model for each target column
for target_col in target_cols:
    print(f"Finding best model for target column: {target_col}")

    if target_col not in config.TARGET_COLS:
        raise ValueError(f"Invalid target column: {target_col}")

    run_output_dir, run_base_name = modelling.setup_run_metadata(target_col)

    modelling.wandb_sweep(
        processed_df,
        target_col,
        feature_cols=config.ALL_FEATURE_COLS,
        trials=args.wandb_sweep_count,
        run_base_name=run_base_name,
        wandb_mode=args.wandb_mode,
    )
