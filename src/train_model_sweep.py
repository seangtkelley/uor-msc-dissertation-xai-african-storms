#!/usr/bin/env python

"""train_model_sweep.py: Modified training script for W&B sweeps"""

__author__ = "Sean Kelley"
__version__ = "0.1.0"

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
    "--target_col_name",
    type=str,
    choices=config.TARGET_COL_NAMES,
    help="Name of the target column in the dataset",
)
parser.add_argument(
    "--target_all",
    action="store_true",
    help="Train model on all target columns",
    default=False,
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

if args.target_all and args.target_col_name is not None:
    parser.error("--target_all cannot be used with --target_col_name")
elif not args.target_all and args.target_col_name is None:
    parser.error(
        "--target_col_name must be specified if --target_all is not used"
    )

# load the processed dataset
print("Loading processed dataset...")
processed_df = pd.read_csv(
    config.PROCESSED_DATASET_PATH, parse_dates=["timestamp"]
)

# define target columns
target_cols: List[str] = (
    config.TARGET_COL_NAMES if args.target_all else [args.target_col_name]
)

# find best model for each target column
for target_col in target_cols:
    print(f"Finding best model for target column: {target_col}")

    run_output_dir, run_base_name = modelling.setup_run_metadata(target_col)

    modelling.wandb_sweep(
        processed_df,
        target_col,
        feature_cols=config.FEATURE_COL_NAMES,
        trials=args.wandb_sweep_count,
        run_base_name=run_base_name,
        wandb_mode=args.wandb_mode,
    )
