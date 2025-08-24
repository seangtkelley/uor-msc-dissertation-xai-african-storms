#!/usr/bin/env python

"""train_model.py: Script to train model on processed storm dataset given specified parameters"""

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
    help="Mode for Weights & Biases logging",
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
    config.TARGET_COLS
    if args.target_cols is None
    else args.target_cols.split(",")
)

# train model for each target column
for target_col in target_cols:
    print(f"Training model for target column: {target_col}")

    if target_col not in config.TARGET_COLS:
        raise ValueError(f"Invalid target column: {target_col}")

    run_output_dir, run_base_name = modelling.setup_run_metadata(target_col)

    # init Weights & Biases
    wandb_run = modelling.init_wandb(
        run_name_base=run_base_name,
        wandb_mode=args.wandb_mode,
    )

    # separate features and target variable
    X, y = modelling.get_features_and_target(processed_df, target_col)

    # train the model
    modelling.train_model(X, y, wandb_run=wandb_run)

    # finish the W&B run
    wandb_run.finish()
