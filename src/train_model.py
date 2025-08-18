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
from pathlib import Path
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
    "--model_type",
    type=str,
    choices=["xgboost"],
    default="xgboost",
    help="Type of model to train",
)
parser.add_argument(
    "--output_model_dir",
    type=str,
    default=str(config.MODEL_OUTPUT_DIR),
    help="Path to save the trained model",
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
    "--test_size",
    type=float,
    default=0.2,
    help="Proportion of the dataset to include in the test split",
)
parser.add_argument(
    "--val_size",
    type=float,
    default=0.2,
    help="Proportion of the training set to include in the validation split",
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

# set hyperparameters for the model
random_state = None
if args.model_type == "xgboost":
    # load hyperparameters from config
    hyperparams = config.XGB_HYPERPARAMS.copy()
    random_state = hyperparams.get("random_state", None)
else:
    raise ValueError(f"Unsupported model type: {args.model_type}")

# load the processed dataset
print("Loading processed dataset...")
processed_df = pd.read_csv(
    config.PROCESSED_DATASET_PATH, parse_dates=["timestamp"]
)

# define target columns
target_cols: List[str] = (
    config.TARGET_COL_NAMES if args.target_all else [args.target_col_name]
)

# train model for each target column
for target_col in target_cols:
    print(f"Training model for target column: {target_col}")

    # set run name with current timestamp and update output model directory
    run_timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_model_dir = Path("./models")
    if args.output_model_dir is not None:
        output_model_dir = Path(args.output_model_dir)

    # create run output dir based on target column and timestamp
    run_output_dir = output_model_dir / target_col / run_timestamp_str

    # ensure output path exists
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # create run base name
    run_base_name = f"{target_col}_{run_timestamp_str}"

    # init Weights & Biases
    wandb_run = modelling.init_wandb(
        run_name_base=run_base_name,
        wandb_mode=args.wandb_mode,
    )

    # separate features and target variable
    X, y = modelling.separate_features_and_target(processed_df, target_col)

    # train the model
    modelling.train_model(
        X,
        y,
        args.val_size,
        args.test_size,
        hyperparams,
        random_state=random_state,
        wandb_run=wandb_run,
        model_output_dir=output_model_dir,
    )
