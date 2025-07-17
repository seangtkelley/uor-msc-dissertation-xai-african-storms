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
import json
from pathlib import Path
from typing import List

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

import config

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
    "--hyperparameter_json",
    type=str,
    default=str(config.HYPERPARAMETER_JSON_PATH),
    help="Path to JSON file containing hyperparameters for model training",
)
parser.add_argument(
    "--train_parameters_json",
    type=str,
    default=str(config.TRAIN_PARAMETERS_JSON_PATH),
    help="Path to JSON file containing training parameters for model training",
)
parser.add_argument(
    "--output_model_dir",
    type=str,
    default=str(config.OUTPUT_MODEL_DIR),
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
    "--random_state",
    type=int,
    help="Random state for train/test split",
)
args = parser.parse_args()

if args.target_all and args.target_col_name is not None:
    parser.error("--target_all cannot be used with --target_col_name")

# set run name with current timestamp and update output model directory
run_name = f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
output_model_dir = None
if args.output_model_dir is None:
    output_model_dir = Path("./models") / run_name
else:
    output_model_dir = Path(args.output_model_dir) / run_name

# ensure output model path exists
output_model_dir.mkdir(parents=True, exist_ok=True)

# load hyperparameters from JSON file
with open(args.hyperparameter_json, "r") as f:
    hyperparams = json.load(f)

# load training parameters from JSON file
with open(args.train_parameters_json, "r") as f:
    train_params = json.load(f)

# load the processed dataset
print("Loading processed dataset...")
processed_df = pd.read_csv(
    config.PROCESSED_DATASET_PATH, parse_dates=["timestamp"]
)

target_cols: List[str] = (
    config.TARGET_COL_NAMES if args.target_all else [args.target_col_name]
)

# train model for each target column
for target_col in target_cols:
    print(f"Training model for target column: {target_col}")

    # separate features and target variable
    feature_cols = config.FEATURE_COL_NAMES
    feature_cols.remove(target_col)
    X = processed_df[feature_cols]
    y = processed_df[target_col]

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    # create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # train/val split
    evals = None
    if args.val_size is not None:
        # adjust val size for train size
        val_size = args.val_size / (1 - args.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=args.random_state
        )

        # create DMatrix for validation set
        dval = xgb.DMatrix(X_val, label=y_val)

        # add to evals for validation
        evals = [(dval, "val")]

    if (
        train_params.get("early_stopping_rounds", None) is not None
        and evals is None
    ):
        raise ValueError(
            "Early stopping requires a validation set. Please provide a validation set."
        )

    # train the model
    model = xgb.train(hyperparams, dtrain, evals=evals, **train_params)

    # save the model
    model_path = output_model_dir / f"{target_col}_model.xgb"
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
