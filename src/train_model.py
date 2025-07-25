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
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from wandb.integration.xgboost import WandbCallback
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping

import config
import wandb

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

# set run name with current timestamp and update output model directory
run_name = f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
output_model_dir = None
if args.output_model_dir is None:
    output_model_dir = Path("./models") / run_name
else:
    output_model_dir = Path(args.output_model_dir) / run_name

# ensure output model path exists
output_model_dir.mkdir(parents=True, exist_ok=True)

# set hyperparameters for the model
if args.model_type == "xgboost":
    hyperparams = config.XGB_HYPERPARAMS.copy()
else:
    raise ValueError(f"Unsupported model type: {args.model_type}")

# initialize Weights & Biases
wandb.init(
    entity=config.WANDB_ENTITY,
    project=config.WANDB_PROJECT,
    name=run_name,
    mode=args.wandb_mode,
)

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
    feature_cols = config.FEATURE_COL_NAMES.copy()
    feature_cols.remove(target_col)
    X = processed_df[feature_cols]
    y = processed_df[target_col]

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    # train/val split
    if args.val_size is None or args.val_size <= 0 or args.val_size >= 1:
        raise ValueError(
            "val_size must be between 0 and 1 (exclusive) for early stopping"
        )

    # adjust val size for train size
    # source: https://datascience.stackexchange.com/a/15136
    val_size = args.val_size / (1 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=args.random_state
    )

    # add callback to log metrics to Weights & Biases
    callbacks = [
        EarlyStopping(**config.XGB_EARLY_STOPPING_PARAMS),
        WandbCallback(log_model=True),
    ]

    # init the model
    model = XGBRegressor(**hyperparams, callbacks=callbacks)

    # train the model
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    # evaluate the model on the test set
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"test-rmse: {rmse}")

    # log evaluation metric to Weights & Biases
    wandb.log({"test-rmse": rmse})

    # save the model
    model_path = output_model_dir / f"{target_col}_model.json"
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

    # finish the Weights & Biases run
    wandb.finish()
