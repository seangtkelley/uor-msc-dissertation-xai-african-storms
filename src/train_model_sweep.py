#!/usr/bin/env python

"""train_model_sweep.py: Modified training script for W&B sweeps"""

__author__ = "Sean Kelley"
__version__ = "0.1.0"

import argparse
import uuid
from pathlib import Path
from typing import List

import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import KFold
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
    "--wandb_mode",
    type=str,
    choices=["online", "offline", "disabled"],
    default="disabled",
    help="Mode for W&B logging",
)
parser.add_argument(
    "--wandb_sweep_count",
    type=int,
    default=config.WANDB_DEFAULT_SWEEP_COUNT,
    help="Number of runs for the W&B sweep",
)
args = parser.parse_args()

if args.target_all and args.target_col_name is not None:
    parser.error("--target_all cannot be used with --target_col_name")
elif not args.target_all and args.target_col_name is None:
    parser.error(
        "--target_col_name must be specified if --target_all is not used"
    )

# set run name with current timestamp and update output model directory
run_name_base = f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
output_model_dir = Path("./models") / run_name_base
if args.output_model_dir is not None:
    output_model_dir = Path(args.output_model_dir) / run_name_base

# ensure output model path exists
output_model_dir.mkdir(parents=True, exist_ok=True)

random_state = None
if args.model_type == "xgboost":
    # model specific code goes here
    # hyperparams are controlled by W&B sweep in this script
    pass
else:
    raise ValueError(f"Unsupported model type: {args.model_type}")

# load the processed dataset
print("Loading processed dataset...")
processed_df = pd.read_csv(
    config.PROCESSED_DATASET_PATH, parse_dates=["timestamp"]
)


def train_model(target_col: str, output_model_dir: Path = output_model_dir):
    """
    Train an XGBoost model on the processed dataset for a specific target column.
    """
    # initialize Weights & Biases
    short_guid = uuid.uuid4().hex[:8]
    run_name = f"{run_name_base}_{short_guid}_{target_col}"
    wandb.init(name=run_name, mode=args.wandb_mode)

    # get random state from W&B config
    random_state = wandb.config.get("random_state", None)

    # Separate features and target variable
    feature_cols = config.FEATURE_COL_NAMES.copy()
    feature_cols.remove(target_col)
    X = processed_df[feature_cols]
    y = processed_df[target_col]

    # Cross-validation setup
    kfold = KFold(**config.CV_PARAMS, random_state=random_state)
    cv_scores = []
    cv_models = []

    # perform cross-validation
    for train_idx, val_idx in kfold.split(X, y):
        # create train and val sets
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # add callback to log metrics to W&B
        callbacks = [
            EarlyStopping(**config.XGB_EARLY_STOPPING_PARAMS),
            WandbCallback(),
        ]

        # init the model
        model = XGBRegressor(**wandb.config.as_dict(), callbacks=callbacks)

        # train the model
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        # save the best score and model
        cv_scores.append(model.best_score)
        cv_models.append(model)

    # find the lowest val error from the cross-validation
    best_cv_score = min(cv_scores)
    best_cv_model = cv_models[cv_scores.index(best_cv_score)]

    # log the best score across all cv folds to W&B for the sweep
    wandb.log({"val-rmse": best_cv_score})

    # save the model to the output directory
    model_path = output_model_dir / f"{run_name}_model.json"
    best_cv_model.save_model(str(model_path))
    print(f"Model saved to {model_path}")

    # upload the model to W&B
    wandb.save(str(model_path), base_path=args.output_model_dir)


target_cols: List[str] = (
    config.TARGET_COL_NAMES if args.target_all else [args.target_col_name]
)

# find best model for each target column
for target_col in target_cols:
    print(f"Finding best model for target column: {target_col}")

    # setup hyperparameters sweep
    sweep_id = wandb.sweep(
        config.WANDB_SWEEP_CONFIG,
        entity=config.WANDB_ENTITY,
        project=config.WANDB_PROJECT,
    )

    # run the sweep
    wandb.agent(
        sweep_id=sweep_id,
        function=lambda col=target_col,: train_model(col),
        count=args.wandb_sweep_count,
    )
