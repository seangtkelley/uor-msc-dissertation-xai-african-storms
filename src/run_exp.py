import argparse

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
    "--exp_name",
    type=str,
    help="Name of the experiment",
)
parser.add_argument(
    "--trials",
    type=int,
    help="Number of trials for hyperparameter sweep",
)
parser.add_argument(
    "--wandb_mode",
    type=str,
    choices=["online", "offline", "disabled"],
    default="online",
    help="Mode for Weights & Biases logging",
)
args = parser.parse_args()

# load the processed dataset
print("Loading processed dataset...")
processed_df = pd.read_csv(
    config.PROCESSED_DATASET_PATH, parse_dates=["timestamp"]
)

# get experiment config
exp_config = config.EXPERIMENT_CONFIG[args.exp_name]

# run the experiment
modelling.run_experiment(
    exp_name=args.exp_name,
    processed_df=processed_df,
    first_points_only=exp_config["first_points_only"],
    target_col=exp_config["target_col"],
    feature_cols=exp_config["feature_cols"],
    trials=args.trials,
    wandb_mode=args.wandb_mode,
)
