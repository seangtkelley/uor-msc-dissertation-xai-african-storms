import argparse

import pandas as pd
from dotenv import load_dotenv

import config
from utils import experiments

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
    "--wandb_mode",
    type=str,
    choices=["online", "offline", "disabled"],
    default="disabled",
    help="Mode for Weights & Biases logging",
)
args = parser.parse_args()

# load the processed dataset
print("Loading processed dataset...")
processed_df = pd.read_csv(
    config.PROCESSED_DATASET_PATH, parse_dates=["timestamp"]
)

# run the experiment
experiments.all_experiments[args.exp_name](
    processed_df=processed_df, wandb_mode=args.wandb_mode
)
