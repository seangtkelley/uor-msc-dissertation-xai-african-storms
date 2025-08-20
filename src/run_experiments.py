import pandas as pd
from dotenv import load_dotenv

import config
from utils import modelling

load_dotenv()

# load the processed dataset
print("Loading processed dataset...")
processed_df = pd.read_csv(
    config.PROCESSED_DATASET_PATH, parse_dates=["timestamp"]
)

# ==============================================================================
#  Experiment 1
#  Target: storm_min_bt
#  Features: All columns except leakage
# ==============================================================================
exp1_target_col = "storm_min_bt"
run_output_dir, run_base_name = modelling.setup_run_metadata(exp1_target_col)
modelling.wandb_sweep(
    processed_df=processed_df,
    target_col=exp1_target_col,
    feature_cols=config.FEATURE_COLS,
    run_base_name=run_base_name,
    wandb_mode="online",
)

# ==============================================================================
#  Experiment 2
#  Target: storm_min_bt
#  Features: ERA5 meteorological cols only
# ==============================================================================
exp2_target_col = "storm_min_bt"
run_output_dir, run_base_name = modelling.setup_run_metadata(exp2_target_col)
modelling.wandb_sweep(
    processed_df=processed_df,
    target_col=exp2_target_col,
    feature_cols=config.ERA5_MET_FEATURE_COLS,
    run_base_name=run_base_name,
    wandb_mode="online",
)
