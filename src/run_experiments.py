import multiprocessing

import pandas as pd
from dotenv import load_dotenv

import config
from utils import experiments

load_dotenv()

# load the processed dataset
print("Loading processed dataset...")
processed_df = pd.read_csv(
    config.PROCESSED_DATASET_PATH, parse_dates=["timestamp"]
)

# run all experiments in parallel
num_cores = max(1, multiprocessing.cpu_count() // 2)
with multiprocessing.Pool(num_cores) as pool:
    pool.starmap(
        lambda func, df: func(df),
        [(func, processed_df) for func in experiments.all_experiments],
    )
