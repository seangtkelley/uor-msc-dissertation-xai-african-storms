#!/usr/bin/env python

"""eval_exps.py: Script to evaluate experiments"""

__author__ = "Sean Kelley"
__copyright__ = "Copyright 2025, University of Reading"
__credits__ = ["Sean Kelley"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Sean Kelley"
__email__ = "s.g.t.kelley@student.reading.ac.uk"
__status__ = "Development"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

import config
from utils import modelling, plotting

load_dotenv()

# mpl.rcParams["agg.path.chunksize"] = 10000

sns.set_theme(style="darkgrid")

# load the processed dataset
print("Loading processed dataset...")
processed_df = pd.read_csv(
    config.PROCESSED_DATASET_PATH, parse_dates=["timestamp"]
)


# evaluate each experiment
for exp_name, exp_config in config.EXPERIMENT_CONFIG.items():
    print(f"Evaluating experiment: {exp_name}")

    exp_config = config.EXPERIMENT_CONFIG[exp_name]

    best_run = modelling.get_best_run_from_exp(exp_name)

    best_model = modelling.get_model_from_run(best_run)

    test_idx = best_run.config.get("test_dataset_index")

    test_df = processed_df.iloc[test_idx]

    if exp_config["feature_cols"] == "all":
        feature_cols = config.ALL_FEATURE_COLS
    elif exp_config["feature_cols"] == "era5":
        feature_cols = config.ERA5_MET_FEATURE_COLS
    else:
        raise ValueError(
            f"Unknown feature column set: {exp_config['feature_cols']}"
        )

    X_test, y_test = modelling.get_features_and_target(
        test_df, exp_config["target_col"], feature_cols
    )

    y_pred = best_model.predict(X_test)

    test_rmse = root_mean_squared_error(y_test, y_pred)

    # plot predictions vs actual using matplotlib
    plt.figure(figsize=(12, 12))
    plt.scatter(y_pred, y_test, s=10)

    # Regression line and R value using sklearn
    lr = LinearRegression()
    lr.fit(y_pred.reshape(-1, 1), y_test.to_numpy().reshape(-1, 1))
    reg_line = lr.predict(np.unique(y_pred).reshape(-1, 1))
    r_value = lr.score(y_pred.reshape(-1, 1), y_test.to_numpy().reshape(-1, 1))

    plt.plot(
        np.unique(y_pred),
        reg_line,
        label=f"Regression line (R²={r_value:.2f})",
        color="black",
        linestyle="--",
    )

    plt.title(f"Predictions vs Actual for {exp_name}")
    plt.xlabel("Predicted Value (K)")
    plt.ylabel("Actual Value (K)")
    plt.legend()

    plotting.save_plot(f"predictions_vs_actual_{exp_name}.png")

    break
