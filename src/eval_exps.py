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

sns.set_theme(style="darkgrid")

# load the processed dataset
print("Loading processed dataset...")
processed_df = pd.read_csv(
    config.PROCESSED_DATASET_PATH, parse_dates=["timestamp"]
)

for exp_group_name, exp_names in config.EXPERIMENT_GROUPS.items():

    fig, axs = plt.subplots(1, len(exp_names), figsize=(12, 6 * len(exp_names)))

    # evaluate each experiment in the group
    for i, exp_name in enumerate(exp_names):
        print(f"Evaluating experiment: {exp_name}")

        # get exp config
        exp_config = config.EXPERIMENT_CONFIG[exp_name]

        # get best run from all sweeps
        best_run = modelling.get_best_run_from_exp(exp_name)

        # get best model from best run
        best_model = modelling.get_model_from_run(best_run)

        # get test dataset index from best run config
        test_idx = best_run.config.get("test_dataset_index")

        # select test dataset
        test_df = processed_df.iloc[test_idx]

        # use first points only for all storm aggregate exps for fair comparison
        if exp_name.startswith("storm_"):
            test_df = test_df.groupby("storm_id").first()

        # determine feature columns based on experiment config
        if exp_config["feature_cols"] == "all":
            feature_cols = config.ALL_FEATURE_COLS
        elif exp_config["feature_cols"] == "era5":
            feature_cols = config.ERA5_MET_FEATURE_COLS
        else:
            raise ValueError(
                f"Unknown feature column set: {exp_config['feature_cols']}"
            )

        # separate features and target
        X_test, y_test = modelling.get_features_and_target(
            test_df, exp_config["target_col"], feature_cols
        )

        # make predictions on test set
        y_pred = best_model.predict(X_test)

        # calculate RMSE
        test_rmse = root_mean_squared_error(y_test, y_pred)

        # calculate standard deviation of test target
        test_std = np.std(y_test)

        # plot predictions vs actual using matplotlib
        axs[i].scatter(y_pred, y_test, s=10)

        # Regression line and R value using sklearn
        lr = LinearRegression()
        lr.fit(y_pred.reshape(-1, 1), y_test.to_numpy().reshape(-1, 1))
        reg_line = lr.predict(np.unique(y_pred).reshape(-1, 1))
        r_squared = lr.score(
            y_pred.reshape(-1, 1), y_test.to_numpy().reshape(-1, 1)
        )

        # plot regression line
        axs[i].plot(
            np.unique(y_pred),
            reg_line,
            label=f"Regression line (R²={r_squared:.2f})",
            color="black",
            linestyle="--",
        )

        axs[i].set_title(f"Model Verification for {exp_name}")
        axs[i].set_xlabel("Predicted Value (K)")
        axs[i].set_ylabel("Actual Value (K)")
        axs[i].legend()

    plotting.save_plot(f"{exp_group_name}.png", config.EXPERIMENT_FIGURES_DIR)

    break
