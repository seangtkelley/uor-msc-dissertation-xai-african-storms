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
import shap
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

    fig = plt.figure(figsize=(16, 6 * len(exp_names)))

    # evaluate each experiment in the group
    for i, exp_name in enumerate(exp_names):
        print(f"Evaluating experiment: {exp_name}")

        # get exp config
        exp_config = config.EXPERIMENT_CONFIG[exp_name]

        # get best run from all sweeps
        best_run = modelling.get_best_run_from_exp(exp_name)

        # get the model from best run
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
        ax_pred = fig.add_subplot(2, len(exp_names), i + 1)
        ax_pred.scatter(y_pred, y_test, s=10)

        # Regression line and R value using sklearn
        lr = LinearRegression()
        lr.fit(y_pred.reshape(-1, 1), y_test.to_numpy().reshape(-1, 1))
        reg_line = lr.predict(np.unique(y_pred).reshape(-1, 1))
        r_squared = lr.score(
            y_pred.reshape(-1, 1), y_test.to_numpy().reshape(-1, 1)
        )

        # plot regression line
        ax_pred.plot(
            np.unique(y_pred),
            reg_line,
            label=f"Regression line (R²={r_squared:.2f})",
            color="black",
            linestyle="--",
        )
        ax_pred.set_title(f"Model Verification for {exp_name}")
        ax_pred.set_xlabel(f"Predicted Value ({exp_config['target_units']})")
        ax_pred.set_ylabel(f"Actual Value ({exp_config['target_units']})")
        ax_pred.legend()

        # sample X_test for faster shap value calc
        X_test_sample = X_test.sample(
            frac=0.2, random_state=config.RANDOM_STATE
        )

        # cast bool to int
        X_test_sample = X_test_sample.astype(
            {
                col: int
                for col in X_test_sample.select_dtypes(include="bool").columns
            }
        )

        # get shap values for test set
        explainer = shap.TreeExplainer(best_model, X_test_sample)
        explanation = explainer(X_test_sample)

        # plot SHAP summary plot
        ax_shap = fig.add_subplot(2, len(exp_names), len(exp_names) + i + 1)
        shap.plots.beeswarm(
            explanation,
            show=False,
            plot_size=None,
            ax=ax_shap,
            group_remaining_features=False,
            max_display=12,
        )
        ax_shap.set_title(f"SHAP Beeswarm Plot for {exp_name}")
        ax_shap.set_xlabel(f"SHAP value ({exp_config['target_units']})")
        ax_shap.tick_params(axis="y", labelsize=10)

    plotting.save_plot(f"{exp_group_name}.png", config.EXPERIMENT_FIGURES_DIR)

    break
