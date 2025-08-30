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


import argparse
import pickle

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from dotenv import load_dotenv
from matplotlib.colors import TwoSlopeNorm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

import config
from utils import modelling, plotting

load_dotenv()

sns.set_theme(style="darkgrid")

# parse cli arguments
parser = argparse.ArgumentParser(description="Evaluate experiment groups")
parser.add_argument(
    "--exp_group_names",
    type=str,
    help="Experiments to evaluate, separated by commas. If not specified, all will be evaluated.",
)
parser.add_argument(
    "--shap_sample",
    type=float,
    help="Proportion of test samples to use for SHAP value calculation",
)
parser.add_argument(
    "--save_shap",
    action="store_true",
    help="Save SHAP values to file",
)
parser.add_argument(
    "--load_shap",
    action="store_true",
    help="Load SHAP values from file",
)
args = parser.parse_args()

# load the processed dataset
print("Loading processed dataset...")
processed_df = pd.read_csv(
    config.PROCESSED_DATASET_PATH, parse_dates=["timestamp"]
)

# select experiment groups to evaluate
if args.exp_group_names is not None:
    exp_group_names = args.exp_group_names.split(",")
    exp_groups = {
        name: config.EXPERIMENT_GROUPS[name] for name in exp_group_names
    }
else:
    exp_groups = config.EXPERIMENT_GROUPS

print(f"Evaluating {', '.join(exp_groups.keys())}...")
for exp_group_name, exp_names in exp_groups.items():

    # init exp group fig directory
    exp_group_fig_dir = config.EXPERIMENT_FIGURES_DIR / exp_group_name
    exp_group_fig_dir.mkdir(parents=True, exist_ok=True)

    # init dirs for geographic and temporal correlation plots
    exp_group_geo_corr_fig_dir = (
        config.EXPERIMENT_FIGURES_DIR / exp_group_name / "geographic_corr"
    )
    exp_group_geo_corr_fig_dir.mkdir(parents=True, exist_ok=True)
    exp_group_temp_corr_fig_dir = (
        config.EXPERIMENT_FIGURES_DIR / exp_group_name / "temporal_corr"
    )
    exp_group_temp_corr_fig_dir.mkdir(parents=True, exist_ok=True)

    # init exp group summary fig
    exp_group_sum_fig = plt.figure(figsize=(16, 6 * len(exp_names)))

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
            # using head(1) instead of first() to preserve original index
            test_df = test_df.groupby("storm_id").head(1)

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

        # print RMSE and standard deviation
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test target standard deviation: {test_std:.4f}")

        # plot predictions vs actual using matplotlib
        ax_pred = exp_group_sum_fig.add_subplot(2, len(exp_names), i + 1)
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

        # by default, use entire test set
        X_test_sample = X_test
        if args.load_shap:
            # load shap values from pickled file
            shap_values_path = (
                config.SHAP_VALUES_DIR / f"{exp_name}_shap_explanation.pkl"
            )
            with open(shap_values_path, "rb") as f:
                explanation = pickle.load(f)
        else:
            if args.shap_sample is not None and args.shap_sample < 1.0:
                # sample X_test for faster shap value calc
                X_test_sample = X_test.sample(
                    frac=args.shap_sample, random_state=config.RANDOM_STATE
                )

            # cast bool to int as SHAP TreeExplainer requires numeric inputs
            X_test_sample = X_test_sample.astype(
                {
                    col: int
                    for col in X_test_sample.select_dtypes(
                        include="bool"
                    ).columns
                }
            )

            # get shap values for test sample
            explainer = shap.TreeExplainer(best_model, X_test_sample)
            explanation = explainer(X_test_sample)

            if args.save_shap:
                # ensure shap values directory exists
                config.SHAP_VALUES_DIR.mkdir(parents=True, exist_ok=True)

                # save shap values to pickled file
                shap_values_path = (
                    config.SHAP_VALUES_DIR / f"{exp_name}_shap_explanation.pkl"
                )
                with open(shap_values_path, "wb") as f:
                    pickle.dump(explanation, f)

        # plot SHAP summary plot
        ax_shap = exp_group_sum_fig.add_subplot(
            2, len(exp_names), len(exp_names) + i + 1
        )
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

        # convert shap values to dataframe
        shap_df = pd.DataFrame(
            explanation.values,
            columns=X_test_sample.columns,
            index=X_test_sample.index,
        )

        # merge shap values dataframe with geo_temp_cols from X_test dataframe
        geo_temp_cols = ["lon", "lat", "eat_hours", "date_angle"]
        merge_df = test_df.loc[X_test_sample.index, geo_temp_cols].merge(
            shap_df[
                [col for col in shap_df.columns if col not in geo_temp_cols]
            ],
            left_index=True,
            right_index=True,
        )

        # calculate correlation
        corr_matrix = merge_df.corr()

        # plot heat map of correlations
        plt.figure(figsize=(10, int(0.3 * len(corr_matrix))))
        sns.heatmap(
            corr_matrix[geo_temp_cols],
            annot=True,
            fmt=".2f",
            cmap=config.CORR_HEATMAP_CMAP,
            center=0,
            cbar_kws={"label": "Correlation"},
            vmin=-1,
            vmax=1,
        )
        plt.title(
            f"Heatmap of {exp_name} SHAP Value Correlations with Geo-Temporal Features"
        )
        plotting.save_plot(
            f"{exp_name}_shap_correlation_heatmap.png",
            exp_group_fig_dir,
        )

        # get absolute correlations with lon and lat
        corr_with_lon = corr_matrix["lon"].abs()
        corr_with_lat = corr_matrix["lat"].abs()

        # combine and get top features (excluding lon and lat themselves)
        n_top_features = 5
        combined_corr = corr_with_lon.add(corr_with_lat, fill_value=0)
        combined_corr = combined_corr.drop(["lon", "lat"], errors="ignore")
        top_geo_corr_features = (
            combined_corr.sort_values(ascending=False)
            .head(n_top_features)
            .index
        )
        for feature in top_geo_corr_features:
            n_bins = 50
            binned2d_mean = (
                merge_df.groupby(
                    [
                        pd.cut(merge_df["lat"], bins=n_bins),
                        pd.cut(merge_df["lon"], bins=n_bins),
                    ],
                    observed=False,
                )[feature]
                .mean()
                .reset_index(name=f"{feature}_mean")
            )

            binned2d_mean["center_lat"] = (
                binned2d_mean["lat"].apply(lambda x: x.mid).astype(float)
            )
            binned2d_mean["center_lon"] = (
                binned2d_mean["lon"].apply(lambda x: x.mid).astype(float)
            )

            mean_grid = (
                binned2d_mean[f"{feature}_mean"]
                .to_numpy()
                .reshape(
                    n_bins,
                    n_bins,
                )
            )

            plt.figure(figsize=(10, 6))
            ax = plotting.init_map(extent=config.STORM_DATA_EXTENT)

            pcolormesh = ax.pcolormesh(
                binned2d_mean["center_lon"].unique(),
                binned2d_mean["center_lat"].unique(),
                mean_grid,
                cmap=config.SHAP_MAP_CMAP,
                transform=ccrs.PlateCarree(),
                vmin=-max(abs(np.nanmin(mean_grid)), np.nanmax(mean_grid)),
                vmax=max(abs(np.nanmin(mean_grid)), np.nanmax(mean_grid)),
            )
            cbar = plt.colorbar(
                pcolormesh,
                ax=ax,
                orientation="horizontal",
                pad=0.1,
                aspect=40,
                shrink=0.63,
            )
            cbar.set_label(f"Mean SHAP Value ({exp_config['target_units']})")
            plotting.add_borders(ax)
            plotting.add_gridlines(ax)

            plt.title(f"Mean SHAP Value of {feature} over Map for {exp_name}")
            plotting.save_plot(
                f"{exp_name}_shap_{feature}_map.png", exp_group_geo_corr_fig_dir
            )

        # for n_top_features abs corr with eat_hours, bar plot with mean per hour
        top_hour_corr_features = (
            corr_matrix["eat_hours"]
            .abs()
            .sort_values(ascending=False)
            .iloc[1 : n_top_features + 1]
            .index
        )
        for feature in top_hour_corr_features:
            mean_per_hour = (
                merge_df.groupby("eat_hours")[feature].mean().reset_index()
            )

            centers = mean_per_hour[feature]
            m = np.max(np.abs(centers))  # Symmetric range around zero
            norm = TwoSlopeNorm(vmin=-m, vcenter=0, vmax=m)
            cmap = plt.get_cmap(config.SHAP_MAP_CMAP)
            colors = [cmap(norm(val)) for val in centers]

            plt.figure(figsize=(10, 6))
            ax = sns.barplot(
                data=mean_per_hour,
                x="eat_hours",
                y=feature,
                hue="eat_hours",
                palette=colors,
                legend=False,
            )
            plt.title(f"Mean SHAP Value of {feature} by Hour")
            plt.xlabel("Hour (UTC+3)")
            plt.ylabel(f"Mean SHAP Value ({exp_config['target_units']})")
            ax.set_xticks(range(24 * 4))
            ax.set_xticklabels(
                [
                    f"{h//4}:{(h%4)*15:02d}" if h % 4 == 0 else ""
                    for h in range(24 * 4)
                ],
                rotation=45,
            )
            plotting.save_plot(
                f"{exp_name}_shap_{feature}_by_hour.png",
                exp_group_temp_corr_fig_dir,
            )

        top_date_corr_features = (
            corr_matrix["date_angle"]
            .abs()
            .sort_values(ascending=False)
            .iloc[1 : n_top_features + 1]
            .index
        )
        merge_df["timestamp"] = test_df.loc[X_test_sample.index, "timestamp"]
        for feature in top_date_corr_features:
            mean_per_day = (
                merge_df.groupby(merge_df["timestamp"].dt.dayofyear)[feature]
                .mean()
                .reset_index()
            )

            centers = mean_per_day[feature]
            m = np.max(np.abs(centers))  # Symmetric range around zero
            norm = TwoSlopeNorm(vmin=-m, vcenter=0, vmax=m)
            cmap = plt.get_cmap(config.SHAP_MAP_CMAP)
            colors = [cmap(norm(val)) for val in centers]

            plt.figure(figsize=(10, 6))
            ax = sns.barplot(
                data=mean_per_day,
                x="timestamp",
                y=feature,
                hue="timestamp",
                palette=colors,
                legend=False,
                edgecolor="none",
            )

            plt.title(f"Mean SHAP Value of {feature} over Year")
            plt.xlabel("Day of Year")
            plt.ylabel(f"Mean SHAP Value ({exp_config['target_units']})")
            daysinyear = mean_per_day["timestamp"].nunique()
            ax.set_xticks(range(1, daysinyear + 1))
            ax.set_xticklabels(
                [
                    str(day) if day % 30 == 0 else ""
                    for day in range(1, daysinyear + 1)
                ]
            )
            plotting.save_plot(
                f"{exp_name}_shap_{feature}_by_day_over_year.png",
                exp_group_temp_corr_fig_dir,
            )

            mean_per_week = (
                merge_df.groupby(merge_df["timestamp"].dt.isocalendar().week)[
                    feature
                ]
                .mean()
                .reset_index()
            )

            centers = mean_per_week[feature]
            m = np.max(np.abs(centers))  # Symmetric range around zero
            norm = TwoSlopeNorm(vmin=-m, vcenter=0, vmax=m)
            cmap = plt.get_cmap(config.SHAP_MAP_CMAP)
            colors = [cmap(norm(val)) for val in centers]

            plt.figure(figsize=(10, 6))
            ax = sns.barplot(
                data=mean_per_week,
                x="week",
                y=feature,
                hue="week",
                palette=colors,
                legend=False,
            )

            plt.title(f"Mean SHAP Value of {feature} over Year")
            plt.xlabel("Week of Year")
            plt.ylabel(f"Mean SHAP Value ({exp_config['target_units']})")
            weeksinyear = mean_per_week["week"].nunique()
            ax.set_xticks(range(1, weeksinyear + 1))
            ax.set_xticklabels(
                [
                    str(week) if week % 4 == 0 else ""
                    for week in range(1, weeksinyear + 1)
                ]
            )
            plotting.save_plot(
                f"{exp_name}_shap_{feature}_by_week_over_year.png",
                exp_group_temp_corr_fig_dir,
            )

    plotting.save_plot(f"{exp_group_name}_summary.png", exp_group_fig_dir)
