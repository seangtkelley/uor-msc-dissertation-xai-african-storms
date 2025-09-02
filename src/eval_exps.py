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

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from dotenv import load_dotenv
from scipy.stats import circstd
from sklearn.metrics import root_mean_squared_error

import config
from utils import explaining, modelling, plotting, processing

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
    default=False,
    help="Save SHAP values to file",
)
parser.add_argument(
    "--load_shap",
    action="store_true",
    default=False,
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

        # calculate RMSE and std dev
        if exp_config["target_units"] == "degrees":
            y_pred_deg = (y_pred + 360) % 360
            y_true_deg = y_test.to_numpy() % 360
            test_rmse = modelling.rmse(y_true_deg, y_pred_deg)
            test_std = circstd(y_true_deg, high=360)
        else:
            test_rmse = root_mean_squared_error(y_test, y_pred)
            test_std = np.std(y_test)

        # print RMSE and standard deviation
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test target standard deviation: {test_std:.4f}")

        if exp_name.startswith("storm_") and not all(
            test_df["storm_obs_idx"] == 0
        ):
            # get first points for fair comparison of metrics
            first_points_test_df = test_df[test_df["storm_obs_idx"] == 0]
            X_test_first_points = X_test.loc[first_points_test_df.index]
            y_test_first_points = y_test.loc[first_points_test_df.index]

            # make predictions on first points
            y_pred_first_points = best_model.predict(X_test_first_points)

            # calculate RMSE and std dev for first points
            if exp_config["target_units"] == "degrees":
                y_pred_fp_deg = (y_pred_first_points + 360) % 360
                y_true_fp_deg = y_test_first_points.to_numpy() % 360
                test_rmse_first_points = modelling.rmse(
                    y_true_fp_deg, y_pred_fp_deg
                )
                test_std_first_points = circstd(y_true_fp_deg, high=360)
            else:
                test_rmse_first_points = root_mean_squared_error(
                    y_test_first_points, y_pred_first_points
                )
                test_std_first_points = np.std(y_test_first_points)

            print(f"Test RMSE (first points): {test_rmse_first_points:.4f}")
            print(
                f"Test target standard deviation (first points): {test_std_first_points:.4f}"
            )

        # plot model verification and compute R2
        if exp_config["target_units"] == "degrees":
            y_pred_deg = (y_pred + 360) % 360
            y_true_deg = y_test.to_numpy() % 360
            r_squared = modelling.circ_r2(y_true_deg, y_pred_deg)
            modelling.plot_model_verification(
                exp_name,
                exp_config["target_units"],
                y_true_deg,
                y_pred_deg,
                ax=exp_group_sum_fig.add_subplot(2, len(exp_names), i + 1),
                title=f"{chr(i+97)}) Model Verification for {exp_name}",
                r_squared=r_squared,
            )
        else:
            r_squared = modelling.plot_model_verification(
                exp_name,
                exp_config["target_units"],
                y_test.to_numpy(),
                y_pred,
                ax=exp_group_sum_fig.add_subplot(2, len(exp_names), i + 1),
                title=f"{chr(i+97)}) Model Verification for {exp_name}",
            )

        print(f"R-squared: {r_squared:.4f}")

        if r_squared < config.R_SQUARED_THRESHOLD:
            continue

        # by default, use entire test set
        X_test_sample = X_test
        if args.load_shap:
            print("Loading SHAP values...")
            X_test_sample, explanation = explaining.load_shap_for_exp(
                exp_name, X_test
            )
        else:
            print("Calculating SHAP values...")
            X_test_sample, explanation = explaining.calc_shap_values(
                best_model, X_test, sample_frac=args.shap_sample
            )

            if args.save_shap:
                explaining.save_shap_for_exp(
                    exp_name, X_test_sample, explanation
                )

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
        ax_shap.set_title(f"{chr(i+97+2)}) SHAP Beeswarm Plot for {exp_name}")
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
            agg_lon, agg_lat, agg_grid = processing.calc_2d_agg(
                merge_df, feature
            )
            plotting.plot_2d_agg_map(
                agg_lon,
                agg_lat,
                agg_grid,
                cmap=config.SHAP_MAP_CMAP,
                sym_cmap_centre=0.0,
                cbar_label=f"Mean SHAP Value ({exp_config['target_units']})",
                cbar_aspect=40,
                cbar_shrink=0.63,
                title=f"Mean SHAP Value of {feature} over Map for {exp_name}",
                filename=f"{exp_name}_shap_{feature}_map.png",
                save_dir=exp_group_geo_corr_fig_dir,
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

            explaining.plot_shap_over_time(
                mean_per_hour,
                agg_x="eat_hours",
                agg_y=feature,
                xtick_interval=4,
                xtick_offset=0,
                xtick_convert=lambda x: f"{x//4}:00",
                xtick_rotation=45,
                title=f"Mean SHAP Value of {feature} by Hour",
                xlabel="Time (UTC+3)",
                ylabel=f"Mean SHAP Value ({exp_config['target_units']})",
                filename=f"{exp_name}_shap_{feature}_by_hour.png",
                save_dir=exp_group_temp_corr_fig_dir,
            )

        # add timestamp to merge_df for easier grouping
        merge_df["timestamp"] = test_df.loc[X_test_sample.index, "timestamp"]

        # for n_top_features abs corr with date_angle, bar plot with mean per day and week of year
        top_date_corr_features = (
            corr_matrix["date_angle"]
            .abs()
            .sort_values(ascending=False)
            .iloc[1 : n_top_features + 1]
            .index
        )
        for feature in top_date_corr_features:
            mean_per_day = (
                merge_df.groupby(merge_df["timestamp"].dt.dayofyear)[feature]
                .mean()
                .reset_index()
            )

            explaining.plot_shap_over_time(
                mean_per_day,
                agg_x="timestamp",
                agg_y=feature,
                edgecolor="none",
                xtick_interval=30,
                title=f"Mean SHAP Value of {feature} over Year",
                xlabel="Day of Year",
                ylabel=f"Mean SHAP Value ({exp_config['target_units']})",
                filename=f"{exp_name}_shap_{feature}_by_day_over_year.png",
                save_dir=exp_group_temp_corr_fig_dir,
            )

            mean_per_week = (
                merge_df.groupby(merge_df["timestamp"].dt.isocalendar().week)[
                    feature
                ]
                .mean()
                .reset_index()
            )

            explaining.plot_shap_over_time(
                mean_per_week,
                agg_x="week",
                agg_y=feature,
                xtick_interval=4,
                title=f"Mean SHAP Value of {feature} over Year",
                xlabel="Week of Year",
                ylabel=f"Mean SHAP Value ({exp_config['target_units']})",
                filename=f"{exp_name}_shap_{feature}_by_week_over_year.png",
                save_dir=exp_group_temp_corr_fig_dir,
            )

        # plot mean SHAP value maps by hour for features with high geo and hour correlation
        for feature in set(top_geo_corr_features).intersection(
            set(top_hour_corr_features)
        ):
            fig, axs = plt.subplots(
                2,
                3,
                figsize=(10, 6),
                subplot_kw={"projection": ccrs.PlateCarree()},
            )
            axs = axs.flatten()

            # symmetrical cmap
            m = max(
                abs(np.percentile(merge_df[feature], 1)),
                abs(np.percentile(merge_df[feature], 99)),
            )

            for idx, hour in enumerate(range(0, 24, 4)):
                hour_df = merge_df[merge_df["eat_hours"] == hour]
                if hour_df.empty:
                    continue
                agg_lon, agg_lat, agg_grid = processing.calc_2d_agg(
                    hour_df, feature, n_bins=25
                )
                axs[idx] = plotting.init_map(
                    axs[idx], extent=config.STORM_DATA_EXTENT
                )
                plotting.plot_2d_agg_map(
                    agg_lon,
                    agg_lat,
                    agg_grid,
                    ax=axs[idx],
                    cmap=config.SHAP_MAP_CMAP,
                    vmin=-m,
                    vmax=m,
                    add_cbar=False,
                    small_grid_labels=True,
                    title=f"{chr(idx+97)}) {hour}:00",
                )

            # single cbar for whole image
            fig.subplots_adjust(
                bottom=0.11, top=0.93, left=0.07, right=0.97, hspace=0.08
            )
            cbar_ax = fig.add_axes(
                (0.07, 0.07, 0.86, 0.025)
            )  # [left, bottom, width, height]
            cbar = fig.colorbar(
                axs[-1].collections[0], cax=cbar_ax, orientation="horizontal"
            )
            cbar.set_label(f"Mean SHAP Value ({exp_config['target_units']})")

            fig.suptitle(
                f"{exp_name}: Mean SHAP Value of {feature} by Hour over Map",
                fontsize=17,
                y=0.97,
            )
            plotting.save_plot(
                f"{exp_name}_shap_{feature}_map_by_hour.png",
                exp_group_geo_corr_fig_dir,
                tight=False,
            )

        # plot mean SHAP value maps by month for features with high geo and date correlation
        for feature in set(top_geo_corr_features).intersection(
            set(top_date_corr_features)
        ):
            fig, axs = plt.subplots(
                3,
                4,
                figsize=(10, 8),
                subplot_kw={"projection": ccrs.PlateCarree()},
            )
            axs = axs.flatten()

            # symmetrical cmap
            m = max(
                abs(np.percentile(merge_df[feature], 1)),
                abs(np.percentile(merge_df[feature], 99)),
            )

            for idx, month in enumerate(range(1, 13)):
                month_df = merge_df[merge_df["timestamp"].dt.month == month]
                if month_df.empty:
                    continue
                agg_lon, agg_lat, agg_grid = processing.calc_2d_agg(
                    month_df, feature, n_bins=25
                )
                axs[idx] = plotting.init_map(
                    axs[idx], extent=config.STORM_DATA_EXTENT
                )
                plotting.plot_2d_agg_map(
                    agg_lon,
                    agg_lat,
                    agg_grid,
                    ax=axs[idx],
                    cmap=config.SHAP_MAP_CMAP,
                    vmin=-m,
                    vmax=m,
                    add_cbar=False,
                    small_grid_labels=True,
                    title=f"{chr(idx+97)}) {pd.Timestamp(month=month, day=1, year=2000).strftime('%b')}",
                )

            # single cbar for whole image
            fig.subplots_adjust(
                bottom=0.11, top=0.93, left=0.07, right=0.97, hspace=0.08
            )
            cbar_ax = fig.add_axes(
                (0.07, 0.07, 0.86, 0.025)
            )  # [left, bottom, width, height]
            cbar = fig.colorbar(
                axs[-1].collections[0], cax=cbar_ax, orientation="horizontal"
            )
            cbar.set_label(f"Mean SHAP Value ({exp_config['target_units']})")

            fig.suptitle(
                f"{exp_name}: Mean SHAP Value of {feature} by Month over Map",
                fontsize=17,
                y=0.97,
            )
            plotting.save_plot(
                f"{exp_name}_shap_{feature}_map_by_month.png",
                exp_group_geo_corr_fig_dir,
                tight=False,
            )

    plotting.save_plot(f"{exp_group_name}_summary.png", exp_group_fig_dir)
