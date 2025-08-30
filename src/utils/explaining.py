#!/usr/bin/env python

"""explaining.py: Helper functions for generating model explanations"""

__author__ = "Sean Kelley"
__copyright__ = "Copyright 2025, University of Reading"
__credits__ = ["Sean Kelley"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Sean Kelley"
__email__ = "s.g.t.kelley@student.reading.ac.uk"
__status__ = "Development"

import pickle
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib.colors import TwoSlopeNorm

import config
from utils import plotting


def get_shap_path_for_exp(exp_name: str) -> Path:
    """
    Get the file path for SHAP values of a given experiment.

    :param exp_name: The name of the experiment.
    :return: The file path.
    """
    return config.SHAP_VALUES_DIR / f"{exp_name}_shap_explanation.pkl"


def load_shap_for_exp(
    exp_name: str, df: pd.DataFrame
) -> tuple[pd.DataFrame, shap.Explanation]:
    """
    Load SHAP values for a given experiment.

    :param exp_name: The name of the experiment.
    :param df: The data used for explanation.
    :return: Filtered data and SHAP explanation object.
    """
    # load shap values from pickled file
    with open(get_shap_path_for_exp(exp_name), "rb") as f:
        df_index, explanation = pickle.load(f)

    return df.loc[df_index], explanation


def save_shap_for_exp(
    exp_name: str, df: pd.DataFrame, explanation: shap.Explanation
) -> None:
    """
    Save SHAP values for a given experiment.

    :param exp_name: The name of the experiment.
    :param df: The data used for explanation.
    :param explanation: SHAP explanation object to save.
    """
    # ensure shap values directory exists
    config.SHAP_VALUES_DIR.mkdir(parents=True, exist_ok=True)

    # save shap values to pickled file
    with open(get_shap_path_for_exp(exp_name), "wb") as f:
        pickle.dump((df.index, explanation), f)


def calc_shap_values(
    model, df, sample_frac: Optional[float] = None
) -> tuple[pd.DataFrame, shap.Explanation]:
    """
    Calculate SHAP values for a given model and dataset.

    :param model: The model to explain.
    :param df: The data to use for explanation.
    :param sample_frac: Optional fraction of data to sample for SHAP calculation.
    :return: SHAP values object.
    """
    if sample_frac is not None:
        # sample the data
        df = df.sample(frac=sample_frac, random_state=config.RANDOM_STATE)

    # cast bool to int as SHAP TreeExplainer requires numeric inputs
    df_sample = df.astype(
        {col: int for col in df.select_dtypes(include="bool").columns}
    )

    explainer = shap.TreeExplainer(model, df_sample)
    return df_sample, explainer(df_sample)


def plot_shap_over_time(
    temp_agg, feature, exp_name, exp_config, exp_group_temp_corr_fig_dir
):
    centers = temp_agg[feature]
    m = np.max(np.abs(centers))  # Symmetric range around zero
    norm = TwoSlopeNorm(vmin=-m, vcenter=0, vmax=m)
    cmap = plt.get_cmap(config.SHAP_MAP_CMAP)
    colors = [cmap(norm(val)) for val in centers]

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=temp_agg,
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
    daysinyear = temp_agg["timestamp"].nunique()
    ax.set_xticks(range(1, daysinyear + 1))
    ax.set_xticklabels(
        [str(day) if day % 30 == 0 else "" for day in range(1, daysinyear + 1)]
    )
    plotting.save_plot(
        f"{exp_name}_shap_{feature}_by_day_over_year.png",
        exp_group_temp_corr_fig_dir,
    )
