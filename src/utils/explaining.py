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
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib.axes import Axes
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
    temp_agg_df: pd.DataFrame,
    agg_x: str,
    agg_y: str,
    ax: Optional[Axes] = None,
    edgecolor: Optional[str] = None,
    xtick_interval: Optional[int] = None,
    xtick_offset: int = 1,
    xtick_convert: Callable[[int], str] = str,
    xtick_rotation: int = 0,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    filename: Optional[str] = None,
    save_dir: Optional[Path] = None,
):
    """
    Plot aggregated SHAP values over time (or another sequential variable centred at zero) using a bar plot.

    :param temp_agg_df: DataFrame containing aggregated SHAP values to plot.
    :param agg_x: Column name in temp_agg_df to use for the x-axis (e.g., time or interval).
    :param agg_y: Column name in temp_agg_df to use for the y-axis (SHAP value to plot).
    :param ax: Optional matplotlib Axes to plot on. If None, a new figure is created.
    :param edgecolor: Optional color for bar edges.
    :param xtick_interval: If set, only every nth x-tick is labeled.
    :param xtick_offset: Offset for x-tick labeling (default: 1).
    :param xtick_convert: Function to convert x-tick values to labels (default: str).
    :param xtick_rotation: Rotation angle for x-tick labels (default: 0).
    :param title: Optional plot title.
    :param xlabel: Optional x-axis label.
    :param ylabel: Optional y-axis label.
    :param filename: If provided, the plot is saved to this filename.
    :param save_dir: Directory to save the plot if filename is provided.
    """
    # create custom color palette centred at zero
    m = np.max(np.abs(temp_agg_df[agg_y]))
    norm = TwoSlopeNorm(vmin=-m, vcenter=0, vmax=m)
    cmap = plt.get_cmap(config.SHAP_MAP_CMAP)
    colors = [cmap(norm(val)) for val in temp_agg_df[agg_y]]

    # init plot if not axis provided
    if ax is None:
        plt.figure(figsize=(10, 6))

    # plot bars with custom colors
    ax = sns.barplot(
        data=temp_agg_df,
        x=agg_x,
        y=agg_y,
        hue=agg_x,
        palette=colors,
        legend=False,
        edgecolor=edgecolor,
        ax=ax,
    )

    if xtick_interval is not None:
        # reduce xticks to every nth interval
        intervals = temp_agg_df[agg_x].nunique()
        ax.set_xticks(range(intervals))
        ax.set_xticklabels(
            [
                (
                    xtick_convert(interval)
                    if interval % xtick_interval == 0
                    else ""
                )
                for interval in range(xtick_offset, intervals + xtick_offset)
            ],
            rotation=xtick_rotation,
        )

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if filename is not None and save_dir is not None:
        plotting.save_plot(filename, save_dir)
    else:
        plt.show()
