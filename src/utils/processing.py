#!/usr/bin/env python

"""processing.py: Helper functions for processing data"""

__author__ = "Sean Kelley"
__copyright__ = "Copyright 2025, University of Reading"
__credits__ = ["Sean Kelley"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Sean Kelley"
__email__ = "s.g.t.kelley@student.reading.ac.uk"
__status__ = "Development"

import numpy as np
import pandas as pd
import xarray as xr
from pint import Quantity
from scipy.stats import gaussian_kde

import config


def rename_columns(df, column_map: dict[str, str]) -> pd.DataFrame:
    """
    Rename columns in a DataFrame based on a provided mapping.

    :param df: The DataFrame whose columns are to be renamed.
    :param column_map: A dictionary mapping old column names to new column names.
    :return: A DataFrame with renamed columns.
    :rtype: pd.DataFrame
    """
    column_map = {k: v for k, v in column_map.items() if k in df.columns}
    return df.rename(columns=column_map)


def closest_indices(values: np.ndarray, search_space: np.ndarray) -> np.ndarray:
    """
    Find the indices of the closest values in a search space for each value in the input array.

    :param values: Array of values to find closest indices for.
    :param search_space: Array of values to search within.
    :return: Indices of the closest values in the search space.
    """
    # find the indices of the closest values in the search space
    return np.abs(search_space[:, None] - values).argmin(axis=0)


def get_era5_indices_for_coords(
    df: pd.DataFrame,
    era5_lons: np.ndarray,
    era5_lats: np.ndarray,
) -> pd.DataFrame:
    """
    Get the indices of the nearest ERA5 grid points for given longitudes and latitudes.

    :param df: DataFrame containing 'x' and 'y' columns representing longitudes and latitudes.
    :param era5_lons: Array of ERA5 longitudes.
    :param era5_lats: Array of ERA5 latitudes.
    :return: DataFrame with additional columns 'era5_x_idx' and 'era5_y_idx'
             containing the indices of the closest ERA5 grid points.
    :rtype: pd.DataFrame
    """
    df["era5_x_idx"] = closest_indices(df["x"].to_numpy(), era5_lons)
    df["era5_y_idx"] = closest_indices(df["y"].to_numpy(), era5_lats)
    return df


def get_orography_features(
    df: pd.DataFrame, height: Quantity, anor: xr.Dataset
) -> pd.DataFrame:
    """
    Calculate orography features for the dataset.

    This function computes the orography height and anor (anomaly of orography) features
    for the storm dataset. It uses the ERA5 data to calculate these features based on
    the storm coordinates.
    """
    df["orography_height"] = height[df["era5_y_idx"], df["era5_x_idx"]]
    df["anor"] = anor["anor"].values[df["era5_y_idx"], df["era5_x_idx"]]
    return df


def calc_kde(lons, lats) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the 2D kernel density estimate (KDE) for the given longitudes and latitudes.

    :param lons: Array of longitudes.
    :param lats: Array of latitudes.
    :return: Tuple containing the KDE values and the grid points.
    """
    # stack lons and lats for kde calculation
    xy = np.vstack([lons, lats])

    # calculate the 2D KDE
    kde = gaussian_kde(xy)

    # create a grid for the KDE
    xmin, xmax = config.STORM_DATA_EXTENT[0], config.STORM_DATA_EXTENT[1]
    ymin, ymax = config.STORM_DATA_EXTENT[2], config.STORM_DATA_EXTENT[3]
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

    # evaluate the KDE on the grid
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kde(positions).T, X.shape)

    # normalize Z for consistent color scaling
    Z /= Z.max()

    return X, Y, Z
