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
from metpy.calc import geopotential_to_height
from pandarallel import pandarallel
from pint import Quantity
from scipy.stats import gaussian_kde

import config

pandarallel.initialize(progress_bar=True)


def load_geop_and_calc_elevation() -> tuple[xr.Dataset, Quantity]:
    """
    Load and cache geopotential data and height.

    :return: Tuple containing the geopotential dataset and height as a Quantity.
    :rtype: tuple[xr.Dataset, Quantity]
    """
    # load geopotential data
    geop = xr.open_dataset(config.DATA_DIR / "std" / "geop.nc")
    geop_vals = geop["geop"].values.squeeze()

    # convert geopotential values to a Quantity
    # source: https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.geopotential_to_height.html#metpy.calc.geopotential_to_height
    geop_units = Quantity(geop_vals, "m^2/s^2")

    # convert geopotential to height
    height = geopotential_to_height(geop_units)

    return geop, height


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


def get_orography_features(
    processed_df: pd.DataFrame,
    geop: xr.Dataset,
    height: Quantity,
    anor: xr.Dataset,
) -> pd.DataFrame:
    """
    Calculate orography features for the dataset.

    :param processed_df: DataFrame containing storm data with 'x' and 'y' columns for longitude and latitude.
    :param geop: Geopotential dataset containing 'geop' variable.
    :param height: Height calculated from geopotential data.
    :param anor: Dataset containing subgrid orography angle data.
    :return: DataFrame with additional columns for orography height and subgrid orography angle (anor).
    :rtype: pd.DataFrame
    """

    def get_orography_at_lon_lat(row):
        # find the closest point in the geopotential data
        closest = geop.sel(
            longitude=row["x"], latitude=row["y"], method="nearest"
        )

        # get the indices of the closest point
        i = np.where(
            np.isclose(geop.latitude.values, closest.latitude.values.item())
        )[0][0]
        j = np.where(
            np.isclose(geop.longitude.values, closest.longitude.values.item())
        )[0][0]

        # return the geopotential height at the closest point
        return height[i, j].magnitude

    processed_df["orography_height"] = processed_df.parallel_apply(
        get_orography_at_lon_lat, axis=1
    )
    processed_df["anor"] = processed_df.parallel_apply(
        lambda row: anor.sel(
            longitude=row["x"], latitude=row["y"], method="nearest"
        )["anor"].item(),
        axis=1,
    )
    return processed_df


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
