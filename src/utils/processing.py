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
import pyproj

import config

# initialize pandarallel for parallel processing with progress bar
pandarallel.initialize(progress_bar=True)

# initialize the geodesic calculator with WGS84 ellipsoid
# TODO: is there a better projection for East Africa?
geod = pyproj.Geod(ellps='WGS84')


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
    # extract longitude and latitude arrays
    longitudes = processed_df["x"].values
    latitudes = processed_df["y"].values

    # perform batch indexing for geopotential height
    closest_geop = geop.sel(longitude=longitudes, latitude=latitudes, method="nearest")
    closest_lat_indices = closest_indices(closest_geop.latitude.values, geop.latitude.values)
    closest_lon_indices = closest_indices(closest_geop.longitude.values, geop.longitude.values)
    processed_df["orography_height"] = height[closest_lat_indices, closest_lon_indices].magnitude

    # perform batch indexing for subgrid orography angle (anor)
    closest_anor = anor.sel(longitude=longitudes, latitude=latitudes, method="nearest")
    processed_df["anor"] = closest_anor["anor"].values

    return processed_df

def calc_storm_distances_and_bearings(processed_df: pd.DataFrame) -> pd.DataFrame:
    processed_df["distance_from_prev"] = np.nan
    processed_df["bearing_from_prev"] = np.nan
    processed_df["storm_straight_line_distance"] = np.nan
    processed_df["storm_bearing"] = np.nan
    # TODO: can this be vectorized or parallelized?
    for _, group in processed_df.groupby("storm_id"):
        for i in range(1, len(group)):
            # get previous and current point coords
            prev_lon, prev_lat = group.iloc[i - 1]["y"], group.iloc[i - 1]["x"]
            curr_lon, curr_lat = group.iloc[i]["y"], group.iloc[i]["x"]

            # calc forward azimuth, back azimuth, and distance
            fwd_azimuth, _, distance_m = geod.inv(
                prev_lon, prev_lat, curr_lon, curr_lat
            )

            # update the dataframe with the calculated values
            processed_df.loc[group.index[i], "distance_from_prev"] = distance_m / 1000
            processed_df.loc[group.index[i], "bearing_from_prev"] = fwd_azimuth + 180

        # get first and last point coords
        first_lon, first_lat = group.iloc[0]["y"], group.iloc[0]["x"]
        last_lon, last_lat = group.iloc[-1]["y"], group.iloc[-1]["x"]
        
        # calc forward azimuth, back azimuth, and distance
        fwd_azimuth, _, distance_m = geod.inv(
            first_lon, first_lat, last_lon, last_lat
        )

        # write the storm direction and distance to the entire group
        processed_df.loc[group.index, "storm_straight_line_distance"] = distance_m / 1000
        processed_df.loc[group.index, "storm_bearing"] = fwd_azimuth + 180

    # fill in NaN values in distance_from_prev with 0 for the first point in each storm
    # Note: leave bearing_from_prev as NaN for the first point in each storm as the storm has not yet moved
    processed_df["distance_from_prev"] = processed_df["distance_from_prev"].fillna(0)
    
    # calc storm distance traversed via cumulative sum of distance_from_prev
    processed_df["storm_distance_traversed"] = (
        processed_df.groupby("storm_id")["distance_from_prev"]
        .cumsum()
        .fillna(0)
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
