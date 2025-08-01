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

from typing import Callable, Optional

import numpy as np
import pandas as pd
import psutil
import pyproj
import xarray as xr
from metpy.calc import geopotential_to_height
from pandarallel import pandarallel
from pint import Quantity
from scipy.stats import gaussian_kde

import config

# fallback to 6 workers if cpu_count is None
nb_workers = psutil.cpu_count(logical=False) or 12
# use half the physical cores for parallel processing
nb_workers //= 2
# initialize pandarallel for parallel processing
pandarallel.initialize(progress_bar=True, nb_workers=nb_workers)

# initialize the geodesic calculator with WGS84 ellipsoid
# TODO: is there a better projection for East Africa?
geod = pyproj.Geod(ellps="WGS84")


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


def closest_indices(
    values: np.typing.ArrayLike, search_space: np.ndarray
) -> np.ndarray:
    """
    Find the indices of the closest values in a search space for each value in the input array.

    :param values: Array of values to find closest indices for.
    :param search_space: Array of values to search within.
    :return: Indices of the closest values in the search space.
    """
    return np.abs(search_space[:, None] - values).argmin(axis=0)


def get_orography_features(
    processed_df: pd.DataFrame,
    geop: xr.Dataset,
    height: Quantity,
    anor: xr.Dataset,
) -> pd.DataFrame:
    """
    Calculate orography features for the dataset.

    :param processed_df: DataFrame containing storm data. Must include 'lon' and 'lat' columns representing longitude and latitude, respectively.
    :param geop: Geopotential dataset containing 'geop' variable.
    :param height: Height calculated from geopotential data.
    :param anor: Dataset containing subgrid orography angle data.
    :return: DataFrame with additional columns for orography height and subgrid orography angle (anor).
    :rtype: pd.DataFrame
    """
    # extract longitude and latitude arrays
    # source: https://stackoverflow.com/questions/40544846/read-multiple-coordinates-with-xarray#62784295
    lons = xr.DataArray(processed_df["lon"].to_numpy())
    lats = xr.DataArray(processed_df["lat"].to_numpy())

    # perform batch indexing for geopotential height
    closest_geop = geop.sel(longitude=lons, latitude=lats, method="nearest")
    closest_lat_indices = closest_indices(
        closest_geop.latitude.values, geop.latitude.values
    )
    closest_lon_indices = closest_indices(
        closest_geop.longitude.values, geop.longitude.values
    )
    processed_df["orography_height"] = height[
        closest_lat_indices, closest_lon_indices
    ].magnitude

    # perform batch indexing for subgrid orography angle (anor)
    closest_anor = anor.sel(longitude=lons, latitude=lats, method="nearest")
    processed_df["anor"] = closest_anor["anor"].values.squeeze()

    return processed_df


def calc_over_land_features(
    processed_df: pd.DataFrame,
    lsm: xr.Dataset,
) -> pd.DataFrame:
    """
    Calculate over land features for the dataset.
    :param processed_df: DataFrame containing storm data. Must include the following columns:
        - 'lon': Longitude values.
        - 'lat': Latitude values.
        - 'storm_id': Unique identifier for each storm, used for grouping operations.
        - 'timestamp': Timestamps for each observation, used for time-based calculations.
    :param lsm: Land-sea mask dataset containing 'lsm' variable.
    :return: DataFrame with additional columns for over land status and accumulated land time.
    :rtype: pd.DataFrame
    """
    # extract longitude and latitude arrays
    # source: https://stackoverflow.com/questions/40544846/read-multiple-coordinates-with-xarray#62784295
    lons = xr.DataArray(processed_df["lon"].to_numpy())
    lats = xr.DataArray(processed_df["lat"].to_numpy())

    # perform batch indexing for land-sea mask
    closest_lsm = lsm.sel(longitude=lons, latitude=lats, method="nearest").isel(
        valid_time=0
    )
    # add over land status to the DataFrame and convert to boolean (True for land, False for sea)
    processed_df["over_land"] = (
        closest_lsm["lsm"].values.squeeze().round().astype(bool)
    )

    # calculate accumulated land time
    processed_df["acc_land_time"] = (
        # find the difference in seconds between consecutive timestamps in a storm
        processed_df.groupby("storm_id")["timestamp"]
        .diff()
        .dt.total_seconds()
        .fillna(0)
        # multiply by over_land to only count time when over land
        * processed_df["over_land"]
    )
    # cumulative sum in hours
    processed_df["acc_land_time"] = (
        processed_df.groupby("storm_id")["acc_land_time"].cumsum() / 3600
    )

    # calculate storm total land time
    processed_df["storm_total_land_time"] = processed_df.groupby("storm_id")[
        "acc_land_time"
    ].transform("max")

    return processed_df


def calc_storm_distances_and_bearings(
    processed_df: pd.DataFrame,
) -> pd.DataFrame:
    # init columns
    processed_df["distance_from_prev"] = np.nan
    processed_df["bearing_from_prev"] = np.nan
    processed_df["storm_straight_line_distance"] = np.nan
    processed_df["storm_bearing"] = np.nan

    # calculate distances and bearings between consecutive points
    fwd_azimuths, _, distances_m = geod.inv(
        processed_df["lon"][:-1],
        processed_df["lat"][:-1],
        processed_df["lon"][1:],
        processed_df["lat"][1:],
    )

    # update the df with the calculated values
    processed_df.loc[processed_df.index[1:], "distance_from_prev"] = (
        distances_m / 1000
    )
    processed_df.loc[processed_df.index[1:], "bearing_from_prev"] = (
        fwd_azimuths % 360  # normalize to [0, 360)
    )

    # get storm init and end indices
    storm_groups = processed_df.groupby("storm_id")
    storm_inits = storm_groups.head(1)
    storm_ends = storm_groups.tail(1)

    # calculate straight-line distance and bearing for the entire storm
    fwd_azimuths, _, distances_m = geod.inv(
        storm_inits["lon"],
        storm_inits["lat"],
        storm_ends["lon"],
        storm_ends["lat"],
    )
    processed_df.loc[storm_inits.index, "storm_straight_line_distance"] = (
        distances_m / 1000
    )
    processed_df.loc[storm_inits.index, "storm_bearing"] = (
        fwd_azimuths % 360  # normalize to [0, 360)
    )

    # fill in first point for each storm with 0 distance_from_prev
    processed_df.loc[storm_inits.index, "distance_from_prev"] = 0

    # fill in first point bearing_from_prev for each storm with the overall storm bearing
    # this should be more meaningful than 0 as that would imply all storms initialize moving north
    processed_df.loc[storm_inits.index, "bearing_from_prev"] = processed_df.loc[
        storm_inits.index, "storm_bearing"
    ]

    # fill in missing values for storm distances and bearings by forward filling
    processed_df["storm_straight_line_distance"] = processed_df[
        "storm_straight_line_distance"
    ].ffill()
    processed_df["storm_bearing"] = processed_df["storm_bearing"].ffill()

    # calc distance traversed via cumulative sum of distance_from_prev
    processed_df["distance_traversed"] = (
        processed_df.groupby("storm_id")["distance_from_prev"]
        .cumsum()
        .fillna(0)
    )

    # get total storm distance traversed
    processed_df["storm_distance_traversed"] = (
        processed_df.groupby("storm_id")["distance_traversed"]
        .transform("max")
        .fillna(0)
    )

    return processed_df


def calc_temporal_rate_of_change(
    processed_df: pd.DataFrame,
    col_name: str,
    time_interval: Optional[pd.Timedelta] = None,
    new_col_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate the rate of change of a specified column over a given time interval.

    :param processed_df: DataFrame containing storm data.
    :param col_name: Name of the column to calculate the rate of change for.
    :param time_interval: Time interval over which to smooth the rate of change. If None, no smoothing is applied.
    :param new_col_name: Optional name for the new column to store the rate of change.
                         If None, defaults to "d{col_name}_dt".
    :return: A DataFrame with the calculated rate of change added as a new column.
    :rtype: pd.DataFrame
    """
    if new_col_name is None:
        new_col_name = "d" + col_name + "_dt"

    # calculate the rate of change of the specified column
    processed_df[new_col_name] = processed_df[col_name].diff() / (
        processed_df["timestamp"].diff().dt.total_seconds()
    )

    # fill the rate of change column with 0 for the first point in each storm
    storm_inits_idx = processed_df.groupby("storm_id").head(1).index
    processed_df.loc[storm_inits_idx, new_col_name] = 0

    if time_interval is not None:
        # calculate a rolling mean of the rate of change over the specified time interval for each storm group
        processed_df[new_col_name] = (
            processed_df.set_index("timestamp")
            .groupby("storm_id")[new_col_name]
            .rolling(time_interval)
            .mean()
            .values
        )

    return processed_df


def calc_spatiotemporal_mean_at_point(
    timestamp: pd.Timestamp,
    lon: float,
    lat: float,
    dataset: xr.Dataset,
    variable_name: str,
    radius_km: float = 400,
    timedelta: Optional[pd.Timedelta] = None,
    invariant: bool = False,
    variable_bounds: Optional[tuple[float, float]] = None,
) -> np.floating:
    """
    Calculate the spatial mean of a specified variable from an xarray dataset
    within a specified radius from a storm's location.

    :param timestamp: Timestamp of the storm's location.
    :param lon: Longitude of the storm's location.
    :param lat: Latitude of the storm's location.
    :param dataset: xarray Dataset containing the variable to calculate the spatial mean for.
    :param variable_name: Name of the variable in the dataset to calculate the spatial mean for.
    :param radius_km: Radius in kilometres for the spatial mean calculation.
    :param timedelta: Time delta to consider for the spatial mean calculation.
    :param invariant: If True, the variable is invariant in time (e.g., static data).
    :param variable_bounds: Optional tuple of (lower, upper) bounds to filter the variable values.
    :return: Spatial mean of the specified variable within the radius.
    :rtype: np.floating
    """
    # convert radius to meters
    radius_m = radius_km * 1000

    # vars for dataset longitude and latitude arrays
    dataset_lons = dataset["longitude"].values
    dataset_lats = dataset["latitude"].values

    # find the grid cell closest to the storm's location
    lon_idx = closest_indices([lon], dataset_lons).item()
    lat_idx = closest_indices([lat], dataset_lats).item()

    # find the grid height and width near the storm's location
    _, _, grid_width_m = geod.inv(
        dataset_lons[lon_idx],
        dataset_lats[lat_idx],
        dataset_lons[min(lon_idx + 1, len(dataset_lons) - 1)],
        dataset_lats[lat_idx],
    )
    _, _, grid_height_m = geod.inv(
        dataset_lons[lon_idx],
        dataset_lats[lat_idx],
        dataset_lons[lon_idx],
        dataset_lats[min(lat_idx + 1, len(dataset_lats) - 1)],
    )

    # calculate the number of grid cells in the radius for a square-ish area
    area_width_cells = int(radius_m // grid_width_m)
    area_height_cells = int(radius_m // grid_height_m)

    # calculate the indices of the grid cells within the radius
    lon_start = max(0, lon_idx - area_width_cells)
    lon_end = min(len(dataset_lons), lon_idx + area_width_cells + 1)
    lat_start = max(0, lat_idx - area_height_cells)
    lat_end = min(len(dataset_lats), lat_idx + area_height_cells + 1)

    # extract the relevant grid cells
    var_over_grid = dataset.isel(
        longitude=slice(lon_start, lon_end),
        latitude=slice(lat_start, lat_end),
        missing_dims="warn",
    )

    # if invariant, take the first time step
    if invariant:
        var_over_grid = var_over_grid.isel(valid_time=0)
    else:
        # otherwise, select the time range around the storm's timestamp
        if timedelta is None:
            var_over_grid = var_over_grid.sel(
                valid_time=timestamp, method="nearest"
            )
        else:
            var_over_grid = var_over_grid.sel(
                valid_time=slice(timestamp, timestamp + timedelta)
            )

    # get variable values over the grid cells
    var_over_grid = var_over_grid[variable_name].values

    # filter values using variable_bounds if provided
    if variable_bounds is not None:
        lower, upper = variable_bounds
        var_over_grid = var_over_grid[
            (var_over_grid >= lower) & (var_over_grid <= upper)
        ]

    # return the mean over all the grid cells, ignoring NaNs unless all are NaN
    return np.nanmean(var_over_grid)


def calc_spatiotemporal_mean(
    processed_df: pd.DataFrame,
    filename_prefix: str,
    variable_name: str,
    new_col_name: str,
    mask: Optional[xr.DataArray] = None,
    squeeze_dims: list[str] = [],
    radius_km: float = 400,
    timedelta: Optional[pd.Timedelta] = None,
    invariant: bool = False,
    variable_bounds: Optional[tuple[float, float]] = None,
    fillna_val: Optional[float] = None,
    unit_conv_func: Optional[Callable] = None,
) -> pd.DataFrame:
    """
    Calculate the spatial mean of a specified variable from an xarray dataset
    for each storm in the DataFrame.

    :param processed_df: DataFrame containing storm data. Must include 'timestamp', 'lon', and 'lat' columns.
    :param filename_prefix: Prefix for the dataset filenames to load.
    :param variable_name: Name of the variable in the dataset to calculate the spatial mean for
    :param new_col_name: Name of the new column to store the spatial mean values.
    :param mask: Optional mask to apply to the variable values (e.g., land-sea mask).
    :param squeeze_dims: List of dimensions to squeeze from the dataset before processing.
    :param radius_km: Radius in kilometres for the spatial mean calculation.
    :param timedelta: Time delta to consider for the spatial mean calculation.
    :param invariant: If True, the variable is invariant in time (e.g., static data).
    :param variable_bounds: Optional tuple of (lower, upper) bounds to filter the variable values.
    :param fillna_val: Optional value to fill NaN values in the new column.
    :param unit_conv_func: Optional function to convert the units of the spatial mean values
                          (e.g., from Kelvin to Celsius).
    :return: DataFrame with the new column containing the spatial mean values.
    :rtype: pd.DataFrame
    """
    # init the new column for the spatial mean
    processed_df[new_col_name] = np.nan

    # group storm data by year
    grouped = processed_df.groupby(processed_df["timestamp"].dt.year)

    # for each year, calculate the spatial mean
    for year, group in grouped:
        print(f"Processing year: {year}")

        # load the dataset for the year
        dataset = xr.open_dataset(
            config.DATA_DIR / "std" / f"{filename_prefix}{year}.nc"
        )

        # if mask is provided, apply it to the dataset over the entire grid
        if mask is not None:
            dataset = dataset.where(mask)

        # squeeze dimensions if specified
        if len(squeeze_dims) > 0:
            dataset = dataset.squeeze(dim=squeeze_dims)

        # calculate the spatial mean at each point
        processed_df.loc[group.index, new_col_name] = group.parallel_apply(  # type: ignore
            lambda row: calc_spatiotemporal_mean_at_point(
                row["timestamp"],
                row["lon"],
                row["lat"],
                dataset,
                variable_name,
                radius_km=radius_km,
                timedelta=timedelta,
                invariant=invariant,
                variable_bounds=variable_bounds,
            ),
            axis=1,
        )

        # clear the dataset from memory
        dataset.close()

    # fill any remaining NaN values with fillna_val if provided
    if fillna_val is not None:
        processed_df[new_col_name] = processed_df[new_col_name].fillna(
            fillna_val
        )

    # apply the unit conversion function if provided
    if unit_conv_func is not None:
        processed_df[new_col_name] = processed_df[new_col_name].apply(
            unit_conv_func
        )

    return processed_df


def interpolate_storm_to_n_points(group, n_points=10):
    """
    Interpolate a storm track to exactly n points

    :param group: DataFrame group representing a single storm.
    :param n_points: Number of points to interpolate the storm track to.
    :return: DataFrame with interpolated points and their storm stage.
    :rtype: pd.DataFrame
    """
    # ignore storms with less than 2 points
    if len(group) < 2:
        return group

    # create normalized time indices from 0 to 1 (start to end of storm)
    # for both the original and target time
    normalized_time = np.linspace(0, 1, len(group))
    target_time = np.linspace(0, 1, n_points)

    # interpolate each numeric column
    interpolated_data = {}
    for col in group.select_dtypes(include=[np.number]).columns:
        interpolated_data[col] = np.interp(
            target_time, normalized_time, group[col]
        )

    # add back storm_id and create storm stage and storm progress columns
    interpolated_data["storm_id"] = group["storm_id"].iloc[0]
    interpolated_data["storm_stage"] = np.arange(
        n_points
    )  # eg: 0-9 for 10 points
    interpolated_data["storm_progress"] = target_time  # eg: 0.0 to 1.0

    return pd.DataFrame(interpolated_data)


def sample_storm_at_quantiles(group, n_points=10):
    """
    Sample storm at specific quantiles of its duration

    :param group: DataFrame group representing a single storm.
    :param n_points: Number of points to sample from the storm.
    :return: DataFrame with sampled points and their storm stage.
    :rtype: pd.DataFrame
    """
    # ignore storms with less than 2 points
    if len(group) < 2:
        return group

    # calculate quantiles for sampling
    quantiles = np.linspace(0, 1, n_points)
    indices = (quantiles * (len(group) - 1)).astype(int)

    # select the points at the calculated indices
    sampled = group.iloc[indices].copy()

    # create storm stage and storm progress columns
    sampled["storm_stage"] = np.arange(n_points)
    sampled["storm_progress"] = quantiles

    return sampled


def interpolate_all_storms(
    df: pd.DataFrame, n_points: int = 10
) -> pd.DataFrame:
    """
    Interpolate all storms in the DataFrame to have exactly n_points.

    :param df: DataFrame containing storm data with 'storm_id' and 'timestamp' columns.
    :param n_points: Number of points to interpolate each storm to.
    :return: DataFrame with interpolated storm tracks.
    :rtype: pd.DataFrame
    """
    # group by storm_id and apply interpolation
    interpolated_df = (
        df.groupby("storm_id", group_keys=False)
        .parallel_apply(lambda x: interpolate_storm_to_n_points(x, n_points))  # type: ignore
        .reset_index(drop=True)
    )

    return interpolated_df


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
