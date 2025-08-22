#!/usr/bin/env python

"""create_dataset.py: Script to create a processed dataset from raw storm database and era5 data"""

__author__ = "Sean Kelley"
__copyright__ = "Copyright 2025, University of Reading"
__credits__ = ["Sean Kelley"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Sean Kelley"
__email__ = "s.g.t.kelley@student.reading.ac.uk"
__status__ = "Development"

import argparse
import os
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

import config
from utils import processing

# parse cli arguments
parser = argparse.ArgumentParser(
    description="Create processed dataset from raw storm database and era5 data"
)
parser.add_argument(
    "--recalc",
    type=str,
    help="Features to recalculate, separated by commas. If not specified, only features that are not already present in the dataset will be recalculated.",
)
args = parser.parse_args()

RECALC_FEATURES = args.recalc.split(",") if args.recalc is not None else []
RECALC_ALL = args.recalc == "all"

if RECALC_ALL:
    print("Recalculating all features...")
elif len(RECALC_FEATURES) > 0:
    print("Recalculating features:", RECALC_FEATURES, "...")
else:
    print(
        "Only recalculating features that are not already present in the dataset..."
    )


def should_recalc(
    column_names: str | Iterable[str], df_cols: Iterable[str]
) -> bool:
    """
    Check if columns should be recalculated.

    :param column_names: Column names to check.
    :param df_cols: List of columns in the processed dataframe.
    :return: True if the column should be recalculated, False otherwise.
    """
    if isinstance(column_names, str):
        column_names = [column_names]

    return RECALC_ALL or any(
        [
            column_name in RECALC_FEATURES or column_name not in df_cols
            for column_name in column_names
        ]
    )


# load the raw storm database and rename the columns
raw_df = pd.read_csv(config.RAW_STORM_DB_PATH, parse_dates=["timestamp"])
raw_df = processing.rename_columns(raw_df, column_map=config.COL_RENAME_MAP)

# check if the processed dataset already exists
processed_df = None
if not RECALC_ALL and os.path.exists(config.PROCESSED_DATASET_PATH):
    print("Loading existing processed dataset...")

    # load the existing processed dataset
    processed_df = pd.read_csv(
        config.PROCESSED_DATASET_PATH, parse_dates=["timestamp"]
    )

    # get columns from raw_df that aren't in processed_df (excluding merge keys)
    merge_keys = ["storm_id", "timestamp"]
    raw_cols_to_add = [
        col
        for col in raw_df.columns
        if col not in processed_df.columns and col not in merge_keys
    ]

    # select only the merge keys and new columns from raw_df
    raw_df_subset = raw_df[merge_keys + raw_cols_to_add]

    # merge with processed_df
    processed_df = processed_df.merge(raw_df_subset, on=merge_keys, how="left")
else:
    print("Creating processed dataset from scratch...")
    if not RECALC_ALL:
        print(
            "Warning: args.recalc was not set but existing processed dataset does not exist."
        )

    # overwrite arg since the processed dataset does not exist
    RECALC_ALL = True

    # processed_df will start as the raw dataframe
    processed_df = raw_df.copy()

# sort by storm_id and timestamp to ensure consistent order before processing
processed_df = processed_df.sort_values(by=["storm_id", "timestamp"])

if should_recalc("date_angle", processed_df.columns):
    print("Calculating date angle...")

    day_of_year = processed_df["timestamp"].dt.dayofyear
    days_in_year = (
        processed_df["timestamp"]
        .dt.is_leap_year.replace({True: 366, False: 365})
        .infer_objects(copy=False)  # type: ignore
    )
    processed_df["date_angle"] = (day_of_year / days_in_year) * 360

if should_recalc(
    ["orography_height", "anor", "upslope_bearing", "slope_angle"],
    processed_df.columns,
):
    print("Calculating orography features...")

    # load geopotential data and convert to height
    geop, height = processing.load_geop_and_calc_elevation()

    # load subgrid orography angle data
    anor = xr.open_dataset(config.DATA_DIR / "std" / "anor.nc")

    # add orography features to the dataframe
    processed_df = processing.get_orography_features(
        processed_df, geop, height, anor
    )

    # close datasets
    geop.close()
    anor.close()

if should_recalc("area", processed_df.columns):
    print("Calculating storm area...")

    # convert storm area from number of pixels to km^2
    # area is given in pixels. 18 pixels is roughly 350 km^2 (18.7 km x 18.7 km)
    processed_df["area"] = (raw_df["area"] / 18) * (18.7**2)

if should_recalc("storm_max_area", processed_df.columns):
    print("Calculating storm maximum area...")

    # calculate the maximum storm area for each storm
    processed_df["storm_max_area"] = processed_df.groupby("storm_id")[
        "area"
    ].transform("max")

if should_recalc(
    ["over_land", "acc_land_time", "storm_total_land_time", "mean_land_frac"],
    processed_df.columns,
):
    print("Calculating land mask features...")

    # load the land mask dataset
    lsm = xr.open_dataset(config.DATA_DIR / "std" / "lsm.nc")

    # calculate over land features
    processed_df = processing.calc_over_land_features(processed_df, lsm)

    # calculate the land fraction
    processed_df["mean_land_frac"] = np.nan
    processed_df["mean_land_frac"] = processed_df.parallel_apply(  # type: ignore
        lambda row: processing.calc_spatiotemporal_agg_at_point(
            row["timestamp"], row["lon"], row["lat"], lsm, "lsm", invariant=True
        ),
        axis=1,
    )

    # close dataset
    lsm.close()

if should_recalc(
    [
        "distance_from_prev",
        "distance_to_next",
        "bearing_from_prev",
        "bearing_to_next",
        "storm_straight_line_distance",
        "storm_bearing",
        "storm_distance_traversed",
    ],
    processed_df.columns,
):
    print("Calculating storm distances and bearings...")

    # calculate the distance and bearing from the previous point for each storm
    processed_df = processing.calc_storm_distances_and_bearings(processed_df)

if should_recalc(
    ["storm_min_bt", "storm_min_bt_reached"], processed_df.columns
):
    print("Calculating storm minimum cloudtop brightness...")

    # calculate the minimum cloudtop brightness for each storm
    processed_df["storm_min_bt"] = processed_df.groupby("storm_id")[
        "min_bt"
    ].transform("min")

    # check if the storm minimum cloudtop brightness was reached
    processed_df["storm_min_bt_reached"] = None
    min_indices = processed_df[
        processed_df["min_bt"] == processed_df["storm_min_bt"]
    ].index
    processed_df.loc[min_indices, "storm_min_bt_reached"] = True

    # set storm_min_bt_reached to False for storm initial points if the storm
    # minimum cloudtop brightness was not reached at the initial point
    storm_init_indices = processed_df.groupby("storm_id").head(1).index
    processed_df.loc[
        [idx for idx in storm_init_indices if idx not in min_indices],
        "storm_min_bt_reached",
    ] = False

    # forward fill the remaining values
    processed_df["storm_min_bt_reached"] = (
        processed_df["storm_min_bt_reached"].astype(bool).ffill()
    )

if should_recalc("dmin_bt_dt", processed_df.columns):
    print("Calculating the rate of change of minimum cloudtop brightness...")

    processed_df = processing.calc_temporal_rate_of_change(
        processed_df, "min_bt", pd.Timedelta(days=1)
    )

if should_recalc("dmean_bt_dt", processed_df.columns):
    print("Calculating the rate of change of mean cloudtop brightness...")

    processed_df = processing.calc_temporal_rate_of_change(
        processed_df, "mean_bt", pd.Timedelta(days=1)
    )

if should_recalc("mean_prcp_400", processed_df.columns):
    print("Calculating mean precipitation...")

    processed_df = processing.calc_spatiotemporal_agg(
        processed_df,
        "prcp_tot_",
        "prcp",
        "mean_prcp_400",
        timedelta=pd.Timedelta(hours=6),
        fillna_val=0.0,
        unit_conv_func=lambda x: x * 1000.0,
    )

if should_recalc("mean_land_skt", processed_df.columns):
    print("Calculating mean land skin temperature...")

    # load the land sea mask
    lsm = xr.open_dataset(config.DATA_DIR / "std" / "lsm.nc")
    land_mask = (
        lsm["lsm"]
        .isel(valid_time=0)
        .squeeze()
        .drop_vars("valid_time")
        .round()
        .astype(bool)
    )

    processed_df = processing.calc_spatiotemporal_agg(
        processed_df,
        "skt_sfc_",
        "skt",
        "mean_land_skt",
        mask=land_mask,
        variable_bounds=config.EARTH_TEMP_BOUNDS,
    )

    # close dataset
    lsm.close()

if should_recalc("mean_sst", processed_df.columns):
    print("Calculating mean sea surface temperature...")

    # load the land sea mask
    lsm = xr.open_dataset(config.DATA_DIR / "std" / "lsm.nc")
    land_mask = (
        lsm["lsm"]
        .isel(valid_time=0)
        .squeeze()
        .drop_vars("valid_time")
        .round()
        .astype(bool)
    )

    processed_df = processing.calc_spatiotemporal_agg(
        processed_df,
        "sst_sfc_",
        "sst",
        "mean_sst",
        mask=~land_mask,  # invert mask for sea surface temperature
        variable_bounds=config.EARTH_TEMP_BOUNDS,
    )

    # close dataset
    lsm.close()

if should_recalc("mean_skt", processed_df.columns):
    print("Calculating mean skin temperature...")

    # if mean_land_skt or mean_sst is nan, use the other, otherwise calculate the mean of the two
    processed_df["mean_skt"] = np.where(
        processed_df["mean_land_skt"].isna(),
        processed_df["mean_sst"],
        np.where(
            processed_df["mean_sst"].isna(),
            processed_df["mean_land_skt"],
            (processed_df["mean_land_skt"] + processed_df["mean_sst"]) / 2,
        ),
    )

pressure_levels = [200, 500, 850]
for level in pressure_levels:
    if should_recalc(f"mean_u{level}", processed_df.columns):
        print(f"Calculating mean zonal wind speed at {level} hPa...")

        processed_df = processing.calc_spatiotemporal_agg(
            processed_df,
            f"uwnd_{level}_",
            "uwnd",
            f"mean_u{level}",
            squeeze_dims=["pressure_level"],
            fillna_val=0.0,
        )

for level in pressure_levels:
    if should_recalc(f"mean_v{level}", processed_df.columns):
        print(f"Calculating mean meridional wind speed at {level} hPa...")

        processed_df = processing.calc_spatiotemporal_agg(
            processed_df,
            f"vwnd_{level}_",
            "vwnd",
            f"mean_v{level}",
            squeeze_dims=["pressure_level"],
            fillna_val=0.0,
        )

if should_recalc("mean_swvl1", processed_df.columns):
    print("Calculating mean volumetric soil moisture layer 1...")

    processed_df = processing.calc_spatiotemporal_agg(
        processed_df,
        "swvl1_d1_",
        "swvl1",
        "mean_swvl1",
    )

if should_recalc("mean_swvl2", processed_df.columns):
    print("Calculating mean volumetric soil moisture layer 2...")

    processed_df = processing.calc_spatiotemporal_agg(
        processed_df,
        "swvl2_d2_",
        "swvl2",
        "mean_swvl2",
    )

for level in pressure_levels:
    if should_recalc(f"mean_q_{level}", processed_df.columns):
        print(f"Calculating mean specific humidity at {level} hPa...")

        processed_df = processing.calc_spatiotemporal_agg(
            processed_df,
            f"shum_{level}_",
            "shum",
            f"mean_q_{level}",
            squeeze_dims=["pressure_level"],
        )

if should_recalc("mean_cape", processed_df.columns):
    print("Calculating mean convective available potential energy (CAPE)...")

    processed_df = processing.calc_spatiotemporal_agg(
        processed_df,
        "cape_0_",
        "cape",
        "mean_cape",
    )

olr_percents = [90, 75, 50]
for percent in olr_percents:
    if should_recalc(f"olr_{percent}", processed_df.columns):
        print(f"Calculating {percent}th percentile of negative OLR...")

        processed_df = processing.calc_spatiotemporal_agg(
            processed_df,
            f"olr_toa_",
            "olr",
            f"olr_{percent}",
            agg_func=lambda x: np.nanpercentile(x, percent),
        )

if should_recalc("wind_direction_850", processed_df.columns):
    print("Calculating wind direction...")

    processed_df = processing.calc_wind_direction(processed_df)

if should_recalc("wind_angle_upslope", processed_df.columns):
    print("Calculating wind direction relative to upslope direction...")

    # first, rotate the wind direction to point to where the wind is going to
    # as wind_direction_850 is defined as the direction the wind is coming from
    # then, rotate the wind direction to be relative to the upslope direction
    # e.g. wind is going upslope: wind_angle_upslope = 0
    # e.g. wind is going downslope: wind_angle_upslope = 180
    # e.g. wind is going cross-slope: wind_angle_upslope = 90
    # finally, ensure the direction is in the range [0, 360)
    processed_df["wind_angle_upslope"] = (
        ((processed_df["wind_direction_850"] + 180) % 360)
        - processed_df["upslope_bearing"]
    ) % 360

if should_recalc("mean_tcwv", processed_df.columns):
    print("Calculating mean total column water vapour (TCWV)...")

    processed_df = processing.calc_spatiotemporal_agg(
        processed_df,
        "tcwv_tot_",
        "tcwv",
        "mean_tcwv",
    )

if should_recalc(
    [
        "mean_u_shear_850_500",
        "mean_v_shear_850_500",
        "mean_u_shear_850_200",
        "mean_v_shear_850_200",
    ],
    processed_df.columns,
):
    print("Calculating vertical wind shear...")

    processed_df = processing.calc_vertical_wind_shear(processed_df)

if should_recalc("domain_mean_u500", processed_df.columns):
    print("Calculating domain mean 500 hPa zonal wind...")

    processed_df = processing.calc_spatiotemporal_agg(
        processed_df,
        "uwnd_500_",
        "uwnd",
        "domain_mean_u500",
        radius_km=np.inf,
        squeeze_dims=["pressure_level"],
    )

if should_recalc("domain_mean_tcwv", processed_df.columns):
    print("Calculating domain mean total column water vapour (TCWV)...")

    processed_df = processing.calc_spatiotemporal_agg(
        processed_df, "tcwv_tot_", "tcwv", "domain_mean_tcwv", radius_km=np.inf
    )

if should_recalc("domain_mean_cape", processed_df.columns):
    print(
        "Calculating domain mean convective available potential energy (CAPE)..."
    )

    processed_df = processing.calc_spatiotemporal_agg(
        processed_df, "cape_0_", "cape", "domain_mean_cape", radius_km=np.inf
    )

# select only the columns that are in the config
processed_df = processed_df[
    [col for col in config.DATASET_COLS if col in processed_df.columns]
]

# save the processed dataset
print("Saving processed dataset...")
processed_df.to_csv(config.PROCESSED_DATASET_PATH, index=False)
