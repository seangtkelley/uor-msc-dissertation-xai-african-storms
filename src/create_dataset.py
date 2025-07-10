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

import pandas as pd
import xarray as xr

import config
from utils import processing

# parse cli arguments
parser = argparse.ArgumentParser(
    description="Create processed dataset from raw storm database and era5 data"
)
parser.add_argument(
    # if set to False, the script will only recalculate features that are not already present in the dataset
    "--recalc_all",
    action="store_true",
    help="Recalculate all features",
    default=False,
)
args = parser.parse_args()

if args.recalc_all:
    print("Recalculating all features...")
else:
    print(
        "Only recalculating features that are not already present in the dataset..."
    )

# load the raw storm database and rename the columns
raw_df = pd.read_csv(config.RAW_STORM_DB_PATH)
raw_df = processing.rename_columns(raw_df, column_map=config.COL_RENAME_MAP)

# convert the timestamp to a datetime object
raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], utc=True)

# check if the processed dataset already exists
processed_df = None
if not args.recalc_all and os.path.exists(config.PROCESSED_DATASET_PATH):
    print("Loading existing processed dataset...")

    # load the existing processed dataset
    processed_df = pd.read_csv(config.PROCESSED_DATASET_PATH)

    # convert the timestamp to a datetime object
    processed_df["timestamp"] = pd.to_datetime(
        processed_df["timestamp"], utc=True
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
    if not args.recalc_all:
        print(
            "Warning: args.recalc_all was set to False but existing processed dataset does not exist."
        )

    # overwrite arg since the processed dataset does not exist
    args.recalc_all = True

    # processed_df will start as the raw dataframe
    processed_df = raw_df.copy()

# sort by storm_id and timestamp to ensure consistent order before processing
processed_df = processed_df.sort_values(by=["storm_id", "timestamp"])

if args.recalc_all or (
    "orography_height" not in processed_df.columns
    or "anor" not in processed_df.columns
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

if args.recalc_all or "area" not in processed_df.columns:
    print("Calculating storm area...")

    # convert storm area from number of pixels to km^2
    # area is given in pixels. 18 pixels is roughly 350 km^2 (18.7 km x 18.7 km)
    processed_df["area"] = (raw_df["area"] / 18) * (18.7**2)

if args.recalc_all or "storm_max_area" not in processed_df.columns:
    print("Calculating storm maximum area...")

    # calculate the maximum storm area for each storm
    processed_df["storm_max_area"] = processed_df.groupby("storm_id")[
        "area"
    ].transform("max")

if args.recalc_all or (
    "distance_from_prev" not in processed_df.columns
    or "bearing_from_prev" not in processed_df.columns
    or "storm_straight_line_distance" not in processed_df.columns
    or "storm_bearing" not in processed_df.columns
    or "storm_distance_traversed" not in processed_df.columns
):
    print("Calculating storm distances and bearings...")

    # calculate the distance and bearing from the previous point for each storm
    processed_df = processing.calc_storm_distances_and_bearings(processed_df)

# select only the columns that are in the config
processed_df = processed_df[
    [col for col in config.DATASET_COL_NAMES if col in processed_df.columns]
]

# save the processed dataset
print("Saving processed dataset...")
processed_df.to_csv(config.PROCESSED_DATASET_PATH, index=False)
