#!/bin/bash

# load jaspy module for parallel command
module load jaspy

# get current directory of this file
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# build python path
PYTHON_PATH="$HOME/.conda/envs/uor-msc-dissertation-xai-african-storms/bin/python"

# names of experiments to run separated by new lines
EXPERIMENTS="storm_max_intensity_all
storm_max_intensity_all_first_points
storm_max_intensity_era5
storm_max_intensity_era5_first_points
storm_direction_all
storm_direction_all_first_points
storm_direction_era5
storm_direction_era5_first_points
obs_intensification_all
obs_intensification_era5
obs_next_direction_all
obs_next_direction_era5
obs_next_distance_all
obs_next_distance_era5
obs_precipitation_all
obs_precipitation_era5"

EXPERIMENT_COUNT=$(echo "$EXPERIMENTS" | wc -l)

# run run_exp.py sequentially for each experiment
for exp in $EXPERIMENTS; do
    $PYTHON_PATH $DIR/run_exp.py --exp_name $exp
done
