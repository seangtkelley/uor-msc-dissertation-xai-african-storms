#!/bin/bash

# load jaspy module for parallel command
module load jaspy

# get current directory of this file
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# build python path
PYTHON_PATH="$HOME/.conda/envs/uor-msc-dissertation-xai-african-storms/bin/python"

# names of experiments to run separated by new lines
EXPERIMENTS="storm_max_intensity_all
storm_max_intensity_era5
storm_direction_all
storm_direction_era5
point_intensification_all
point_intensification_era5
point_next_direction_all
point_next_direction_era5
point_next_distance_all
point_next_distance_era5
point_precipitation_all
point_precipitation_era5"

EXPERIMENT_COUNT=$(echo "$EXPERIMENTS" | wc -l)

# run run_exp.py in parallel for each experiment
echo "$EXPERIMENTS" | parallel -j $EXPERIMENT_COUNT $PYTHON_PATH $DIR/run_exp.py --exp_name {}