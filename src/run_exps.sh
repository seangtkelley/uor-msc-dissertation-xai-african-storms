#!/bin/bash

# load jaspy module for parallel command
module load jaspy

# get current directory of this file
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# names of experiments to run separated by new lines
EXPERIMENTS="max_intensity_era5
direction_era5
direction_all
max_intensity_all
intensification_era5
intensification_all
next_direction_all
next_direction_era5
next_distance_all
next_distance_era5
precipitation_era5
precipitation_all"

# run run_exp.py in parallel for each experiment
echo "$EXPERIMENTS" | parallel -j 4 $HOME/.conda/envs/uor-msc-dissertation-xai-african-storms/bin/python $DIR/run_exp.py --exp_name {}