#!/usr/bin/env python

"""train_model.py: Script to train model on processed storm dataset given specified parameters"""

__author__ = "Sean Kelley"
__copyright__ = "Copyright 2025, University of Reading"
__credits__ = ["Sean Kelley"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Sean Kelley"
__email__ = "s.g.t.kelley@student.reading.ac.uk"
__status__ = "Development"

import argparse

import xgboost as xgb

import config

# parse cli arguments
parser = argparse.ArgumentParser(
    description="Train model on processed storm dataset given specified parameters"
)
parser.add_argument(
    "--model_type",
    type=str,
    choices=["xgboost"],
    default="xgboost",
    help="Type of model to train",
)
parser.add_argument(
    "--hyperparameter_json",
    type=str,
    default=config.HYPERPARAMETER_JSON_PATH,
    help="Path to JSON file containing hyperparameters for model training",
)
parser.add_argument(
    "--output_model_path",
    type=str,
    default=config.OUTPUT_MODEL_PATH,
    help="Path to save the trained model",
)
args = parser.parse_args()
