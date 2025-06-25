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
from scipy.stats import gaussian_kde

import config


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
