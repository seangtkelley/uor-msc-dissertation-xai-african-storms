#!/usr/bin/env python

"""plotting.py: Helper functions for plotting data"""

__author__ = "Sean Kelley"
__copyright__ = "Copyright 2025, University of Reading"
__credits__ = ["Sean Kelley"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Sean Kelley"
__email__ = "s.g.t.kelley@student.reading.ac.uk"
__status__ = "Development"


from typing import Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import cartopy.crs as ccrs
import cartopy.feature as cf
from scipy.stats import gaussian_kde

import config


def plot_kde_map(
    lons: Union[np.ndarray, pd.Series],
    lats: Union[np.ndarray, pd.Series],
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    alpha: float = 1.0,
    add_colorbar: bool = True,
    contour_lines_only: bool = False,
) -> None:
    """Plot a KDE map of the given data.

    :param lons: Array of longitudes.
    :param lats: Array of latitudes.
    :param ax: Matplotlib axis to plot on. If None, a new figure and
        axis will be created.
    :param title: Title for the plot.
    :param alpha: Transparency level for the KDE map.
    :return: The axis with the KDE map plotted.
    """
    # 2D kernel density estimation
    xy = np.vstack([lons, lats])
    kde = gaussian_kde(xy)
    xmin, xmax = config.DATA_EXTENT[0], config.DATA_EXTENT[1]
    ymin, ymax = config.DATA_EXTENT[2], config.DATA_EXTENT[3]
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kde(positions).T, X.shape)

    # normalize Z for consistent color scaling
    Z /= Z.max()

    # setup plot of east africa
    if ax is None:
        plt.figure(figsize=(5, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())

    # ax.set_extent(config.MAP_AREA_EXTENT, crs=ccrs.PlateCarree())
    ax.coastlines(resolution="50m", color="black", linewidth=1)
    ax.add_feature(cf.BORDERS, linewidth=0.5)
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    # add filled contours and contour lines
    ax.contour(
        X,
        Y,
        Z,
        levels=15,
        colors="k",
        linewidths=0.5,
        alpha=0.5 * alpha if not contour_lines_only else 1.0,
    )
    if not contour_lines_only:
        ctf = ax.contourf(X, Y, Z, levels=15, cmap="YlOrBr", alpha=alpha)
        if add_colorbar:
            cbar = plt.colorbar(
                ctf, ax=ax, orientation="horizontal", pad=0.1, aspect=50
            )
            cbar.set_label("Density")

    if title:
        ax.set_title(title)
