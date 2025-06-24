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

import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from metpy.calc import geopotential_to_height
from metpy.units import units

import config

# scope cmap terrain to start at green
TERRAIN_CMAP = ListedColormap(
    plt.get_cmap("terrain")(np.linspace(0.25, 1, plt.get_cmap("terrain").N))
)

geop = xr.open_dataset(config.DATA_DIR / "std" / "geop.nc")
geop_vals = geop["geop"].values.squeeze()

# source: https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.geopotential_to_height.html#metpy.calc.geopotential_to_height
geopot = units.Quantity(geop_vals, "m^2/s^2")
height = geopotential_to_height(geopot)


def plot_kde_map(
    lons: Union[np.ndarray, pd.Series],
    lats: Union[np.ndarray, pd.Series],
    kde: np.ndarray,
    ax: Optional[Axes] = None,
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

    # setup plot of east africa
    if ax is None:
        plt.figure(figsize=(5, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())

    # add filled contours and contour lines
    ax.contour(
        lons,
        lats,
        kde,
        levels=15,
        colors="k",
        linewidths=0.5,
        alpha=0.5 * alpha if not contour_lines_only else 1.0,
    )
    if not contour_lines_only:
        ctf = ax.contourf(
            lons, lats, kde, levels=15, cmap="YlOrBr", alpha=alpha
        )
        if add_colorbar:
            cbar = plt.colorbar(
                ctf, ax=ax, orientation="horizontal", pad=0.1, aspect=50
            )
            cbar.set_label("Density")

    # add other map features
    add_borders(ax)
    add_gridlines(ax)


def init_map(
    ax: Optional[Axes] = None,
    projection: ccrs.Projection = ccrs.PlateCarree(),
    extent: Optional[tuple[float, float, float, float]] = None,
) -> Axes:
    """Initialize a map with the given extent and projection.

    :param extent: Tuple of (lon_min, lon_max, lat_min, lat_max) to set the map extent.
    :param projection: Cartopy projection to use for the map.
    :return: Matplotlib axis with the initialized map.
    """
    if ax is None:
        ax = plt.axes(projection=projection)
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())  # type: ignore
    return ax


def add_water_features(
    ax: Axes,
) -> None:
    """Add water features to the given axis.

    :param ax: Matplotlib axis to add the water features to.
    """
    ax.add_feature(  # type: ignore
        cf.NaturalEarthFeature(
            "physical",
            "ocean",
            scale="110m",
            edgecolor="none",
            facecolor=cf.COLORS["water"],
        )
    )
    ax.add_feature(cf.LAKES, color=cf.COLORS["water"])  # type: ignore
    ax.add_feature(cf.RIVERS, color=cf.COLORS["water"])  # type: ignore


def add_borders(
    ax: Axes,
    color: str = "black",
) -> None:
    """Add borders to the given axis.

    :param ax: Matplotlib axis to add the borders to.
    """
    ax.coastlines(resolution="50m", color=color, linewidth=1)  # type: ignore
    ax.add_feature(cf.BORDERS, color=color, linewidth=0.5)  # type: ignore


def add_gridlines(
    ax: Axes,
) -> None:
    """Add gridlines to the given axis.

    :param ax: Matplotlib axis to add the gridlines to.
    """
    gl = ax.gridlines(draw_labels=True)  # type: ignore
    gl.top_labels = False
    gl.right_labels = False


def add_geopotential_height(ax: Axes, add_colorbar: bool = False) -> None:
    """Add geopotential height contours to the given axis.

    :param ax: Matplotlib axis to add the geopotential height contours to.
    :param add_colorbar: Whether to add a colorbar for the geopotential height.
    """
    terrain = ax.pcolormesh(
        geop["longitude"],
        geop["latitude"],
        height,
        cmap=TERRAIN_CMAP,
        transform=ccrs.PlateCarree(),
    )
    if add_colorbar:
        cbar = plt.colorbar(
            terrain, ax=ax, orientation="horizontal", pad=0.1, aspect=50
        )
        cbar.set_label("Elevation (m)")


def add_all_map_features(
    ax: Axes,
) -> None:
    """Add all map features to the given axis.

    :param ax: Matplotlib axis to add the map features to.
    """
    add_water_features(ax)
    add_borders(ax)
    add_gridlines(ax)


def save_plot(
    filename: str,
    dpi: int = 300,
    show: bool = True,
) -> None:
    """Save the current plot to a file and optionally show it.

    :param filename: Filename to save the plot to.
    :param dpi: Dots per inch for the saved figure.
    :param show: Whether to show the plot after saving.
    """
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / filename, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
