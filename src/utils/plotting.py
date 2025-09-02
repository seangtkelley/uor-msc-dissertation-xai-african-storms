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


from pathlib import Path
from typing import Optional

import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, ListedColormap
from pint import Quantity

import config

# scope cmap terrain to start at green
TERRAIN_CMAP = ListedColormap(
    plt.get_cmap("terrain")(np.linspace(0.25, 1, plt.get_cmap("terrain").N))
)


def init_map(
    ax: Optional[Axes] = None,
    projection: ccrs.Projection = ccrs.PlateCarree(),
    extent: Optional[tuple[float, float, float, float]] = None,
) -> Axes:
    """
    Initialize a map with the given extent and projection.

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
    """
    Add water features to the given axis.

    :param ax: Matplotlib axis to add the water features to.
    """
    ax.add_feature(  # type: ignore
        cf.NaturalEarthFeature(
            "physical",
            "ocean",
            scale="110m",
            edgecolor="face",
            facecolor=cf.COLORS["water"],
        )
    )
    ax.add_feature(cf.LAKES, edgecolor="face", facecolor=cf.COLORS["water"])  # type: ignore
    ax.add_feature(cf.RIVERS, edgecolor=cf.COLORS["water"])  # type: ignore


def add_borders(
    ax: Axes,
    edgecolor: str = "black",
) -> None:
    """
    Add borders to the given axis.

    :param ax: Matplotlib axis to add the borders to.
    """
    ax.add_feature(cf.COASTLINE, edgecolor=edgecolor, linewidth=1)  # type: ignore
    ax.add_feature(cf.BORDERS, edgecolor=edgecolor, linewidth=0.5)  # type: ignore


def add_gridlines(
    ax: Axes,
    small_labels: bool = False,
) -> None:
    """
    Add gridlines to the given axis.

    :param ax: Matplotlib axis to add the gridlines to.
    :param small_labels: Whether to reduce the font size of the grid labels.
    """
    gl = ax.gridlines(color="lightgray", draw_labels=True)  # type: ignore
    gl.top_labels = False
    gl.right_labels = False

    if small_labels:
        gl.xlabel_style = {"fontsize": 8}
        gl.ylabel_style = {"fontsize": 8}


def add_geopotential_height(
    geop: xr.Dataset, height: Quantity, ax: Axes, add_colorbar: bool = False
) -> None:
    """
    Add geopotential height contours to the given axis.

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
    """
    Add all map features to the given axis.

    :param ax: Matplotlib axis to add the map features to.
    """
    add_water_features(ax)
    add_borders(ax)
    add_gridlines(ax)


def save_plot(
    filename: str,
    directory: Path = config.EXPLORATION_FIGURES_DIR,
    dpi: int = 300,
    tight: bool = True,
    show: bool = False,
) -> None:
    """
    Save the current plot to a file and optionally show it.

    :param filename: Filename to save the plot to.
    :param directory: Directory to save the plot in.
    :param dpi: Dots per inch for the saved figure.
    :param show: Whether to show the plot after saving.
    """
    if tight:
        plt.tight_layout()
    plt.savefig(directory / filename, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_kde_map(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    ax: Optional[Axes] = None,
    alpha: float = 1.0,
    contour_lines_only: bool = False,
    add_colorbar: bool = True,
    colorbar_padding: float = 0.1,
) -> None:
    """
    Plot the result of scipy.stats.gaussian_kde on a map.

    :param X: Meshgrid of longitudes.
    :param Y: Meshgrid of latitudes.
    :param Z: KDE values corresponding to the meshgrid.
    :param ax: Matplotlib axis to plot on. If None, a new figure and axis will be created.
    :param alpha: Transparency level for the KDE map.
    :param contour_lines_only: If True, only plot contour lines without filled contours.
    :param add_colorbar: Whether to add a colorbar to the plot.
    :param colorbar_padding: Padding for the colorbar.
    """
    # init axis if not provided
    if ax is None:
        plt.figure(figsize=(5, 5))
        ax = init_map()

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
                ctf,
                ax=ax,
                orientation="horizontal",
                pad=colorbar_padding,
                aspect=50,
            )
            cbar.set_label("Density")

    # add other map features
    add_borders(ax)
    add_gridlines(ax)


def plot_2d_agg_map(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    ax: Optional[Axes] = None,
    cmap: Optional[str | Colormap] = None,
    sym_cmap_centre: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    add_cbar: bool = True,
    cbar_pad: float = 0.1,
    cbar_aspect: float = 20,
    cbar_shrink: float = 1.0,
    cbar_label: Optional[str] = None,
    cbar_value_labels: Optional[dict] = None,
    small_grid_labels: bool = False,
    title: Optional[str] = None,
    filename: Optional[str] = None,
    save_dir: Optional[Path] = None,
) -> None:
    """
    Plot a 2D aggregated map.

    :param x: 1D array of x-coordinates (e.g., longitudes).
    :param y: 1D array of y-coordinates (e.g., latitudes).
    :param grid: 2D array of aggregated values.
    :param ax: Matplotlib axis to plot on. If None, a new figure and axis will be created.
    :param cmap: Colormap to use for the plot.
    :param sym_cmap_centre: Centre value for symmetrical colormap.
    :param add_cbar: Whether to add a colorbar to the plot.
    :param cbar_pad: Padding for the colorbar.
    :param cbar_aspect: Aspect ratio for the colorbar.
    :param cbar_shrink: Shrink factor for the colorbar.
    :param cbar_label: Label for the colorbar.
    :param cbar_value_labels: Labels for the colorbar max and min values.
    :param small_grid_labels: Whether to make small grid labels.
    :param title: Title for the plot.
    :param filename: Filename to save the plot.
    :param save_dir: Directory to save the plot.
    """
    # init axis if not provided
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = init_map(extent=config.STORM_DATA_EXTENT)

    if sym_cmap_centre is not None:
        # symmetrical cmap centred at specified value
        m = max(
            abs(np.nanmin(grid) - sym_cmap_centre),
            abs(np.nanmax(grid) - sym_cmap_centre),
        )
        vmin, vmax = sym_cmap_centre - m, sym_cmap_centre + m
    elif vmin is not None and vmax is not None:
        # use specified colormap limits
        vmin, vmax = vmin, vmax
    else:
        # use default colormap limits
        vmin, vmax = None, None

    # plot aggregated values as colormesh
    pcolormesh = ax.pcolormesh(
        x,
        y,
        grid,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
    )
    if add_cbar:
        cbar = plt.colorbar(
            pcolormesh,
            ax=ax,
            orientation="horizontal",
            pad=cbar_pad,
            aspect=cbar_aspect,
            shrink=cbar_shrink,
        )
        if cbar_label is not None:
            cbar.set_label(cbar_label)
        else:
            cbar.set_label(f"Aggregated Value")
        if cbar_value_labels is not None:
            cbar.ax.text(
                -0.05,
                0.5,
                cbar_value_labels["negative"],
                va="center",
                ha="right",
                transform=cbar.ax.transAxes,
            )

            cbar.ax.text(
                1.05,
                0.5,
                cbar_value_labels["positive"],
                va="center",
                ha="left",
                transform=cbar.ax.transAxes,
            )

    # add other map features
    add_borders(ax)
    add_gridlines(ax, small_labels=small_grid_labels)

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f"Aggregated Value over Map")

    if filename is not None and save_dir is not None:
        save_plot(filename, save_dir)
    else:
        plt.show()
