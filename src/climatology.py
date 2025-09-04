from typing import Optional
import xarray as xr
import config
from utils import plotting
import matplotlib.pyplot as plt
import numpy as np

years = [2014, 2015, 2016, 2017, 2018, 2019]


def load_climatology_data(
    filename: str, squeeze: Optional[bool] = True
) -> xr.Dataset:
    dataset = xr.open_dataset(
        config.DATA_DIR / "std" / f"{filename}{years[0]}.nc"
    )
    for year in years[1:]:
        next_dataset = xr.open_dataset(
            config.DATA_DIR / "std" / f"{filename}{year}.nc"
        )
        dataset = xr.concat([dataset, next_dataset], dim="valid_time")
    if squeeze:
        dataset = dataset.squeeze("pressure_level")
    dataset = dataset.sortby("valid_time")
    dataset = dataset.mean(dim="valid_time")
    return dataset


print("Loading data...")
dataset_uwind = load_climatology_data("uwnd_850_")
dataset_vwind = load_climatology_data("vwnd_850_")
dataset_shum = load_climatology_data("shum_850_")
dataset_swvl = load_climatology_data("swvl1_d1_", squeeze=False)
dataset_skt = load_climatology_data("skt_sfc_", squeeze=False)

# Plot humidity
print("Plotting mean humidity...")
ax = plotting.init_map()
dataset_shum["shum"].plot.contourf(
    ax=ax,
    levels=10,
    alpha=0.5,
    cmap="Blues",
    cbar_kwargs={
        "label": "Mean Specific Humidity (kg/kg)",
        "location": "bottom",
        "shrink": 0.8,
        "orientation": "horizontal",
    },
)
plotting.add_gridlines(ax)
plotting.add_borders(ax)
ax.set_title("")
plotting.save_plot("mean_shum_850.png", show=True)

# Plot skin surface temperature
print("Plotting skin surface temperature...")
ax = plotting.init_map()
dataset_skt["skt"].plot.contourf(
    ax=ax,
    levels=10,
    alpha=0.5,
    cmap="YlOrRd_r",
    cbar_kwargs={
        "label": "Mean Skin Surface Temperature (°K)",
        "location": "bottom",
        "shrink": 0.8,
        "orientation": "horizontal",
    },
)
plotting.add_gridlines(ax)
plotting.add_borders(ax)
ax.set_title("")
plotting.save_plot("mean_skt.png", show=True)

# Plot mean soil water volume
print("Plotting mean soil water volume...")
ax = plotting.init_map()
dataset_swvl["swvl1"].plot.contourf(
    ax=ax,
    levels=10,
    alpha=0.5,
    cmap="Blues",
    cbar_kwargs={
        "label": "Mean Soil Water Volume (m³/m³)",
        "location": "bottom",
        "shrink": 0.8,
        "orientation": "horizontal",
    },
)
plotting.add_gridlines(ax)
plotting.add_borders(ax)
ax.set_title("")
plotting.save_plot("mean_swvl1.png", show=True)


# Plot wind
print("Plotting mean wind speed...")
step = 3
ax = plotting.init_map()
wind_lon = dataset_uwind["longitude"][::step]
wind_lat = dataset_uwind["latitude"][::step]
uwind = dataset_uwind["uwnd"][::step, ::step]
vwind = dataset_vwind["vwnd"][::step, ::step]

wind_speed = np.sqrt(uwind**2 + vwind**2)
Q = plt.quiver(
    wind_lon,
    wind_lat,
    uwind,
    vwind,
    wind_speed,
    cmap="plasma",
)
plt.title("    ")
plt.quiverkey(Q, 0.85, 1.04, 10, r"$m/s$", labelpos="E")
plt.colorbar(Q, label="Wind Speed (m/s)", orientation="horizontal", shrink=0.8)
plotting.add_gridlines(ax)
plotting.add_borders(ax)
plotting.save_plot("mean_wind.png", show=True)

# Plot humidity and wind
print("Plotting mean humidity and wind speed...")
plt.figure(figsize=(10, 8))
ax = plotting.init_map()
dataset_shum["shum"].plot.contourf(
    ax=ax,
    levels=10,
    alpha=0.5,
    cmap="Blues",
    cbar_kwargs={
        "label": "Mean Specific Humidity (kg/kg)",
        "location": "left",
        "shrink": 0.5,
        "orientation": "vertical",
    },
)
Q = plt.quiver(
    wind_lon,
    wind_lat,
    uwind,
    vwind,
    wind_speed,
    cmap="plasma",
)
plt.title("    ")
plt.quiverkey(Q, 1.05, 1.05, 5, r"5 m/s", labelpos="E")
plt.colorbar(
    Q,
    label="Wind Speed (m/s)",
    orientation="vertical",
    shrink=0.5,
    location="right",
)

plotting.add_gridlines(ax, small_labels=True)
plotting.add_borders(ax)
plotting.save_plot("mean_humidity_wind.png", show=True)
