# %%
import warnings

import cartopy.crs as ccrs
import eofs
import kaleido
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import xarray as xr
from scipy.stats import circmean

import config
from utils import plotting, processing

# %%
warnings.filterwarnings("ignore", message=".*Downloading.*", module="cartopy")

# %%
sns.set_theme(style="darkgrid")

# %%
kaleido.get_chrome_sync()

# %%
print("Loading processed dataset and displaying first few rows.")
df = pd.read_csv(config.PROCESSED_DATASET_PATH, parse_dates=["timestamp"])
df.head()

# %%
print("Grouping storms by storm_id and getting initial points.")
storm_groups = df.groupby("storm_id")
storm_inits = storm_groups.head(1)

# %%
print("Plotting histogram of storm initial locations.")
plt.figure(figsize=(10, 6))
ax = plotting.init_map()
plotting.add_borders(ax, edgecolor="white")
plotting.add_gridlines(ax)

# https://www.julienphalip.com/blog/drawing-geographical-density-maps-with-matplotlib/
plt.hist2d(storm_inits["lon"], storm_inits["lat"], bins=50, cmap="viridis")
plt.colorbar(label="Storm Count")

plt.title("Storm Initial Locations Histogram")
plotting.save_plot("storm_init_hist2d.png")

# %%
print("Performing 2D kernel density estimation for storm initial locations.")
X, Y, Z = processing.calc_kde(storm_inits["lon"], storm_inits["lat"])

plt.figure(figsize=(10, 10))
ax = plotting.init_map()

# add filled contours and contour lines
plotting.plot_kde_map(X, Y, Z, ax=ax, colorbar_padding=0.05)

ax.set_title("Storm Initial Locations")
plotting.save_plot("storm_init_kde.png")

# %%
print("Loading geopotential height and calculating elevation.")
geop, height = processing.load_geop_and_calc_elevation()

# %%
print("Plotting orography, storm initial, and end locations KDE.")
fig, axs = plt.subplots(
    1, 3, figsize=(15, 5), subplot_kw={"projection": ccrs.PlateCarree()}
)

# plot the orography
plotting.init_map(axs[0], extent=config.STORM_DATA_EXTENT)
plotting.add_geopotential_height(geop, height, axs[0], add_colorbar=True)
plotting.add_all_map_features(axs[0])
axs[0].set_title("a) Orography (Geopotential Height)")

# plot storm initial locations
storm_inits_x, storm_inits_y, storm_inits_kde = processing.calc_kde(
    storm_inits["lon"], storm_inits["lat"]
)
plotting.init_map(axs[1], extent=config.STORM_DATA_EXTENT)
plotting.plot_kde_map(storm_inits_x, storm_inits_y, storm_inits_kde, ax=axs[1])
axs[1].set_title("b) Storm Initial Locations")

# plot storm end locations
storm_ends = storm_groups.tail(1)
storm_ends_x, storm_ends_y, storm_ends_kde = processing.calc_kde(
    storm_ends["lon"], storm_ends["lat"]
)
plotting.init_map(axs[2], extent=config.STORM_DATA_EXTENT)
plotting.plot_kde_map(storm_ends_x, storm_ends_y, storm_ends_kde, ax=axs[2])
axs[2].set_title("c) Storm End Locations")

plotting.save_plot("orography_storm_init_end_kde.png")

# %%
print("Plotting storm initial location density contours with orography.")
plt.figure(figsize=(10, 6))
ax = plotting.init_map(extent=config.STORM_DATA_EXTENT)
plotting.add_geopotential_height(geop, height, ax)
plotting.add_all_map_features(ax)

# plot storm initial locations kde contours
plotting.plot_kde_map(
    storm_inits_x,
    storm_inits_y,
    storm_inits_kde,
    ax=ax,
    contour_lines_only=True,
)

ax.set_title("Storm Initial Location Density Contours with Orography")
plotting.save_plot("orography_storm_init_locations.png")

# %%
print("Loading angle of sub-gridscale orography dataset.")
anor = xr.open_dataset(config.DATA_DIR / "std" / "anor.nc")

# %%
print("Plotting angle of sub-gridscale orography and storm locations KDE.")
fig, axs = plt.subplots(
    1, 3, figsize=(15, 5), subplot_kw={"projection": ccrs.PlateCarree()}
)

# plot the orography
plotting.init_map(axs[0], extent=config.STORM_DATA_EXTENT)
terrain = axs[0].pcolormesh(
    anor["longitude"],
    anor["latitude"],
    anor["anor"][0],
    cmap="plasma",
    transform=ccrs.PlateCarree(),
)
cbar = plt.colorbar(
    terrain, ax=axs[0], orientation="horizontal", pad=0.1, aspect=50
)
cbar.set_label("Angle (radians)")
plotting.add_all_map_features(axs[0])
axs[0].set_title("a) Angle of sub-gridscale orography")

# plot storm initial locations
storm_inits_x, storm_inits_y, storm_inits_kde = processing.calc_kde(
    storm_inits["lon"], storm_inits["lat"]
)
plotting.init_map(axs[1], extent=config.STORM_DATA_EXTENT)
plotting.plot_kde_map(storm_inits_x, storm_inits_y, storm_inits_kde, ax=axs[1])
axs[1].set_title("b) Storm Initial Locations")

# plot storm end locations
storm_ends_x, storm_ends_y, storm_ends_kde = processing.calc_kde(
    storm_ends["lon"], storm_ends["lat"]
)
plotting.init_map(axs[2], extent=config.STORM_DATA_EXTENT)
plotting.plot_kde_map(storm_ends_x, storm_ends_y, storm_ends_kde, ax=axs[2])
axs[2].set_title("c) Storm End Locations")

plotting.save_plot("anor_storm_init_end_kde.png")

# %%
print("Binning storm initial locations by latitude and longitude.")
eat_mode_lon, eat_mode_lat, eat_mode_grid = processing.calc_2d_agg(
    storm_inits, "eat_hours", agg_func=lambda x: pd.Series.mode(x).iloc[0]
)
eat_mean_lon, eat_mean_lat, eat_mean_grid = processing.calc_2d_agg(
    storm_inits,
    "eat_hours",
    # use circular mean as distance between 23:59h and 00:00h is only 1 minute, not 24 hours
    agg_func=lambda x: circmean(x, high=24),
)

# %%
print("Plotting storm initial locations with mode of EAT hours.")
plotting.plot_2d_agg_map(
    eat_mode_lon,
    eat_mode_lat,
    eat_mode_grid,
    cmap="twilight_shifted",
    cbar_aspect=40,
    cbar_shrink=0.63,
    cbar_label="Time of Day (EAT UTC+3)",
    title="Storm Initial Locations with Mode of EAT Hours",
    filename="storm_init_eat_hours_mode_by_loc.png",
    save_dir=config.EXPLORATION_FIGURES_DIR,
)

# %%
print("Plotting orography, storm initial EAT hour mode, and mean.")
fig, axs = plt.subplots(
    1, 3, figsize=(15, 5), subplot_kw={"projection": ccrs.PlateCarree()}
)

# plot the orography
plotting.init_map(axs[0], extent=config.STORM_DATA_EXTENT)
plotting.add_geopotential_height(geop, height, axs[0], add_colorbar=True)
plotting.add_all_map_features(axs[0])
axs[0].set_title("a) Orography (Geopotential Height)")

# plot eat hours mode
plotting.plot_2d_agg_map(
    eat_mode_lon,
    eat_mode_lat,
    eat_mode_grid,
    ax=axs[1],
    cmap="twilight_shifted",
    cbar_aspect=50,
    cbar_label="Time of Day (EAT UTC+3)",
    title="b) Storm Init EAT Hours Mode",
)

# plot eat hours mean
plotting.plot_2d_agg_map(
    eat_mean_lon,
    eat_mean_lat,
    eat_mean_grid,
    ax=axs[2],
    cmap="twilight_shifted",
    cbar_aspect=50,
    cbar_label="Time of Day (EAT UTC+3)",
    title="c) Storm Init EAT Hours Mean",
)

plotting.save_plot("orography_storm_init_eat_hours_mode_mean.png")

# %%
print(
    "Plotting storm tracks for 98th percentile duration storms with orography."
)
plt.figure(figsize=(10, 6))
ax = plotting.init_map(extent=config.STORM_DATA_EXTENT)
plotting.add_geopotential_height(geop, height, ax)
plotting.add_all_map_features(ax)

duration_thres = df["storm_total_duration"].quantile(0.98)
p98_duration_df = df[df["storm_total_duration"] >= duration_thres]
for storm, group in p98_duration_df.groupby("storm_id"):
    ax.plot(
        group["lon"],
        group["lat"],
        color="red",
        linewidth=0.5,
        transform=ccrs.PlateCarree(),
    )

ax.set_title("98% Percentile Duration Storm Tracks with Orography")
plotting.save_plot("orography_storm_tracks_p98_duration.png")

# %%
print(
    "Performing KDE for initial locations of 98th percentile duration storms."
)
p98_duration_inits = (
    p98_duration_df.sort_values(["timestamp"]).groupby("storm_id").first()
)
X, Y, Z = processing.calc_kde(
    p98_duration_inits["lon"], p98_duration_inits["lat"]
)

plt.figure(figsize=(10, 10))
ax = plotting.init_map()

# add filled contours and contour lines
plotting.plot_kde_map(X, Y, Z, ax=ax, colorbar_padding=0.05)

ax.set_title("Storm Initial Locations for 98% Percentile Duration Storms")
plotting.save_plot("p98_duration_storm_init_kde.png")

# %%
print("Plotting storm tracks for 98th percentile area storms with orography.")
plt.figure(figsize=(10, 6))
ax = plotting.init_map(extent=config.STORM_DATA_EXTENT)
plotting.add_geopotential_height(geop, height, ax)
plotting.add_all_map_features(ax)

area_thres = df["area"].quantile(0.98)
p98_area_df = df[df["area"] >= area_thres]
for storm, group in p98_area_df.groupby("storm_id"):
    ax.plot(
        group["lon"],
        group["lat"],
        color="red",
        linewidth=0.5,
        transform=ccrs.PlateCarree(),
    )

ax.set_title("98% Percentile Area Storm Tracks with Orography")
plotting.save_plot("orography_storm_tracks_p98_area.png")

# %%
print("Performing KDE for initial locations of 98th percentile area storms.")
p98_area_inits = (
    p98_area_df.sort_values(["timestamp"]).groupby("storm_id").first()
)
X, Y, Z = processing.calc_kde(p98_area_inits["lon"], p98_area_inits["lat"])

plt.figure(figsize=(10, 10))
ax = plotting.init_map()

# add filled contours and contour lines
plotting.plot_kde_map(X, Y, Z, ax=ax, colorbar_padding=0.05)

ax.set_title("Storm Initial Locations for 98% Percentile Area Storms")
plotting.save_plot("p98_area_storm_init_kde.png")

# %%
print("Plotting histogram of storm area with percentile lines.")
plt.figure(figsize=(10, 6))
plt.hist(df["area"], bins=50)
plt.yscale("log")

# plot vertical lines for 75, 90, and 98th percentiles
plt.axvline(
    df["area"].quantile(0.75),
    color="orange",
    linestyle="--",
    label="75th Percentile",
)
plt.axvline(
    df["area"].quantile(0.90),
    color="red",
    linestyle="--",
    label="90th Percentile",
)
plt.axvline(
    df["area"].quantile(0.98),
    color="purple",
    linestyle="--",
    label="98th Percentile",
)
plt.legend()

plt.xlabel("Area (km²)")
plt.ylabel("Frequency")
plt.title("Storm Area Histogram")
plotting.save_plot("storm_area_hist.png")

# %%
print("Plotting histogram of zonal speed for 98th percentile area storms.")
plt.figure(figsize=(10, 6))
plt.hist(p98_area_df["zonal_speed"], bins=50)
plt.xlabel("Zonal Speed (m/s)")
plt.ylabel("Frequency")
plt.title("Storm Zonal Speed Histogram for 98% Percentile Area Storms")
plotting.save_plot("p98_area_storm_zonal_speed_hist.png")

# %%
print("Defining mapping from degrees to cardinal directions.")
degrees_to_cardinal_map = {
    90: "E",
    67.5: "ENE",
    45: "NE",
    22.5: "NNE",
    0: "N",
    337.5: "NNW",
    315: "NW",
    292.5: "WNW",
    270: "W",
    247.5: "WSW",
    225: "SW",
    202.5: "SSW",
    180: "S",
    157.5: "SSE",
    135: "SE",
    112.5: "ESE",
}
cardinal_directions = list(degrees_to_cardinal_map.values())

# %%
print("Converting storm bearing to closest cardinal direction.")
df["storm_closest_cardinal_direction"] = (
    ((df["storm_bearing"] % 360) + (22.5 / 2)).floordiv(22.5) * 22.5 % 360
)
df["storm_closest_cardinal_direction"] = (
    df["storm_closest_cardinal_direction"]
    .map(degrees_to_cardinal_map)
    .astype("category")
)

# %%
print("Plotting polar chart of storm cardinal directions distribution.")
fig = go.Figure()

storm_inits = df.iloc[storm_inits.index]
fig.add_trace(
    go.Scatterpolar(
        r=storm_inits["storm_closest_cardinal_direction"].value_counts(
            normalize=True
        )[cardinal_directions],
        theta=cardinal_directions,
        fill="toself",
        name="All Storms",
    )
)

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
        )
    ),
)

fig.write_image(
    config.EXPLORATION_FIGURES_DIR
    / "storm_cardinal_directions_distribution.png"
)

# %%
print(
    "Plotting polar chart of storm cardinal directions distribution for different storm categories."
)
fig = go.Figure()

fig.add_trace(
    go.Scatterpolar(
        r=storm_inits["storm_closest_cardinal_direction"].value_counts(
            normalize=True
        )[cardinal_directions],
        theta=cardinal_directions,
        fill="toself",
        name="All Storms",
    )
)
p98_area_df = storm_inits[storm_inits["storm_max_area"] >= area_thres]
fig.add_trace(
    go.Scatterpolar(
        r=p98_area_df["storm_closest_cardinal_direction"].value_counts(
            normalize=True
        )[cardinal_directions],
        theta=cardinal_directions,
        fill="toself",
        name="98% Percentile Max Area Storms",
    )
)
p98_duration_df = storm_inits[
    storm_inits["storm_total_duration"] >= duration_thres
]
fig.add_trace(
    go.Scatterpolar(
        r=p98_duration_df["storm_closest_cardinal_direction"].value_counts(
            normalize=True
        )[cardinal_directions],
        theta=cardinal_directions,
        fill="toself",
        name="98% Percentile Duration Storms",
    )
)
p98_min_bt_df = storm_inits[
    storm_inits["storm_min_bt"] <= storm_inits["storm_min_bt"].quantile(0.02)
]
fig.add_trace(
    go.Scatterpolar(
        r=p98_min_bt_df["storm_closest_cardinal_direction"].value_counts(
            normalize=True
        )[cardinal_directions],
        theta=cardinal_directions,
        fill="toself",
        name="98% Percentile Min BT Storms",
    )
)

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
        )
    ),
    showlegend=True,
)

fig.write_image(
    config.EXPLORATION_FIGURES_DIR
    / "storm_categories_cardinal_directions_distribution.png"
)

# %%
print("Plotting histogram of storm total duration.")
plt.figure(figsize=(10, 6))
plt.hist(df["storm_total_duration"], bins=50)

plt.xlabel("Duration (hours)")
plt.ylabel("Frequency")
plt.title("Storm Duration Histogram")
plotting.save_plot("storm_duration_hist.png")

# %%
print(
    "Plotting subplots comparing storm straight line and traversed distances."
)
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# 98th percentile duration storms
axs[0].hist(
    p98_duration_df["storm_straight_line_distance"],
    bins=50,
    alpha=0.5,
    label="Straight Line Distance",
)
axs[0].hist(
    p98_duration_df["storm_distance_traversed"],
    bins=50,
    alpha=0.5,
    label="Distance Traversed",
)
axs[0].set_ylabel("Frequency")
axs[0].set_title("a) 98% Percentile Duration Storms")
axs[0].legend()

# 98th percentile area storms
axs[1].hist(
    p98_area_df["storm_straight_line_distance"],
    bins=50,
    alpha=0.5,
    label="Straight Line Distance",
)
axs[1].hist(
    p98_area_df["storm_distance_traversed"],
    bins=50,
    alpha=0.5,
    label="Distance Traversed",
)
axs[1].set_ylabel("Frequency")
axs[1].set_title("b) 98% Percentile Area Storms")

# 2% lowest storm_min_bt storms
min_bt_thres = df["storm_min_bt"].quantile(0.02)
p98_min_bt_df = df[df["storm_min_bt"] <= min_bt_thres]
axs[2].hist(
    p98_min_bt_df["storm_straight_line_distance"],
    bins=50,
    alpha=0.5,
    label="Straight Line Distance",
)
axs[2].hist(
    p98_min_bt_df["storm_distance_traversed"],
    bins=50,
    alpha=0.5,
    label="Distance Traversed",
)

axs[2].set_ylabel("Frequency")
axs[2].set_title("c) 2% Coldest Min BT Storms")

# All storms
axs[3].hist(
    df["storm_straight_line_distance"],
    bins=50,
    alpha=0.5,
    label="Straight Line Distance",
)
axs[3].hist(
    df["storm_distance_traversed"],
    bins=50,
    alpha=0.5,
    label="Distance Traversed",
)
axs[3].set_ylabel("Frequency")
axs[3].set_xlabel("Distance (km)")
axs[3].set_title("d) All Storms")

plt.suptitle("Storm Distance Comparison Histograms")

plt.tight_layout()
plotting.save_plot("storm_distance_hist_subplots.png")

# %%
print("Plotting histogram of storm zonal speed.")
plt.figure(figsize=(10, 6))
plt.hist(df["zonal_speed"], bins=50)

plt.xlabel("Zonal Speed (m/s)")
plt.ylabel("Frequency")
plt.title("Storm Zonal Speed Histogram")
plotting.save_plot("storm_zonal_speed_hist.png")

# %%
print("Plotting histogram of storm meridional speed.")
plt.figure(figsize=(10, 6))
plt.hist(df["meridional_speed"], bins=50)

plt.xlabel("Meridional Speed (m/s)")
plt.ylabel("Frequency")
plt.title("Storm Meridional Speed Histogram")
plotting.save_plot("storm_meridional_speed_hist.png")

# %%
print("Plotting histogram of storm speed (magnitude).")
plt.figure(figsize=(10, 6))
plt.hist(np.sqrt(df["zonal_speed"] ** 2 + df["meridional_speed"] ** 2), bins=50)

plt.xlabel("Speed (m/s)")
plt.ylabel("Frequency")
plt.title("Storm Speed Histogram")
plotting.save_plot("storm_speed_hist.png")

# %%
print("Calculating correlation matrix for numeric features.")
df_corr = df.select_dtypes(include=[np.number]).corr()

# %%
print("Filtering for highly correlated features.")
df_corr = df_corr - np.eye(df_corr.shape[0])
df_corr = df_corr[df_corr.abs() > 0.5]
df_corr = df_corr.dropna(how="all")

# %%
print("Plotting heatmap of feature correlations.")
plt.figure(figsize=(40, 30))
sns.heatmap(
    df_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmax=1,
    vmin=-1,
)
plotting.save_plot("feature_correlation_heatmap.png")

# %%
# print("Plotting heatmap of orography, anor, and storm duration correlations.")
# plt.figure(figsize=(40, 30))
# sns.heatmap(
#     df[["orography_height", "anor", "storm_total_duration"]].corr(),
#     annot=True,
#     cmap="coolwarm",
# )
# plotting.save_plot("orography_anor_correlation_heatmap.png")

# %%
# print("Plotting mutual information heatmap for all numeric features.")
# sns.heatmap(
#     df.select_dtypes(include=[np.number]).corr(
#         method=lambda x, y: mutual_info_regression(
#             x.reshape(-1, 1), y, discrete_features=False
#         )[0]
#     ),
#     annot=True,
#     cmap="coolwarm",
# )
# plotting.save_plot("mutual_info_heatmap.png")

# %%
# print("Plotting mutual information heatmap for orography, anor, and duration.")
# sns.heatmap(
#     df[["orography_height", "anor", "storm_total_duration"]].corr(
#         method=lambda x, y: mutual_info_regression(
#             x.reshape(-1, 1), y, discrete_features=False
#         )[0]
#     ),
#     annot=True,
#     cmap="coolwarm",
# )
# plotting.save_plot("orography_anor_mutual_info_heatmap.png")

# %%
print(
    "Calculating storm counts per day and plotting with MJO phase and amplitude."
)
df["date"] = pd.to_datetime(df["timestamp"]).dt.date
storm_counts = df.groupby("date")["storm_id"].nunique().reset_index()
storm_counts.columns = ["date", "storm_count"]

fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
sns.lineplot(data=storm_counts, x="date", y="storm_count", ax=axs[0])
axs[0].set_ylabel("a) Number of Storms")

mjo_phases = df.groupby("date")["mjo_phase"].first().reset_index()
sns.lineplot(data=mjo_phases, x="date", y="mjo_phase", ax=axs[1])
axs[1].set_ylabel("b) MJO Phase")

mjo_ampls = df.groupby("date")["mjo_amplitude"].first().reset_index()
sns.lineplot(data=mjo_ampls, x="date", y="mjo_amplitude", ax=axs[2])
axs[2].set_ylabel("c) MJO Amplitude")
plt.xticks(rotation=45)
plt.xlabel("Date")

plt.suptitle("Storm Counts, MJO Phases, and Amplitudes Over Time")

plotting.save_plot("storm_counts_per_day_vs_mjo.png")

# %%
print(
    "Converting 'over_land' and 'storm_min_bt_reached' columns to integer type."
)
df["over_land"] = df["over_land"].astype(int)
df["storm_min_bt_reached"] = df["storm_min_bt_reached"].astype(int)

# %%
print(
    "Interpolating all storms to have 11 points each for life cycle analysis."
)
# interpolate all storms to have 11 points each
# 11 ensures that there are points at 0%, 100%, and 10% intervals
n_points = 11
storm_10_points_df = processing.interpolate_all_storms(df, n_points=n_points)

# %%
print("Reshaping min_bt column to 2D array for EOF analysis.")
# reshape the min_bt column to a 2D array with shape (num_storms, n_points)
min_bt_by_storm_arr = (
    storm_10_points_df["min_bt"]
    .to_numpy()
    .reshape(storm_10_points_df["storm_id"].nunique(), n_points)
)

# %%
print("Performing EOF analysis on minimum BT by storm.")
# perform EOF analysis on the minimum BT by storm
solver = eofs.standard.Eof(min_bt_by_storm_arr)
pcs = solver.pcs(pcscaling=0)  # 0 means unscaled pcs
eofs_list = solver.eofs(
    neofs=10, eofscaling=1
)  # 1 means normalised EOF(divided by the square-root of their eignevalues)
variance_fractions = solver.varianceFraction()

# %%
print("Plotting first 5 EOFs of minimum brightness temperature.")
# plot the first 5 EOFs
plt.figure(figsize=(10, 6))
for i in range(5):
    plt.plot(
        eofs_list[i, :],
        label=f"EOF {i + 1} ({variance_fractions[i] * 100:.2f}% variance)",
    )
plt.xlabel("Storm Life Cycle (%)")
plt.xticks(
    np.arange(0, n_points, 1),
    labels=[f"{round(x/10, 2)*100}" for x in range(0, n_points)],
)
plt.ylabel("EOF Value")
plt.title("First 5 EOFs of Minimum Brightness Temperature")
plt.legend()
plotting.save_plot("eofs_min_bt.png")

# %%
print(
    "Grouping storms by land/sea and day/night initiation for life cycle analysis."
)
# get the first point of each storm
storm_first_points = storm_10_points_df.groupby("storm_id").first()

# create land/sea groups
land_storms = storm_first_points[storm_first_points["over_land"] == True].index
sea_storms = storm_first_points[storm_first_points["over_land"] == False].index

# create time groups (nighttime: < 8 or >= 20, daytime: 8 <= eat_hours < 20)
night_storms = storm_first_points[
    (storm_first_points["eat_hours"] < 8)
    | (storm_first_points["eat_hours"] >= 20)
].index
day_storms = storm_first_points[
    (storm_first_points["eat_hours"] >= 8)
    & (storm_first_points["eat_hours"] < 20)
].index

# add group labels to the interpolated dataset
storm_10_points_df["land_init"] = storm_10_points_df["storm_id"].isin(
    land_storms
)
storm_10_points_df["night_init"] = storm_10_points_df["storm_id"].isin(
    night_storms
)

# %%
print(
    "Calculating mean minimum BT over storm life cycle for each initiation type."
)
land_init_points = storm_10_points_df[storm_10_points_df["land_init"] == True]
land_init_min_bt_mean = (
    land_init_points["min_bt"]
    .to_numpy()
    .reshape(land_init_points["storm_id"].nunique(), n_points)
    .mean(axis=0)
)

sea_init_points = storm_10_points_df[storm_10_points_df["land_init"] == False]
sea_init_min_bt_mean = (
    sea_init_points["min_bt"]
    .to_numpy()
    .reshape(sea_init_points["storm_id"].nunique(), n_points)
    .mean(axis=0)
)
day_init_points = storm_10_points_df[storm_10_points_df["night_init"] == False]
day_init_min_bt_mean = (
    day_init_points["min_bt"]
    .to_numpy()
    .reshape(day_init_points["storm_id"].nunique(), n_points)
    .mean(axis=0)
)
night_init_points = storm_10_points_df[storm_10_points_df["night_init"] == True]
night_init_min_bt_mean = (
    night_init_points["min_bt"]
    .to_numpy()
    .reshape(night_init_points["storm_id"].nunique(), n_points)
    .mean(axis=0)
)

# %%
print("Plotting mean minimum BT over storm life cycle by initiation type.")
fig, ax = plt.subplots(figsize=(10, 6))

# plot the mean minimum brightness temperature over the storm life cycle
ax.plot(
    land_init_min_bt_mean,
    "tab:green",
    linewidth=2,
    label="Land Storms (first point over land)",
)
ax.plot(
    sea_init_min_bt_mean,
    "tab:blue",
    linewidth=2,
    label="Sea Storms (first point over water)",
)
ax.plot(
    night_init_min_bt_mean,
    "tab:purple",
    linewidth=2,
    label="Night Storms (EAT < 8 or ≥ 20)",
)
ax.plot(
    day_init_min_bt_mean,
    "tab:orange",
    linewidth=2,
    label="Day Storms (8 ≤ EAT < 20)",
)
ax.set_xlabel("Storm Life Cycle (%)")
ax.set_xticks(
    np.arange(0, n_points, 1),
    labels=[f"{round(x/10, 2)*100}" for x in range(0, n_points)],
)
ax.set_ylabel("Mean Minimum Brightness Temperature (K)")
ax.set_title("Mean Min BT Over Storm Life Cycle by Initiation Type")
ax.legend()

plt.tight_layout()
plotting.save_plot("min_bt_over_lifecycle_by_init_type.png")

# %%
print("Binning tcwv by latitude and longitude.")
tcwv_mean_lon, tcwv_mean_lat, tcwv_mean_grid = processing.calc_2d_agg(
    df,
    "domain_mean_tcwv"
)

# %%
print("Plotting Mean of TCWV by Location.")
plotting.plot_2d_agg_map(
    tcwv_mean_lon,
    tcwv_mean_lat,
    tcwv_mean_grid,
    cmap=config.DEFAULT_MAP_CMAP,
    cbar_aspect=40,
    cbar_shrink=0.63,
    cbar_label="Mean Total Column Water Vapour",
    title="Mean of TCWV by Location.",
    filename="tcwv_mean_by_loc.png",
    save_dir=config.EXPLORATION_FIGURES_DIR,
)

# %%
print("Binning u850 by latitude and longitude.")
u850_mean_lon, u850_mean_lat, u850_mean_grid = processing.calc_2d_agg(
    df,
    "mean_u850"
)

# %%
print("Plotting Mean of u850 by Location.")
plotting.plot_2d_agg_map(
    u850_mean_lon,
    u850_mean_lat,
    u850_mean_grid,
    cmap=config.DEFAULT_MAP_CMAP,
    cbar_aspect=40,
    cbar_shrink=0.63,
    cbar_label="Mean 850 hPa zonal wind",
    title="Mean of 850 hPa zonal wind by Location.",
    filename="u850_mean_by_loc.png",
    save_dir=config.EXPLORATION_FIGURES_DIR,
)

# %%
print("Binning v850 by latitude and longitude.")
v850_mean_lon, v850_mean_lat, v850_mean_grid = processing.calc_2d_agg(
    df,
    "mean_v850"
)

# %%
print("Plotting Mean of v850 by Location.")
plotting.plot_2d_agg_map(
    v850_mean_lon,
    v850_mean_lat,
    v850_mean_grid,
    cmap=config.DEFAULT_MAP_CMAP,
    cbar_aspect=40,
    cbar_shrink=0.63,
    cbar_label="Mean 850 hPa meridional wind",
    title="Mean of 850 hPa meridional wind by Location.",
    filename="v850_mean_by_loc.png",
    save_dir=config.EXPLORATION_FIGURES_DIR,
)
