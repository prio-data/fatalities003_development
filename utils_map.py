import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
matplotlib.pyplot.switch_backend('Agg') # Important! Disable GUI windows so that thread won't break down
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import requests
from PIL import Image
from pathlib import Path


def plot_cm_map(df, month, step):
    cm_path = Path('./GeospatialData/countries.shp')
    cm_file = gpd.read_file(cm_path)

    df_cm = df.reset_index()
    df_cm = pd.merge(df_cm, cm_file, on='country_id')
    gdf_cm = gpd.GeoDataFrame(df_cm, crs=cm_file.crs)

    # Get the data for the specific month
    gdf_cm_m = gdf_cm.loc[gdf_cm['month_id'] == month]
    pred_min = gdf_cm_m[step].min()
    pred_max = gdf_cm_m[step].max()
    # print(pred_max, pred_min)

    fig, ax = plt.subplots(1, 1, figsize=(24, 16))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.5)
    cax.tick_params(labelsize=20)

    # Customize plot
    cm_file.boundary.plot(ax=ax, linewidth=0.3)
    plot = gdf_cm_m.plot(column=step, cmap='viridis', legend=True,
                         norm=colors.SymLogNorm(linthresh=1, vmin=pred_min, vmax=pred_max), cax=cax,
                         legend_kwds={"label": " ", "orientation": "horizontal"}, ax=ax)

    # Download and display the VIEWS logo image
    logo_url = "https://cdn.cloud.prio.org/images/c784369fb4ae42acb7ee882e91056d92.png?x=800&"
    response = requests.get(logo_url, stream=True)

    if response.status_code == 200:
        logo_img = Image.open(response.raw)
        logo_ax = fig.add_axes([0.16, 0.28, 0.1, 0.1])  # Define the position and size of the logo [left, bottom, width, height]
        logo_ax.imshow(logo_img)
        logo_ax.axis('off')  # Turn off axis labels and ticks for the logo
    else:
        print("Failed to download the logo image")

    ax.set_title(f'Predicted Fatalities (Month {month})', fontsize=25, y=1)

    # Access the colorbar associated with the plot
    colorbar = plot.get_figure().get_axes()[1]

    # Customize tick labels
    colorbar.xaxis.set_major_formatter(ticker.ScalarFormatter())  # Customize tick labels

    # Remove the degrees (latitude and longitude labels) from the plot
    ax.set_xticks([])
    ax.set_yticks([])

    plt.close()

    return fig


def plot_pgm_map(df, month, step):
    # Define the bounding box coordinates for Africa and the Middle East (latitude and longitude)
    xmin, xmax, ymin, ymax = -18.5, 64.0, -35.5, 43.0

    cm_path = Path('./GeospatialData/countries.shp')
    cm_file = gpd.read_file(cm_path)
    pgm_path = Path('./GeospatialData/priogrid.shp')
    pgm_file = gpd.read_file(pgm_path)

    df_pgm = df.reset_index()
    df_pgm = pd.merge(df_pgm, pgm_file, left_on='priogrid_id', right_on='priogrid_i') # Note that in predictions the coloumn name is priogrid_id instead of priogrid_gid
    gdf_pgm = gpd.GeoDataFrame(df_pgm, crs=pgm_file.crs)

    gdf_pgm_m = gdf_pgm.loc[gdf_pgm['month_id'] == month]
    pred_min = gdf_pgm_m[step].min()
    pred_max = gdf_pgm_m[step].max()

    # Create subplot
    fig, ax = plt.subplots(1, 1, figsize=(24, 16))  # Customize layout and size

    # Customize legend
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.5)
    cax.tick_params(labelsize=20)

    # Customize plot
    ax.set_xlim(xmin, xmax)  # Set the longitude limits
    ax.set_ylim(ymin, ymax)  # Set the latitude limits
    pgm_file.boundary.plot(ax=ax, linewidth=0.2)  # Plot pgm boundaries
    plot = gdf_pgm_m.plot(column=step, cmap='viridis', legend=True,
                                           norm=colors.SymLogNorm(linthresh=1, vmin=pred_min,
                                                                  vmax=pred_max), cax=cax,
                                           legend_kwds={"label": " ", "orientation": "horizontal"}, ax=ax,
                                           linewidth=0.2, edgecolor='#FF000000')


    # Download and display the VIEWS logo image
    logo_url = "https://cdn.cloud.prio.org/images/c784369fb4ae42acb7ee882e91056d92.png?x=800&"
    response = requests.get(logo_url, stream=True)

    if response.status_code == 200:
        logo_img = Image.open(response.raw)
        logo_ax = fig.add_axes([0.28, 0.18, 0.1, 0.1])  # Define the position and size of the logo [left, bottom, width, height]
        logo_ax.imshow(logo_img)
        logo_ax.axis('off')  # Turn off axis labels and ticks for the logo
    else:
        print("Failed to download the logo image")

    ax.set_title(f'Predicted Fatalities (Month {month})', fontsize=25, y=1)

    # Access the colorbar associated with the plot
    colorbar = plot.get_figure().get_axes()[1]

    # Customize tick labels
    colorbar.xaxis.set_major_formatter(ticker.ScalarFormatter())  # Customize tick labels

    # Remove the degrees (latitude and longitude labels) from the plot
    ax.set_xticks([])
    ax.set_yticks([])

    # Add country borders overlay
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    cm_file.boundary.plot(ax=ax, linewidth=1.1, color='grey')  # use cm unit map

    plt.close()

    return fig
