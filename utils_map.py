import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
        logo_ax = fig.add_axes(
            [0.16, 0.28, 0.1, 0.1])  # Define the position and size of the logo [left, bottom, width, height]
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