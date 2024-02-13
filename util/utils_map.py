import numpy as np
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

class GeoPlotter:
    # Define the bounding box coordinates for Africa and the Middle East (latitude and longitude)
    XMIN, XMAX, YMIN, YMAX = -18.5, 64.0, -35.5, 43.0

    def __init__(self, cm_path='./GeospatialData/countries.shp', pgm_path='./GeospatialData/priogrid.shp'):
        self.cm_path = Path(cm_path)
        self.pgm_path = Path(pgm_path)
        self.cm_file = gpd.read_file(self.cm_path)
        self.pgm_file = gpd.read_file(self.pgm_path)


    def plot_cm_map(self, df, month, step, transform):
        df_cm = df.reset_index()
        df_cm = pd.merge(df_cm, self.cm_file, on='country_id')
        gdf_cm = gpd.GeoDataFrame(df_cm, crs=self.cm_file.crs)
        gdf_cm_m = gdf_cm.loc[gdf_cm['month_id'] == month]

        # Create subplot
        fig, axes = plt.subplots(2, 2, figsize=(24, 16))
        fig.suptitle(f'Level: cm, Month: {month}, Transform: {transform}', fontsize=25)
        self.plot_fatality(gdf_cm_m, axes[0,0], step, 'cm')
        self.plot_absolute_error(gdf_cm_m, axes[0,1], step, 'cm')
        self.plot_squared_error(gdf_cm_m, axes[1,0], step, 'cm')
        self.plot_squared_error(gdf_cm_m, axes[1,1], step, 'cm', if_log=True)

        # # Download and display the VIEWS logo image
        # logo_url = "https://cdn.cloud.prio.org/images/c784369fb4ae42acb7ee882e91056d92.png?x=800&"
        # response = requests.get(logo_url, stream=True)
        #
        # if response.status_code == 200:
        #     logo_img = Image.open(response.raw)
        #     logo_ax = fig.add_axes(
        #         [0.28, 0.18, 0.1, 0.1])  # Define the position and size of the logo [left, bottom, width, height]
        #     logo_ax.imshow(logo_img)
        #     logo_ax.axis('off')  # Turn off axis labels and ticks for the logo
        # else:
        #     print("Failed to download the logo image")

        plt.close()
        return fig

    def plot_pgm_map(self, df, month, step, transform):
        df_pgm = df.reset_index()
        df_pgm = pd.merge(df_pgm, self.pgm_file, left_on='priogrid_id', right_on='priogrid_i') # Note that in predictions the coloumn name is priogrid_id instead of priogrid_gid
        gdf_pgm = gpd.GeoDataFrame(df_pgm, crs=self.pgm_file.crs)
        gdf_pgm_m = gdf_pgm.loc[gdf_pgm['month_id'] == month]

        # Create subplot
        fig, axes = plt.subplots(2, 2, figsize=(24, 16))
        fig.suptitle(f'Level: pgm, Month: {month}, Transform: {transform}', fontsize=25)
        self.plot_fatality(gdf_pgm_m, axes[0,0], step, 'pgm')
        self.plot_absolute_error(gdf_pgm_m, axes[0,1], step, 'pgm')
        self.plot_squared_error(gdf_pgm_m, axes[1,0], step, 'pgm')
        self.plot_squared_error(gdf_pgm_m, axes[1,1], step, 'pgm', if_log=True)

        plt.close()
        return fig

    def add_cax(self, ax):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.5)
        cax.tick_params(labelsize=20)
        return cax

    def add_cm_plot(self, gdf, column, vmin, vmax, ax, title):
        cax = self.add_cax(ax)
        self.cm_file.boundary.plot(ax=ax, linewidth=0.3)
        plot = gdf.plot(column=column, cmap='viridis', legend=True,
                        norm=colors.SymLogNorm(linthresh=1, vmin=vmin, vmax=vmax), cax=cax,
                        legend_kwds={"label": " ", "orientation": "horizontal"}, ax=ax)
        ax.set_title(title, fontsize=15, y=1)
        # Remove the degrees (latitude and longitude labels) from the plot
        ax.set_xticks([])
        ax.set_yticks([])

        colorbar = plot.get_figure().get_axes()[1]
        colorbar.xaxis.set_major_formatter(ticker.ScalarFormatter())

    def choose_scale_colormap_pgm_plot(self, vmin, vmax, title):
        #note that the function should be standardized to pick up only Fatality type values
        
        if title = 'Predicted Fatalities':
            standard_scale = [0,100,300,1000,3000]
            standard_labels = ['0','100', '300', '1000', '3000']
            map_dictionary = dict(zip(standard_scale, standard_labels))

            vmin=min(map_dictionary)
            vmax=max(map_dictionary)

            #ensure that the custom colormap is made
            color_list = [plt.cm.jet(key/max(map_dictionary.keys())) for key in map_dictionary.keys()]
            custom_cmap = ListedColormap(color_list)
            cmap = custom_cmap

        else: 
            vmin=vmin
            vmax=vmax
            cmap = 'viridis'
            map_dictionary = 'ignore'


        return vmin, vmax, cmap, map_dictionary

    def add_pgm_plot(self, gdf, column, vmin, vmax, ax, title):
            
        vmin, vmax, cmap, map_dictionary = choose_scale_colormap_pgm_plot(self, vmin, vmax, title)

        norm = SymLogNorm(linthresh=1, vmin=vmin, vmax=vmax)

        cax = self.add_cax(ax)
        self.pgm_file.boundary.plot(ax=ax, linewidth=0.2)  # Plot pgm boundaries
        self.cm_file.boundary.plot(ax=ax, linewidth=1.1, color='grey')  # use cm unit map
        plot = gdf.plot(column=column, cmap=cmap, legend=True,
                        norm=norm, cax=cax,
                        legend_kwds={"label": " ", "orientation": "horizontal"}, ax=ax,
                        linewidth=0.2, edgecolor='#FF000000')
        ax.set_title(title, fontsize=15, y=1)
        ax.set_xticks([])
        ax.set_yticks([])
        # Add country borders overlay
        ax.set_xlim(self.XMIN, self.XMAX)
        ax.set_ylim(self.YMIN, self.YMAX)

        #use the custom cmap but only for fatalities
        if map_dictionary == 'ignore':
            colorbar = plot.get_figure().get_axes()[1]
            colorbar.xaxis.set_major_formatter(ticker.ScalarFormatter())
        else:
            #Customize tick labels but have the same SymLogNorm as above
            formatter = ScalarFormatter()
            formatter.set_locs(list(map_dictionary.values()))
            formatter.set_ticklabels(list(map_dictionary.keys()))

            colorbar = plot.get_figure().get_axes()[1]
            colorbar.xaxis.set_major_formatter(formatter)
            colorbar.set_norm(norm) 
            colorbar.set_ticks(list(custom_dict.values())) 

    def plot_fatality(self, gdf_m, ax, step, level):
        pred_min = gdf_m[step].min()
        pred_max = gdf_m[step].max()
        # print(pred_max, pred_min)
        title = 'Predicted Fatalities'
        column = step

        if level == 'cm':
            self.add_cm_plot(gdf_m, column, pred_min, pred_max, ax, title)
        elif level == 'pgm':
            self.add_pgm_plot(gdf_m, column, pred_min, pred_max, ax, title)

    def plot_absolute_error(self, gdf_m, ax, step, level):
        gdf_m['absolute_error'] = abs(gdf_m['ged_sb_dep'] - gdf_m[step])
        ae_min = gdf_m['absolute_error'].min()
        ae_max = gdf_m['absolute_error'].max()
        title = 'Absolute Error'
        column = 'absolute_error'

        if level == 'cm':
            self.add_cm_plot(gdf_m, column, ae_min, ae_max, ax, title)
        elif level == 'pgm':
            self.add_pgm_plot(gdf_m, column, ae_min, ae_max, ax, title)

    def plot_squared_error(self, gdf_m, ax, step, level, if_log=False):
        if if_log:
            gdf_m['squared_error'] = np.square(np.log(gdf_m['ged_sb_dep']+1) - np.log(gdf_m[step]+1))
            title = 'Squared Logarithmic Error'
        else:
            gdf_m['squared_error'] = np.square(gdf_m['ged_sb_dep'] - gdf_m[step])
            title = 'Squared Error'
        se_min = gdf_m['squared_error'].min()
        se_max = gdf_m['squared_error'].max()
        column = 'squared_error'

        if level == 'cm':
            self.add_cm_plot(gdf_m, column, se_min, se_max, ax, title)
        elif level == 'pgm':
            self.add_pgm_plot(gdf_m, column, se_min, se_max, ax, title)