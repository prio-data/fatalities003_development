import warnings
from pathlib import Path
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
# Important! Disable GUI windows so that thread won't break down
matplotlib.pyplot.switch_backend('Agg')
warnings.filterwarnings("ignore")


class GeoPlotter:
    """
    This class is used to plot the maps for the predictions and the errors. It uses the geopandas library to read the shapefiles for the countries and the priogrids. The class has two methods, plot_cm_map and plot_pgm_map, which are used to plot the maps for the cm and pgm levels, respectively. The methods take the following parameters:
    - df: A pandas DataFrame containing the predictions and the errors.
    - month: An integer representing the month for which the maps are to be plotted.
    - step: A string representing the step for which the maps are to be plotted.
    - transform: A string representing the transformation to be used for the predictions and the errors.
    """
    # Define the bounding box coordinates for Africa and the Middle East (latitude and longitude)
    XMIN, XMAX, YMIN, YMAX = -18.5, 64.0, -35.5, 43.0

    def __init__(self, cm_path='./GeospatialData/countries.shp', pgm_path='./GeospatialData/priogrid.shp') -> None:
        self.cm_path = Path(cm_path)
        self.pgm_path = Path(pgm_path)
        self.cm_file = gpd.read_file(self.cm_path)
        self.pgm_file = gpd.read_file(self.pgm_path)

    def plot_cm_map(self, df, month, step, transform):
        """
        This method is used to plot the maps for the cm level. It takes the following parameters:
        - df: A pandas DataFrame containing the predictions and the errors.
        - month: An integer representing the month for which the maps are to be plotted.
        - step: A string representing the step for which the maps are to be plotted.
        - transform: A string representing the transformation to be used for the predictions and the errors.
        """

        df_cm = df.reset_index()
        df_cm = pd.merge(df_cm, self.cm_file, on='country_id')
        gdf_cm = gpd.GeoDataFrame(df_cm, crs=self.cm_file.crs)
        gdf_cm_m = gdf_cm.loc[gdf_cm['month_id'] == month]

        # Create subplot
        fig, axes = plt.subplots(2, 2, figsize=(24, 16))
        fig.suptitle(
            f'Level: cm, Month: {month}, Transform: {transform}', fontsize=25)
        self.plot_fatality(gdf_cm_m, axes[0, 0], step, 'cm')
        self.plot_absolute_error(gdf_cm_m, axes[0, 1], step, 'cm')
        self.plot_squared_error(gdf_cm_m, axes[1, 0], step, 'cm')
        self.plot_squared_error(gdf_cm_m, axes[1, 1], step, 'cm', if_log=True)
        plt.close()
        return fig

    def plot_pgm_map(self, df, month, step, transform):
        """
        This method is used to plot the maps for the pgm level. It takes the following parameters:
        - df: A pandas DataFrame containing the predictions and the errors.
        - month: An integer representing the month for which the maps are to be plotted.
        - step: A string representing the step for which the maps are to be plotted.
        - transform: A string representing the transformation to be used for the predictions and the errors.
        """
        df_pgm = df.reset_index()
        # Note that in predictions the coloumn name is priogrid_id instead of priogrid_gid
        df_pgm = pd.merge(df_pgm, self.pgm_file,
                          left_on='priogrid_id', right_on='priogrid_i')
        gdf_pgm = gpd.GeoDataFrame(df_pgm, crs=self.pgm_file.crs)
        gdf_pgm_m = gdf_pgm.loc[gdf_pgm['month_id'] == month]

        # Create subplot
        fig, axes = plt.subplots(2, 2, figsize=(24, 16))
        fig.suptitle(
            f'Level: pgm, Month: {month}, Transform: {transform}', fontsize=25)
        self.plot_fatality(gdf_pgm_m, axes[0, 0], step, 'pgm')
        self.plot_absolute_error(gdf_pgm_m, axes[0, 1], step, 'pgm')
        self.plot_squared_error(gdf_pgm_m, axes[1, 0], step, 'pgm')
        self.plot_squared_error(
            gdf_pgm_m, axes[1, 1], step, 'pgm', if_log=True)

        plt.close()
        return fig

    def add_cax(self, ax):
        """
        This method is used to add a colorbar to the map. It takes the following parameters:
        - ax: The axes object to which the colorbar is to be added.
        """
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.5)
        cax.tick_params(labelsize=20)
        return cax

    def add_cm_plot(self, gdf, column, vmin, vmax, ax, title) -> None:
        """
        This method is used to add a plot for the cm level. It takes the following parameters:
        - gdf: A geopandas GeoDataFrame containing the predictions and the errors.
        - column: A string representing the column to be plotted.
        - vmin: A float representing the minimum value for the colorbar.
        - vmax: A float representing the maximum value for the colorbar.
        - ax: The axes object to which the plot is to be added.
        - title: A string representing the title of the plot.
        """
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

    def add_pgm_plot(self, gdf, column, vmin, vmax, ax, title) -> None:
        """
        This method is used to add a plot for the pgm level. It takes the following parameters:
        - gdf: A geopandas GeoDataFrame containing the predictions and the errors.
        - column: A string representing the column to be plotted.
        - vmin: A float representing the minimum value for the colorbar.
        - vmax: A float representing the maximum value for the colorbar.
        - ax: The axes object to which the plot is to be added.
        - title: A string representing the title of the plot.
        """
        cax = self.add_cax(ax)
        self.pgm_file.boundary.plot(
            ax=ax, linewidth=0.2)  # Plot pgm boundaries
        self.cm_file.boundary.plot(
            ax=ax, linewidth=1.1, color='grey')  # use cm unit map
        plot = gdf.plot(column=column, cmap='viridis', legend=True,
                        norm=colors.SymLogNorm(linthresh=1, vmin=vmin, vmax=vmax), cax=cax,
                        legend_kwds={"label": " ", "orientation": "horizontal"}, ax=ax,
                        linewidth=0.2, edgecolor='#FF000000')
        ax.set_title(title, fontsize=15, y=1)
        ax.set_xticks([])
        ax.set_yticks([])
        # Add country borders overlay
        ax.set_xlim(self.XMIN, self.XMAX)
        ax.set_ylim(self.YMIN, self.YMAX)

        colorbar = plot.get_figure().get_axes()[1]
        colorbar.xaxis.set_major_formatter(
            ticker.ScalarFormatter())  # Customize tick labels

    def plot_fatality(self, gdf_m, ax, step, level) -> None:
        """
        This method is used to plot the predictions for the fatalities. It takes the following parameters:
        - gdf_m: A geopandas GeoDataFrame containing the predictions and the errors.
        - ax: The axes object to which the plot is to be added.
        - step: A string representing the step for which the predictions are to be plotted.
        - level: A string representing the level for which the predictions are to be plotted.
        """
        pred_min = gdf_m[step].min()
        pred_max = gdf_m[step].max()
        # print(pred_max, pred_min)
        title = 'Predicted Fatalities'
        column = step

        if level == 'cm':
            self.add_cm_plot(gdf_m, column, pred_min, pred_max, ax, title)
        elif level == 'pgm':
            self.add_pgm_plot(gdf_m, column, pred_min, pred_max, ax, title)

    def plot_absolute_error(self, gdf_m, ax, step, level) -> None:
        """
        This method is used to plot the absolute error for the predictions. It takes the following parameters:
        - gdf_m: A geopandas GeoDataFrame containing the predictions and the errors.
        - ax: The axes object to which the plot is to be added.
        - step: A string representing the step for which the absolute error is to be plotted.
        - level: A string representing the level for which the absolute error is to be plotted.
        """
        gdf_m['absolute_error'] = abs(gdf_m['ged_sb_dep'] - gdf_m[step])
        ae_min = gdf_m['absolute_error'].min()
        ae_max = gdf_m['absolute_error'].max()
        title = 'Absolute Error'
        column = 'absolute_error'

        if level == 'cm':
            self.add_cm_plot(gdf_m, column, ae_min, ae_max, ax, title)
        elif level == 'pgm':
            self.add_pgm_plot(gdf_m, column, ae_min, ae_max, ax, title)

    def plot_squared_error(self, gdf_m, ax, step, level, if_log=False) -> None:
        """
        This method is used to plot the squared error for the predictions. It takes the following parameters:
        - gdf_m: A geopandas GeoDataFrame containing the predictions and the errors.
        - ax: The axes object to which the plot is to be added.
        - step: A string representing the step for which the squared error is to be plotted.
        - level: A string representing the level for which the squared error is to be plotted.
        - if_log: A boolean representing whether the squared error is to be plotted on a logarithmic scale.
        """
        if if_log:
            gdf_m['squared_error'] = np.square(
                np.log(gdf_m['ged_sb_dep']+1) - np.log(gdf_m[step]+1))
            title = 'Squared Logarithmic Error'
        else:
            gdf_m['squared_error'] = np.square(
                gdf_m['ged_sb_dep'] - gdf_m[step])
            title = 'Squared Error'
        se_min = gdf_m['squared_error'].min()
        se_max = gdf_m['squared_error'].max()
        column = 'squared_error'

        if level == 'cm':
            self.add_cm_plot(gdf_m, column, se_min, se_max, ax, title)
        elif level == 'pgm':
            self.add_pgm_plot(gdf_m, column, se_min, se_max, ax, title)
