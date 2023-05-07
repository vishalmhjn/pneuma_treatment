# This code plots the time-series postion data of 
# vehicles in the data on a map for a holositic overview
#  of the data. 

# The generated images can be further combined (using external tools) 
# to make a video or gif (such as displayed in the readme file),
# which really looks nice.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from contextily import add_basemap
from shapely.geometry import Point
from geopandas import GeoDataFrame

SIZE = 20
plt.rc('font', size=SIZE)
plt.rc('axes', titlesize=SIZE)
plt.rc('axes', labelsize=SIZE)
plt.rc('xtick', labelsize=SIZE)
plt.rc('ytick', labelsize=SIZE)
plt.rc('legend', fontsize=SIZE)
plt.rc('figure', titlesize=SIZE)

def scale_to_unity(s):
    return (s - s.min()) / (np.std(s))

def series_to_colors(s, cmap='YlOrRd'):
    import matplotlib.pyplot as plt
    color_map = plt.get_cmap(cmap)
    return s, color_map

# specify the path to the data you want to visualize
# the data should have at least these three columns
# lon	: Longitude
# lat	: Latitude
# frame	: time sequence

# change the path to the data that you want to visualize
df = pd.read_csv('../data/sample_data.csv')

# Starting point of the time sequence
frame = 0
while frame < df.frame.max():
    
    temp_sample_one = df[(df.frame>frame) & (df.frame<frame + 0.05)]
    temp_sample = temp_sample_one
    
    f, ax = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True)
    
    # Extents or bounding box of the map will adjust according to the data
    min_x = df.lon.min()
    max_x = df.lon.max()
    min_y = df.lat.min()
    max_y = df.lat.max()

    geometry = [Point(xy) for xy in zip(temp_sample.lon, temp_sample.lat)]
    gdf = temp_sample.drop(['lon', 'lat'], axis=1)
    gdf = GeoDataFrame(gdf, crs="EPSG:4326", geometry=geometry)

    # color scheme
    c_s, cmap = series_to_colors(temp_sample.v, cmap="RdYlGn")
    
    axis = ax.scatter(x='lon', 
                      y='lat', 
                      data=temp_sample, 
                      c=c_s, 
                      alpha=1,
                      s=5,
                      vmin=0,
                      vmax=60,
                      cmap = cmap
                    )
    
    # plot North direction arrow
    x, y, arrow_length = 0.1, 0.95, 0.1
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                arrowprops=dict(facecolor='black', width=5, headwidth=15),
                ha='center', va='center', fontsize=20,
                xycoords=ax.transAxes)

    cbar = plt.colorbar(axis, ax = ax,fraction=0.046, pad=0.0)
    
    cbar.set_label('Speed in Kmph', rotation=90, labelpad=20)    
    ax.set_ylim([min_y, max_y])
    ax.set_xlim([min_x, max_x])
    
    # add the basemap e.g., from OSM layer
    add_basemap(ax, crs = "EPSG:4326", alpha=0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title("Time: "+str(round(frame,2)))

    plt.tight_layout()
    # resolution of the images can be adjusted using dpi, 
    # which will also effect the filesize.
    plt.savefig("../plots/map_visualization/"+str(frame)+".png", dpi=100)
    plt.close()
    # Interval between the time sequence
    frame+=0.5
    