#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script

Provide a full running example on how to use the localization simulator.


__author__ = "Axel Torchy"
__copyright__ = "Copyright 2020, Wizzilab"
__credits__ = ["Axel Torchy", "Wizzilab"]
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Axel Torchy"
__email__ = "axel@wizzilab.com"
__status__ = "Production"
"""


from typing import Union, Sequence, Dict
import json
import numpy as np
import genenv
import genmeas
import matplotlib.pyplot as plt



# %% Simulation parameters

# Grid bounds
x_min, x_max = 1, 30
y_min, y_max = 1, 20
z_min, z_max = 0.5, 3
step = 0.5

# Number of tries for each grid position
N_tries = 3

# Noise parameters, in meters
mu_noise = 0.08
sigma_noise = 0.15
noise_model = "gaussian"

# Ranging distance (first inventory) = max distance for an anchor to be visible
ranging_dst = 30

# Name of the file to which the localization results will be saved
filename = "loc.json"
# If replace_file is True, each simulation will overwrite the previous results.
# Keep it to False to superimpose the results of several settings.

# Number of anchors pre-ranging
N_anchors_pre = 8

# Number of anchors post-ranging
N_anchors_post = 5

replace_file = False

# Optimization parameters
bnds = ((None, None), (None, None), (None, None))

# Name of the file where to save the anchors' location information
anchors_filename = "anchors_dump.json"

anchors_list = [
    [0 ,5.12, 3],       # 0
    [0, 10, 3],         # 1
    [0, 25.31, 3],      # 2
    [7.35, 10, 3],      # 3
    [7.2, 25.31, 3],    # 4
    [7.35, 5.12, 3],    # 5
    [10.08, 25.31, 3],  # 6
    [21, 25.31, 3],     # 7
    [24.2, 21, 3],      # 8
    [10.5, 5.12, 3],    # 9
    [19, 5.12, 3],      # 10
    [24, 6.9, 3],       # 11
    [27.5, 17, 3],      # 12
    [30.7, 11.9, 3],    # 13
    [14.9, 14.4, 3],    # 14
    [21.6, 13.9, 3],    # 15
    [0, 21, 3],         # 16
    [7.2, 21, 3],       # 17
    [0, 15.5, 3],       # 18
    [7.2, 15.5, 3]      # 19
    ]



if __name__ == "__main__":
    # %% BLOCK 1: environment generation
    print("Generating environment.")
    anchors = genenv.generate_anchors_from_list(anchors_list)
    for a in anchors:
        print(f"== Anchor {a}: \t{anchors[a]}")
    
    # Save anchors to file, so it can be used again with the
    # genenv.generate_anchors_from_json_file function
    genenv.save_anchors_to_json_file(anchors, anchors_filename)
    
    # Can be loaded using:
    #   anchors = genenv.load_anchors_from_json_file(anchors_filename)
    
    # Use randomly distributed anchors:
    # anchors = genenv.generate_uniform_anchors(0,5,0,10,2,4,20)
    
    # %% Visualization of the warehouse environment, specific to this warehouse
    # image.
    def xy_to_ij(x,y):
        """Returns the coordinates on the warehouse picture."""
        # 10.92 m = 860 px
        # 25.32 = 1994 px
        conv_factor = 1994. / 25.32
        return (65 + conv_factor * x, 2058 - conv_factor*y)

    warehouse = plt.imread("warehouse.png")
    f1 = plt.imshow(warehouse)
    plt.title("Warehouse - anchor positions (2D)")
    f1.axes.get_xaxis().set_visible(False)
    f1.axes.get_yaxis().set_visible(False)
    for ID, anc in anchors.items():
        
        pos = xy_to_ij(anc['x'], anc['y'])
        f2 = plt.scatter(pos[0], pos[1], color="red", marker="D", s=60)
        f2.axes.get_xaxis().set_visible(True)
        f2.axes.get_yaxis().set_visible(True)
        f2 = plt.scatter(pos[0], pos[1], color="red", marker="D", s=60)
        plt.annotate("A %d" % ID, (pos[0], pos[1]), textcoords="offset points",
                     xytext=(0,7), ha='center', color="red")
        labels = np.arange(-2, 33, 2)
        locs = labels * 1994 / 25.32 + 65
        plt.xticks(locs, labels)
        labels = np.arange(-2, 29, 2)
        locs = 2058 - labels * 1994 / 25.32
        plt.yticks(locs, labels)
    
    plt.show()
    
    
    # %% BLOCK 2: Generate measurements for each point on the grid and
    # geolocate the tag over the grid
    x_grid, y_grid, z_grid = genenv.generate_grid(x_min, x_max, y_min, y_max, z_min, z_max, step)
    
    
    # Several simulations can be run successively in order to compare the
    # performance of the localization with different parameters.
    
    options = {
        'ranging_method':   "UWB_TWR",
        'initial_pos':      "bary_z0",
        'noise_model':      noise_model,
        'noise_params':     {"mu": mu_noise, "sigma": sigma_noise},
        'anchor_selection': "nearest",
        'optimization':     genmeas.cost_function,
        'bounds':           bnds,
        'ranging_distance': ranging_dst,
        'N_anchors_pre':    N_anchors_pre,
        'N_anchors_post':   N_anchors_post
        }

    
    genmeas.locate_grid(filename,
                   replace_file,
                   anchors,
                   x_grid,
                   y_grid,
                   z_grid,
                   N_tries,
                   options
                   )
    
                    
                    
    