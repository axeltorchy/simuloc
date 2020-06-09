#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script

Provide a full running example on how to use the localization simulator.


Author:         "Axel Torchy"
Copyright:      "Copyright 2020, Wizzilab"
Credits:        ["Axel Torchy", "Wizzilab"]
Version:        "1.0"
Email:          "axel@wizzilab.com"
Status:         "Production"
"""

# Simulator imports
from simulator import genenv, genmeas, genstats

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


# Improves quality of the graphs

rcParams['figure.dpi'] = 150
size_factor = 1.2
rcParams['figure.figsize'] = [size_factor * 8.0, size_factor * 6.0]
rcParams['lines.markersize'] = 6.0 

# %% Simulation parameters

do_plot = True

# Grid bounds
x_min, x_max = 1, 30
y_min, y_max = 1, 20
z_min, z_max = 0.5, 3
step = 2

# Number of tries for each grid position
N_tries = 3

# Noise parameters, in meters
mu_noise = 0.05
sigma_noise = 0.15
noise_model = "gaussian"

# Ranging distance (first inventory) = max distance for an anchor to be visible
ranging_dst = 30

# Name of the file to which the localization results will be saved
filename = "localization_log.json"
# If replace_file is True, each simulation will overwrite the previous results.
# Keep it to False to superimpose the results of several settings.

# Number of anchors pre-ranging.
# Choose 0 for no pre-random selection i.e. all the anchors within the ranging
# distance will be kept if N_anchors_pre == 0
N_anchors_pre = 8

# Number of anchors post-ranging. 
N_anchors_post = 6

replace_file = False

# Optimization parameters
# The bounds are of the form: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
# To be in plane constrained mode, choose   z_min = z_max = constrained_alt
# Tolerance should be small enough (no bigger than 1e-5)
bnds = ((None, None), (None, None), (0.5, 3))
tolerance = 1e-7

# Name of the file where to save the anchors' location information
anchors_filename = "anchors_dump.json"

anchors_list = [
    [0 ,5.12, 2],       # 0
    [0, 10, 2.5],         # 1
    [0, 25.31, 3],      # 2
    [7.35, 10, 2],      # 3
    [7.2, 25.31, 2],    # 4
    [7.35, 5.12, 3.5],    # 5
    [10.08, 25.31, 2],  # 6
    [21, 25.31, 4],     # 7
    [24.2, 21, 3],      # 8
    [10.5, 5.12, 2],    # 9
    [19, 5.12, 3.5],      # 10
    [24, 6.9, 2],       # 11
    [27.5, 17, 1.5],      # 12
    [30.7, 11.9, 3.5],    # 13
    [14.9, 14.4, 4],    # 14
    [21.6, 13.9, 2],    # 15
    [0, 21, 3.5],         # 16
    [7.2, 21, 2],       # 17
    [0, 15.5, 3],       # 18
    [7.2, 15.5, 3.5]      # 19
    ]



if __name__ == "__main__":
    # %% BLOCK 1: environment generation
    print("Generating environment.")
    anchors = genenv.generate_anchors_from_list(anchors_list)
    
    # # If more variability is wanted on the altitudes of the anchors.
    # # It should improve the results of the final localization, especially on
    # # the z coordinate.
    # genenv.random_z_coordinate(anchors, z_min = 0., z_max = 6)
    
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
    
    # Remember to specify "simu_name" in order to be able to differenciate
    # different simulations that are going to be plotted on the same graph.
    
    
    # First simulation :
    simu_name = "low noise"
    
    options = {
        'method':   "UWB_TWR",
        'initial_pos':      {'type': "bary_z0", 'initial_z': 0},
        'noise_model':      noise_model,
        'noise_params':     {"mu": mu_noise, "sigma": sigma_noise},
        'anchor_selection': "nearest",
        'optimization':     "basic",
        'tolerance':        tolerance,
        'bounds':           bnds,
        'ranging_distance': ranging_dst,
        'N_anchors_pre':    N_anchors_pre,
        'N_anchors_post':   N_anchors_post
        }

    
    genmeas.locate_grid(filename,
                        simu_name,
                        replace_file,
                        anchors,
                        x_grid,
                        y_grid,
                        z_grid,
                        N_tries,
                        options
                        )
    
    # Second simulation :
    # Another simulation with increased noise to compare the performance.
    simu_name = "high noise"
    
    sigma_noise = 2. * sigma_noise
    options = {
        'method':   "UWB_TWR",
        'initial_pos':      {'type': "bary_z0", 'initial_z': 0},
        'noise_model':      noise_model,
        'noise_params':     {"mu": mu_noise, "sigma": sigma_noise},
        'anchor_selection': "nearest",
        'optimization':     "basic",
        'tolerance':        tolerance,
        'bounds':           bnds,
        'ranging_distance': ranging_dst,
        'N_anchors_pre':    N_anchors_pre,
        'N_anchors_post':   N_anchors_post
        }

    
    genmeas.locate_grid(filename,
                        simu_name,
                        replace_file,
                        anchors,
                        x_grid,
                        y_grid,
                        z_grid,
                        N_tries,
                        options
                        )
    
    
    data, metadata = genstats.read_log_file(filename)
    genstats.print_stats(data, metadata, do_plot)
    
    
    
    
    
    
    
                    
                    
    