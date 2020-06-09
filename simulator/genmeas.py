#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""genmeas module
Measurement and localization / optimization module

Provide functions to generate noisy measurements for several technologies and
save the results to a text file to be used to produce statistics.


Author:         "Axel Torchy"
Copyright:      "Copyright 2020, Wizzilab"
Credits:        ["Axel Torchy", "Wizzilab"]
Version:        "1.0"
Email:          "axel@wizzilab.com"
Status:         "Production"
"""

import numpy as np
import json
from scipy.optimize import minimize
from random import sample
from itertools import combinations
from heapq import nsmallest 

from time import time

# List of public objects of the module, as interpreted by "import *"

__all__ = ['locate_grid']

def cost_function(point, anchors, weight_method):
    x = point[0]
    y = point[1]
    z = point[2]

    s = 0
    
    for i in anchors:
        x_i = anchors[i]['x']
        y_i = anchors[i]['y']
        z_i = anchors[i]['z']
        
        d_i = np.sqrt(np.sum((np.array([x,y,z])-np.array([x_i, y_i, z_i]))**2))
        
        # Two possible weight methods
        if weight_method == 1: # proportional to the inverse of squared dist
            s += ((anchors[i]['dst']) - d_i)**2 / ((anchors[i]['dst'])**2)
        else:
            s += ((anchors[i]['dst']) - d_i)**2
    
    return s


# To be verified
    
# def gradient_cost_function(point, anchors, weight_method):
#     x = point[0]
#     y = point[1]
#     z = point[2]
#     G = np.zeros(3)
#     for i in anchors:
#         x_i = anchors[i]['x']
#         y_i = anchors[i]['y']
#         z_i = anchors[i]['z']
#         d_i = np.sqrt(np.sum((np.array([x,y,z])-np.array([x_i, y_i, z_i]))**2))
#         G[0] += -2 * (anchors[i]['dst'] - d_i) * (x-x_i) / d_i
#         G[1] += -2 * (anchors[i]['dst'] - d_i) * (y-y_i) / d_i
#         G[2] += -2 * (anchors[i]['dst'] - d_i) * (z-z_i) / d_i
        
#     return G
    

def anchor_selection_nearest(tag_pos, anchors, N):
    """Returns the N nearest anchors."""
    
    m = len(anchors)
    if N > m:
        raise Exception("Impossible to select",N,"among",m,"anchors.")
    
    distances = {i: np.linalg.norm(tag_pos - np.array([anchors[i]['x'], anchors[i]['y'], anchors[i]['z']])) for i in anchors }
    
    selected_IDs = nsmallest(N, distances, key = distances.get)
        
    selected_anchors = {}
            
    for i in range(N):
        selected_anchors[selected_IDs[i]] = anchors[selected_IDs[i]]
    
    return selected_anchors



def anchor_selection_variance_z(anchors, N):
    """Returns N anchors that maximize the variance along the z direction."""
    # First, the anchors have to be sorted
    m = len(anchors)
    if N > m:
        raise Exception("Impossible to select",N,"among",m,"anchors.")
    z_values = []
    z_indices = []
    for ID in anchors:
        z_values.append(anchors[ID]['z'])
        z_indices.append(ID)
    
    ind = np.argsort(z_values)
    
    max_var = 0
    max_var_k = 0

    for k in range(1,N):
        k_min = [z_values[ind[i]] for i in range(k)]
        Nk_max = [z_values[ind[i]] for i in range(m-(N-k),m)]
        v = np.var(np.concatenate((k_min, Nk_max)))
        if v > max_var:
            max_var = v
            max_var_k = k
    
    
    selected_anchors = {}
    for i in range(max_var_k):
        selected_anchors[z_indices[ind[i]]] = anchors[z_indices[ind[i]]]
        
    for i in range(m-(N-max_var_k), m):
        selected_anchors[z_indices[ind[i]]] = anchors[z_indices[ind[i]]]
    
    return selected_anchors
    


def anchor_selection_random(anchors, N):
    """Returns a random subset of N anchors among those presented."""
    if N > len(anchors):
        raise Exception("N="+str(N)+" is greater than the number of anchors ("
                        + str(len(anchors))+")")
    # Unpacking to get the anchors IDs = dict keys
    IDs = [*anchors]
    
    ID_selection = sample(IDs, N)
    selected_anchors =  {}
    for i in ID_selection:
        selected_anchors[i] = anchors[i]
        
    return selected_anchors



    
def volume_tetrahedron(vertices):
    """Returns the volume of the tetrahedron formed by 4 points
    Vertices is a 2D array of size 4 x 3"""
    a = np.array(vertices[0])
    b = np.array(vertices[1])
    c = np.array(vertices[2])
    d = np.array(vertices[3])
    return 1/6 * abs((a-d).dot(np.cross(b-d,c-d)))


def anchor_selection_det_covariance(anchors, N, N_tries=100, cov_3D=True):
    """Returns N anchors that maximize the determinant of the covariance
    matrix. If N_tries = 0, the function goes through all N_anchors choose N
    combinations.
    
    If cov_3D is set to True, the covariance matrix of the XYZ coordinates is
    computed. If false, only the XY coordinates are used (and thus, only a
    2 x 2 covariance matrix is obtained).
    
    If N_tries == 0, all combinations will be tested. If N_tries > 0, only
    this amount of random tries will be tested."""
    
    covariance = 0
    selected_anchors = None
    ID_selected = None
    vertices = np.zeros((N,3)) if cov_3D else np.zeros((N,2))
    
    if N_tries == 0:
        # Return list of tuples containing the IDs of the anchors of the
        # combination.
        combi = combinations(anchors, N)
        for c in combi:
            for i in range(N):
                vertices[i][0] = anchors[c[i]]['x']
                vertices[i][1] = anchors[c[i]]['y']
                if cov_3D:
                    vertices[i][2] = anchors[c[i]]['z']
                
            cov = np.linalg.det(np.cov(vertices.T))
          
            if cov > covariance:
                covariance = cov
                ID_selected = c
                    
        selected_anchors = {}        
        for ID in ID_selected:
            selected_anchors[ID] = anchors[ID]
    
    
    elif N_tries > 0:
        for i in range(N_tries):
            sel = anchor_selection_random(anchors, N)
            count = 0
            for a in sel:
                vertices[count][0] = sel[a]['x']
                vertices[count][1] = sel[a]['y']
                if cov_3D:
                    vertices[count][2] = sel[a]['z']
                count += 1
            
            cov = np.linalg.det(np.cov(vertices.T))
            if cov > covariance:
                covariance = cov
                selected_anchors = sel
    else:
        raise Exception(f"N_tries = {N_tries} but must be a non-negative int.")
    
    return selected_anchors
        
    

def anchor_selection_tetrahedron(anchors, N_tries=100):
    """Returns 4 anchors that maximize the volume of the 3D tetrahedron
    formed by the anchors.
    
    If N_tries == 0, all combinations will be tested. If N_tries > 0, only
    this amount of random tries will be tested."""

    if len(anchors) < 4:
        raise Exception(f"The number of anchors ({len(anchors)}) must be at least 4.")
    
    volume = 0
    selected_anchors = None
    vertices = np.zeros((4,3))
    
    # 1st method: try all combinations of 4 anchors.
    if N_tries == 0:
        # Return list of tuples containing the IDs of the anchors of the
        # combination.
        combi = combinations(anchors, 4)
        for c in combi:
            for i in range(len(c)):
                vertices[i][0] = anchors[c[i]]['x']
                vertices[i][1] = anchors[c[i]]['y']
                vertices[i][2] = anchors[c[i]]['z']
                
            vol = volume_tetrahedron(vertices)
            # Try to weigh using the variance on z
            # vol *= np.var(vertices.T[2])
            
            if vol > volume:
                volume = vol
                selected_anchors = {}
                for ID in c:
                    selected_anchors[ID] = anchors[ID]
                    
    # 2nd method: try a certain number of 4 combinations and choose the best
    else:
        for i in range(N_tries):
            sel = anchor_selection_random(anchors, 4)
                       
            count = 0
            for a in sel:
                vertices[count][0] = sel[a]['x']
                vertices[count][1] = sel[a]['y']
                vertices[count][2] = sel[a]['z']
                count += 1
            vol = volume_tetrahedron(vertices)
            vol *= np.var(vertices.T[2])
            if vol > volume:
                volume = vol
                selected_anchors = sel
        
    return selected_anchors


def ranging(tag_pos: np.ndarray, anchors: dict, ranging_dst: float):
    """Scans the anchors and returns those in the specified range.
     
    Parameters
    ----------
    tag_pos : tag position (numpy array of length 3)
    anchors : dict of anchors (list of dicts) {ID: {'x', 'y', 'z'}}
    ranging_dst -- ranging distance
    
    Returns
    ----------
    selected_anchors : dict of anchors with only those within the ranging dist
    """
    selected_anchors = {}
    for i in anchors:
        a = anchors[i]
        sqdst = (a['x'] - tag_pos[0])**2 + (a['y'] - tag_pos[1])**2 + (a['z'] - tag_pos[2])**2
        if sqdst <= ranging_dst**2:
            selected_anchors[i] = anchors[i]
            
    return selected_anchors


def generate_UWB_TWR(tag_pos, anchors, noise_model, noise_params):
    """Generates ranging distances for UWB_TWR measurements for each anchor
    and adds noise following the given noise model and parameters.
     
    Parameters
    ----------
    tag_pos : tag position (numpy array of length 3)
    anchors : dict of anchors (list of dicts) {ID: {'x', 'y', 'z'}}
    noise_model : only "gaussian" is supported
    noise_params : mu and sigma for the noise
    
    Returns
    ----------
    None. Adds a "dst" field to each anchor.
    """
    for i in anchors:
        anc_pos = np.array([anchors[i]['x'], anchors[i]['y'], anchors[i]['z']])
        dst = np.linalg.norm(tag_pos - anc_pos)
        
        # Error generation
        err = 0
        if noise_model == "gaussian":
            err = np.random.normal(noise_params['mu'], noise_params['sigma'])
        else:
            raise Exception(f"Noise model not supported: {noise_model}.")
        anchors[i]['dst'] = dst + err
        anchors[i]['dst'] = np.floor(100 * anchors[i]['dst']) / 100
    
    return

def locate_grid(filename: str, simu_name: str, replace_file: bool, anchors: dict,
                x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray,
                N_tries: int, options: dict) -> None:
    """
    Scan the given 3D grid of positions and append localization results and
    information frto the specified file. 
    
    The execution of this function might take a long time. A percentage of the
    progression is printed every ten percent.
     
    Parameters
    ----------
    filename : name of the file to save the localization results
    simu_name : simulation name, used by the statistics module to distinguish
            simulations that are plotted on the same graphs
    replace_file : if True, the file will be replaced every time this function
        is called. If False, it will append the results to the existing file
        (if it already exists, otherwise will create a new one)
    x_grid, y_grid, z_grid : 1D numpy arrays containing the ticks of the X, Y
        and Z axes where the tag is to be localized
    N_tries : number of successive localizations (and measurement generations)
        to be run at each position of the grid
    
    Returns
    ----------
    None. This function writes the results directly to the text file. 
    """
    
    # Generate unique ID for the simulation results, based on the simulation
    # beginning timestamp
    simu_uid = int(np.floor(time()))
    
    
    try:
        open_mode = "a+"
        if replace_file:
            open_mode = "w+"
        outfile = open(filename, open_mode)
        options_outfile = open("options_" + filename, open_mode)
        to_write = {}
        to_write['simu_uid'] = simu_uid
        to_write['simu_name'] = simu_name
        to_write['options'] = options
        to_write['anchors'] = anchors
        to_write['x_grid'] = x_grid.tolist()
        to_write['y_grid'] = y_grid.tolist()
        to_write['z_grid'] = z_grid.tolist()
        to_write['N_tries'] = N_tries
        json.dump(to_write, options_outfile)
        options_outfile.write('\n')
        options_outfile.close()
    except:
        raise Exception(f"An error occured when opening file {outfile}.")
    
    N_tot_steps = len(x_grid) * len(y_grid) * len(z_grid) * N_tries
    
    step_count = 0
    
    result = {}
    
    print("="*80)
    print(f"== Starting localization on grid - simulation: {simu_name}")
    
    for i in range(len(x_grid)):
        for j in range(len(y_grid)):
            for k in range(len(z_grid)):
                for n in range(N_tries):
                    step_count += 1
                    if (step_count / N_tot_steps) % 0.1 < ((step_count-1)/N_tot_steps) % 0.1:
                        print(f"{((step_count / N_tot_steps) // 0.1) * 10} %")
                    
                    x = x_grid[i]
                    y = y_grid[j]
                    z = z_grid[k]
                    tag_pos = np.array([x, y, z])
                    
                    anchors_ranging = ranging(tag_pos, anchors, options['ranging_distance'])
                    
                    N_anchors_pre = min(len(anchors_ranging), options['N_anchors_pre'])
                    N_anchors_post = min(len(anchors_ranging), options['N_anchors_post'])
                    
                    IDs_anchors_ranging = [*anchors_ranging]
                    
                    # If N_anchors_pre is zero, there is no random pre-selection
                    # of anchors. Otherwise, only N_anchors_pre of those within
                    # ranging distance will be kept, with probability
                    # proportional to the inverse of the tag-anchor distance
                    pre_selected_anchors = {}
                    
                    if N_anchors_pre > 0:
                        distances = np.array([np.linalg.norm(tag_pos - np.array([anchors_ranging[a]['x'], anchors_ranging[a]['y'], anchors_ranging[a]['z']])) for a in IDs_anchors_ranging])
                        
                        ID = np.random.choice(IDs_anchors_ranging, N_anchors_pre, replace=False,
                                              p=(1./distances) / sum(1./distances))
                        for ii in ID:
                            pre_selected_anchors[int(ii)] = anchors_ranging[ii]
                    else:
                        pre_selected_anchors = anchors_ranging
                    
                    bnds = options['bounds']
                    
                    # Anchor selection procedure
                    # It is done _before_ the ranging measurements, just like
                    # what would happen in real life if we want to avoid
                    # unnecessary energy consumption and do the ranging a
                    # posteriori, after selecting a reduced subset of anchors.
                    
                    if options['anchor_selection'] == "nearest":
                        post_selected_anchors = anchor_selection_nearest(tag_pos, pre_selected_anchors, N_anchors_post)
                    elif options['anchor_selection'] == "variance_z":
                        post_selected_anchors = anchor_selection_variance_z(pre_selected_anchors, N_anchors_post)
                    elif options['anchor_selection'] == "det_covariance_2D":
                        post_selected_anchors = anchor_selection_det_covariance(pre_selected_anchors, N_anchors_post, N_tries=0, cov_3D=False)
                    elif options['anchor_selection'] == "det_covariance_3D":
                        post_selected_anchors = anchor_selection_det_covariance(pre_selected_anchors, N_anchors_post, N_tries=0, cov_3D=True)
                    elif options['anchor_selection'] == "random":
                        post_selected_anchors = anchor_selection_random(pre_selected_anchors, N_anchors_post)
                    else:
                        raise Exception(f"Anchor selection method not supported: {options['anchor_selection']}.")
                    
                    
                    
                    # Measurement generation
                    # It will add the measurements directly in the
                    # post_selected_anchorsdict by adding another field.
                    if options['method'] == "UWB_TWR":
                        generate_UWB_TWR(tag_pos, post_selected_anchors,
                                         options['noise_model'], options['noise_params'])
                    else:
                        raise Exception(f"Measurement method not supported: {options['method']}.")
                    
                    # Choice of the initial guess
                    if options['initial_pos']['type'] == "bary_z0":
                        initial_guess = np.sum([[post_selected_anchors[i]['x'], post_selected_anchors[i]['y'], post_selected_anchors[i]['z']] for i in post_selected_anchors], axis=0)/len(post_selected_anchors)
                        initial_guess[2] = float(options['initial_pos']['initial_z'])
                    
                    elif options['initial_pos']['type'] == "weighted_bary":
                        initial_guess = np.sum([[post_selected_anchors[i]['x'], post_selected_anchors[i]['y'], post_selected_anchors[i]['z']] for i in post_selected_anchors], axis=0)/len(post_selected_anchors)
                    else:
                        raise Exception("Initial position option not supported.")
                        
                    
                    if options['optimization'] == "basic":
                        res = minimize(cost_function,
                                       initial_guess,
                                       args=(post_selected_anchors, 0),
                                       method='L-BFGS-B',
                                       bounds=bnds,
                                       tol=options['tolerance'],
                                       #jac=gradient_cost_function,
                                       options={'disp': False})
                    else:
                        raise Exception(f"Optimization option not supported: {options['optimization']}.")
                    
                    # Store all the localization information in a dict to be
                    # dumped as a JSON object in the log file to generate the
                    # stats.
                    # 
                    # The options are the same for all the localization steps
                    # of the simulation and can thus be retrieved thanks to the
                    # simu_uid in another file (beginning by "options_").
                    
                    # This is redundant with the options + post_selected_anchors information
                    # result['initial_pos'] = {'x': initial_guess[0],
                    #                          'y': initial_guess[1],
                    #                          'z': initial_guess[2]}
                    result['simu_uid'] = simu_uid
                    result['res_pos'] = {'x': res.x[0],
                                         'y': res.x[1],
                                         'z': res.x[2]}
                    result['tag_pos'] = {'x': tag_pos[0],
                                         'y': tag_pos[1],
                                         'z': tag_pos[2]}
                    
                    # These include the ranging distances
                    result['post_selected_anchors'] = post_selected_anchors
                    
                    json.dump(result, outfile)
                    outfile.write('\n')
        
    outfile.close()  

    print("Done.")
    return        
                    