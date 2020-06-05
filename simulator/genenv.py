#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environment anchor generation module

Provide functions to generate settings and save it to a JSON file.


__author__ = "Axel Torchy"
__copyright__ = "Copyright 2020, Wizzilab"
__credits__ = ["Axel Torchy", "Wizzilab"]
__version__ = "1.0"
__maintainer__ = "Axel Torchy"
__email__ = "axel@wizzilab.com"
__status__ = "Production"

"""

from typing import Union, Dict
import numpy as np
import json

# List of public objects of the module, as interpreted by "import *"

__all__ = ['generate_anchors_from_list',
           'generate_anchors_from_json_file',
           'generate_manual_anchors',
           'generate_uniform_anchors',
           'save_anchors_to_json_file',
           'generate_grid']




def generate_anchors_from_list(anc_list: Union[list, np.ndarray]) -> dict:
    """
    Generate a dict containing the 3D coordinates of the anchors from
    the list of XYZ coordinates.
     
    Parameters
    ----------
    anc_list : N x 3 list or numpy array containing the X,Y,Z anchor coord.
    
    Returns
    ----------
    anchors : dict of dict containing the anchors, indexed by their position as
    integers in the anc_list array.
    """
    
    anchors = {i: { 'x': anc_list[i][0],
                    'y': anc_list[i][1],
                    'z': anc_list[i][2]} for i in range(len(anc_list))}
    
    return anchors


def generate_anchors_from_json_file(filename: str) -> dict:
    """
    Generate a dict containing the 3D coordinates of the anchors from a pre-
    recorded JSON file.
     
    Parameters
    ----------
    filename : name of the JSON file. The object should have the same structure
    as the anchors dictionaries as explained in the README.
    
    Returns
    ----------
    anchors : dict of dict containing the anchors as indexed in the JSON file.
    """
    try:
        with open(filename, 'r') as json_file:
            anchors = json.load(json_file)
    except EnvironmentError:
        raise Exception(f"Unable to read file: {filename}")
        
    return anchors
    

def save_anchors_to_json_file(anchors: Dict[int, dict], filename: str) -> None:
    """
    Save a current anchor (dict) configuration to a JSON file.
     
    Parameters
    ----------
    anchors : dictionary of anchors matching the correct structure (see README)
    filename : name of the JSON file. 
    """
    try:
        with open(filename, 'w') as fp:
            json.dump(anchors, fp)
    except EnvironmentError:
        raise Exception(f"Unable to write to file: {filename}")
    
    return


def generate_manual_anchors() -> dict:
    """
    Prompts the user to manually and successively enter the X,Y,Z coordinates
    of the anchors.
     
    Parameters
    ----------
    None
    
    Returns
    ----------
    anchors : dict of dict containing the anchors, index from 0 to N_anchors-1
    The coordinates are truncated (centimeter precision).
    """
    
    print("You will be asked to manually input the XYZ coordinates of the anchors.")
    print("You can stop at any moment by typing 'q' or 'Q' instead of a float.")
    print("Then, the current anchor will be dismissed and only the completed anchors will be saved.")
    
    anchors = {}
    i = 0
    
    while True:
        print(f"==== Anchor {i}:", end="")
        x = input("Please enter the X coordinate.   ")
        if x == "q" or x == "Q":
            break
        
        try:
            x = float(x)
        except ValueError:
            print("Please enter a float for the X coordinate. Try again.")
            continue
        
        y = input("Please enter the Y coordinate.   ")
        if y == "q" or y == "Q":
            break
        
        try:
            y = float(y)
        except ValueError:
            print("Please enter a float for the Y coordinate. Try again.")
            continue
        
        z = input("Please enter the Z coordinate.   ")
        if z == "q" or z == "Q":
            break
        
        try:
            z = float(z)
        except ValueError:
            print("Please enter a float for the Z coordinate. Try again.")
            continue
        
        anchors[i] = {'x': x,
                      'y': y,
                      'z': z}
        
        print(f"  ** Anchor {i} added: (x, y, z) = ({x}, {y}, {z})")
        
        i += 1
        
    return anchors 


def generate_uniform_anchors(x_min: float, x_max: float,
                             y_min: float, y_max: float,
                             z_min: float, z_max: float,
                             N_anchors: int) -> dict:
    """
    Generate a dict containing the 3D coordinates of the anchors inside the 3D
    cuboid, following a uniform random distribution.
     
    Parameters
    ----------
    x_min, x_max : bounds of the 3D cuboid on the X axis
    y_min, y_max : bounds of the 3D cuboid on the Y axis
    z_min, z_max : bounds of the 3D cuboid on the Z axis
    N_anchors : number of anchors to be generated
    
    Returns
    ----------
    anchors : dict of dict containing the anchors, index from 0 to N_anchors-1
    The coordinates are truncated (centimeter precision).
    """
    
    anchors = {}
    
    for i in range(N_anchors):
        coord = np.array([x_min, y_min, z_min]) + np.random.rand(3) * np.array(
            [x_max-x_min, y_max-y_min, z_max-z_min])
        coord = np.floor(coord * 100) / 100
        anchors[i] = {'x': coord[0],
                      'y': coord[1],
                      'z': coord[2]}
    
    
    return anchors
        

    # To do: various ways of random generation.
    # * beta-Ginibre process
    # * equally spaced
    # * ...


def generate_grid(x_min, x_max, y_min, y_max, z_min, z_max, step):
    """
    Generates a grid for positions to be tested. The step is assumed to be 
    the same over all 3 dimensions.
    Returns a numpy 3D meshgrid.
    """
    
    x_ = np.arange(x_min, x_max, step)
    y_ = np.arange(y_min, y_max, step)
    z_ = np.arange(z_min, z_max, step)
        
    return x_, y_, z_