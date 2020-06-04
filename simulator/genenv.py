#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:44:17 2020


__author__ = "Rob Knight, Gavin Huttley, and Peter Maxwell"
__copyright__ = "Copyright 2007, The Cogent Project"
__credits__ = ["Rob Knight", "Peter Maxwell", "Gavin Huttley",
                    "Matthew Wakefield"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Rob Knight"
__email__ = "rob@spot.colorado.edu"
__status__ = "Production"
@author: axel
"""

from typing import Union
import numpy as np
import json

# List of public objects of the module, as interpreted by "import *"

__all__ = ['generate_anchors_from_list',
           'generate_anchors_from_json_file',
           'generate_random_anchors',
           'save_anchors_to_json_file']




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
    recorder JSON file.
     
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
    

def save_anchors_to_json_file(anchors: dict, filename: str) -> None:
    try:
        with open(filename, 'w') as fp:
            json.dump(anchors, fp)
    except EnvironmentError:
        raise Exception(f"Unable to write to file: {filename}")
    
    return


def generate_random_anchors() -> dict:
    # To do: various ways of random generation.
    # * beta-Ginibre process
    # * 3D uniform distribution
    # * equally spaced
    
    raise Exception("Unimplemented.")
    
    return {}
        