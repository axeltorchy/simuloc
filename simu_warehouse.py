#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:02:15 2020

@author: axel
"""

import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D 
from scipy.optimize import minimize
from scipy.special import comb
from random import sample, seed
from itertools import combinations

matplotlib.rcParams['figure.dpi'] = 150
size_factor = 1.2
matplotlib.rcParams['figure.figsize'] = [size_factor * 8.0, size_factor * 6.0]
matplotlib.rcParams['lines.markersize'] = 6.0 


# For reproducibility
np.random.seed(0)
seed(0)

# Simulation parameters
default_z   = 2         # default anchor altitude
min_z       = 1.
max_z       = 5.
ranging_dst = 20
noise_std   = 0.05      # standard deviation gaussian noise rangings
noise_mean  = 0.00     # mean gaussian noise rangings

# If True, altitude will be chosen randomly, uniformly between min_z and max_z
random_z  = True

use_mean_z = False


tag_pos = np.array([2, 3, 2])

def ranging(tag_pos, anchors, ranging_dst):
    """Scans the anchors and returns the IDs of those in the
    specified range.
    
    Keyword arguments:
    tag_pos -- tag position (numpy array of length 3)
    anchors -- list of anchors (list of dicts) {ID: {'x', 'y', 'z'}}
    ranging_dst -- ranging distance
    """
    ID = []
    for i in anchors:
        a = anchors[i]
        sqdst = (a['x'] - tag_pos[0])**2 + (a['y'] - tag_pos[1])**2 + (a['z'] - tag_pos[2])**2
        if sqdst <= ranging_dst**2:
            ID.append(i)
    return np.array(ID)



def cost_function(point, anchors, weight_method, constrained_alt):
    x = point[0]
    y = point[1]
    z = point[2]
    if constrained_alt > 0:
        #print("Constrained altitude:",constrained_alt)
        z = constrained_alt

    s = 0
    
    for i in anchors:
        x_i = anchors[i]['x']
        y_i = anchors[i]['y']
        z_i = anchors[i]['z']
        
        # Explore influence if squared or not squared
        # d_i = np.sqrt(np.sum((np.array([x,y,z])-np.array([x_i, y_i, z_i]))**2))
        # s += ((anchors[i]['dst'] - d_i))**2
        d_i = np.sqrt(np.sum((np.array([x,y,z])-np.array([x_i, y_i, z_i]))**2))
        
        if weight_method == 1:
            s += ((anchors[i]['dst']) - d_i)**2 / ((anchors[i]['dst'])**2)
        else:
            s += ((anchors[i]['dst']) - d_i)**2
            #s += ((anchors[i]['dst'])**2 - d_i**2)**2
    
    return s

def gradient_cost_function(point, anchors, weight_method, constrained_alt):
    x = point[0]
    y = point[1]
    z = point[2]
    G = np.zeros(3)
    for i in anchors:
        x_i = anchors[i]['x']
        y_i = anchors[i]['y']
        z_i = anchors[i]['z']
        d_i = np.sqrt(np.sum((np.array([x,y,z])-np.array([x_i, y_i, z_i]))**2))
        G[0] += -2 * (anchors[i]['dst'] - d_i) * (x-x_i) / d_i
        G[1] += -2 * (anchors[i]['dst'] - d_i) * (y-y_i) / d_i
        G[2] += -2 * (anchors[i]['dst'] - d_i) * (z-z_i) / d_i
        
    return G
    

def anchor_selection_nearest(anchors, N):
    """Returns the N nearest anchors."""
    m = len(anchors)
    if N > m:
        raise Exception("Impossible to select",N,"among",m,"anchors.")
    
    min_dsts = 10e8 * np.ones(N)
    min_indx = np.zeros(N, dtype=int)
    
    for i in anchors:
        dst = anchors[i]['dst']
        ind_max = np.argmax(min_dsts)
        if min_dsts[ind_max] > dst:
            min_dsts[ind_max] = dst
            min_indx[ind_max] = i
        
    selected_anchors = {}
            
    for i in range(N):
        selected_anchors[min_indx[i]] = anchors[min_indx[i]]
    
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
    # print("z_values")
    # print(z_values)
    # print("ind")
    # print(ind)

    for k in range(1,N):
        k_min = [z_values[ind[i]] for i in range(k)]
        Nk_max = [z_values[ind[i]] for i in range(m-(N-k),m)]
        v = np.var(np.concatenate((k_min, Nk_max)))
        if v > max_var:
            max_var = v
            max_var_k = k
    
    #print("N=",N,"max_var_k:",max_var_k)
    
    selected_anchors = {}
    for i in range(max_var_k):
        selected_anchors[z_indices[ind[i]]] = anchors[z_indices[ind[i]]]
        
    for i in range(m-(N-max_var_k), m):
        selected_anchors[z_indices[ind[i]]] = anchors[z_indices[ind[i]]]
    
    #print(selected_anchors)
    return selected_anchors
    



def anchor_selection_random(anchors, N):
    """Returns a random subset of N anchors among those presented."""
    if N > len(anchors):
        raise Exception("N="+str(N)+" is greater than the number of anchors ("
                        + str(len(anchors))+")")
    IDs = []
    for i in anchors:
        IDs.append(i)
    ID_selection = sample(IDs, N)
    selected_anchors =  {}
    for i in ID_selection:
        selected_anchors[i] = anchors[i]
        
    return selected_anchors



    
def volume_tetrahedron(vertices):
    # Vertices is a 2D array of size 4x3
    
    # If dilatation of the z coordinate
    # alpha = 10
    # min_z = vertices[0][2]
    # for j in range(1, len(vertices)):
    #     if vertices[j][2] < min_z:
    #         min_z = vertices[j][2]
            
    # for j in range(len(vertices)):
    #     vertices[j][2] = min_z + alpha * (vertices[j][2] - min_z)
        
    a = np.array(vertices[0])
    b = np.array(vertices[1])
    c = np.array(vertices[2])
    d = np.array(vertices[3])
    return 1/6 * abs((a-d).dot(np.cross(b-d,c-d)))


def anchor_selection_det_covariance(anchors, N, N_tries=100, cov_3D=True):
    """Returns N anchors that maximize the determinant of the covariance
    matrix. If N_tries = 0, the function goes through all N_anchors choose N
    combinations."""
    #N_tries = 300 # Number of random N-anchor subsets to be tested
    covariance = 0
    selected_anchors = None
    ID_selected = None
    vertices = np.zeros((N,3)) if cov_3D else np.zeros((N,2))
    
    if N_tries == 0:
        # Return list of tuples containing the IDs of the anchors of the
        # combination.
        combi = combinations(anchors, N)
        for c in combi:
            for i in range(len(c)):
                vertices[i][0] = anchors[c[i]]['x']
                vertices[i][1] = anchors[c[i]]['y']
                if cov_3D:
                    vertices[i][2] = anchors[c[i]]['z']
                
                
            # print(vertices)
            # renormalisation
            # for i in range(3 if cov_3D else 2):
            #     i_min, i_max = min([vertices[j][i] for j in range(N)]), max([vertices[j][i] for j in range(N)])
            #     for j in range(N):
            #         vertices[j][i] = vertices[j][i] / (i_max - i_min) + i_min / (i_min - i_max)
            # print(vertices)
            # print(np.var(vertices.T[0]))
            # print(np.var(vertices.T[1]))
            # print(np.var(vertices.T[2]))
            # raise Exception("STOP")
            
            #cov = np.power(np.var(vertices.T[0]) + np.var(vertices.T[1]) +  20 * np.var(vertices.T[2]), 1/3)
            
            # cov = (max([vertices[i][0] for i in range(N)]) - min([vertices[i][0] for i in range(N)])) * (
            #     max([vertices[i][1] for i in range(N)]) - min([vertices[i][1] for i in range(N)])) * (
            #         max([vertices[i][2] for i in range(N)]) - min([vertices[i][2] for i in range(N)]))

            cov = np.linalg.det(np.cov(vertices.T))
            
            # cov = np.prod([np.var(vertices.T[i]) for i in range(len(vertices.T))])
                         
            
            if cov > covariance:
                covariance = cov
                ID_selected = c
                    
        selected_anchors = {}        
        for ID in ID_selected:
            selected_anchors[ID] = anchors[ID]
    else:
        for i in range(N_tries):
            sel = anchor_selection_random(anchors, N)
            count = 0
            for a in sel:
                vertices[count][0] = sel[a]['x']
                vertices[count][1] = sel[a]['y']
                vertices[count][2] = sel[a]['z']
                count += 1
            #print(np.cov(vertices.T))
            cov = np.linalg.det(np.cov(vertices.T))
            cov *= np.var(vertices.T[2])**4
            #print(cov)
            if cov > covariance:
                covariance = cov
                selected_anchors = sel
            
    return selected_anchors
        
    

def anchor_selection_tetrahedron(anchors, N_tries=20):
    """Returns 4 anchors that maximize the volume of the 3D tetrahedron
    formed by the anchors."""
    # 1st method: try a certain number of 4 combinations and choose the best
    #N_tries = 100
    volume = 0
    selected_anchors = None
    vertices = np.zeros((4,3))
    
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
            vol *= np.var(vertices.T[2])
            
            if vol > volume:
                volume = vol
                selected_anchors = {}
                for ID in c:
                    selected_anchors[ID] = anchors[ID]
                
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
            

def initialize_anchors(anchors, N_x=5, N_y=5, d=20, min_z=1, max_z=3, random_z=True, std_z=0.1):
    """Adds the grid of anchors with random altitude between min_z and max_z
    (uniform) if random_z is True."""
    for i in range(N_x):
        for j in range(N_y):
            z = default_z
            if random_z:
                #z = min_z + np.random.rand() * (max_z - min_z)
                #z = np.random.normal((max_z + min_z)/2, (max_z - min_z) / 0.5)
                if (i+j) % 2 == 0:
                    z = np.random.normal(min_z, std_z)
                    z = 4
                else:
                    z = np.random.normal(max_z, std_z)
                    z= 1
                z = np.random.normal(4,0.01)
                
                z = min_z + np.random.rand() * (max_z - min_z)
                z = np.floor(z*100)/100
            anchors[i*N_y+j] = {'x': i*d,
                                'y': j*d,
                                'z': z}
    return



def plot_setting(tag_pos, anchors, title="Setting"):
    plt.figure(1)
    plt.axis('equal')
    plt.title(title)
    plt.scatter(tag_pos[0], tag_pos[1], marker="s", color="red")
    # plt.annotate("Tag", (tag_pos[0], tag_pos[1]), textcoords="offset points",
    #             xytext=(0,7), ha='center')
    circle = plt.Circle((tag_pos[0], tag_pos[1]),
                        ranging_dst,
                        color='b', fill=True, alpha=0.1)
    ax = plt.gca()
    ax.add_artist(circle)
    for i in range(len(anchors)):
        a = anchors[i]
        plt.scatter(a['x'], a['y'], marker="D", color="maroon")
        plt.annotate(str(i),(a['x'],a['y']), textcoords="offset points",
                     xytext=(0,7), ha='center')
        plt.annotate("z = "+str(a['z']),(a['x'],a['y']), textcoords="offset points",
                     xytext=(0,-15), ha='center', fontsize=8.5)
    
    plt.show()



def anchors_from_IDs(anchors, ID, tag_pos):
    distances = {}
    errors = {}
    selected_anchors = {}
    # print(anchors)
    # raise Exception("SOP")
    for i in ID:
        a = anchors[i]
        x = a['x']
        y = a['y']
        z = a['z']
        anchor = np.array([x,y,z])
        err = np.random.normal(noise_mean, noise_std)
        distances[i] = np.linalg.norm(anchor - tag_pos) + err
        errors[i] = err
        # a['dst'] = distances[i]
        a['dst'] = np.round(100*distances[i])/100
        selected_anchors[i] = a
    return selected_anchors, distances, errors


def normal_density(anchors, x, y, z, sigma):
    res = 1.
    for a in anchors:
        dst = anchors[a]['dst']
        x_i = anchors[a]['x']
        y_i = anchors[a]['y']
        z_i = anchors[a]['z']
        d = np.linalg.norm(np.array([x, y, z]) - np.array([x_i, y_i, z_i]))
        res *= np.exp(-(d - dst)**2 / sigma **2)
    return res




anchors = {}
# initialize_anchors(anchors, N_x, N_y, d, min_z, max_z, random_z)

def xy_to_ij(x,y):
    """Returns the coordinates on the warehouse picture."""
    # 10.92 m = 860 px
    # 25.32 = 1994 px
    conv_factor = 1994. / 25.32
    return (65 + conv_factor * x, 2058 - conv_factor*y)

def ij_to_xy(i,j):
    conv_factor = 25.32 / 1994
    return ((i-65)*conv_factor, (2058 - j)*conv_factor)
    

# 18 anchors cover a surface of more than 500 square meters
anchors_locations = [
    [0 ,5.12, 3],     # 0
    [0, 10, 3],         # 1
    [0, 25.31, 3],         # 2
    [7.35, 10, 3],       # 3
    [7.2, 25.31, 3],      # 4
    [7.35, 5.12, 3],      # 5
    [10.08, 25.31, 3],  # 6
    [21, 25.31, 3],     # 7
    [24.2, 21, 3],      # 8
    [10.5, 5.12, 3],      # 9
    [19, 5.12, 3],      # 10
    [24, 6.9, 3],       # 11
    [27.5, 17, 3],      # 12
    [30.7, 11.9, 3],     # 13
    [14.9, 14.4, 3],     # 14
    [21.6, 13.9, 3],     # 15
    [0, 21, 3],     # 16
    [7.2, 21, 3],     # 17
    [0, 15.5, 3],     # 18
    [7.2, 15.5, 3]     # 19
    ]

# if __name__ == "__main__":
print("Start Warehouse simulation.")
warehouse = plt.imread("warehouse.png")
f1 = plt.imshow(warehouse)
plt.title("Warehouse - anchor positions (2D)")
f1.axes.get_xaxis().set_visible(False)
f1.axes.get_yaxis().set_visible(False)
for i in range(len(anchors_locations)):
    pos = xy_to_ij(anchors_locations[i][0], anchors_locations[i][1])
    f2 = plt.scatter(pos[0], pos[1], color="red", marker="D", s=60)
    f2.axes.get_xaxis().set_visible(True)
    f2.axes.get_yaxis().set_visible(True)
    f2 = plt.scatter(pos[0], pos[1], color="red", marker="D", s=60)
    plt.annotate("A %d" % i, (pos[0], pos[1]), textcoords="offset points",
                 xytext=(0,7), ha='center', color="red")
    labels = np.arange(-2, 33, 2)
    locs = labels * 1994 / 25.32 + 65
    plt.xticks(locs, labels)
    labels = np.arange(-2, 29, 2)
    locs = 2058 - labels * 1994 / 25.32
    plt.yticks(locs, labels)
    
plt.show()


# Possible tag positions to be localized

x_tag = np.arange(0.2, 7.1, 0.4)
y_tag = np.arange(5.2, 25.2, 0.4)

x_tag = np.arange(0.2, 30.1, 0.5)
y_tag = np.arange(5.2, 25.2, 0.5)
z_tag = np.arange(0.2, 2.1, 0.4)

x_tag = np.arange(0.2, 30.1, 2)
y_tag = np.arange(5.2, 25.2, 2)
z_tag = np.arange(0.2, 2.1, 0.4)

#X,Y = np.meshgrid(x_tag, y_tag)
#pos = xy_to_ij(X, Y)

#plt.scatter(pos[0], pos[1], marker="+", color="blue", alpha=0.5, s=10)
    

anchors = {}

# Initialize anchors
for i in range(len(anchors_locations)):
    anchors_locations[i][2] = min_z + np.random.rand() * (max_z - min_z)
    anchors[i] = {'x': anchors_locations[i][0],
                  'y': anchors_locations[i][1],
                  'z': anchors_locations[i][2]}
import pprint
pp = pprint.PrettyPrinter(indent=4)

bnds = ((None, None), (None, None), (None, None))
bnds = ((None, None), (None, None), (None, None))
tolerance = 1e-7

SSE = {"det_covariance_2D":    0,
       "det_covariance_3D":    0,
       "variance_z":        0,
       "random":            0,
       "nearest":           0
       }

SSE_XY = {"det_covariance_2D": 0,
          "det_covariance_3D": 0,
          "variance_z":     0,
          "random":         0,
          "nearest":        0
          }

errors_rec = {"det_covariance_2D": [],
              "det_covariance_3D": [],
              "variance_z":        [],
              "random":            [],
              "nearest":           []
              }

errors_rec_XY = {"det_covariance_2D": [],
                 "det_covariance_3D": [],
                 "variance_z":        [],
                 "random":            [],
                 "nearest":           []
                 }


errors_rec_cor_z = {"det_covariance_2D": [],
              "det_covariance_3D": [],
              "variance_z":        [],
              "random":            [],
              "nearest":           []
              }

SSE_cor_z = {"det_covariance_2D":    0,
       "det_covariance_3D":    0,
       "variance_z":        0,
       "random":            0,
       "nearest":           0
       }

mean_distance = {"det_covariance_2D": [],
                 "det_covariance_3D": [],
                 "variance_z":        [],
                 "random":            [],
                 "nearest":           []
                 }



count_iterations = 0

sum_d = 0

z_guess = {"det_covariance_2D": [],
                 "det_covariance_3D": [],
                 "variance_z":        [],
                 "random":            [],
                 "nearest":           []
                 }
z_guess_cor = {"det_covariance_2D": [],
                 "det_covariance_3D": [],
                 "variance_z":        [],
                 "random":            [],
                 "nearest":           []
                 }


outfile = open("guesses.json", "w")
recordings = {}

for i in range(len(x_tag)):
    if i%10 == 0:
        print("x =",x_tag[i])
    for j in range(len(y_tag)):
        for k in range(len(z_tag)):
            N = 2
            for n in range(N):
                
                
                # arithmetic / weighted mean z coordinate
                def arithmetic_mean(anc, res_xyz):
                    # res_xyz = res.x from minimization function
                    z_sum = 0
                    z_rk_sum = 0
                    rk_sum = 0
                    z_count = 0
                    x, y = res_xyz[0], res_xyz[1]
                    for a in anc:
                        x_i = anc[a]['x']
                        y_i = anc[a]['y']
                        z_i = anc[a]['z']
                        dst = anc[a]['dst']
                        
                        # complex square root, retain modulus
                        delta_z_k = np.real( np.sqrt( dst**2 - (x - x_i)**2 - (y - y_i)**2 + 0j) )
                        
                        # anchor is considered to be always above tag 
                        z_sum += z_i - delta_z_k
                        z_rk_sum += (z_i - delta_z_k)/dst
                        z_count += 1
                        rk_sum += 1. / dst
                    
                    z = z_sum / z_count
                    z = z_rk_sum / rk_sum
                    return np.array([x, y, z])
                
                x = x_tag[i]
                y = y_tag[j]
                z = 1.6
                z = z_tag[k]
                tag_pos = np.array([x, y, z])
                
                
                bnds = ((None, None), (None, None), (z, z))
                
                
                #print("(x , y) = (%f , %f)" % (x, y))
                ID = ranging(tag_pos, anchors, ranging_dst)
                
                nb_tot_anchors = min(len(ID), 6)
                nb_anchors = min(len(ID), 4)
                
                distances = np.array([np.linalg.norm(tag_pos - np.array(anchors_locations[i])) for i in ID])
                #ID = np.random.choice(ID, min(len(ID), 8), replace=False)
                ID = np.random.choice(ID, nb_tot_anchors, replace=False, p=(1./distances) /sum(1./distances))
                
                sum_d += sum([np.linalg.norm(tag_pos - np.array(anchors_locations[i])) for i in ID])/len(ID)
                
                selected_anchors, distances, errors = anchors_from_IDs(anchors, ID, tag_pos)
                
                # pp.pprint(selected_anchors)
                
                # Initial guess = barycenter
                initial_guess = tag_pos + 0.5 * np.random.rand(3)
                initial_guess = np.sum([anchors_locations[i] for i in ID], axis=0)/len(ID)
                initial_guess[2] = 0
        
                anc = anchor_selection_variance_z(selected_anchors, nb_anchors)
                res = minimize(cost_function,
                               initial_guess,
                               args=(anc, 0, 0),
                               method='L-BFGS-B',
                               bounds=bnds,
                               tol=tolerance,
                               #jac=gradient_cost_function,
                               options={'disp': False})
                
                z_guess['variance_z'].append(res.x[2])
                
                SSE['variance_z'] += np.sum((res.x - tag_pos)**2)
                SSE_XY['variance_z'] += np.sum((res.x[0:2] - tag_pos[0:2])**2)
                errors_rec['variance_z'].append(np.sum((res.x - tag_pos)**2))
                errors_rec_XY['variance_z'].append(np.sum((res.x[0:2] - tag_pos[0:2])**2))
                
                # Correction of z coordinate
                res_cor = arithmetic_mean(anc, res.x)
                z_guess_cor['variance_z'].append(res_cor[2])
                
                SSE_cor_z['variance_z'] += np.sum((res_cor - tag_pos)**2)
                errors_rec_cor_z['variance_z'].append(np.sum((res_cor - tag_pos)**2))
                
                mean_distance['variance_z'].append(np.mean([anc[a]['dst'] for a in anc]))
                
                
                anc = anchor_selection_random(selected_anchors, nb_anchors)
                res = minimize(cost_function,
                               initial_guess,
                               args=(anc, 0, 0),
                               method='L-BFGS-B',
                               bounds=bnds,
                               tol=tolerance,
                               #jac=gradient_cost_function,
                               options={'disp': False})
                
                z_guess['random'].append(res.x[2])
                
                SSE['random'] += np.sum((res.x - tag_pos)**2)
                SSE_XY['random'] += np.sum((res.x[0:2] - tag_pos[0:2])**2)
                errors_rec['random'].append(np.sum((res.x - tag_pos)**2))
                errors_rec_XY['random'].append(np.sum((res.x[0:2] - tag_pos[0:2])**2))
                
                # Correction of z coordinate
                res_cor = arithmetic_mean(anc, res.x)
                z_guess_cor['random'].append(res_cor[2])
                
                SSE_cor_z['random'] += np.sum((res_cor - tag_pos)**2)
                errors_rec_cor_z['random'].append(np.sum((res_cor - tag_pos)**2))
                
                mean_distance['random'].append(np.mean([anc[a]['dst'] for a in anc]))
                
                
                
                
                anc = anchor_selection_det_covariance(selected_anchors, nb_anchors, 0, True)
                res = minimize(cost_function,
                               initial_guess,
                               args=(anc, 0, 0),
                               method='L-BFGS-B',
                               bounds=bnds,
                               tol=tolerance,
                               #jac=gradient_cost_function,
                               options={'disp': False})
                
                z_guess['det_covariance_3D'].append(res.x[2])
                
                SSE['det_covariance_3D'] += np.sum((res.x - tag_pos)**2)
                SSE_XY['det_covariance_3D'] += np.sum((res.x[0:2] - tag_pos[0:2])**2)
                errors_rec['det_covariance_3D'].append(np.sum((res.x - tag_pos)**2))
                errors_rec_XY['det_covariance_3D'].append(np.sum((res.x[0:2] - tag_pos[0:2])**2))
                
                # Correction of z coordinate
                res_cor = arithmetic_mean(anc, res.x)
                z_guess_cor['det_covariance_3D'].append(res_cor[2])
                
                SSE_cor_z['det_covariance_3D'] += np.sum((res_cor - tag_pos)**2)
                errors_rec_cor_z['det_covariance_3D'].append(np.sum((res_cor - tag_pos)**2))
                
                mean_distance['det_covariance_3D'].append(np.mean([anc[a]['dst'] for a in anc]))
                
                
                # 2D
                anc = anchor_selection_det_covariance(selected_anchors, nb_anchors, 0, False)
                res = minimize(cost_function,
                               initial_guess,
                               args=(anc, 0, 0),
                               method='L-BFGS-B',
                               bounds=bnds,
                               tol=tolerance,
                               #jac=gradient_cost_function,
                               options={'disp': False})
                
                z_guess['det_covariance_2D'].append(res.x[2])
                
                SSE['det_covariance_2D'] += np.sum((res.x - tag_pos)**2)
                SSE_XY['det_covariance_2D'] += np.sum((res.x[0:2] - tag_pos[0:2])**2)
                errors_rec['det_covariance_2D'].append(np.sum((res.x - tag_pos)**2))
                errors_rec_XY['det_covariance_2D'].append(np.sum((res.x[0:2] - tag_pos[0:2])**2))
                
                # Correction of z coordinate
                res_cor = arithmetic_mean(anc, res.x)
                z_guess_cor['det_covariance_2D'].append(res_cor[2])
                
                SSE_cor_z['det_covariance_2D'] += np.sum((res_cor - tag_pos)**2)
                errors_rec_cor_z['det_covariance_2D'].append(np.sum((res_cor - tag_pos)**2))
                
                mean_distance['det_covariance_2D'].append(np.mean([anc[a]['dst'] for a in anc]))
                
                
                
                
                anc = anchor_selection_nearest(selected_anchors, nb_anchors)
                res = minimize(cost_function,
                               initial_guess,
                               args=(anc, 0, 0),
                               method='L-BFGS-B',
                               bounds=bnds,
                               tol=tolerance,
                               #jac=gradient_cost_function,
                               options={'disp': False})
                
                z_guess['nearest'].append(res.x[2])
                
                SSE['nearest'] += np.sum((res.x - tag_pos)**2)
                SSE_XY['nearest'] += np.sum((res.x[0:2] - tag_pos[0:2])**2)
                errors_rec['nearest'].append(np.sum((res.x - tag_pos)**2))
                errors_rec_XY['nearest'].append(np.sum((res.x[0:2] - tag_pos[0:2])**2))
                
                # Correction of z coordinate
                res_cor = arithmetic_mean(anc, res.x)
                z_guess_cor['nearest'].append(res_cor[2])
                
                SSE_cor_z['nearest'] += np.sum((res_cor - tag_pos)**2)
                errors_rec_cor_z['nearest'].append(np.sum((res_cor - tag_pos)**2))
                
                mean_distance['nearest'].append(np.mean([anc[a]['dst'] for a in anc]))
                
                
                recording = {'x_tag':   x,
                             'y_tag':   y,
                             'z_tag':   z,
                             'x_guess': res.x[0],
                             'y_guess': res.x[1],
                             'z_guess': res.x[2],
                             'method':  'nearest'
                             }
                
                
                recordings[count_iterations] = recording
                
                count_iterations += 1

json.dump(recordings, outfile)
outfile.close()

print("\tMean tag-anchor distance:",sum_d/count_iterations)
print("")

fignb = 0


# %% Plot stats and diagrams
for l in ["det_covariance_2D", "det_covariance_3D", "variance_z", "random", "nearest"]:
    print("== %s" % l)
    print("\t3D:")
    print("\t  Sum of Squared Errors (SSE):", SSE[l])
    print("\t  Mean Squared Error (MSE):", SSE[l]/count_iterations)
    print("\t  Median Squared Error:",np.median(errors_rec[l]))
    print("\t  Max / Min of Squared Error:",np.max(errors_rec[l]),",",np.min(errors_rec[l]))
    print("\t  Z correction - Sum of Squared Errors (SSE):", SSE_cor_z[l])
    print("\t  Z correction - Mean Squared Error (MSE):", SSE_cor_z[l]/count_iterations)
    print("\t  Z correction - Median Squared Error:",np.median(errors_rec_cor_z[l]))
    print("\t  Z correction - Max / Min of Squared Error:",np.max(errors_rec_cor_z[l]),",",np.min(errors_rec_cor_z[l]))
    
    print("\t2D:")
    print("\t  Sum of XY Squared Errors (SSE)", SSE_XY[l])
    print("\t  XY Mean Squared Error (MSE)", SSE_XY[l]/count_iterations)
    print("\t  Median Squared Error:",np.median(errors_rec_XY[l]))
    print("\t  Max / Min of Squared Error:",np.max(errors_rec_XY[l]),",",np.min(errors_rec_XY[l]))
    
    # 2D error
    plt.figure(0)
    values, base = np.histogram(errors_rec_XY[l], bins=5000)
    cumulative = np.cumsum(values)
    plt.plot(base[:-1], cumulative/count_iterations, label=l)
    plt.figure(1)
    values, base = np.histogram(np.sqrt(errors_rec_XY[l]), bins=5000)
    cumulative = np.cumsum(values)
    plt.plot(base[:-1], cumulative/count_iterations, label=l)
    
    # 3D error
    plt.figure(2)
    values, base = np.histogram(errors_rec[l], bins=5000)
    cumulative = np.cumsum(values)
    p2 = plt.plot(base[:-1], cumulative/count_iterations, label=l)
    plt.figure(3)
    values, base = np.histogram(np.sqrt(errors_rec[l]), bins=5000)
    cumulative = np.cumsum(values)
    p3 = plt.plot(base[:-1], cumulative/count_iterations, label=l)
    
    # 3D error with z correction
    plt.figure(2)
    values, base = np.histogram(errors_rec_cor_z[l], bins=5000)
    cumulative = np.cumsum(values)
    plt.plot(base[:-1], cumulative/count_iterations, '--', label=l, color = p2[0].get_color())
    plt.figure(3)
    values, base = np.histogram(np.sqrt(errors_rec_cor_z[l]), bins=5000)
    cumulative = np.cumsum(values)
    plt.plot(base[:-1], cumulative/count_iterations, '--', label=l, color = p3[0].get_color())
    
    
    # 3D error
    plt.figure(4)
    values, base = np.histogram(errors_rec[l], bins=5000)
    cumulative = np.cumsum(values)
    p2 = plt.plot(base[:-1], cumulative/count_iterations, label=l)
    plt.figure(5)
    values, base = np.histogram(np.sqrt(errors_rec[l]), bins=5000)
    cumulative = np.cumsum(values)
    p3 = plt.plot(base[:-1], cumulative/count_iterations, label=l)
    
    # 3D error with z correction
    plt.figure(6)
    values, base = np.histogram(errors_rec_cor_z[l], bins=5000)
    cumulative = np.cumsum(values)
    plt.plot(base[:-1], cumulative/count_iterations, label=l)
    plt.figure(7)
    values, base = np.histogram(np.sqrt(errors_rec_cor_z[l]), bins=5000)
    cumulative = np.cumsum(values)
    plt.plot(base[:-1], cumulative/count_iterations, label=l)
    

plt.figure(0)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("Squared error in the XY plane ($m^2$)")
plt.ylabel("Cumulative proportion")
plt.title("Cumulative distribution functions")
plt.legend()
    
plt.figure(1)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("Absolute error in the XY plane ($m$)")
plt.ylabel("Cumulative proportion")
plt.title("Cumulative distribution functions")
plt.legend()

plt.figure(2)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("Squared error in the 3D plane (with [dotted line] and without [plain] Z correction) ($m^2$)")
plt.ylabel("Cumulative proportion")
plt.title("Cumulative distribution functions")
plt.legend()

plt.figure(3)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("Absolute error in the 3D plane (with [dotted line] and without [plain] Z correction) ($m$)")
plt.ylabel("Cumulative proportion")
plt.title("Cumulative distribution functions")
plt.legend()

plt.figure(4)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("Squared error in the 3D plane ($m^2$)")
plt.ylabel("Cumulative proportion")
plt.title("Cumulative distribution functions")
plt.legend()

plt.figure(5)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("Absolute error in the 3D plane ($m$)")
plt.ylabel("Cumulative proportion")
plt.title("Cumulative distribution functions")
plt.legend()

plt.figure(6)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("Squared error in the 3D plane with z correction ($m^2$)")
plt.ylabel("Cumulative proportion")
plt.title("Cumulative distribution functions")
plt.legend()

plt.figure(7)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("Absolute error in the 3D plane with z correction ($m$)")
plt.ylabel("Cumulative proportion")
plt.title("Cumulative distribution functions")
plt.legend()
plt.show()

# %% z coordinate

plt.figure(7)
i = 0
fig, axs = plt.subplots(2, 2)
for l, pltnb in [("det_covariance_2D", (0,0)), ("variance_z", (0,1)),  ("random", (1,0)), ("nearest", (1,1))]:
    #plt.scatter(np.arange(len(z_guess[l])), z_guess[l], marker="+", label=l)
    #plt.scatter(i * np.ones(len(z_guess[l])), z_guess[l], marker="+", label=l)
    axs[pltnb].hist(z_guess[l], bins=np.arange(-10,12,0.04), alpha=0.4, label="without correction")
    axs[pltnb].hist(z_guess_cor[l], bins=np.arange(-10,12,0.04), alpha=0.4, label="with z correction")
    axs[pltnb].set_title("z guess: " + l)
    axs[pltnb].set_xlabel("z coordinate")
    axs[pltnb].set_ylabel("# guesses")
    axs[pltnb].legend()
    i += 1
fig.tight_layout(pad=2.0)
plt.show()


# %% mean distances 
print("== MEAN TAG-ANCHOR DISTANCES")
for l in ["det_covariance_2D", "det_covariance_3D", "variance_z", "random", "nearest"]:
    print("\tMean:\t"+l+":",np.mean(mean_distance[l]))
    print("\tMedian:\t"+l+":",np.median(mean_distance[l]))