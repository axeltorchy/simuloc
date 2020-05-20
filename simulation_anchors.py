#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:02:15 2020

@author: axel
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D 
from scipy.optimize import minimize
from scipy.special import comb
from random import sample, seed
from itertools import combinations

matplotlib.rcParams['figure.dpi'] = 120
size_factor = 1.2
matplotlib.rcParams['figure.figsize'] = [size_factor * 8.0, size_factor * 6.0]
matplotlib.rcParams['lines.markersize'] = 6.0 


# For reproducibility
np.random.seed(0)
seed(0)

# Simulation parameters
N_x         = 6         # number of anchors on a row
N_y         = 6         # number of anchors on a column
d           = 80        # grid periodicity
default_z   = 2         # default anchor altitude
min_z       = 1.
max_z       = 4.
ranging_dst = 160
noise_std   = 0.08      # standard deviation gaussian noise rangings
noise_mean  = -0.00     # mean gaussian noise rangings

# If True, altitude will be chosen randomly, uniformly between min_z and max_z
random_z  = True

use_mean_z = False


tag_pos = np.array([2, 3, 2])

def ranging(tag_pos, anchors, ranging_dst):
    """Scans the anchors and returns the IDs of those in the
    specified range.
    
    Keyword arguments:
    tag_pos -- tag position (numpy array of length 3)
    anchors -- list of anchors (list of dicts)
    ranging_dst -- ranging distance
    """
    ID = []
    for i in anchors:
        a = anchors[i]
        x = a['x']
        y = a['y']
        z = a['z']
        anchor = np.array([x,y,z])
        dst = np.linalg.norm(anchor - tag_pos)
        # print("Distance to anchor "+str(i)+" = " + str(dst))
        if dst <= ranging_dst:
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


def anchor_selection_det_covariance(anchors, N, N_tries=100):
    """Returns N anchors that maximize the determinant of the covariance
    matrix. If N_tries = 0, the function goes through all N_anchors choose N
    combinations."""
    #N_tries = 300 # Number of random N-anchor subsets to be tested
    covariance = 0
    selected_anchors = None
    ID_selected = None
    vertices = np.zeros((N,3))
    
    if N_tries == 0:
        # Return list of tuples containing the IDs of the anchors of the
        # combination.
        combi = combinations(anchors, N)
        for c in combi:
            for i in range(len(c)):
                vertices[i][0] = anchors[c[i]]['x']
                vertices[i][1] = anchors[c[i]]['y']
                vertices[i][2] = anchors[c[i]]['z']
                
                
            # print(vertices)
            # renormalisation
            # for i in range(3):
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
        a['dst'] = distances[i]
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
initialize_anchors(anchors, N_x, N_y, d, min_z, max_z, random_z)

plot_setting(tag_pos, anchors)


initial_guess = [d*N_x/2, d*N_y/2, default_z]

# Selection of the anchors (to be incorporated in a function)

ID = ranging(tag_pos, anchors, ranging_dst)
selected_anchors, distances, errors = anchors_from_IDs(anchors, ID, tag_pos)

# print("Anchors:", anchors)
# print("IDs:", ID)
# print("Selected anchors:", selected_anchors)

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tag_pos[0],tag_pos[1],tag_pos[2], marker='s', color='red')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
print("Real position: \t x="+str(tag_pos[0])+" \t y="+str(tag_pos[1])+" \t z="+str(tag_pos[2]))

for N in range(4,len(selected_anchors)):
    sel_anchors = anchor_selection_variance_z(selected_anchors, N)
    
    # Bounds only for L-BFGS-B, TNC and SLSQP
    bnds = ((-d, d*N_x), (-d, d*N_y), (-3, 10))
    bnds = ((None, None), (None, None), (None, None))
    res = minimize(cost_function,
                   initial_guess,
                   args=(sel_anchors, 0, 0),
                   method='L-BFGS-B',
                   bounds=bnds,
                   tol=1e-4,
                   options={'disp': False})
    
    sol = np.round(100*res.x)/100
    # print("Algo solution: \t x="+str(sol[0])+" \t y="+str(sol[1])+" \t z="+str(sol[2]),
    #       "Euclidean error", np.linalg.norm(res.x - tag_pos))
    ax.scatter(sol[0],sol[1],sol[2], marker='o', color='blue')
    # print("SSE =",cost_function(res.x,sel_anchors,0,0))
    # print("OPT =",cost_function(tag_pos, sel_anchors, 0, 0))
    
for N in range(60,100):
    sel_anchors = anchor_selection_tetrahedron(selected_anchors, N)
    print(selected_anchors)
    print("A")
    print(sel_anchors)
    
    bnds = ((None, None), (None, None), (None, None))
    res = minimize(cost_function,
                   initial_guess,
                   args=(sel_anchors, 0, 0),
                   method='L-BFGS-B',
                   bounds=bnds,
                   tol=1e-6,
                   options={'disp': False})
    
    sol = np.round(100*res.x)/100
    # print("Algo solution: \t x="+str(sol[0])+" \t y="+str(sol[1])+" \t z="+str(sol[2]),
    #       "Euclidean error", np.linalg.norm(res.x - tag_pos))
    ax.scatter(sol[0],sol[1],sol[2], marker='o', color='green')

initial_guess = tag_pos + 0.01 * np.ones(3)

for i in range(2):
    print("   Intial guess:",initial_guess)
    N_tries = 100
    
    np.random.seed(i)
    seed(i)
    anchors = {}
    initialize_anchors(anchors, N_x, N_y, d, min_z, max_z, random_z)
    ID = ranging(tag_pos, anchors, ranging_dst)
    selected_anchors, distances, errors = anchors_from_IDs(anchors, ID, tag_pos)
    # print("Anchors:", anchors)
    # print("IDs:", ID)
    # print("Selected anchors:", selected_anchors)
    tolerance = 1e-10
    bnds = ((None, None), (None, None), (None, None))
    print("=== Try",i)
    print(selected_anchors)
    
    print("  * DET COVARIANCE")
    np.random.seed(i)
    seed(i)
    anc = anchor_selection_det_covariance(selected_anchors, 4, N_tries)
    res = minimize(cost_function,
                   initial_guess,
                   args=(anc, 0, 0),
                   method='L-BFGS-B',
                   bounds=bnds,
                   tol=tolerance,
                   jac=gradient_cost_function,
                   options={'disp': False})
    
    res_x = res.x[0]
    res_y = res.x[1]
    res_z = res.x[2]
    
    if use_mean_z:
        z_sum = 0
        z_rk_sum = 0
        rk_sum = 0
        z_count = 0
        for a in anc:
            x_i = anc[a]['x']
            y_i = anc[a]['y']
            z_i = anc[a]['z']
            dst = anc[a]['dst']
            
            # complex square root, retain modulus
            delta_z_k = np.real( np.sqrt(dst**2 - (res_x - x_i)**2 - (res_y - y_i)**2 + 0j) )
            
            if z_i > 2.1: # anchor is above tag
                z_sum += z_i - delta_z_k
                z_rk_sum += (z_i - delta_z_k) / dst
                z_count += 1
                rk_sum += 1. / dst
            elif z_i < 1.9: # anchor is below tag
                z_sum += z_i + delta_z_k
                z_rk_sum += (z_i + delta_z_k) / anc[a]['dst']
                rk_sum += 1. / dst
                z_count += 1
            else:
                print("Not included.")
        
        res_z = z_rk_sum / rk_sum
    
    #print(anc)
    mean_alt = 0
    for j in anc:
        mean_alt += anc[j]['z']
    mean_alt = mean_alt / len(anc)
        
    print("Mean altitude anchors:",mean_alt)
    print("\t",np.sort(list(anc.keys())))
    #print("\t", anc)
    print("\t",np.round(100*res.x)/100)
    print("\t",res.fun)
    print("\talt z:",res_z)
    print("\t",cost_function(tag_pos,anc, 0, 0))
    print("\tError (3D):", np.round(100 * np.linalg.norm(res.x - tag_pos))/100)
    
    
    
    print("  * VOL TETRAHEDRON")
    
    np.random.seed(i)
    seed(i)
    anc = anchor_selection_tetrahedron(selected_anchors, N_tries)
    res = minimize(cost_function,
                   initial_guess,
                   args=(anc, 0, 0),
                   method='L-BFGS-B',
                   bounds=bnds,
                   tol=tolerance,
                   jac=gradient_cost_function,
                   options={'disp': False})
    #print(anc)
    print("\t",np.sort(list(anc.keys())))
    print("\t",np.round(100*res.x)/100)
    print("\t",res.fun) # equivalent to cost_function(res.x, anc, 0, 0) 
    print("\t",cost_function(tag_pos,anc, 0, 0))
    print("\tError (3D):", np.round(100 * np.linalg.norm(res.x - tag_pos))/100)
    
    
    
    print("  * MAX VARIANCE Z")
    
    np.random.seed(i)
    seed(i)
    anc = anchor_selection_variance_z(selected_anchors, 4)
    res = minimize(cost_function,
                   initial_guess,
                   args=(anc, 0, 0),
                   method='L-BFGS-B',
                   bounds=bnds,
                   tol=tolerance,
                   jac=gradient_cost_function,
                   options={'disp': False})
    print("\t",np.sort(list(anc.keys())))
    print("\t",np.round(100*res.x)/100)
    print("\t",res.fun)
    print("\tError (3D):", np.round(100 * np.linalg.norm(res.x - tag_pos))/100)

# import time

# tolerance = 1e-8
# t0 = time.time()
# print("With jacobian:")
# for i in range(100):
#     #np.random.seed(i)
#     #seed(i)
#     res = minimize(cost_function,
#                initial_guess,
#                args=(anchor_selection_det_covariance(selected_anchors, 4), 0, 0),
#                method='L-BFGS-B',
#                bounds=bnds,
#                tol=tolerance,
#                jac=gradient_cost_function,
#                options={'disp': False})
# print("\tElapsed time:",time.time()-t0,"for 200 iterations.")

# t0 = time.time()
# print("Without jacobian:")
# for i in range(100):
#     #np.random.seed(i)
#     #seed(i)
#     res = minimize(cost_function,
#                initial_guess,
#                args=(anchor_selection_det_covariance(selected_anchors, 4), 0, 0),
#                method='L-BFGS-B',
#                bounds=bnds,
#                tol=tolerance,
#                options={'disp': False})
# print("\tElapsed time:",time.time()-t0,"for 200 iterations.")
    

# # Tests comparaison volume tétraèdre et matrice de covariance
# V1_l = []
# V2_l = []
# for i in range(2000):
#     X = 100*np.random.rand(4,3)
#     V1 = volume_tetrahedron(X)
#     V2 = np.linalg.det(np.cov(X.T))
#     # print(V1,V2)
#     V1_l.append(V1)
#     V2_l.append(V2)

# plt.figure(10)

# z = np.polyfit(V1_l, V2_l, 4)
# p = np.poly1d(z)
# xvalues = np.linspace(0,max(V1_l),100)
# plt.plot(V1_l, V2_l, '.', xvalues, p(xvalues), "-")
# print("p",z)



# Estimate performance of localisation:
# Comparison between random selection of 4 anchors, selection based on Z
# variance and selection based on the tetrahedron volume

x_tag = np.arange(0.25, d*(N_x - 1), round(d/1))
y_tag = np.arange(0.25, d*(N_y - 1), round(d/1))
x_tag = np.arange(2.25, d*(N_x - 1), round(d/1))
y_tag = np.arange(2.25, d*(N_y - 1), round(d/1))
z_tag = np.arange(0.2, 2.5, 2)

N_simu = 2
N_steps = len(x_tag) * len(y_tag) * len(z_tag)
tolerance = 1e-8

N_tries = 200


from time import sleep


for n in range(N_simu):
    step_count = 0
    err_3D_cov = np.zeros(N_steps)
    err_3D_rand = np.zeros(N_steps)
    err_3D_rand_zcor = np.zeros(N_steps)
    err_3D_var_z = np.zeros(N_steps)
    mean_dst_cov = np.zeros(N_steps)
    mean_dst_rand = np.zeros(N_steps)
    mean_dst_var_z = np.zeros(N_steps)

    print("Simulation",n+1)
    # For reproducibility
    np.random.seed(n)
    seed(n)
    anchors = {}
    initialize_anchors(anchors, N_x, N_y, d, min_z, max_z, random_z)
    
    plot_setting(np.array([x_tag[len(x_tag)//2], y_tag[len(y_tag)//2], z_tag[len(z_tag)//2]]), anchors, title="Setting "+str(n))
    
    for i in range(len(x_tag)):
        print("x =",x_tag[i])
        
        for j in range(len(y_tag)):
            
            for k in range(len(z_tag)):
                tag_pos = np.array([x_tag[i], y_tag[j], z_tag[k]])
                
                

                
                
                initial_guess = tag_pos + 0.1 * np.random.rand(3)
                
                
                
                
                ID = ranging(tag_pos, anchors, ranging_dst)
                selected_anchors, distances, errors = anchors_from_IDs(anchors, ID, tag_pos)
                
                # preferred number of anchors for ranging
                nb_anchors = min(6, len(selected_anchors))
                # print(nb_anchors)
                       
                #constrained_alt = tag_pos[2]
                constrained_alt = 0
                # constrained_alt = z_tag[k]
                
                # # DET COVARIANCE
                
                anc = anchor_selection_det_covariance(selected_anchors, nb_anchors, 0)
                
               
                                
                #anc = anchor_selection_tetrahedron(selected_anchors, 0)
                res = minimize(cost_function,
                                initial_guess,
                                args=(anc, 0, constrained_alt),
                                method='L-BFGS-B',
                                bounds=bnds,
                                tol=tolerance,
                                options={'disp': False})
                res.x[2] = constrained_alt if constrained_alt > 0 else res.x[2]
                
                
                err = np.round(100 * np.linalg.norm(res.x - tag_pos))/100
                #err = np.round(100 * np.linalg.norm(res.x[0:2] - tag_pos[0:2]))/100
                err_3D_cov[step_count] = err
                mean_dst_cov[step_count] = np.mean([anc[i]['dst'] for i in anc])

                
                # plt.figure(42)
                # x = np.linspace(0, 15, 201)
                # y = np.linspace(0, 15, 201)
                # X,Y = np.meshgrid(x,y)
                # Z = normal_density(anc, X, Y, tag_pos[2], noise_std)
                # plt.pcolor(X, Y, Z)
                # plt.scatter(tag_pos[0], tag_pos[1], marker="+")
                # plt.scatter(res.x[0], res.x[1], marker="*")
                # for a in anc:
                #     plt.scatter(anc[a]['x'], anc[a]['y'], marker="D", color="g")
                    
                # plt.show()




                # VAR Z
                anc = anchor_selection_variance_z(selected_anchors, nb_anchors)
                res = minimize(cost_function,
                               initial_guess,
                               args=(anc, 0, 0),
                               method='L-BFGS-B',
                               bounds=bnds,
                               tol=tolerance,
                               jac=gradient_cost_function,
                               options={'disp': False})
                
                res.x[2] = constrained_alt if constrained_alt > 0 else res.x[2]
                
                err = np.round(100 * np.linalg.norm(res.x - tag_pos))/100
                #err = np.round(100 * np.linalg.norm(res.x[0:2] - tag_pos[0:2]))/100
                err_3D_var_z[step_count] = err
                mean_dst_var_z[step_count] = np.mean([anc[i]['dst'] for i in anc])
                
                
                
                # RANDOM SELECTION
                # remember to check influence of number of anchors in ranging
                #anc = anchor_selection_nearest(selected_anchors, nb_anchors)
                anc = anchor_selection_random(selected_anchors, nb_anchors)
                 #print(anc)
                res = minimize(cost_function,
                               initial_guess,
                               args=(anc, 0, 0),
                               method='L-BFGS-B',
                               bounds=bnds,
                               tol=tolerance,
                               jac=gradient_cost_function,
                               options={'disp': False})
                res.x[2] = constrained_alt if constrained_alt > 0 else res.x[2]
                
                err = np.round(100 * np.linalg.norm(res.x - tag_pos))/100
                #err = np.round(100 * np.linalg.norm(res.x[0:2] - tag_pos[0:2]))/100
                err_3D_rand[step_count] = err
                mean_dst_rand[step_count] = np.mean([anc[i]['dst'] for i in anc])
                     
                
                
                res_x = res.x[0]
                res_y = res.x[1]
                res_z = res.x[2]
                if True:
                    z_sum = 0
                    z_rk_sum = 0
                    rk_sum = 0
                    z_count = 0
                    for a in anc:
                        x_i = anc[a]['x']
                        y_i = anc[a]['y']
                        z_i = anc[a]['z']
                        dst = anc[a]['dst']
                        
                        # complex square root, retain modulus
                        delta_z_k = np.real( np.sqrt(dst**2 - (res_x - x_i)**2 - (res_y - y_i)**2 + 0j) )
                        
                        if z_i >= tag_pos[2]: # anchor is above tag
                            z_sum += z_i - delta_z_k
                            z_rk_sum += (z_i - delta_z_k) / dst
                            z_count += 1
                            rk_sum += 1. / dst
                        elif z_i < tag_pos[2]: # anchor is below tag
                            z_sum += z_i + delta_z_k
                            z_rk_sum += (z_i + delta_z_k) / anc[a]['dst']
                            rk_sum += 1. / dst
                            z_count += 1
                        else:
                            print("Not included. z_i=",z_i)
                    
                    res_z = z_sum / z_count  
                    res_z = z_rk_sum / rk_sum  
                
                    res.x[2] = res_z
                    err = np.round(100 * np.linalg.norm(res.x - tag_pos))/100
                    #err = np.round(100 * np.linalg.norm(res.x[0:2] - tag_pos[0:2]))/100
                    err_3D_rand_zcor[step_count] = err

                
                
                step_count += 1

    print("== MEAN 3D ERROR")
    print("Det covariance:\t",np.mean(err_3D_cov))
    print("Random choice:\t",np.mean(err_3D_rand))
    print("Rand (z corr.):\t",np.mean(err_3D_rand_zcor))
    print("Variance on Z: \t", np.mean(err_3D_var_z))
    
    print("== STD 3D ERROR")
    print("Det covariance:\t", np.std(err_3D_cov))
    print("Random choice:\t", np.std(err_3D_rand))
    print("Rand (z corr.):\t", np.std(err_3D_rand_zcor))
    print("Variance on Z: \t", np.std(err_3D_var_z))
    
    print("== MEAN TAG-ANCHOR DST")
    print("Det covariance:\t",np.mean(mean_dst_cov))
    print("Random choice:\t",np.mean(mean_dst_rand))
    print("Variance on Z: \t", np.mean(mean_dst_var_z))
    


noise_std = 0.15

ARANGE = np.arange(2, 2, 1)
error = np.zeros(len(ARANGE))
error2 = np.zeros(len(ARANGE))

count = 0
anchors = {}
tag_pos = np.array([19, 1, 0.5])

for diversity in ARANGE:
    print(diversity)    

    for i in range(200):
        # print("=== "+str(i)+ " ===")
        # print(anchors)
        # print([anchors[a]['z'] for a in anchors])
        anchors = {}
        initialize_anchors(anchors, 3, 3, 20, 1, 1 + diversity, True, 0)
        ID = ranging(tag_pos, anchors, 60)
        bnds = ((None, None), (None, None), (0, 1))
        selected_anchors, distances, errors = anchors_from_IDs(anchors, ID, tag_pos)
        
        constrained_alt = 0
        
        res = minimize(cost_function,
                        initial_guess,
                        args=(selected_anchors, 0, constrained_alt),
                        method='L-BFGS-B',
                        bounds=bnds,
                        tol=1e-9,
                        jac=gradient_cost_function,
                        options={'disp': False})
        res.x[2] = constrained_alt if constrained_alt > 0 else res.x[2]
        
        
        err = np.round(100 * np.linalg.norm(res.x - tag_pos))/100
        
        error[count] = error[count] + (err - error[count])/(i + 1)
        
        # print("res.x",res.x)
        # print("err",err)
        
        # plt.figure(43)
        # x = np.linspace(0, N_x*d, 201)
        # y = np.linspace(0, N_y*d, 201)
        # X,Y = np.meshgrid(x,y)
        # Z = normal_density(selected_anchors, X, Y, tag_pos[2], noise_std)
        # plt.pcolor(X, Y, np.log(Z + 1e-300))
        # plt.scatter(tag_pos[0], tag_pos[1], marker="+")
        # # plt.scatter(res.x[0], res.x[1], marker="*")
        # anc = selected_anchors
        # for a in anc:
        #     plt.scatter(anc[a]['x'], anc[a]['y'], marker="D", color="g")
            
        # plt.show()
    
    count += 1