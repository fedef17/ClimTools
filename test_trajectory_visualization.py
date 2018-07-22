#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import pickle
import os
import datetime
from shutil import copy2

from matplotlib import pyplot as plt
from numpy import linalg as LA
import netCDF4 as nc
from netCDF4 import Dataset, num2date
import cartopy.crs as ccrs

import Tkinter
import tkMessageBox
from copy import deepcopy as cp

import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.animation import ImageMagickWriter as gif_maker

sys.path.insert(0,'/home/fedefab/Scrivania/Research/Post-doc/git/WRtool/CLUS_tool/')
sys.path.insert(0,'/home/fedefab/Scrivania/Research/Post-doc/git/WRtool/CLUS_tool/WRtool/')
import lib_WRtool as lwr
import clus_manipulate as clum
import readsavencfield as rsnc

numens = 1
numpcs = 4
numclus = 4
dim = 3
timeout = 100

cart1 = '/home/fedefab/Scrivania/Research/Post-doc/Prima_WRTool/Visual_traj/ERA_ref/'
nameout = 'zg500_day_ERAInterim_144x73_1ens_DJF_EAT_1979-2008'

ifile = cart1 + 'area_anomaly_'+nameout+'_{}.nc'.format(0)
_ = rsnc.read3Dncfield(ifile)
anomalies_ERA = _[0]

solver_ERA = lwr.read_out_compute(cart1, nameout, numpcs)

_, ind_ERA, clus_ERA, _ = lwr.read_out_clustering(cart1, nameout, numpcs, numclus)


pvec_anom_ERA = np.array(clum.compute_pvectors(dim, solver_ERA, anomalies_ERA))
pvec_clus_ERA = np.array(clum.compute_pvectors(dim, solver_ERA, clus_ERA))


cart2 = '/home/fedefab/Scrivania/Research/Post-doc/Prima_WRTool/Visual_traj/OUT_ECEARTH_EC1279/'
nameout = 'zg500_day_ECEARTH_144x73_1ens_DJF_EAT_1979-2008'

ifile = cart2+'area_anomaly_'+nameout+'_{}.nc'.format(0)
_ = rsnc.read3Dncfield(ifile)
anomalies_EC = _[0]

solver_EC = lwr.read_out_compute(cart2, nameout, numpcs)

# _, ind_EC, clus_EC, _ = lwr.read_out_clustering(cart, nameout, numpcs, numclus)
_, ind_EC, clus_EC = lwr.read_out_cluscompare(cart2, nameout, numpcs, numclus)

pvec_anom_EC = np.array(clum.compute_pvectors(dim, solver_ERA, anomalies_EC))
pvec_clus_EC = np.array(clum.compute_pvectors(dim, solver_ERA, clus_EC))


cmap = cm.get_cmap('jet')
colors = [cmap(0.0), cmap(0.2), cmap(0.8), cmap(1.0)]

plt.ion()

if dim == 2:
    fig = plt.figure(figsize=(8, 6), dpi=150)
    for clu in range(numclus):
        cluok = (ind_ERA == clu)
        color = colors[clu]
        plt.scatter(pvec_anom_ERA[cluok, 0], pvec_anom_ERA[cluok, 1], color = color, s = 2)
    plt.scatter(pvec_clus_ERA[:,0], pvec_clus_ERA[:,1], color = 'red', s = 20)

    fig2 = plt.figure(figsize=(8, 6), dpi=150)
    for clu in range(numclus):
        cluok = (ind_EC == clu)
        color = colors[clu]
        plt.scatter(pvec_anom_EC[cluok, 0], pvec_anom_EC[cluok, 1], color = color, s = 2)

    plt.scatter(pvec_clus_EC[:,0], pvec_clus_EC[:,1], color = 'blue', s = 20)
    plt.scatter(pvec_clus_ERA[:,0], pvec_clus_ERA[:,1], color = 'red', s = 20)
elif dim == 3:
    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    for clu in range(numclus):
        cluok = (ind_ERA == clu)
        color = colors[clu]
        ax.scatter(pvec_anom_ERA[cluok, 0], pvec_anom_ERA[cluok, 1], pvec_anom_ERA[cluok, 2], s = 2, color = color)

    ax.set_xlabel('EOF 1')
    ax.set_ylabel('EOF 2')
    ax.set_zlabel('EOF 3')

    ax.scatter(pvec_clus_ERA[:,0], pvec_clus_ERA[:,1], pvec_clus_ERA[:,2], color = 'red', s = 20)

    fig2 = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig2.add_subplot(111, projection='3d')

    for clu in range(numclus):
        cluok = (ind_EC == clu)
        color = colors[clu]
        ax.scatter(pvec_anom_EC[cluok, 0], pvec_anom_EC[cluok, 1], pvec_anom_EC[cluok, 2], s = 2, color = color)

    ax.set_xlabel('EOF 1')
    ax.set_ylabel('EOF 2')
    ax.set_zlabel('EOF 3')

    ax.scatter(pvec_clus_ERA[:,0], pvec_clus_ERA[:,1], pvec_clus_ERA[:,2], color = 'red', s = 20)
    ax.scatter(pvec_clus_EC[:,0], pvec_clus_EC[:,1], pvec_clus_EC[:,2], color = 'blue', s = 20)

#data = pvec_anom_ERA
data = pvec_anom_EC

print(data.shape)
fig = plt.figure(figsize=(8, 6), dpi=150)
ax = fig.add_subplot(111, projection='3d')

def update_lines(num, data, line, testa, ind_col, colors, numclus, timout):
    if num > timeout:
        init = num-timeout
    elif num < 5:
        num = 5
        init = 0
    else:
        init = 0

    color = colors[int(ind_col[num])]

    line.set_data(data[init:num, :2].T)
    line.set_3d_properties(data[init:num, 2])
    line.set_color(color)

    # testa.set_color(color)
    # #scat.set_sizes(rain_drops['size'])
    # # print(data[num, :])
    # # testa.set_offsets([data[num, 0], data[num, 1], data[num, 2]])

    testa._offsets3d = (np.array([data[num-1, 0]]), np.array([data[num-1, 1]]), np.array([data[num-1, 2]]))

    #(data[num, 0], data[num, 1], data[num, 2])
    testa._facecolor3d = color
    testa._edgecolor3d = color

    # testa = ax.scatter(data[num, 0], data[num, 1], data[num, 2], color = colors[int(ind_ERA[num])], s = 20)

    return


ax.scatter(pvec_clus_ERA[:,0], pvec_clus_ERA[:,1], pvec_clus_ERA[:,2], color = 'red', s = 20)

line = ax.plot(data[0:5, 0], data[0:5, 1], data[0:5, 2], color = 'grey')[0]
testa = ax.scatter(data[4, 0], data[4, 1], data[4, 2], color = colors[int(ind_ERA[4])], s = 30)

ax.set_title('EAT geopotential 3D Trajectory')

ax.set_xlim3d([-5000., 5000.])
ax.set_xlabel('EOF 1')

ax.set_ylim3d([-5000., 5000.])
ax.set_ylabel('EOF 2')

ax.set_zlim3d([-5000., 5000.])
ax.set_zlabel('EOF 3')

# Creating the Animation object
# line_ani = animation.FuncAnimation(fig, update_lines, len(data[:,0]), fargs=(data, line, testa, ind_ERA, colors, numclus, timeout), interval=50, blit=False)

#metadata = dict(title='ERA trajectory in 3D EOF space', artist='Federico Fabiano')
metadata = dict(title='EC trajectory in 3D EOF space', artist='Federico Fabiano')
writer = gif_maker(fps = 20, metadata = metadata)

with writer.saving(fig, cart2 + "EC_trajectory.gif", 100):
    for i in range(len(data[:,0])):
        print(i)
        update_lines(i, data, line, testa, ind_ERA, colors, numclus, timeout)
        writer.grab_frame()
