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

import tkinter
import tkinter.messagebox
from copy import deepcopy as cp

import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.animation import ImageMagickFileWriter, FFMpegFileWriter, AVConvFileWriter

#cart_out_WRtool = '/home/fabiano/Research/lavori/WeatherRegimes/OUT_WRTOOL/'
cart_out_WRtool = '/home/fedefab/Scrivania/Research/Post-doc/Prima_WRTool/Visual_traj/'
#cart_git_WRtool = '/home/fabiano/Research/git/WRtool/'
cart_git_WRtool = '/home/fedefab/Scrivania/Research/Post-doc/git/WRtool/'

sys.path.insert(0,cart_git_WRtool+'CLUS_tool/')
sys.path.insert(0,cart_git_WRtool+'CLUS_tool/WRtool/')
import lib_WRtool as lwr
import clus_manipulate as clum
import readsavencfield as rsnc

numens = 1
numpcs = 4
numclus = 4
dim = 3
timeout = 100
alpha = 0.3
alpha2 = 0.6

cart1 = cart_out_WRtool+'ERA_ref/'
nameout = 'zg500_day_ERAInterim_144x73_1ens_DJF_EAT_1979-2008'

ifile = cart1 + 'area_anomaly_'+nameout+'_{}.nc'.format(0)
_ = rsnc.read3Dncfield(ifile)
anomalies_ERA = _[0]
dates = _[3]

solver_ERA = lwr.read_out_compute(cart1, nameout, numpcs)

_, ind_ERA, clus_ERA, _ = lwr.read_out_clustering(cart1, nameout, numpcs, numclus)


pvec_anom_ERA = np.array(clum.compute_pvectors(dim, solver_ERA, anomalies_ERA))
pvec_clus_ERA = np.array(clum.compute_pvectors(dim, solver_ERA, clus_ERA))


cart2 = cart_out_WRtool+ 'OUT_ECEARTH_EC1279/'
nameout = 'zg500_day_ECEARTH_144x73_1ens_DJF_EAT_1979-2008'

ifile = cart2+'area_anomaly_'+nameout+'_{}.nc'.format(0)
_ = rsnc.read3Dncfield(ifile)
anomalies_EC = _[0]

solver_EC = lwr.read_out_compute(cart2, nameout, numpcs)

# _, ind_EC, clus_EC, _ = lwr.read_out_clustering(cart, nameout, numpcs, numclus)
_, ind_EC, clus_EC = lwr.read_out_cluscompare(cart2, nameout, numpcs, numclus)

pvec_anom_EC = np.array(clum.compute_pvectors(dim, solver_ERA, anomalies_EC))
pvec_clus_EC = np.array(clum.compute_pvectors(dim, solver_ERA, clus_EC))


cmap = cm.get_cmap('nipy_spectral')

plt.ion()
# all_col = [cmap(co) for co in np.linspace(0,1,20)]
# plt.figure()
# for co, pu in zip(all_col, range(len(all_col))):
#     plt.scatter(pu*1.0/20, 0., color = co, s = 100)

colors = [cmap(0.25), cmap(0.85), cmap(0.05), cmap(0.45)]


if dim == 2:
    fig = plt.figure(figsize=(8, 6), dpi=150)
    for clu in range(numclus):
        cluok = (ind_ERA == clu)
        color = colors[clu]
        plt.scatter(pvec_anom_ERA[cluok, 0], pvec_anom_ERA[cluok, 1], color = color, s = 1)
    plt.scatter(pvec_clus_ERA[:,0], pvec_clus_ERA[:,1], color = colors, s = 30)

    fig2 = plt.figure(figsize=(8, 6), dpi=150)
    for clu in range(numclus):
        cluok = (ind_EC == clu)
        color = colors[clu]
        plt.scatter(pvec_anom_EC[cluok, 0], pvec_anom_EC[cluok, 1], color = color, s = 1)

    plt.scatter(pvec_clus_EC[:,0], pvec_clus_EC[:,1], color = colors, s = 10)
    plt.scatter(pvec_clus_ERA[:,0], pvec_clus_ERA[:,1], color = colors, s = 30)
elif dim == 3:
    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    for clu in range(numclus):
        cluok = (ind_ERA == clu)
        color = colors[clu]
        ax.scatter(pvec_anom_ERA[cluok, 0], pvec_anom_ERA[cluok, 1], pvec_anom_ERA[cluok, 2], s = 1, color = color, alpha = alpha)

    ax.set_xlabel('EOF 1')
    ax.set_ylabel('EOF 2')
    ax.set_zlabel('EOF 3')

    ax.scatter(pvec_clus_ERA[:,0], pvec_clus_ERA[:,1], pvec_clus_ERA[:,2], color = colors, s = 40)

    fig.savefig(cart1+'ERA_cloudplot_clusters.pdf', format = 'pdf')

    fig2 = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig2.add_subplot(111, projection='3d')

    for clu in range(numclus):
        cluok = (ind_EC == clu)
        color = colors[clu]
        ax.scatter(pvec_anom_EC[cluok, 0], pvec_anom_EC[cluok, 1], pvec_anom_EC[cluok, 2], s = 1, color = color, alpha = alpha)

    ax.set_xlabel('EOF 1')
    ax.set_ylabel('EOF 2')
    ax.set_zlabel('EOF 3')

    ax.scatter(pvec_clus_ERA[:,0], pvec_clus_ERA[:,1], pvec_clus_ERA[:,2], color = colors, s = 40)
    ax.scatter(pvec_clus_EC[:,0], pvec_clus_EC[:,1], pvec_clus_EC[:,2], color = colors, s = 20)

    fig2.savefig(cart2+'EC_cloudplot_clusters.pdf', format = 'pdf')


data = pvec_anom_ERA
nome = 'ERA'
cart_out = cart1
save = False

#data = pvec_anom_EC
#nome = 'EC'
#cart_out = cart2

print(data.shape)
fig = plt.figure(figsize=(8, 6), dpi=150)
ax = fig.add_subplot(111, projection='3d')

def update_lines(num, data, line, testa, ind_col, colors, numclus, timeout, showdate, dates):
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

    nudate = dates[num-1].ctime()[4:11]+dates[num-1].ctime()[-5:]
    showdate.set_text(nudate)

    # testa = ax.scatter(data[num, 0], data[num, 1], data[num, 2], color = colors[int(ind_ERA[num])], s = 20)

    return


ax.scatter(pvec_clus_ERA[:,0], pvec_clus_ERA[:,1], pvec_clus_ERA[:,2], color = colors, s = 30)

nudate = dates[4].ctime()[4:11]+dates[4].ctime()[-5:]
showdate = ax.text(-4000., 4000., 3000., nudate, bbox=dict(facecolor='lightsteelblue', edgecolor='black', boxstyle='round,pad=1'))

line = ax.plot(data[0:5, 0], data[0:5, 1], data[0:5, 2], color = 'grey', alpha = alpha2)[0]
testa = ax.scatter(data[4, 0], data[4, 1], data[4, 2], color = colors[int(ind_ERA[4])], s = 30)

ax.set_title('EAT geopotential 3D Trajectory')

ax.set_xlim3d([-4000., 4000.])
ax.set_xlabel('EOF 1')

ax.set_ylim3d([-4000., 4000.])
ax.set_ylabel('EOF 2')

ax.set_zlim3d([-4000., 4000.])
ax.set_zlabel('EOF 3')


if save:
    metadata = dict(title='{} trajectory in 3D EOF space'.format(nome), artist='Federico Fabiano')
    #writer = AVConvFileWriter(fps = 20, metadata = metadata)
    #writer = FFMpegFileWriter(fps = 20, metadata = metadata)
    writer = ImageMagickFileWriter(fps = 20, metadata = metadata)
    with writer.saving(fig, cart_out + "{}_trajectory.gif".format(nome), 100):
        for i in range(len(data[:,0])):
            print(i)
            update_lines(i, data, line, testa, ind_ERA, colors, numclus, timeout, showdate, dates)
            writer.grab_frame()
else:
    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, update_lines, len(data[:,0]), fargs=(data, line, testa, ind_ERA, colors, numclus, timeout, showdate, dates), interval=50, blit=False)
