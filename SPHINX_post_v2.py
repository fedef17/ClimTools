#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os

from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as PathEffects

import netCDF4 as nc
import cartopy.crs as ccrs
import pandas as pd

from numpy import linalg as LA
from eofs.standard import Eof
from scipy import stats
import itertools as itt

from sklearn.cluster import KMeans
import ctool
import ctp

from datetime import datetime
import pickle

import climtools_lib as ctl
import climdiags as cd

from copy import deepcopy as cp

##############################################
############ INPUTS #########
erafile = '/home/fabiano/data/OBS/ERA/ERAInterim/zg500/zg500_Aday_ERAInterim_2deg_1979-2014.nc'
ERA_ref_EAT = cd.WRtool_from_file(erafile, 'DJFM', 'EAT', extract_level_4D = 50000., numclus = 4, run_significance_calc = False)
print('Transition matrix (reference)\n')
print(ERA_ref_EAT['trans_matrix'])
print('\n')

cart_out = '/home/fabiano/Research/lavori/SPHINX_for_lisboa/WRtool/transitions/'
if not os.path.exists(cart_out): os.mkdir(cart_out)
fil_res = '/home/fabiano/Research/lavori/SPHINX_for_lisboa/WRtool/results_SPHINX_definitivo_light.p'

results = pickle.load(open(fil_res))

area = 'EAT'
print('Transition matrix (all members)\n')
print(results[area]['all']['trans_matrix'])
print('\n')

lab_base = [results[area]['ens{}'.format(i)]['labels'] for i in [0, 1, 2]]
dates_base = [results[area]['ens{}'.format(i)]['dates'] for i in [0, 1, 2]]
pcs_base = [results[area]['ens{}'.format(i)]['pcs'] for i in [0, 1, 2]]

lab_stoc = [results[area]['ens{}'.format(i)]['labels'] for i in [3, 4, 5]]
dates_stoc = [results[area]['ens{}'.format(i)]['dates'] for i in [3, 4, 5]]
pcs_stoc = [results[area]['ens{}'.format(i)]['pcs'] for i in [3, 4, 5]]

lab_all = lab_base + lab_stoc
dates_all = dates_base + dates_stoc
pcs_all = pcs_base + pcs_stoc

trans_base_matrix = ctl.calc_regime_transmatrix(3, lab_base, dates_base)
#trans_base = ctl.calc_regime_transmatrix(3, lab_base, dates_base, pcs_base)

trans_stoc_matrix = ctl.calc_regime_transmatrix(3, lab_stoc, dates_stoc)
#trans_stoc = ctl.calc_regime_transmatrix(3, lab_stoc, dates_stoc, pcs_stoc)

print('Transition matrix (base)\n')
print(trans_base_matrix)
print('\n')

print('Transition matrix (stoc)\n')
print(trans_stoc_matrix)
print('\n')

lab_hist = []
dates_hist = []
lab_fut = []
dates_fut = []
for la,da in zip(lab_all, dates_all):
    okhist = da < pd.Timestamp(2006, 1, 1, 0)
    lab_hist.append(la[okhist])
    dates_hist.append(da[okhist])
    lab_fut.append(la[~okhist])
    dates_fut.append(da[~okhist])

trans_hist_matrix = ctl.calc_regime_transmatrix(6, lab_hist, dates_hist)

trans_fut_matrix = ctl.calc_regime_transmatrix(6, lab_fut, dates_fut)

print('Transition matrix (hist)\n')
print(trans_hist_matrix)
print('\n')

print('Transition matrix (fut)\n')
print(trans_fut_matrix)
print('\n')

trans_all = ctl.find_transition_pcs(6, lab_all, dates_all, pcs_all, filter_longer_than = 3, skip_persistence = True)
#
# pickle.dump(trans_all, open(cart_out+'trans_all.p','w'))
# trans_all = pickle.load(open(cart_out+'trans_all.p','r'))

# PLOT TRANSITIONSS
pcscon = np.concatenate(pcs_all)
labcon = np.concatenate(lab_all)


# Proietto sulla nuova base
newbase, new_pcs, new_trans = ctl.rotate_space_interclus_section_3D(results['EAT']['all']['centroids'], 0, 1, pcscon, transitions = trans_all[0,1])

_,_, new_trans_10 = ctl.rotate_space_interclus_section_3D(results['EAT']['all']['centroids'], 0, 1, pcscon, transitions = trans_all[1,0])


ngp = 100
(x0, x1) = (np.percentile(new_pcs[:,0], 1), np.percentile(new_pcs[:,0], 99))
(y0, y1) = (np.percentile(new_pcs[:,1], 1), np.percentile(new_pcs[:,1], 99))
xss = np.linspace(x0,x1,ngp)
yss = np.linspace(y0,y1,ngp)
xi2, yi2 = np.meshgrid(xss, yss)

(z0, z1) = (np.percentile(new_pcs[:,2], 1), np.percentile(new_pcs[:,2], 99))
zss = np.linspace(z0,z1,ngp)
xi3, yi3, zi3 = np.meshgrid(xss, yss, zss)

xib, zib = np.meshgrid(xss, zss)
yic, zic = np.meshgrid(yss, zss)

cordss = [xss, yss, zss]
namz = ['x', 'y', 'z']
coups = [(0,1), (0,2), (1,2)]

for cou in coups:
    regime_pdf_2D = []
    regime_pdf_2D_func = []
    xi, yi = np.meshgrid(cordss[cou[0]], cordss[cou[1]])

    for clus in range(4):
        print(clus)
        okclus = labcon == clus
        okpc = new_pcs[okclus, :]
        kufu = ctl.calc_pdf(okpc[:,cou].T)
        print('fatto kufu\n')
        zi = kufu(np.vstack([xi.flatten(), yi.flatten()]))
        regime_pdf_2D.append(zi)
        regime_pdf_2D_func.append(kufu)

    fig = plt.figure(figsize=(8, 6), dpi=150)
    for li in new_trans:
        li3 = li[:,cou]
        p0 = li3[0]
        dp = li3[1]-li3[0]
        plt.arrow(p0[0], p0[1], dp[0], dp[1], color = 'red')
        #plt.plot(li3[0], li3[1], color = 'grey')

    for li in new_trans_10:
        li3 = li[:,cou]
        p0 = li3[0]
        dp = li3[1]-li3[0]
        plt.arrow(p0[0], p0[1], dp[0], dp[1], color = 'blue')

    for clus, namcm in enumerate(['Purples','Blues','Greens','Oranges']):
        plt.contour(xi, yi, regime_pdf_2D[clus].reshape(xi.shape), cmap = cm.get_cmap(namcm))

    plt.title('Projection on {} and {} rotated eofs'.format(cou[0], cou[1]))
    plt.xlabel('EOF {}'.format(cou[0]))
    plt.ylabel('EOF {}'.format(cou[1]))


# from mpl_toolkits.mplot3d import Axes3D
#
# cou = (1,2)
# okpc = pcscon
# for i in [1., 10., 100., 1000.]:
#     kufu = ctl.calc_pdf(okpc[:, cou].T, bnd_width = i)
#     print('fatto kufu\n')
#     zi = kufu(np.vstack([xi.flatten(), yi.flatten()]))
#     fig = plt.figure(figsize=(8, 6), dpi=150)
#     # plt.title('Projection on {} and {} eofs'.format(cou[0], cou[1]))
#     plt.title('BAND {}'.format(i))
#     ax = fig.add_subplot(111, projection='3d')
#
#     ax.contour(xi, yi, zi.reshape(xi.shape), cmap = cm.get_cmap('Oranges'))
#     ax.plot_surface(xi, yi, zi.reshape(xi.shape), cmap = cm.get_cmap('Oranges'))

sys.exit()

transvecs01 = np.stack([(li[1,:3]-li[0,:3]).T for li in new_trans])
transvecs01_mean = np.mean(transvecs01, axis = 0)

transvecs10 = np.stack([(li[1,:3]-li[0,:3]).T for li in trans_all[1,0]])
transvecs10_mean = np.mean(transvecs10, axis = 0)
