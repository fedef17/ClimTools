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

##############################################
############ INPUTS #########

cart = '/home/fedefab/Scrivania/Research/Post-doc/Primavera/Results_WP2/'
results = pickle.load(open(cart+'res_primavera.p','r'))

cart_ecmwf = '/home/fedefab/Scrivania/Research/Post-doc/Primavera/ECMWF_ens/'
results_ecmwf = pickle.load(open(cart_ecmwf+'res_ecmwf.p','r'))

res_tags = ['labels', 'freq_mem', 'patcor', 'cluspattern', 'significance', 'et', 'cluspattern_area']

mod_tags = ['ECE_HR', 'HadGEM_HR', 'MPI_LR', 'ECE_LR', 'CMCC_LR','CNRM_LR', 'ECMWF_LR', 'CMCC_HR', 'CNRM_HR','ECMWF_HR', 'NCEP', 'MPI_HR', 'HadGEM_LR', 'ERA']
models = ['ECE', 'HadGEM', 'CMCC', 'CNRM', 'MPI', 'ECMWF']
# Significance plot
sig = results['significance']
sig_ecmwf = results_ecmwf['significance']


fig = plt.figure()
lab1 = 'LR'
lab2 = 'HR'
key_HR = ['HR_{}'.format(ke) for ke in range(1,5)]
key_LR = ['LR_{}'.format(ke) for ke in range(1,7)]
sig_HR = [sig_ecmwf[ke] for ke in key_HR]
sig_LR = [sig_ecmwf[ke] for ke in key_LR]
plt.scatter(range(len(sig_LR)), sig_LR, color = 'green', s=30, marker = '$L$', label = 'LR')
plt.scatter(6+np.arange(len(sig_HR)), sig_HR, color = 'orange', s=30, marker = '$H$', label = 'HR')

mean_HR = np.mean(sig_HR)
std_HR = np.std(sig_HR)
mean_LR = np.mean(sig_LR)
std_LR = np.std(sig_LR)
plt.errorbar(6+(len(key_HR)-1)/2., mean_HR, std_HR, color = 'orange', capsize = 5)
plt.scatter(6+(len(key_HR)-1)/2., mean_HR, color = 'orange', s = 20)
plt.errorbar((len(key_LR)-1)/2., mean_LR, std_LR, color = 'green', capsize = 5)
plt.scatter((len(key_LR)-1)/2., mean_LR, color = 'green', s = 20)

plt.legend(fontsize = 'small', loc = 1)
plt.title('Significance of regime structure - ECMWF ensemble')
keytot = key_LR+key_HR
plt.xticks(range(len(keytot)), keytot, size='small')
plt.ylabel('Significance')
fig.savefig(cart+'Significance_ECMWF.pdf')

print('\n \n \n SOSTITUISCO LA MEDIAAA DI ECMWFFFFFFF\n \n \n')
sig['ECMWF_LR'] = mean_LR
sig['ECMWF_HR'] = mean_HR

syms = ['$H$']*len(models) + ['$L$']*len(models) + ['D']
labels = ['ECE', 'HadGEM', 'CMCC', 'CNRM', 'MPI', 'ECMWF']+6*[None]+['NCEP']
colors = ctl.color_set(len(models)+1)

wi = 0.3
#plt.ion()
fig = plt.figure()
lab1 = 'LR'
lab2 = 'HR'
for i, mod in enumerate(models):
    # plt.scatter(i, sig[mod+'_LR'], color = 'green', s=20, marker = '$L$')
    # plt.scatter(i, sig[mod+'_HR'], color = 'orange', s=20, marker = '$H$')
    if i > 0:
        lab1 = None
        lab2 = None
    plt.bar(i-wi/2, sig[mod+'_LR'], width = wi, color = 'green', label = lab1)
    plt.bar(i+wi/2,sig[mod+'_HR'], width = wi,  color = 'orange', label = lab2)
plt.bar(i+1-wi/2, sig['ERA'],width = wi,  color = 'black', label = 'ERA')
plt.bar(i+1+wi/2, sig['NCEP'],width = wi,  color = colors[-1], label = 'NCEP')
# plt.scatter(i+1, sig['ERA'], color = 'black', s=20, marker = 'D')
# plt.scatter(i+1, sig['NCEP'], color = colors[-1], s=15, marker = 'D')
models2 = models+['Obs']
plt.legend(fontsize = 'small', loc = 4)
plt.title('Significance of regime structure - Stream 1')
plt.xticks(range(len(models2)), models2, size='small')
plt.ylabel('Significance')
fig.savefig(cart+'Significance.pdf')


patt_ref = results['cluspattern']['ERA']
lat = np.arange(-90., 91., 2.5)
lon = np.arange(0., 360., 2.5)

patnames = ['NAO +', 'Blocking', 'NAO -', 'Alt. Ridge']
patnames_short = ['NP', 'BL', 'NN', 'AR']
# for tag in mod_tags:
#     print('adesso {}\n'.format(tag))
#     patt = results['cluspattern'][tag]
#     if np.any(np.isnan(patt)):
#         print('There are {} NaNs in this patt.. replacing with zeros\n'.format(np.sum(np.isnan(patt))))
#         patt[np.isnan(patt)] = 0.0
#     cartout = cart+'Model_{}/'.format(tag)
#     if not os.path.exists(cartout): os.mkdir(cartout)
#     filename = cartout+'Allclus_'+tag+'.pdf'
#     ctl.plot_multimap_contour(patt, lat, lon, filename, visualization = 'polar', central_lat_lon = (50.,0.), cmap = 'RdBu_r', title = 'North-Atlantic weather regimes - {}'.format(tag), subtitles = patnames, cb_label = 'Geopotential height anomaly (m)', color_percentiles = (0.5,99.5), fix_subplots_shape = (2,2), number_subplots = False)
#     for patuno, patuno_ref, pp, pps in zip(patt, patt_ref, patnames, patnames_short):
#         nunam = cartout+'clus_'+pps+'_'+tag+'.pdf'
#         ctl.plot_double_sidebyside(patuno, patuno_ref, lat, lon, filename = nunam, visualization = 'polar', central_lat_lon = (50., 0.), title = pp, cb_label = 'Geopotential height anomaly (m)', stitle_1 = tag, stitle_2 = 'ERA', color_percentiles = (0.5,99.5))

# Taylor plots

for num, patt in enumerate(patnames):
    obs = results['cluspattern_area']['ERA'][num, ...]
    modpats_HR = [results['cluspattern_area'][tag+'_HR'][num, ...] for tag in models]
    tags = [tag+'_HR' for tag in models]
    modpats_LR = [results['cluspattern_area'][tag+'_LR'][num, ...] for tag in models]
    tags += [tag+'_LR' for tag in models]
    modpats = modpats_HR+modpats_LR
    if num == 2:
        modpats += [results['cluspattern_area']['NCEP'][3, ...]]
    elif num == 3:
        modpats += [results['cluspattern_area']['NCEP'][2, ...]]
    else:
        modpats += [results['cluspattern_area']['NCEP'][num, ...]]

    tags += ['NCEP']

    colors = ctl.color_set(len(modpats_HR)+1)
    colors = colors[:-1] + colors[:-1] + [colors[-1]]
    syms = ['$H$']*len(modpats_HR) + ['$L$']*len(modpats_HR) + ['D']
    labels = ['ECE', 'HadGEM', 'CMCC', 'CNRM', 'MPI', 'ECMWF']+6*[None]+['NCEP']

    filename = cart+'TaylorPlot_{}.pdf'.format(patnames_short[num])
    label_ERMS_axis = 'Total RMS error (m)'
    label_bias_axis = 'Pattern mean (m)'
    ctl.Taylor_plot(modpats, obs, filename, title = patt, label_bias_axis = label_bias_axis, label_ERMS_axis = label_ERMS_axis, colors = colors, markers = syms, only_first_quarter = True, legend = True, marker_edge = None, labels = labels, obs_label = 'ERA')


colors = 4*['orange']+6*['green']+[colors[-1]]
syms = 4*['$H$'] + 6*['$L$'] + ['D']
labels = ['HR'] + 3*[None] + ['LR'] + 5*[None] + ['NCEP']

for num, patt in enumerate(patnames):
    obs = results['cluspattern_area']['ERA'][num, ...]
    modpats_HR = [results_ecmwf['cluspattern_area'][tag][num, ...] for tag in key_HR]
    modpats_LR = [results_ecmwf['cluspattern_area'][tag][num, ...] for tag in key_LR]
    tags = key_HR+key_LR+['NCEP']
    modpats = modpats_HR+modpats_LR
    if num == 2:
        modpats += [results['cluspattern_area']['NCEP'][3, ...]]
    elif num == 3:
        modpats += [results['cluspattern_area']['NCEP'][2, ...]]
    else:
        modpats += [results['cluspattern_area']['NCEP'][num, ...]]

    filename = cart+'TaylorPlot_{}_ECMWF.pdf'.format(patnames_short[num])
    label_ERMS_axis = 'Total RMS error (m)'
    label_bias_axis = 'Pattern mean (m)'
    ctl.Taylor_plot(modpats, obs, filename, title = patt, label_bias_axis = label_bias_axis, label_ERMS_axis = label_ERMS_axis, colors = colors, markers = syms, only_first_quarter = True, legend = True, marker_edge = None, labels = labels, obs_label = 'ERA')
