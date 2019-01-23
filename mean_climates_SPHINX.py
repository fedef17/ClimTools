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
from scipy import interpolate as itrp
import itertools as itt

from sklearn.cluster import KMeans
import ctool
import ctp

from datetime import datetime
import pickle

import climtools_lib as ctl
import climdiags as cd

from copy import deepcopy as cp

import logging
logging.basicConfig()

##############################################
KtoC = 273.15
tommday = 86400

cart_in = '/data/fabiano/SPHINX/tas_pr_mon/'
cart_in_3d = '/data/fabiano/SPHINX/va_ua_ta_mon/'

cart_out = '/home/fabiano/Research/lavori/SPHINX_for_lisboa/mean_climates/'

varss = ['tas', 'pr', 'tasmax', 'tasmin']
varss3d = ['va', 'ta', 'ua']

cblabels = dict()
for varna in varss+varss3d:
    if 'ta' in varna:
        cblabels[varna] = 'Temp (C)'
cblabels['ua'] = 'u (m/s)'
cblabels['va'] = 'v (m/s)'
cblabels['pr'] = 'pr (mm/day)'

ann = np.arange(1960,2101,20)
annme = [(a1+a2)/2 for a1,a2 in zip(ann[:-1], ann[1:])]

ensmem = ['lcb0','lcb1','lcb2','lcs0','lcs1','lcs2']

seasons = ['DJF', 'JJA']

# Reading lat, lon and level
filena = '{}-1950-2100_{}_mon.nc'.format('lcb0', 'tas')
var, lat, lon, dates, time_units, var_units = ctl.read3Dncfield(cart_in+filena)
filena = '{}_mon_{}_{}.nc'.format('lcb0', 1988, 'ta')
varuna, level, lat, lon, datesuna, time_units, var_units, time_cal = ctl.read4Dncfield(cart_in_3d+filena)

# climat = dict()
# zonal = dict()
# cross3d = dict()
# globalme = dict()
#
# for ens in ensmem:
#     # carico MM di tutti gli ensmems
#     for varna in varss:
#         if ens == 'lcs2' and varna == 'tasmin': continue
#         for seas in seasons+['year']:
#             climat[(ens, varna, seas)] = []
#             zonal[(ens, varna, seas)] = []
#
#         filena = '{}-1950-2100_{}_mon.nc'.format(ens, varna)
#
#         var, lat, lon, dates, time_units, var_units = ctl.read3Dncfield(cart_in+filena)
#         dates_pdh = pd.to_datetime(dates)
#
#         # Global stuff
#         varye, _ = ctl.yearly_average(var, dates)
#         global_mean = ctl.global_mean(varye, lat)
#         globalme[(ens, varna)] = global_mean
#         zonal_mean = ctl.zonal_mean(varye)
#
#         # ref_period = ctl.range_years(1979, 2005)
#         # climat_mon, dates_mon, climat_std = ctl.monthly_climatology(var, dates, dates_range = ref_period)
#         # climat_year = np.mean(climat_mon, axis = 0)
#
#         for a,b in zip(ann[:-1], ann[1:]):
#             am = (a+b)/2
#             cli, datescli, _ = ctl.monthly_climatology(var, dates, dates_range = ctl.range_years(a,b))
#             for seas in seasons:
#                 coso = np.mean(ctl.sel_season(cli, datescli, seas, cut = False)[0], axis = 0)
#                 climat[(ens, varna, seas)].append(coso)
#                 zonal[(ens, varna, seas)].append(np.mean(coso, axis = -1))
#             coso = np.mean(cli, axis = 0)
#             climat[(ens, varna, 'year')].append(coso)
#             zonal[(ens, varna, 'year')].append(np.mean(coso, axis = -1))
#
#         for seas in seasons+['year']:
#             climat[(ens, varna, seas)] = np.stack(climat[(ens, varna, seas)])
#             zonal[(ens, varna, seas)] = np.stack(zonal[(ens, varna, seas)])
#
#     for varna in varss3d:
#         for seas in seasons+['year']:
#             cross3d[(ens, varna, seas)] = []
#
#         for a,b in zip(ann[:-1], ann[1:]):
#             for yea in range(a,b+1):
#                 filena = '{}_mon_{}_{}.nc'.format(ens, yea, varna)
#                 varuna, level, lat, lon, datesuna, time_units, var_units, time_cal = ctl.read4Dncfield(cart_in_3d+filena)
#                 if yea == a:
#                     var = varuna
#                     dates = datesuna
#                 else:
#                     var = np.concatenate([var, varuna], axis = 0)
#                     dates = np.concatenate([dates, datesuna], axis = 0)
#             am = (a+b)/2
#             cli, datescli, _ = ctl.monthly_climatology(var, dates, dates_range = ctl.range_years(a,b))
#             for seas in seasons:
#                 coso = np.mean(ctl.sel_season(cli, datescli, seas, cut = False)[0], axis = 0)
#                 cross3d[(ens, varna, seas)].append(np.mean(coso, axis = -1))
#             coso = np.mean(cli, axis = 0)
#             cross3d[(ens, varna, 'year')].append(np.mean(coso, axis = -1))
#
#         for seas in seasons+['year']:
#             cross3d[(ens, varna, seas)] = np.stack(cross3d[(ens, varna, seas)])
#
# for varna in varss:
#     ensmemok = ensmem
#     if varna == 'tasmin':
#         ensmemok = ensmem[:-1]
#     for seas in seasons+['year']:
#         climat[('base', varna, seas)] = np.mean([climat[(ens, varna, seas)] for ens in ensmemok[:3]], axis = 0)
#         zonal[('base', varna, seas)] = np.mean([zonal[(ens, varna, seas)] for ens in ensmemok[:3]], axis = 0)
#         climat[('stoc', varna, seas)] = np.mean([climat[(ens, varna, seas)] for ens in ensmemok[3:]], axis = 0)
#         zonal[('stoc', varna, seas)] = np.mean([zonal[(ens, varna, seas)] for ens in ensmemok[3:]], axis = 0)
#     globalme[('base', varna)] = np.mean([globalme[(ens, varna)] for ens in ensmemok[:3]], axis = 0)
#     globalme[('stoc', varna)] = np.mean([globalme[(ens, varna)] for ens in ensmemok[3:]], axis = 0)
#
# for varna in varss3d:
#     for seas in seasons+['year']:
#         cross3d[('base', varna, seas)] = np.mean([cross3d[(ens, varna, seas)] for ens in ensmem[:3]], axis = 0)
#         cross3d[('stoc', varna, seas)] = np.mean([cross3d[(ens, varna, seas)] for ens in ensmem[3:]], axis = 0)
#
# for key in globalme.keys():
#     if 'tas' in key or 'tasmax' in key or 'tasmin' in key:
#         globalme[key] = globalme[key]-KtoC
#     if 'pr' in key:
#         globalme[key] = globalme[key]*tommday
#
# for key in climat.keys():
#     if 'tas' in key or 'tasmax' in key or 'tasmin' in key:
#         climat[key] = climat[key]-KtoC
#         zonal[key] = zonal[key]-KtoC
#     if 'pr' in key:
#         climat[key] = climat[key]*tommday
#         zonal[key] = zonal[key]*tommday
#
# for key in cross3d.keys():
#     if 'ta' in key:
#         cross3d[key] = cross3d[key]-KtoC
#
# pickle.dump([globalme, zonal, climat, cross3d], open(cart_out+'out_meanclim_SPHINX.p','w'))
globalme, zonal, climat, cross3d = pickle.load(open(cart_out+'out_meanclim_SPHINX.p','r'))

figures = []
#varlabels = ['tas (K)', 'pr ()']
allyears = np.arange(1950,2101)
#for var, varlab in zip(varss, varlabels):
ylab = ''
for varna in varss:
    ensmemok = ensmem
    if varna == 'tasmin':
        ensmemok = ensmem[:-1]
    fig = plt.figure()
    plt.title('Global mean '+varna)
    for ens in ensmemok[:3]:
        plt.plot(allyears, globalme[(ens, varna)], label = None, color = 'grey', linestyle = '-', linewidth = 0.7)
    plt.plot(allyears, globalme[('base', varna)], label = 'base', linewidth = 2)
    for ens in ensmemok[3:]:
        plt.plot(allyears, globalme[(ens, varna)], label = None, color = 'grey', linestyle = '--', linewidth = 0.7)
    plt.plot(allyears, globalme[('stoc', varna)], label = 'stoc', linewidth = 2)
    plt.legend()
    plt.grid()
    plt.xlabel('Year')
    plt.ylabel(ylab)
    figures.append(fig)

for varna in varss:
    mino = np.min([zonal[key] for key in zonal.keys() if varna in key])
    maxo = np.max([zonal[key] for key in zonal.keys() if varna in key])
    mino = mino - 0.1*abs(maxo-mino)
    maxo = maxo + 0.1*abs(maxo-mino)
    for i, ann in enumerate(annme):
        fig = plt.figure()
        plt.title('Zonal mean {}: {}-{}'.format(varna, ann-10, ann+10))
        cset = ctl.color_set(len(seasons)+1, bright_thres = 1)
        for seas, col in zip(seasons+['year'], cset):
            plt.plot(lat, zonal[('base', varna, seas)][i], label = 'base '+ seas, linestyle = '-', color = col)
            plt.plot(lat, zonal[('stoc', varna, seas)][i], label = 'stoc '+ seas, linestyle = '--', color = col)
        plt.legend(fontsize = 'small')
        plt.ylim(mino, maxo)
        plt.grid()
        plt.xlabel('Lat')
        plt.ylabel(ylab)
        figures.append(fig)

for varna in varss:
    fig = plt.figure()
    plt.title('Arctic mean > 70: {}'.format(varna))
    cset = ctl.color_set(len(seasons)+1, bright_thres = 1)
    for seas, col in zip(seasons+['year'], cset):
        mea = ctl.band_mean_from_zonal(zonal[('base', varna, seas)], lat, 70., 90.)
        plt.plot(annme, mea, label = 'base '+seas, linestyle = '-', color = col)
        mea = ctl.band_mean_from_zonal(zonal[('stoc', varna, seas)], lat, 70., 90.)
        plt.plot(annme, mea, label = 'stoc '+seas, linestyle = '--', color = col)
    plt.legend(fontsize = 'small')
    plt.grid()
    plt.xlabel('Year')
    plt.ylabel(ylab)
    figures.append(fig)

figure_file = cart_out + 'globalzonal_base_vs_stoc.pdf'
ctl.plot_pdfpages(figure_file, figures)
plt.close('all')

figure_file = cart_out + 'diffmaps_base_vs_stoc.pdf'
figure_maps = []
for varna in varss:
    mino = np.percentile([climat[('stoc', varna, seas)]-climat[('base', varna, seas)] for seas in seasons+['year']], 1)
    maxo = np.percentile([climat[('stoc', varna, seas)]-climat[('base', varna, seas)] for seas in seasons+['year']], 99)
    mimax = np.max([abs(mino), abs(maxo)])
    for seas in ['year']+seasons:
        for i, ann in enumerate(annme):
            fig = ctl.plot_map_contour(climat[('stoc', varna, seas)][i]-climat[('base', varna, seas)][i], lat, lon, title = 'diff stoc-base {} {}: {}-{}'.format(varna, seas, ann-10, ann+10), cbar_range = (-mimax, mimax), cb_label = cblabels[varna])
            figure_maps.append(fig)
ctl.plot_pdfpages(figure_file, figure_maps)
plt.close('all')

figure_file = cart_out + 'diffcross_base_vs_stoc.pdf'
figure_cross = []
for varna in varss3d:
    mino = np.percentile([cross3d[('stoc', varna, seas)]-cross3d[('base', varna, seas)] for seas in seasons+['year']], 1)
    maxo = np.percentile([cross3d[('stoc', varna, seas)]-cross3d[('base', varna, seas)] for seas in seasons+['year']], 99)
    mimax = np.max([abs(mino), abs(maxo)])
    for seas in ['year']+seasons:
        for i, ann in enumerate(annme):
            fig = ctl.plot_lat_crosssection(cross3d[('stoc', varna, seas)][i]-cross3d[('base', varna, seas)][i], lat, level, title = 'diff stoc-base {} {}: {}-{}'.format(varna, seas, ann-10, ann+10), cbar_range = (-mimax, mimax), set_logscale_levels = True, cb_label = cblabels[varna])
            figure_cross.append(fig)
ctl.plot_pdfpages(figure_file, figure_cross)
plt.close('all')

for coso in ['base', 'stoc']:
    figure_maps = []
    figure_maps_diff = []
    for varna in varss:
        mino = np.percentile([climat[key] for key in climat.keys() if varna in key], 1)
        maxo = np.percentile([climat[key] for key in climat.keys() if varna in key], 99)
        for seas in ['year']+seasons:
            for i, ann in enumerate(annme):
                print(coso,varna,seas,ann)
                fig = ctl.plot_map_contour(climat[(coso, varna, seas)][i], lat, lon, title = coso+' {} {}: {}-{}'.format(varna, seas, ann-10, ann+10), cbar_range = (mino, maxo), cb_label = cblabels[varna])
                figure_maps.append(fig)
            fig = ctl.plot_map_contour(climat[(coso, varna, seas)][i]-climat[(coso, varna, seas)][0], lat, lon, title = coso+' {} {}: 2100 vs 1960 diff'.format(varna, seas), cb_label = cblabels[varna], plot_anomalies = True)
            figure_maps_diff.append(fig)

    figure_file = cart_out + '{}_maps.pdf'.format(coso)
    ctl.plot_pdfpages(figure_file, figure_maps)
    figure_file = cart_out + '{}_maps_futdiff.pdf'.format(coso)
    ctl.plot_pdfpages(figure_file, figure_maps_diff)
    plt.close('all')

    figure_cross = []
    figure_cross_diff = []
    for varna in varss3d:
        mino = np.percentile([cross3d[key] for key in cross3d.keys() if varna in key], 1)
        maxo = np.percentile([cross3d[key] for key in cross3d.keys() if varna in key], 99)
        mimax = np.max([abs(mino), abs(maxo)])
        if varna == 'ua' or varna == 'va':
            mino = -mimax
            maxo = mimax
        for seas in ['year']+seasons:
            for i, ann in enumerate(annme):
                print(coso,varna,seas,ann)
                fig = ctl.plot_lat_crosssection(cross3d[(coso, varna, seas)][i], lat, level, title = coso+' {} {}: {}-{}'.format(varna, seas, ann-10, ann+10), cbar_range = (mino, maxo), cb_label = cblabels[varna], set_logscale_levels = True)
                figure_cross.append(fig)
            fig = ctl.plot_lat_crosssection(cross3d[(coso, varna, seas)][i]-cross3d[(coso, varna, seas)][0], lat, level, title = coso+' {} {}: 2100 vs 1960 diff'.format(varna, seas), plot_anomalies = True, cb_label = cblabels[varna], set_logscale_levels = True)
            figure_cross_diff.append(fig)

    figure_file = cart_out + '{}_cross.pdf'.format(coso)
    ctl.plot_pdfpages(figure_file, figure_cross)
    figure_file = cart_out + '{}_cross_futdiff.pdf'.format(coso)
    ctl.plot_pdfpages(figure_file, figure_cross_diff)
    plt.close('all')

# faccio mean climates (prec e temp e tasmax e tasmin) ogni:
#       - 20 anni.
#       - JJA DJF year
# faccio medie stoc e base
# mappe + zonal + anomalies + differenze stoc vs base
# grafico temperatura assoluta artico (> 70) anno per anno, DJF e JJA

# cross section 3d di venti e temp, ogni 10 anni sempre
