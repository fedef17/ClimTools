#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import pickle

import climtools_lib as ctl
import climdiags as cd

#######################################

cart_in = '/data-hobbes/fabiano/plasim_lembo/'
cart_out = '/home/fabiano/Research/lavori/plasim_lembo/'
if not os.path.exists(cart_out): os.mkdir(cart_out)

seas = 'DJFM'
area = 'EAT'
erafile = cart_in + 'ERA_remap.nc'
stocfile = cart_in + 'gph500_sppt.nc'
basefile = cart_in + 'gph500_nosppt.nc'

areas = ['EAT', 'PNA']
patnames = dict()
patnames['EAT'] = ['NAO +', 'Sc. Blocking', 'Atl. Ridge', 'NAO -']
patnames['PNA'] = ['Ala. Ridge', 'Pac. Trough', 'Arctic Low', 'Arctic High']
patnames_short = dict()
patnames_short['EAT'] = ['NP', 'BL', 'NN', 'AR']
patnames_short['PNA'] = ['AR', 'PT', 'AL', 'AH']

# for area in areas:
#     ERA_ref = cd.WRtool_from_file(erafile, seas, area, extract_level_4D = 50000., numclus = 4, heavy_output = True, run_significance_calc = False)
#     base = cd.WRtool_from_file(basefile, seas, area, extract_level_4D = 50000., numclus = 4, heavy_output = True, run_significance_calc = False, ref_solver = ERA_ref['solver'], ref_patterns_area = ERA_ref['cluspattern_area'])
#     stoc = cd.WRtool_from_file(stocfile, seas, area, extract_level_4D = 50000., numclus = 4, heavy_output = True, run_significance_calc = False, ref_solver = ERA_ref['solver'], ref_patterns_area = ERA_ref['cluspattern_area'])
#
#     pickle.dump([ERA_ref, base, stoc], open(cart_out+'out_WRtool_{}.p'.format(area), 'w'))

# ERA_ref, base, stoc = pickle.load(open(cart_out+'out_WRtool_{}.p'.format('EAT'), 'r'))
#
# plt.ion()
# fig4 = plt.figure()
# ax = plt.axes(projection = ccrs.NorthPolarStereo(central_longitude = -90))
# ax.set_global()
# ax.set_extent((-180,180,60,90), crs = ccrs.PlateCarree())
# ax.coastlines(linewidth = 2)
#
# # fig = plt.figure()
# # ax = plt.axes(projection = ccrs.Orthographic())
# # ax.set_global()
# # ax.set_extent((-120,120,30,90), crs = ccrs.PlateCarree())
# # ax.coastlines(linewidth = 2)
#
# data = base['cluspattern'][0]
# yi = base['lat']
# xi = base['lon']
# map_plot = ax.contourf(xi, yi, data, transform = ccrs.PlateCarree(), extend = 'both', corner_mask = False)
#
# sys.exit()

clatlo = dict()
clatlo['EAT'] = (70, 0)
clatlo['PNA'] = (70, -90)
for area in areas:
    ERA_ref, base, stoc = pickle.load(open(cart_out+'out_WRtool_{}.p'.format(area), 'r'))
    model = dict()
    model['base'] = base
    model['stoc'] = stoc
    cd.plot_WRtool_results(cart_out, 'base_vs_stoc_{}'.format(area), 2, model, ERA_ref, obs_name = 'ERA', patnames = patnames[area], patnames_short = patnames_short[area], custom_model_colors = ['indianred', 'forestgreen', 'black'], compare_models = [('stoc', 'base')], central_lat_lon = clatlo[area])
