#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os

from matplotlib import pyplot as plt
import netCDF4 as nc
import pandas as pd
from numpy import linalg as LA
from scipy import stats
from scipy import interpolate as itrp
import itertools as itt

from datetime import datetime
import pickle

import climtools_lib as ctl
import climdiags as cd

from copy import deepcopy as cp

###############################################
L = 2501000.0
Rearth = 6371.0e3
##

cart_in = '/home/fabiano/data/SPHINX/heat_flux/fluxes_row/'
cart_in_ref = '/home/fabiano/data/OBS/ERA/ERAInterim/'

file_list = cart_in_ref+'ERAInterim_6hrs_1988_vatazgq.nc'
file_ps = cart_in_ref+'ERAInterim_mon_1988_ps.nc'
cart_out = '/home/fabiano/Research/lavori/SPHINX_for_lisboa/heat_flux/'

# Loading ERA reference
era_fi = 'prova_heatflux_1988_MM.nc'

vars, lat, lon, dates, time_units, var_units, time_cal = ctl.readxDncfield(cart_in_ref+era_fi)
print(vars.keys())

fi = 'northward_water_flux.nc'
var2, lat, lon, dates, time_units, var_units, time_cal = ctl.readxDncfield(cart_in_ref+fi)
print(var2.keys())
vars[var2.keys()[0]] = L*var2[var2.keys()[0]]

era_zonal_factor = 2*np.pi*Rearth*np.cos(np.deg2rad(lat))

era_fluxes_maps = dict()
era_fluxes_zonal = dict()

seasons = ['Feb','DJF', 'JJA']
fluxnames = ['tot', 'SH', 'PE', 'LH']
eraname = {'tot': 'p76.162', 'SH': 'p70.162', 'PE': 'p74.162', 'LH': 'p72.162'}

for flun in fluxnames:
    for seas in seasons:
        era_fluxes_maps[(flun, seas)] = np.mean(ctl.sel_season(vars[eraname[flun]], dates, seas, cut = False)[0], axis = 0)
    era_fluxes_maps[(flun, 'year')] = np.mean(vars[eraname[flun]], axis = 0)

for fu in era_fluxes_maps.keys():
     era_fluxes_zonal[fu] = np.mean(era_fluxes_maps[fu], axis = 1)*era_zonal_factor

###################################################################################

ann = np.arange(1950,2101,10)
annme = [(a1+a2)/2 for a1,a2 in zip(ann[:-1], ann[1:])]
print(annme)

cart_out_results = cart_out + 'out_flux_calc_NEW/'
if not os.path.exists(cart_out_results): os.mkdir(cart_out_results)

seasons = ['DJF', 'JJA']
fluxes_model = dict()
for ens in ['lcb0', 'lcs0']:
    for a,b in zip(annme[:-1], annme[1:]):
        results_year = []
        for yea in range(a,b+1):
            filist = []
            psfile = cart_in+'{}_mon_{}_ps.nc'.format(ens, yea)
            for flun in fluxnames[1:]:
                filist.append(cart_in+'{}_m{}f_y{}.nc'.format(ens, flun.lower(), yea))
            tag = '{}_{}'.format(ens, yea)
            results_year.append(cd.heat_flux_calc(filist, psfile, cart_out_results, tag, zg_in_ERA_units = False, full_calculation = False, seasons = seasons))

        am = (a+b)/2
        for flun in fluxnames:
            for seas in seasons+['year']:
                for taw in ['zonal', 'maps', 'cross']:
                    fluxes_model[(ens, flun, seas, taw, am)] = np.mean([coso[flun][taw][seas] for coso in results_year], axis = 0)

pickle.dump(fluxes_model, open(cart_out+'out_hfc_NEW_lcb0vslcs0.p', 'w'))

seasonsall = ['year']+seasons

figure_file = cart_out + 'hfc_lcb0_vs_lcs0_NEW.pdf'
figures = []
for flun in fluxnames:
    for ann in annme:
        fig = plt.figure()
        plt.title('{} fluxes - {}'.format(flun, ann))
        cset = ctl.color_set(len(seasonsall), bright_thres = 1)
        for seas, col in zip(seasonsall, cset):
            plt.plot(lat, fluxes_model[('lcb0', flun, seas, 'zonal', ann)], label = 'base '+seas, color = col, linewidth = 2.)
            plt.plot(lat, fluxes_model[('lcs0', flun, seas, 'zonal', ann)], label = 'stoc '+seas, color = col, linewidth = 2., linestyle = '--')
            plt.plot(lat, era_fluxes_zonal[(flun, seas)], label = 'ref '+seas, color = col, linewidth = 0.7, linestyle = ':')
        plt.legend()
        plt.grid()
        plt.ylim(limits[('zonal', flun)])
        plt.xlabel('Latitude')
        plt.ylabel('Integrated Net Heat Flux (W)')
        figures.append(fig)

ctl.plot_pdfpages(figure_file, figures)


figure_file = cart_out + 'maps_hfc_lcb0_vs_lcs0_NEW.pdf'
figures = []
for flun in fluxnames:
    for ann in annme:
        data = fluxes_model[('lcs0', flun, seas, 'map', ann)] - fluxes_model[('lcb0', flun, seas, 'map', ann)]
        fig = ctl.plot_map_contour(data, lat, lon, title = 'stoc-base diff: {} - {}'.format(flun, ann), cb_label = 'W/m', cbar_range = limits[('map', flun)])
        figures.append(fig)

ctl.plot_pdfpages(figure_file, figures)


figure_file = cart_out + 'cross_hfc_lcb0_vs_lcs0_NEW.pdf'
figures = []
for flun in fluxnames:
    for ann in annme:
        data = fluxes_model[('lcs0', flun, seas, 'cross', ann)] - fluxes_model[('lcb0', flun, seas, 'cross', ann)]
        fig = ctl.plot_lat_crosssection(data, lat, level, title = 'stoc-base diff: {} - {}'.format(flun, ann), cb_label = 'W/m', cbar_range = limits[('cross', flun)])
        figures.append(fig)

ctl.plot_pdfpages(figure_file, figures)


# figure_file = cart_out + 'ERA_calc6hrs_vs_ref.pdf'
# tag = 'ERA_1988_6hrs'
# #results = cd.heat_flux_calc(file_list, file_ps, cart_out, tag, all_seasons_output = True)
#
# file_list = ['ERAInterim_MM_1988_vt_prod.nc', 'ERAInterim_MM_1988_vz_prod.nc', 'ERAInterim_MM_1988_vq_prod.nc']
# file_list = [cart_in+fi for fi in file_list]
# #results = cd.heat_flux_calc(file_list, file_ps, cart_out, tag, zg_in_ERA_units = True, full_calculation = False)
#
# mrgegegeg = [-3.5e16, 4.5e16]
# results = pickle.load(open(cart_out+'out_hfc_{}_.p'.format(tag), 'r'))
# seasons = ['year', 'JJA', 'DJF']
# print('ejajaajajajajajaj')
#
# figures = []
# for flun in fluxnames:
#     fig = plt.figure()
#     plt.title('{} fluxes - ERAref vs ERAcalc6hrs'.format(flun))
#     cset = ctl.color_set(len(seasons), bright_thres = 1)
#     for seas, col in zip(seasons, cset):
#         plt.plot(lat, results[flun]['zonal'][seas], label = 'calc '+seas, color = col, linewidth = 2.)
#         plt.plot(lat, era_fluxes_zonal[(flun, seas)], label = 'ref '+seas, color = col, linewidth = 0.7, linestyle = ':')
#     plt.legend()
#     plt.grid()
#     plt.ylim(mrgegegeg)
#     plt.xlabel('Latitude')
#     plt.ylabel('Integrated Net Heat Flux (W)')
#     figures.append(fig)
#
# ctl.plot_pdfpages(figure_file, figures)
