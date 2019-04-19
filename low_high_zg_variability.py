#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
from matplotlib import pyplot as plt
from matplotlib import cm
import pickle

import climtools_lib as ctl
import climdiags as cd

from matplotlib.colors import LogNorm

#######################################

season = 'DJF'
area = 'EAT'

cart_out = '/home/fabiano/Research/lavori/PRIMAVERA_bias/'
if not os.path.exists(cart_out): os.mkdir(cart_out)

file_in = '/data-hobbes/fabiano/OBS/ERA/ERA40+Int_daily_1957-2018_zg500_remap25_meters.nc'

var, coords, aux_info = ctl.read_iris_nc(file_in, extract_level_hPa = 500)
lat = coords['lat']
lon = coords['lon']
dates = coords['dates']

var, dates = ctl.sel_time_range(var, dates, ctl.range_years(1979,2014))

mean_field, _ = ctl.seasonal_climatology(var, dates, season)
var_anoms = ctl.anomalies_daily_detrended(var, dates)

# # LOW FREQ VARIABILITY
# var_ullow = ctl.running_mean(var_anoms, 15)
# var_ullow_DJF, dates_DJF = ctl.sel_season(var_ullow, dates, season)
#
# ullowfr_variab = np.std(var_ullow_DJF, axis = 0)
# ullowfr_variab_zonal = ctl.zonal_mean(ullowfr_variab)

# LOW FREQ VARIABILITY
var_low = ctl.running_mean(var_anoms, 5)
var_low_DJF, dates_DJF = ctl.sel_season(var_low, dates, season)

lowfr_variab = np.std(var_low_DJF, axis = 0)
lowfr_variab_zonal = ctl.zonal_mean(lowfr_variab)

# High freq
var_high = var_anoms - var_low
var_high_DJF, dates_DJF = ctl.sel_season(var_high, dates, season)

highfr_variab = np.std(var_high_DJF, axis = 0)
highfr_variab_zonal = ctl.zonal_mean(highfr_variab)

# ctl.plot_map_contour(ullowfr_variab, lat, lon, title = 'ULTRA LOW')
all_figs = []
fig = ctl.plot_map_contour(mean_field, lat, lon, title = 'Mean field - {}'.format('era'), visualization = 'Nstereo', bounding_lat = 20)
all_figs.append(fig)

fig = ctl.plot_map_contour(lowfr_variab, lat, lon, title = 'Low fr var - {}'.format('era'), visualization = 'Nstereo', bounding_lat = 20)
all_figs.append(fig)

fig = ctl.plot_map_contour(highfr_variab, lat, lon, title = 'High fr var - {}'.format('era'), visualization = 'Nstereo', bounding_lat = 20)
all_figs.append(fig)


cart_mods = '/data-hobbes/fabiano/PRIMAVERA/highres_SST/'
listamods = [li.rstrip() for li in open(cart_mods + 'listamods', 'r').readlines()]

mean_field_all = dict()
lowfrvar = dict()
highfrvar = dict()

mean_field_all['era'] = mean_field
lowfrvar['era'] = lowfr_variab
highfrvar['era'] = highfr_variab

for modfil in listamods:
    mod = modfil.split('_')[2]

    var, coords, aux_info = ctl.read_iris_nc(cart_mods + modfil, extract_level_hPa = 500)
    lat = coords['lat']
    lon = coords['lon']
    dates = coords['dates']

    var, dates = ctl.sel_time_range(var, dates, ctl.range_years(1979,2014))

    mean_field, _ = ctl.seasonal_climatology(var, dates, season)
    var_anoms = ctl.anomalies_daily_detrended(var, dates)

    # LOW FREQ VARIABILITY
    var_low = ctl.running_mean(var_anoms, 5)
    var_low_DJF, dates_DJF = ctl.sel_season(var_low, dates, season)

    lowfr_variab = np.std(var_low_DJF, axis = 0)
    lowfr_variab_zonal = ctl.zonal_mean(lowfr_variab)

    # High freq
    var_high = var_anoms - var_low
    var_high_DJF, dates_DJF = ctl.sel_season(var_high, dates, season)

    highfr_variab = np.std(var_high_DJF, axis = 0)
    highfr_variab_zonal = ctl.zonal_mean(highfr_variab)

    # saving
    mean_field_all[mod] = mean_field
    lowfrvar[mod] = lowfr_variab
    highfrvar[mod] = highfr_variab

    # ctl.plot_map_contour(ullowfr_variab, lat, lon, title = 'ULTRA LOW')
    fig = ctl.plot_map_contour(mean_field-mean_field_all['era'], lat, lon, title = 'Mean field - {}'.format(mod), visualization = 'Nstereo', bounding_lat = 20)
    all_figs.append(fig)

    fig = ctl.plot_map_contour(lowfr_variab-lowfrvar['era'], lat, lon, title = 'Low fr var - {}'.format(mod), visualization = 'Nstereo', bounding_lat = 20)
    all_figs.append(fig)

    fig = ctl.plot_map_contour(highfr_variab-highfrvar['era'], lat, lon, title = 'High fr var - {}'.format(mod), visualization = 'Nstereo', bounding_lat = 20)
    all_figs.append(fig)


filename = cart_out + 'all_figs_lowhighvar.pdf'
ctl.plot_pdfpages(filename, all_figs)

pickle.dump([mean_field_all, lowfrvar, highfrvar], open(cart_out + 'out_lowhighvar.p', 'w'))
