#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
from matplotlib import pyplot as plt
import pickle

import climtools_lib as ctl
import climdiags as cd

from matplotlib.colors import LogNorm
import iris

#######################################

season = 'DJF'
area = 'EAT'

cart_out = '/home/fabiano/Research/lavori/WeatherRegimes/taspr_composites_ERA/'
if not os.path.exists(cart_out): os.mkdir(cart_out)

#file_in = '/data-hobbes/fabiano/OBS/ERA/ERA40+Int_daily_1957-2018_zg500_remap25_meters.nc'
file_in = '/data-hobbes/fabiano/OBS/ERA/ERAInterim/zg500/ERAInt_daily_1979-2018_129_zg500_remap25_meters.nc'

print(ctl.datestamp())

# lat = np.arange(-90, 91, 2.5)
# lon = np.arange(0., 360, 2.5)
var, coords, aux_info = ctl.read_iris_nc(file_in, extract_level_hPa = 500)
lat = coords['lat']
lon = coords['lon']
dates = coords['dates']

var_season, dates_season = ctl.sel_season(var, dates, season)
all_years = np.arange(dates[0].year, dates[-1].year+1)

kwar = dict()
kwar['numclus'] = 4
kwar['run_significance_calc'] = False
kwar['numpcs'] = 4
kwar['detrended_eof_calculation'] = False
kwar['detrended_anom_for_clustering'] = False
kwar['nrsamp_sig'] = 500

results_ref = cd.WRtool_core(var_season, lat, lon, dates_season, area, heavy_output = True, **kwar)

# OK. Now I have the regimes. Read the temp/prec.
file_temp_era = '/data-hobbes/fabiano/OBS/ERA/ERAInterim/ERAInt_daily_1979-2018_167.nc'
temp, coords_temp, aux_info_temp = ctl.read_iris_nc(file_temp_era)

seas_temp_mean, seas_temp_std = ctl.seasonal_climatology(temp, coords_temp['dates'], season)
temp_seas, datetemp = ctl.sel_season(temp, coords_temp['dates'], season)
temp_seas_NH, lat, lon = ctl.sel_area(coords_temp['lat'], coords_temp['lon'], temp_seas, 'NH')
del temp, temp_seas
coords_temp['lat'] = lat
coords_temp['lon'] = lon

temp_anoms = ctl.anomalies_daily_detrended(temp_seas_NH, datetemp)

file_temp_ref = iris.load('/data-hobbes/fabiano/OBS/ERA/ERAInterim/ERAInt_daily_1979-2018_167.nc')[0]
# file_prec_era = '/data-hobbes/fabiano/OBS/GPCC/daily/gpcc_daily_EU_1982-2016.nc'
# prec, coords_prec, aux_info_prec = ctl.read_iris_nc(file_prec_era, select_var = 'gpcc full data daily product version 2018 precipitation per grid')
file_prec_era = '/data-hobbes/fabiano/OBS/ERA/ERAInterim/ERAInt_daily_1979-2018_228_pr_daysum_ok.nc'
prec, coords_prec, aux_info_prec = ctl.read_iris_nc(file_prec_era, convert_units_to = 'mm', regrid_to_reference = file_temp_ref)#, select_var = 'gpcc full data daily product version 2018 precipitation per grid')

seas_prec_mean, seas_prec_std = ctl.seasonal_climatology(prec, coords_prec['dates'], season)
prec_seas, dateprec = ctl.sel_season(prec, coords_prec['dates'], season)
prec_seas_NH, lat, lon = ctl.sel_area(coords_prec['lat'], coords_prec['lon'], prec_seas, 'NH')
del prec, prec_seas
coords_prec['lat'] = lat
coords_prec['lon'] = lon

prec_anoms = ctl.anomalies_daily(prec_seas_NH, dateprec, window = 10)

del prec_seas_NH, temp_seas_NH

oklabels, _ = ctl.sel_time_range(results_ref['labels'], results_ref['dates'], dates_range = (dateprec[0], dateprec[-1]))

allfigs = []
# Calculate and visualize composites. NO FILTERS
compos = dict()
compos['temp'] = []
compos['prec'] = []
for reg in range(kwar['numclus']):
    labok = results_ref['labels'] == reg
    cosa = np.mean(temp_anoms[labok, ...], axis = 0)
    compos['temp'].append(cosa)

    fig = ctl.plot_map_contour(cosa, coords_temp['lat'], coords_temp['lon'], title = 'Temp anom - regime {}'.format(reg), plot_margins = (-120, 120, 20, 90), cbar_range = (-5, 5))
    allfigs.append(fig)

    labok = oklabels == reg
    cosa = np.mean(prec_anoms[labok, ...], axis = 0)
    compos['prec'].append(cosa)

    fig = ctl.plot_map_contour(cosa, coords_prec['lat'], coords_prec['lon'], title = 'Prec anom - regime {}'.format(reg), plot_margins = (-120, 120, 20, 90), cbar_range = (-5, 5), cmap = 'RdBu')
    allfigs.append(fig)

# Calculate and visualize composites. FILTER ON EVENT DURATION > 5 days
rsd_tim, rsd_dat, rsd_num = ctl.calc_regime_residtimes(results_ref['labels'], dates = results_ref['dates'])
days_event, length_event = ctl.calc_days_event(results_ref['labels'], rsd_tim, rsd_num)
okleneve, _ = ctl.sel_time_range(length_event, results_ref['dates'], dates_range = (dateprec[0], dateprec[-1]))

compos['temp_filt'] = []
compos['prec_filt'] = []
for reg in range(kwar['numclus']):
    labok = (results_ref['labels'] == reg) & (length_event > 5)
    cosa = np.mean(temp_anoms[labok, ...], axis = 0)
    compos['temp_filt'].append(cosa)

    fig = ctl.plot_map_contour(cosa, coords_temp['lat'], coords_temp['lon'], title = 'Temp anom - regime {} - filt 5 days'.format(reg), plot_margins = (-120, 120, 20, 90), cbar_range = (-5, 5))
    allfigs.append(fig)

    labok = (oklabels == reg) & (okleneve > 5)
    labok = oklabels == reg
    cosa = np.mean(prec_anoms[labok, ...], axis = 0)
    compos['prec_filt'].append(cosa)

    fig = ctl.plot_map_contour(cosa, coords_prec['lat'], coords_prec['lon'], title = 'Prec anom - regime {} - filt 5 days'.format(reg), plot_margins = (-120, 120, 20, 90), cbar_range = (-5, 5), cmap = 'RdBu')
    allfigs.append(fig)

ctl.plot_pdfpages(cart_out + 'temprec_composites_wERA.pdf', allfigs)

pickle.dump(compos, open(cart_out + 'out_composites_ERA_precERA.p', 'w'))
