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
from datetime import datetime

#######################################

print(ctl.datestamp())
season = 'DJF'
area = 'EAT'

cart_out = '/home/fabiano/Research/lavori/WeatherRegimes/taspr_composites_ERA/'
if not os.path.exists(cart_out): os.mkdir(cart_out)

file_in = '/data-hobbes/fabiano/OBS/ERA/ERAInterim/zg500/ERAInt_daily_1979-2018_129_zg500_remap25_meters.nc'

print(ctl.datestamp())

# lat = np.arange(-90, 91, 2.5)
# lon = np.arange(0., 360, 2.5)
var, coords, aux_info = ctl.read_iris_nc(file_in, extract_level_hPa = 500)
lat = coords['lat']
lon = coords['lon']
dates = coords['dates']
print(dates[0], dates[-1], var.shape)

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

sys.exit()

kwar['ref_solver'] = results_ref['solver']
kwar['ref_patterns_area'] = results_ref['cluspattern_area']
kwar['use_reference_eofs'] = True

#file_in = '/data-hobbes/fabiano/OBS/ERA/ERA40+Int_daily_1957-2018_zg500_remap25_meters.nc'
cart_in = '/data-hobbes/fabiano/SPHINX/zg_daily/'
filenam = dict()
# filenam['stoc'] = 'lcs{}-1850-2100-NDJFM_zg500_NH_14473.nc'
# filenam['base'] = 'lcb{}-1850-2100-NDJFM_zg500_NH_14473.nc'
# filenam[('base', 'prtas')] = 'lcb{}-1950-2100-NDJFM_{}_day_NH.nc'
# filenam[('stoc', 'prtas')] = 'lcs{}-1950-2100-NDJFM_{}_day_NH.nc'
filenam['stoc'] = 'lcs{}-1979-2018-NDJFM_zg500_NH_14473.nc'
filenam['base'] = 'lcb{}-1979-2018-NDJFM_zg500_NH_14473.nc'
filenam[('base', 'prtas')] = 'lcb{}-1979-2018-NDJFM_{}_day_NH.nc'
filenam[('stoc', 'prtas')] = 'lcs{}-1979-2018-NDJFM_{}_day_NH.nc'

carttemp = '/data-hobbes/fabiano/SPHINX/tas_pr_day/'

start_year = 1979
end_year = 2018

all_compos = dict()
all_compos['ERA'] = pickle.load(open(cart_out + 'out_composites_ERA_precERA.p'))
[all_compos['base'], all_compos['base_std']] = pickle.load(open(cart_out + 'out_composites_base.p'))
[all_compos['stoc'], all_compos['stoc_std']] = pickle.load(open(cart_out + 'out_composites_stoc.p'))

# num = dict()
# num['base'] = 3
# num['stoc'] = 2
#
# for cos in ['base', 'stoc']:
#     for ind in range(num[cos]):
#         nam = cos+'{}'.format(ind)
#         file_in = cart_in + filenam[cos].format(ind)
#         var, coords, aux_info = ctl.read_iris_nc(file_in, extract_level_hPa = 500)
#         lat = coords['lat']
#         lon = coords['lon']
#         dates = coords['dates']
#
#         var, dates = ctl.sel_time_range(var, dates, ctl.range_years(start_year, end_year))
#         print(dates[0], dates[-1], var.shape)
#
#         var_season, dates_season = ctl.sel_season(var, dates, season)
#         all_years = np.arange(dates[0].year, dates[-1].year+1)
#
#         results = cd.WRtool_core(var_season, lat, lon, dates_season, area, heavy_output = True, **kwar)
#
#         # OK. Now I have the regimes. Read the temp/prec.
#         file_temp_ref = iris.load('/data-hobbes/fabiano/OBS/ERA/ERAInterim/ERAInt_daily_1979-2018_167.nc')[0]
#
#         file_in = carttemp + filenam[(cos, 'prtas')].format(ind, 'tas')
#         temp, coords_temp, aux_info_temp = ctl.read_iris_nc(file_in, regrid_to_reference = file_temp_ref)
#         temp, coords_temp['dates'] = ctl.sel_time_range(temp, coords_temp['dates'], ctl.range_years(start_year, end_year))
#
#         seas_temp_mean, seas_temp_std = ctl.seasonal_climatology(temp, coords_temp['dates'], season)
#         temp_seas, datetemp = ctl.sel_season(temp, coords_temp['dates'], season)
#         temp_seas_NH, lat, lon = ctl.sel_area(coords_temp['lat'], coords_temp['lon'], temp_seas, 'NH')
#         del temp, temp_seas
#         coords_temp['lat'] = lat
#         coords_temp['lon'] = lon
#
#         temp_anoms = ctl.anomalies_daily_detrended(temp_seas_NH, datetemp)
#
#         file_in = carttemp + filenam[(cos, 'prtas')].format(ind, 'pr')
#         prec, coords_prec, aux_info_prec = ctl.read_iris_nc(file_in, convert_units_to = 'kg m-2 day-1', regrid_to_reference = file_temp_ref)
#         prec, coords_prec['dates'] = ctl.sel_time_range(prec, coords_prec['dates'], ctl.range_years(start_year, end_year))
#
#         seas_prec_mean, seas_prec_std = ctl.seasonal_climatology(prec, coords_prec['dates'], season)
#         prec_seas, dateprec = ctl.sel_season(prec, coords_prec['dates'], season)
#         prec_seas_NH, lat, lon = ctl.sel_area(coords_prec['lat'], coords_prec['lon'], prec_seas, 'NH')
#         del prec, prec_seas
#         coords_prec['lat'] = lat
#         coords_prec['lon'] = lon
#
#         prec_anoms = ctl.anomalies_daily(prec_seas_NH, dateprec, window = 10)
#
#         del prec_seas_NH, temp_seas_NH
#
#         oklabels, _ = ctl.sel_time_range(results['labels'], results['dates'], dates_range = (dateprec[0], dateprec[-1]))
#
#         allfigs = []
#         # Calculate and visualize composites. NO FILTERS
#         compos = dict()
#         compos['temp'] = []
#         compos['prec'] = []
#         for reg in range(kwar['numclus']):
#             labok = results['labels'] == reg
#             cosa = np.mean(temp_anoms[labok, ...], axis = 0)
#             compos['temp'].append(cosa)
#
#             fig = ctl.plot_map_contour(cosa, coords_temp['lat'], coords_temp['lon'], title = 'Temp anom - regime {}'.format(reg), plot_margins = (-120, 120, 20, 90), cbar_range = (-5, 5))
#             allfigs.append(fig)
#
#             labok = oklabels == reg
#             cosa = np.mean(prec_anoms[labok, ...], axis = 0)
#             compos['prec'].append(cosa)
#
#             fig = ctl.plot_map_contour(cosa, coords_prec['lat'], coords_prec['lon'], title = 'Prec anom - regime {}'.format(reg), plot_margins = (-120, 120, 20, 90), cbar_range = (-5, 5), cmap = 'RdBu')
#             allfigs.append(fig)
#
#         # Calculate and visualize composites. FILTER ON EVENT DURATION > 5 days
#         rsd_tim, rsd_dat, rsd_num = ctl.calc_regime_residtimes(results['labels'], dates = results['dates'])
#         days_event, length_event = ctl.calc_days_event(results['labels'], rsd_tim, rsd_num)
#         okleneve, _ = ctl.sel_time_range(length_event, results['dates'], dates_range = (dateprec[0], dateprec[-1]))
#
#         compos['temp_filt'] = []
#         compos['prec_filt'] = []
#         for reg in range(kwar['numclus']):
#             labok = (results['labels'] == reg) & (length_event > 5)
#             cosa = np.mean(temp_anoms[labok, ...], axis = 0)
#             compos['temp_filt'].append(cosa)
#
#             fig = ctl.plot_map_contour(cosa, coords_temp['lat'], coords_temp['lon'], title = 'Temp anom - regime {} - filt 5 days'.format(reg), plot_margins = (-120, 120, 20, 90), cbar_range = (-5, 5))
#             allfigs.append(fig)
#
#             labok = (oklabels == reg) & (okleneve > 5)
#             labok = oklabels == reg
#             cosa = np.mean(prec_anoms[labok, ...], axis = 0)
#             compos['prec_filt'].append(cosa)
#
#             fig = ctl.plot_map_contour(cosa, coords_prec['lat'], coords_prec['lon'], title = 'Prec anom - regime {} - filt 5 days'.format(reg), plot_margins = (-120, 120, 20, 90), cbar_range = (-5, 5), cmap = 'RdBu')
#             allfigs.append(fig)
#
#         ctl.plot_pdfpages(cart_out + 'temprec_composites_{}.pdf'.format(nam), allfigs)
#
#         for k in compos:
#             compos[k] = np.stack(compos[k])
#         all_compos[nam] = compos
#
#     compos = dict()
#     compos_std = dict()
#     for k in all_compos[nam]:
#         compos[k] = np.mean([all_compos[cos+'{}'.format(i)][k] for i in range(num[cos])], axis = 0)
#         compos_std[k] = np.std([all_compos[cos+'{}'.format(i)][k] for i in range(num[cos])], axis = 0)
#
#     all_compos[cos] = compos
#     all_compos[cos+'_std'] = compos_std
#
#     pickle.dump([compos, compos_std], open(cart_out + 'out_composites_{}.p'.format(cos), 'wb'))
#
#     for nam in [cos, cos+'_std']:
#         compos = all_compos[nam]
#         for reg in range(kwar['numclus']):
#             cosa = compos['temp'][reg]
#             fig = ctl.plot_map_contour(cosa, coords_temp['lat'], coords_temp['lon'], title = 'Temp anom - regime {}'.format(reg), plot_margins = (-120, 120, 20, 90), cbar_range = (-5, 5))
#             allfigs.append(fig)
#
#             cosa = compos['prec'][reg]
#             fig = ctl.plot_map_contour(cosa, coords_prec['lat'], coords_prec['lon'], title = 'Prec anom - regime {}'.format(reg), plot_margins = (-120, 120, 20, 90), cbar_range = (-5, 5), cmap = 'RdBu')
#             allfigs.append(fig)
#
#         for reg in range(kwar['numclus']):
#             cosa = compos['temp_filt'][reg]
#             fig = ctl.plot_map_contour(cosa, coords_temp['lat'], coords_temp['lon'], title = 'Temp anom - regime {} - filt 5 days'.format(reg), plot_margins = (-120, 120, 20, 90), cbar_range = (-5, 5))
#             allfigs.append(fig)
#
#             cosa = compos['prec_filt'][reg]
#             fig = ctl.plot_map_contour(cosa, coords_prec['lat'], coords_prec['lon'], title = 'Prec anom - regime {} - filt 5 days'.format(reg), plot_margins = (-120, 120, 20, 90), cbar_range = (-5, 5), cmap = 'RdBu')
#             allfigs.append(fig)
#
#         ctl.plot_pdfpages(cart_out + 'temprec_composites_{}.pdf'.format(nam), allfigs)



# QUI FACCIO: pattern correlation, plot_triple_sidebyside, ecc.ecc.
lat = np.arange(0, 90.1, 1.5)
lon = np.arange(0, 360, 1.5)

varunits = dict()
varunits['temp'] = 'K'
varunits['prec'] = 'mm/day'

limitz = dict()
limitz['temp'] = (0.4, 2.)
limitz['prec'] = (0.6, 0.8)

area = 'EAT'

area_compos = dict()
for var in ['temp', 'prec']:
    for k in all_compos:
        if type(all_compos[k][var]) == list:
            all_compos[k][var] = np.stack(all_compos[k][var])
        area_compos[(var, k)], lat_area, lon_area = ctl.sel_area(lat, lon, all_compos[k][var], area)#'Eu')

ncar = np.sqrt(area_compos[(var, k)][0, ...].size)

val_compare = dict()
for cos in ['base', 'stoc']:
    for var in ['temp', 'prec']:
        et, patcor = ctl.calc_RMS_and_patcor(area_compos[(var, cos)], area_compos[(var, 'ERA')])
        val_compare[('RMS', var, cos)] = et/ncar
        val_compare[('patcor', var, cos)] = patcor

        et2, patcor2 = ctl.calc_RMS_and_patcor(area_compos[(var, cos)]+area_compos[(var, cos+'_std')], area_compos[(var, 'ERA')])
        et3, patcor3 = ctl.calc_RMS_and_patcor(area_compos[(var, cos)]-area_compos[(var, cos+'_std')], area_compos[(var, 'ERA')])
        etd = np.array([max([et[i], et2[i], et3[i]])-min([et[i], et2[i], et3[i]]) for i in range(4)])/(2*ncar)
        patd = np.array([max([patcor[i], patcor2[i], patcor3[i]])-min([patcor[i], patcor2[i], patcor3[i]]) for i in range(4)])/2.
        val_compare[('RMS', var, cos+'_std')] = etd
        val_compare[('patcor', var, cos+'_std')] = patd

print(val_compare)
plotmarg = (lon_area[0], lon_area[-1], lat_area[0], lat_area[-1])
allfigs_compare = []

# ellipse plot
pattnames = ['NAO +', 'Sc. Blocking', 'Atl. Ridge', 'NAO -']
colors = ctl.color_set(6)
cols = []
cols.append(np.mean(colors[:3], axis = 0))
cols.append(np.mean(colors[3:], axis = 0))

for var in ['temp', 'prec']:
    fig = plt.figure(figsize = (16,12))
    axes = []
    for i in range(4):
        ax = fig.add_subplot(2, 2, i+1)
        ax.set_title(pattnames[i], fontsize = 18, fontweight = 'bold')

        ax.grid()
        val1 = [val_compare[('patcor', var, cos)][i] for cos in ['base', 'stoc']]
        val2 = [val_compare[('RMS', var, cos)][i] for cos in ['base', 'stoc']]
        val1_err = [val_compare[('patcor', var, cos+'_std')][i] for cos in ['base', 'stoc']]
        val2_err = [val_compare[('RMS', var, cos+'_std')][i] for cos in ['base', 'stoc']]
        ctl.ellipse_plot(val1, val2, val1_err, val2_err, alpha = 0.7, ax = ax, colors = cols, labels = ['base', 'stoc'])

        # for mod, col in zip(all_res.keys(), colors):
        #     pats = [coso['patcor'][i] for coso in all_res[mod]]
        #     rmss = [coso['RMS'][i]/nsqr for coso in all_res[mod]]
        #     ax.scatter(pats, rmss, color = col, s = 5)

        ax.set_xlim(limitz[var][0], 1.0)
        ax.set_ylim(0., limitz[var][1])
        ax.tick_params(labelsize=14)
        plt.gca().invert_xaxis()
        ax.set_xlabel('Pattern correlation', fontsize = 18)
        ax.set_ylabel('RMS ({})'.format(varunits[var]), fontsize = 18)
        axes.append(ax)

    ctl.adjust_ax_scale(axes)

    plt.tight_layout()
    plt.subplots_adjust(top = 0.9)
    plt.suptitle('Regime composites of {}'.format(var), fontsize = 28)
    fig.savefig(cart_out + 'ellipse_plot_{}.pdf'.format(var))
    allfigs_compare.append(fig)

for var in ['temp', 'prec']:
    for i in range(4):
        for cos in ['base', 'stoc']:
            fig = ctl.plot_triple_sidebyside(area_compos[(var, cos)][i], area_compos[(var, 'ERA')][i], lat_area, lon_area, plot_margins = plotmarg, title = '{} - {} vs ERA - {}'.format(var, cos, pattnames[i]), stitle_1 = cos, stitle_2 = 'ERA', cb_label = varunits[var], cbar_range = (-4, 4))
            allfigs_compare.append(fig)
        fig = ctl.plot_triple_sidebyside(area_compos[(var, 'stoc')][i], area_compos[(var, 'base')][i], lat_area, lon_area, plot_margins = plotmarg, title = '{} - stoc vs base - {}'.format(var, pattnames[i]), stitle_1 = 'stoc', stitle_2 = 'base', cb_label = varunits[var], cbar_range = (-4, 4))
        allfigs_compare.append(fig)

ctl.plot_pdfpages(cart_out + 'compos_SPHINX_vs_ERA.pdf', allfigs_compare)
