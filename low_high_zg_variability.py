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

# cart_out = '/home/fabiano/Research/lavori/PRIMAVERA_bias/'
# if not os.path.exists(cart_out): os.mkdir(cart_out)
cart_out = '/home/fedefab/Scrivania/Research/Post-doc/PRIMAVERA_bias/'

mod_HI = ['ECMWF-IFS-HR','HadGEM3-GC31-HM','EC-Earth3-HR','CMCC-CM2-VHR4','MPIESM-1-2-XR']
mod_LO = ['ECMWF-IFS-LR','HadGEM3-GC31-MM', 'EC-Earth3', 'CMCC-CM2-HR4', 'MPIESM-1-2-HR']
mod_short = ['ECMW','Had', 'ECE', 'CMCC', 'MPI']

mod_all = []
for mod1,mod2 in zip(mod_LO, mod_HI):
    mod_all.append(mod1)
    mod_all.append(mod2)

# file_in = '/data-hobbes/fabiano/OBS/ERA/ERA40+Int_daily_1957-2018_zg500_remap25_meters.nc'
#
# var, coords, aux_info = ctl.read_iris_nc(file_in, extract_level_hPa = 500)
# lat = coords['lat']
# lon = coords['lon']
# dates = coords['dates']
#
# var, dates = ctl.sel_time_range(var, dates, ctl.range_years(1979,2014))
#
# mean_field, _ = ctl.seasonal_climatology(var, dates, season)
# var_anoms = ctl.anomalies_daily_detrended(var, dates)
#
# # # LOW FREQ VARIABILITY
# # var_ullow = ctl.running_mean(var_anoms, 15)
# # var_ullow_DJF, dates_DJF = ctl.sel_season(var_ullow, dates, season)
# #
# # ullowfr_variab = np.std(var_ullow_DJF, axis = 0)
# # ullowfr_variab_zonal = ctl.zonal_mean(ullowfr_variab)
#
# # LOW FREQ VARIABILITY
# var_low = ctl.running_mean(var_anoms, 5)
# var_low_DJF, dates_DJF = ctl.sel_season(var_low, dates, season)
#
# lowfr_variab = np.std(var_low_DJF, axis = 0)
# lowfr_variab_zonal = ctl.zonal_mean(lowfr_variab)
#
# # High freq
# var_high = var_anoms - var_low
# var_high_DJF, dates_DJF = ctl.sel_season(var_high, dates, season)
#
# highfr_variab = np.std(var_high_DJF, axis = 0)
# highfr_variab_zonal = ctl.zonal_mean(highfr_variab)

# ctl.plot_map_contour(ullowfr_variab, lat, lon, title = 'ULTRA LOW')


# cart_mods = '/data-hobbes/fabiano/PRIMAVERA/highres_SST/'
# listamods = [li.rstrip() for li in open(cart_mods + 'listamods', 'r').readlines()]

mean_field_all = dict()
lowfrvar = dict()
highfrvar = dict()

mean_field_all, lowfrvar, highfrvar = pickle.load(open(cart_out + 'out_lowhighvar.p', 'r'))
lat = np.arange(-90, 90.1, 2.5)
lon = np.arange(0, 360, 2.5)

areas = dict()
areas['mean field'] = [(0, 360, 50, 90)]
areas['low fr var'] = [(-60, 10, 55, 85), (-180, -130, 40, 75)]
areas['high fr var'] = [(-90, -20, 35, 60), (160, 200, 30, 45)]

i = 0
wi = 0.6
colors = ctl.color_set(6)
oklats = lat >= 40.
for nom, cos in zip(['mean field', 'low fr var', 'high fr var'], [mean_field_all, lowfrvar, highfrvar]):
    for iol, area in enumerate(areas[nom]):
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        var_ref, _, _ = ctl.sel_area(lat, lon, cos['era'], area)

        xti = []

        for mod1, mod2, col in zip(mod_LO, mod_HI, colors):
            var, _, _ = ctl.sel_area(lat, lon, cos[mod1], area)
            rms = ctl.E_rms(var, var_ref)
            patcor = ctl.Rcorr(var, var_ref)
            ax.bar(i, rms, width = wi, color = col)
            ax2.bar(i, 1-patcor, width = wi, color = col)
            xti.append(i+0.35)
            i+=0.7
            var, _, _ = ctl.sel_area(lat, lon, cos[mod2], area)
            rms = ctl.E_rms(var, var_ref)
            patcor = ctl.Rcorr(var, var_ref)
            ax.bar(i, rms, width = wi, color = col)
            ax2.bar(i, 1-patcor, width = wi, color = col)
            i+=1

        ax.set_xticks(xti, minor = False)
        ax.set_xticklabels(mod_short, size='small')
        ax2.set_xticks(xti, minor = False)
        ax2.set_xticklabels(mod_short, size='small')

        ax.set_title(nom)
        ax2.set_title(nom)
        ax.set_ylabel('RMS (m)')
        ax2.set_ylabel('1 - patcor')

        fig.tight_layout()
        fig.savefig(cart_out + nom.split()[0]+'_stat_area{}.pdf'.format(iol))


# mean_field_all['era'] = mean_field
# lowfrvar['era'] = lowfr_variab
# highfrvar['era'] = highfr_variab
mean_field = mean_field_all['era']
lowfr_variab = lowfrvar['era']
highfr_variab = highfrvar['era']

all_figs_mf = []
all_figs_lo = []
all_figs_hi = []

fig = ctl.plot_map_contour(mean_field, lat, lon, title = 'Mean field - {}'.format('era'), visualization = 'Nstereo', bounding_lat = 30, plot_anomalies = False, cbar_range = [5000.,5800.], add_rectangles = areas['mean field'])
all_figs_mf.append(fig)

fig = ctl.plot_map_contour(lowfr_variab, lat, lon, title = 'Low fr var - {}'.format('era'), visualization = 'Nstereo', bounding_lat = 30, plot_anomalies = False, cbar_range = [10.,150.], add_rectangles = areas['low fr var'])
all_figs_lo.append(fig)

fig = ctl.plot_map_contour(highfr_variab, lat, lon, title = 'High fr var - {}'.format('era'), visualization = 'Nstereo', bounding_lat = 30, plot_anomalies = False, cbar_range = [5., 80.], add_rectangles = areas['high fr var'])
all_figs_hi.append(fig)

#for modfil in listamods:
#for mod in np.sort(mean_field_all.keys()):
for mod in mod_all:
    #mod = modfil.split('_')[2]
    if mod == 'era': continue

    # var, coords, aux_info = ctl.read_iris_nc(cart_mods + modfil, extract_level_hPa = 500)
    # lat = coords['lat']
    # lon = coords['lon']
    # dates = coords['dates']
    #
    # var, dates = ctl.sel_time_range(var, dates, ctl.range_years(1979,2014))
    #
    # mean_field, _ = ctl.seasonal_climatology(var, dates, season)
    # var_anoms = ctl.anomalies_daily_detrended(var, dates)
    #
    # # LOW FREQ VARIABILITY
    # var_low = ctl.running_mean(var_anoms, 5)
    # var_low_DJF, dates_DJF = ctl.sel_season(var_low, dates, season)
    #
    # lowfr_variab = np.std(var_low_DJF, axis = 0)
    # lowfr_variab_zonal = ctl.zonal_mean(lowfr_variab)
    #
    # # High freq
    # var_high = var_anoms - var_low
    # var_high_DJF, dates_DJF = ctl.sel_season(var_high, dates, season)
    #
    # highfr_variab = np.std(var_high_DJF, axis = 0)
    # highfr_variab_zonal = ctl.zonal_mean(highfr_variab)
    #
    # # saving
    # mean_field_all[mod] = mean_field
    # lowfrvar[mod] = lowfr_variab
    # highfrvar[mod] = highfr_variab
    mean_field = mean_field_all[mod]
    lowfr_variab = lowfrvar[mod]
    highfr_variab = highfrvar[mod]

    # ctl.plot_map_contour(ullowfr_variab, lat, lon, title = 'ULTRA LOW')
    fig = ctl.plot_map_contour(mean_field, lat, lon, title = 'Mean field - {}'.format(mod), visualization = 'Nstereo', bounding_lat = 30, cbar_range = [5000.,5800.], plot_anomalies = False, add_rectangles = areas['mean field'])
    all_figs_mf.append(fig)

    fig = ctl.plot_map_contour(lowfr_variab, lat, lon, title = 'Low fr var - {}'.format(mod), visualization = 'Nstereo', bounding_lat = 30, cbar_range = [10., 150.], plot_anomalies = False, add_rectangles = areas['low fr var'])
    all_figs_lo.append(fig)

    fig = ctl.plot_map_contour(highfr_variab, lat, lon, title = 'High fr var - {}'.format(mod), visualization = 'Nstereo', bounding_lat = 30, cbar_range = [5., 80], plot_anomalies = False, add_rectangles = areas['high fr var'])
    all_figs_hi.append(fig)

for mod in mod_all:
    mean_field = mean_field_all[mod]
    lowfr_variab = lowfrvar[mod]
    highfr_variab = highfrvar[mod]

    # ctl.plot_map_contour(ullowfr_variab, lat, lon, title = 'ULTRA LOW')
    fig = ctl.plot_map_contour(mean_field-mean_field_all['era'], lat, lon, title = 'Mean field - {}'.format(mod), visualization = 'Nstereo', bounding_lat = 30, cbar_range = [-100, 100], add_rectangles = areas['mean field'])
    all_figs_mf.append(fig)

    fig = ctl.plot_map_contour(lowfr_variab-lowfrvar['era'], lat, lon, title = 'Low fr var - {}'.format(mod), visualization = 'Nstereo', bounding_lat = 30, cbar_range = [-40, 40], add_rectangles = areas['low fr var'])
    all_figs_lo.append(fig)

    fig = ctl.plot_map_contour(highfr_variab-highfrvar['era'], lat, lon, title = 'High fr var - {}'.format(mod), visualization = 'Nstereo', bounding_lat = 30, cbar_range = [-20, 20], add_rectangles = areas['high fr var'])
    all_figs_hi.append(fig)


for mod1, mod2 in zip(mod_LO, mod_HI):
    mod = '-'.join(mod2.split('-')[:-1])
    diff = mean_field_all[mod2] - mean_field_all[mod1]
    fig = ctl.plot_map_contour(diff, lat, lon, title = 'Mean field diff HR-LR - {}'.format(mod), visualization = 'Nstereo', bounding_lat = 30, cbar_range = [-100, 100])
    all_figs_mf.append(fig)

    diff = lowfrvar[mod2] - lowfrvar[mod1]
    fig = ctl.plot_map_contour(diff, lat, lon, title = 'Low fr var  diff HR-LR - {}'.format(mod), visualization = 'Nstereo', bounding_lat = 30, cbar_range = [-40, 40])
    all_figs_lo.append(fig)

    diff = highfrvar[mod2] - highfrvar[mod1]
    fig = ctl.plot_map_contour(diff, lat, lon, title = 'High fr var diff HR-LR - {}'.format(mod), visualization = 'Nstereo', bounding_lat = 30, cbar_range = [-20, 20])
    all_figs_hi.append(fig)

# for mod1, mod2 in zip(mod_LO, mod_HI):
#     mod = '-'.join(mod2.split('-')[:-1])
#     diff = abs(mean_field_all[mod2] - mean_field_all['era']) < abs(mean_field_all[mod1] -mean_field_all['era'])
#     fig = ctl.plot_map_contour(diff, lat, lon, title = 'Mean field - better HR? - {}'.format(mod), visualization = 'Nstereo', bounding_lat = 30)
#     all_figs_mf.append(fig)
#
#     diff = abs(lowfrvar[mod2] - lowfrvar['era']) < abs(lowfrvar[mod1] -lowfrvar['era'])
#     fig = ctl.plot_map_contour(diff, lat, lon, title = 'Low fr var - better HR? - {}'.format(mod), visualization = 'Nstereo', bounding_lat = 30)
#     all_figs_lo.append(fig)
#
#     diff = abs(highfrvar[mod2] - highfrvar['era']) < abs(highfrvar[mod1] -highfrvar['era'])
#     fig = ctl.plot_map_contour(diff, lat, lon, title = 'High fr var - better HR? - {}'.format(mod), visualization = 'Nstereo', bounding_lat = 30)
#     all_figs_hi.append(fig)

filename = cart_out + 'all_figs_mf.pdf'
ctl.plot_pdfpages(filename, all_figs_mf)

filename = cart_out + 'all_figs_lowvar.pdf'
ctl.plot_pdfpages(filename, all_figs_lo)

filename = cart_out + 'all_figs_highvar.pdf'
ctl.plot_pdfpages(filename, all_figs_hi)

#pickle.dump([mean_field_all, lowfrvar, highfrvar], open(cart_out + 'out_lowhighvar.p', 'w'))
