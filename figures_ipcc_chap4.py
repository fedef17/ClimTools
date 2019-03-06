#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
from matplotlib import pyplot as plt
import pickle
import iris
import cartopy.crs as ccrs
import matplotlib.cm as cm

import climtools_lib as ctl
import climdiags as cd

#######################################

# IPCC color palettes

temp_palette = ((103, 0, 31), (178, 24, 43), (214, 96, 77), (244, 165, 130), (253, 219, 199), (247, 247, 247), (209, 229, 240), (146, 197, 222), (67, 147, 195), (33, 102, 172), (5, 48, 97))

prec_palette = ((84, 48, 5), (140, 81, 10), (191, 129, 45), (223, 194, 125), (246, 232, 195), (245, 245, 245), (199, 234, 229), (128, 205, 193), (53, 151, 143), (1, 102, 94), (0, 60, 48))

temp_palette_ex = []
prec_palette_ex = []

for col in temp_palette:
    print(col)
    colnam = '#'
    for pio in col:
        coso = hex(pio)
        if len(coso) == 4:
            colnam += coso[-2:]
        elif len(coso) == 3:
            colnam += ('0'+coso[-1])
    print(colnam)
    temp_palette_ex.append(colnam)

for col in prec_palette:
    colnam = '#'
    for pio in col:
        coso = hex(pio)
        if len(coso) == 4:
            colnam += coso[-2:]
        elif len(coso) == 3:
            colnam += ('0'+coso[-1])
    prec_palette_ex.append(colnam)

temp_palette_ex = [temp_palette_ex[i] for i in np.arange(len(temp_palette_ex))[::-1]]

# read data/models: tas, pr and mslp
cart_in = '/data-hobbes/fabiano/CMIP6/'
cart_out = '/home/fabiano/Research/lavori/IPCC_AR6/'
listafils = os.listdir(cart_in)
print(listafils)
model_names = ['MRI-ESM2-0', 'IPSL-CM6A-LR']

scens = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
hist = 'historical'
miptab = 'Amon'

seasons = ['JJA', 'DJF', 'MAM', 'SON']
futperiod = (2021, 2040)
refperiod = (1990, 2014)

ref_cube = iris.load('/data-hobbes/fabiano/OBS/ERA/ERAInterim/zg500/zg500_Aday_ERAInterim_2deg_1979-2014.nc')[0]

# Loads all variables for all models
models = dict()
for var in ['tas', 'pr', 'psl']:
    for mod in model_names:
        for sc in scens + [hist]:
            filnams = [fi for fi in listafils if (miptab in fi.split('_') and sc in fi.split('_') and mod in fi.split('_') and var in fi.split('_'))]
            if len(filnams) == 1:
                filok = filnams[0]
            else:
                raise ValueError('Files corresponding to {} {} {}: {}\n'.format(var, mod, sc, filnams))
            models[(var, mod, sc)] = iris.load(cart_in + filok)[0]

# Plot temperature/precipitation maps
convunits = dict()
convunits['tas'] = None
convunits['pr'] = 'kg m-2 day-1'

all_res = dict()
for scen in scens:
    for var in ['tas', 'pr']:
        # Calc model mean
        mod_anoms = dict()
        mod_mean_state = dict()
        mod_stddev = dict()
        varanom = dict()
        varmeanstate = dict()
        varstd = dict()
        stpl_mask = dict()

        for seas in seasons:
            mod_anoms[seas] = []
            mod_mean_state[seas] = []
            mod_stddev[seas] = []

        for mod in model_names:
            varfut, cordfut, _ = ctl.transform_iris_cube(models[(var, mod, scen)], regrid_to_reference = ref_cube, convert_units_to = convunits[var])
            varhist, cordhist, _ = ctl.transform_iris_cube(models[(var, mod, hist)], regrid_to_reference = ref_cube, convert_units_to = convunits[var])

            for seas in seasons:
                yr_fut, yr_fut_std = ctl.seasonal_climatology(varfut, cordfut['dates'], seas, dates_range = ctl.range_years(*futperiod))
                yr_hist, yr_hist_std = ctl.seasonal_climatology(varhist, cordhist['dates'], seas, dates_range = ctl.range_years(*refperiod))

                yr_anom = yr_fut - yr_hist

                mod_anoms[seas].append(yr_anom)
                mod_mean_state[seas].append(yr_hist)
                mod_stddev[seas].append(yr_hist_std)

        for seas in seasons:
            mod_anoms[seas] = np.stack(mod_anoms[seas])
            mod_mean_state[seas] = np.stack(mod_mean_state[seas])
            mod_stddev[seas] = np.stack(mod_stddev[seas])
            varanom[seas] = np.mean(mod_anoms[seas], axis = 0)
            #varstd[seas] = np.std(mod_anoms[seas], axis = 0)
            varstd[seas] = np.mean(mod_stddev[seas], axis = 0)
            varmeanstate[seas] = np.mean(mod_mean_state[seas], axis = 0)

        all_res[(var, scen)] = [mod_anoms, mod_mean_state, mod_stddev, varanom, varstd, varmeanstate]

pickle.dump(all_res, open(cart_out + 'all_res_map.p', 'w'))
all_res = pickle.load(open(cart_out + 'all_res_map.p'))

eracub, cords, _ = ctl.transform_iris_cube(ref_cube)
lat = cords['lat']
lon = cords['lon']

cblabels = dict()
cblabels['tas'] = 'Temperature anomaly (K)'
cblabels['pr'] = 'Precipitation anomaly (%)'

cbar_ranges = dict()
cbar_ranges['tas'] = (-3, 3)
cbar_ranges['pr'] = (-50, 50)

paletta = dict()
paletta['tas'] = temp_palette_ex
paletta['pr'] = prec_palette_ex

proj = ccrs.Robinson()
#cmappa = cm.get_cmap('RdBu_r')
cmappa = None

for scen in scens:
    var = 'tas'
    mod_anoms, mod_mean_state, mod_stddev, varanom, varstd, varmeanstate = all_res[(var, scen)]
    #mod_anoms, varmean, varstd = all_res[var]
    stpl_mask = dict()
    for ke in varanom.keys():
        stpl_mask[ke] = np.zeros(varanom[ke].shape)
        oksig = abs(varanom[ke]) > 2*varstd[ke]
        stpl_mask[ke][oksig] = 1

        oksig = abs(varanom[ke]) < varstd[ke]
        stpl_mask[ke][oksig] = -1

    fig = plt.figure(figsize = (16, 12))

    seas = 'JJA'
    ax = fig.add_subplot(221, projection = proj)
    data = varanom[seas]
    map_plot = ctl.plot_mapc_on_ax(ax, data, lat, lon, proj, cmappa, cbar_ranges[var], n_color_levels = len(paletta[var])-1, add_hatching = stpl_mask[seas], hatch_styles = ['///', '', '...'], hatch_levels = [-1.5, -0.5, 0.5, 1.5], colors = paletta[var])
    ax.set_title(r'$\Delta$ Temperature {}'.format(seas))

    ax = fig.add_subplot(222, projection = proj)
    data = varstd[seas]
    map_plot = ctl.plot_mapc_on_ax(ax, data, lat, lon, proj, cmappa, cbar_ranges[var], n_color_levels = len(paletta[var])-1, colors = paletta[var])
    ax.set_title(r'$\sigma$ Temperature {}'.format(seas))

    seas = 'DJF'
    ax = fig.add_subplot(223, projection = proj)
    data = varanom[seas]
    map_plot = ctl.plot_mapc_on_ax(ax, data, lat, lon, proj, cmappa, cbar_ranges[var], n_color_levels = len(paletta[var])-1, add_hatching = stpl_mask[seas], colors = paletta[var], hatch_styles = ['///', '', '...'], hatch_levels = [-1.5, -0.5, 0.5, 1.5])
    ax.set_title(r'$\Delta$ Temperature {}'.format(seas))

    ax = fig.add_subplot(224, projection = proj)
    data = varstd[seas]
    map_plot = ctl.plot_mapc_on_ax(ax, data, lat, lon, proj, cmappa, cbar_ranges[var], n_color_levels = len(paletta[var])-1, colors = paletta[var])
    ax.set_title(r'$\sigma$ Temperature {}'.format(seas))

    cax = plt.axes([0.1, 0.11, 0.8, 0.05]) #horizontal
    cb = plt.colorbar(map_plot, cax = cax, orientation='horizontal')
    cb.ax.tick_params(labelsize=18)
    cb.set_label(cblabels[var], fontsize=20)

    plt.subplots_adjust(left=0.02, bottom=0.2, right=0.98, top=0.88, wspace=0.05, hspace=0.2)
    plt.suptitle('Seasonal mean temperature change (2 models, {}, 2021-2040 vs 1990-2014)'.format(scen))

    fig.savefig(cart_out+'fig_map_{}_{}.pdf'.format(var, scen))


    var = 'pr'
    mod_anoms, mod_mean_state, mod_stddev, varanom, varstd, varmeanstate = all_res[(var, scen)]
    stpl_mask = dict()
    var_perc = dict()
    std_perc = dict()
    for ke in varanom.keys():
        allvp = np.array([modan/modme for modan, modme in zip(mod_anoms[ke], mod_mean_state[ke])])
        var_perc[ke] = 100*np.mean(allvp, axis = 0)
        allvst = np.array([mostd/modme for mostd, modme in zip(mod_stddev[ke], mod_mean_state[ke])])
        std_perc[ke] = 100*np.mean(allvst, axis = 0)
        stpl_mask[ke] = np.zeros(varanom[ke].shape)
        oksig = abs(var_perc[ke]) > 2*std_perc[ke]
        stpl_mask[ke][oksig] = 1

        oksig = abs(var_perc[ke]) < std_perc[ke]
        stpl_mask[ke][oksig] = -1

    fig = plt.figure(figsize = (16, 12))

    for i, seas in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
        ax = fig.add_subplot(2, 2, i+1, projection = proj)
        data = var_perc[seas]
        #data = 100*np.mean([modan/modme for modan, modme in zip(mod_anoms[seas], mod_mean_state[seas])], axis = 0)
        #data = 100*varanom[seas]/varmeanstate[seas]
        map_plot = ctl.plot_mapc_on_ax(ax, data, lat, lon, proj, cmappa, cbar_ranges[var], n_color_levels = len(paletta[var])-1, add_hatching = stpl_mask[seas], colors = paletta[var], hatch_styles = ['///', '', '...'], hatch_levels = [-1.5, -0.5, 0.5, 1.5])
        ax.set_title(r'$\Delta$ Precipitation {}'.format(seas))

    cax = plt.axes([0.1, 0.11, 0.8, 0.05]) #horizontal
    cb = plt.colorbar(map_plot, cax = cax, orientation='horizontal')
    cb.ax.tick_params(labelsize=18)
    cb.set_label(cblabels[var], fontsize=20)

    plt.subplots_adjust(left=0.02, bottom=0.2, right=0.98, top=0.88, wspace=0.05, hspace=0.2)
    plt.suptitle('Seasonal mean precipitation change (2 models, {}, 2021-2040 vs 1990-2014)'.format(scen))

    fig.savefig(cart_out+'fig_map_{}_{}.pdf'.format(var, scen))

# Calc indexes: NAM, SAM, PDO, AMV ...


# Calc teleconnections (other inputs needed.. sst? (tos))
