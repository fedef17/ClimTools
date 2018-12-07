#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
from matplotlib import pyplot as plt
import netCDF4 as nc
#import cartopy.crs as ccrs
from numpy import linalg as LA
import pickle
#from eofs.standard import Eof

sys.path.insert(0,'/home/fabiano/Research/git/WRtool/CLUS_tool/WRtool/')
from readsavencfield import read4Dncfield, save3Dncfield


cart = '/data/fabiano/Medscope/seasonal_forecasts_1d5/'
#cart = '/home/federico/work/MARS_data/MEDSCOPE_seasonal/'

# Open nc file
years = range(1993,2018)
nmon = 7

par = 167

cart2 = cart + 'input_par{}_1ens/'.format(par)

if not os.path.exists(cart2):
    os.mkdir(cart2)
##########################################################
convert = False

if par == 167:
    parname = '2t'
elif par == 228:
    parname = 'tprate'
    convert = True
    convert_factor = 8.64e7
    new_units = 'mm/day'

all_fields = dict()
nc_fields = dict()

dates_clim = dict()

for year in years:
    for seas in ['may','nov']:
        nomefile = 'seasonal_{}_{}_{}.nc'.format(year, seas, par)
        print(nomefile)

        data = nc.Dataset(cart+nomefile)
        field = data.variables[parname]

        gigi = field[:,:,:]
        month_preds = np.array(np.split(gigi, nmon, axis = 0))
        num_ens = month_preds[0].shape[0]

        lat = data.variables['lat'][:]
        lon = data.variables['lon'][:]
        times = np.array(np.split(data.variables['time'][:], nmon, axis = 0))
        time_units = data.variables['time'].units
        time_cal = data.variables['time'].calendar
        var_units = data.variables[parname].units

        if convert:
            month_preds = month_preds * convert_factor
            var_units = new_units

        for num, pred in zip(range(len(month_preds)), month_preds):
            all_fields[(seas, year, num)] = pred



        # Produce 1 nc file for each ensemble member, with all months
        for ens in range(num_ens):
            time = times[:, ens]
            var = month_preds[:, ens, :, :]
            dates = nc.num2date(time,time_units,time_cal)

            filename = 'spred_{}_{}_ens{:02d}.nc'.format(year, seas, ens)
            print(filename)
            filename = cart2 + filename

            save3Dncfield(lat,lon,var,parname,var_units,dates,time_units,time_cal,filename)

        if year == 2000:
            dates_clim[seas] = dates

# Building climatology
# faccio unica matriciona concatenando sull'asse 0, poi medio

climat_mean = dict()
climat_std = dict()

years_clim = range(1993,2017)

for seas in ['may','nov']:
    climat_mean[seas] = []
    climat_std[seas] = []

    for mon in range(nmon):
        matric = all_fields[(seas, years[0], mon)]

        for year in years_clim[1:]:
            print(seas, year, mon)
            matric = np.concatenate([matric, all_fields[(seas, year, mon)]], axis = 0)

        print(matric.shape)
        climat_mean[seas].append(np.mean(matric, axis = 0))
        climat_std[seas].append(np.std(matric, axis = 0))

    climat_mean[seas] = np.array(climat_mean[seas])
    climat_std[seas] = np.array(climat_std[seas])

#pickle.dump([all_fields, climat_mean, climat_std], open(cart+'all_fields_climat_{}.p'.format(par),'wb'), protocol = 2)

for seas in ['may','nov']:
    filename = 'climatology_mean_{}_1993-2016.nc'.format(seas)
    print(filename)
    filename = cart2 + filename
    save3Dncfield(lat,lon,climat_mean[seas],parname,var_units,dates_clim[seas],time_units,time_cal,filename)

    filename = 'climatology_std_{}_1993-2016.nc'.format(seas)
    print(filename)
    filename = cart2 + filename
    save3Dncfield(lat,lon,climat_std[seas],parname,var_units,dates_clim[seas],time_units,time_cal,filename)

print(climat_mean['may'].shape)
print(np.mean(climat_mean['may']))
