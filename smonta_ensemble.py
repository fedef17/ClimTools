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

#cart = '/home/fabiano/Research/lavori/MedscopeEnsClus/DATA/Medscope_seasonal_forecasts/'
cart = '/home/federico/work/MARS_data/MEDSCOPE_seasonal/'

# Open nc file
years = range(1993,2018)
nmon = 7

# par = 167
# parname = '2t'
par = 228
parname = 'tprate'

all_fields = dict()

for year in years:
    for seas in ['may','nov']:
        nomefile = 'seasonal_{}_{}_{}.nc'.format(year, seas, par)
        print(nomefile)

        data = nc.Dataset(cart+nomefile)
        field = data.variables[parname]

        gigi = field[:,:,:]
        month_preds = np.split(gigi, nmon, axis = 0)

        for num, pred in zip(range(len(month_preds)), month_preds):
            all_fields[(seas, year, num)] = pred

# Building climatology
# faccio unica matriciona concatenando sull'asse 0, poi medio

climat_mean = dict()
climat_std = dict()

years_clim = range(1993,2017)

for seas in ['may','nov']:
    seas = 'may'
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

pickle.dump([all_fields, climat_mean, climat_std], open(cart+'all_fields_climat_{}.p'.format(par),'wb'), protocol = 2)

print(climat_mean['may'].shape)
print(np.mean(climat_mean['may']))

