#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
from matplotlib import pyplot as plt
import netCDF4 as nc
import cartopy.crs as ccrs
from numpy import linalg as LA

# Calculate normalization of 2D fields

# Open nc file

cart = '/home/fedefab/Scrivania/Research/Post-doc/Prima_WRTool/OUT_WRTOOL/OUTPUT/OUTnc/'
filename = 'EOFscal2_zg500_day_ERAInterim_obs_144x73_1ens_DJF_EAT_1979-2008_4pcs.nc'

data = nc.Dataset(cart+filename)

#print(data.variables)
field = data.variables['EOFscal2']
print(field.shape)

print('\n Normalizzazione delle EOFscal2\n')

for i in range(4):
    norma = LA.norm(field[i,:,:])
    print(norma)

filename = 'EOFunscal_zg500_day_ERAInterim_obs_144x73_1ens_DJF_EAT_1979-2008_4pcs.nc'

data = nc.Dataset(cart+filename)

#print(data.variables)
eofs_u = data.variables['EOFunscal']
print(eofs_u.shape)

print('\n Normalizzazione delle EOFunscal \n')

for i in range(4):
    norma = LA.norm(eofs_u[i,:,:])
    print(norma)

filename = 'EOFunscal_zg500_day_ECEARTH31_obs_144x73_1ens_DJF_EAT_1979-2008_4pcs.nc'

data = nc.Dataset(cart+filename)

#print(data.variables)
eofs_u_ec = data.variables['EOFunscal']
print(eofs_u_ec.shape)

print('\n Proiezione delle EOFunscal di ECEARTH su quelle di ERA\n')

riga = 'EOF {}: '+ 4*' {:6.2f} '

for i in range(4):
    coef1 = []
    for j in range(4):
        coeff = np.vdot(eofs_u_ec[i,:,:],eofs_u[j,:,:])
        coef1.append(coeff)

    print(riga.format(i+1,*coef1))

riga = 'WR {}: '+ 4*' {:6.2f} '

file1 = 'cluspatternORDasREF_4clus_zg500_day_ECEARTH31_obs_144x73_1ens_DJF_EAT_1979-2008_4pcs.nc'
file2 = 'cluspatternORDasREF_4clus_zg500_day_ERAInterim_obs_144x73_1ens_DJF_EAT_1979-2008_4pcs.nc'

data1 = nc.Dataset(cart+file1)
data2 = nc.Dataset(cart+file2)

#print(data.variables)
field1 = data1.variables['cluspattern']
field2 = data2.variables['cluspattern']

rigaera = 'WR {} era-int: '+ 4*' {:6.2f} '
rigaec = 'WR {} ec-earth: '+ 4*' {:6.2f} '

print('\n Proiezione dei WR di ECEARTH e di ERA sulle EOFunscal di ERA\n')

for i in range(4):
    coef1 = []
    coef2 = []
    for j in range(4):
        coeff = np.vdot(field1[i,:,:],eofs_u[j,:,:])
        coef1.append(coeff)
        coeff = np.vdot(field2[i,:,:],eofs_u[j,:,:])
        coef2.append(coeff)

    print(rigaec.format(i+1,*coef1))
    print(rigaera.format(i+1,*coef2))

print('\n Distanza (norm della diff) dei WR di ECEARTH da quelli di ERA\n')

for i in range(4):
    coef1 = []
    for j in range(4):
        coeff = LA.norm(field1[i,:,:]-field2[j,:,:])
        coef1.append(coeff)
    print(riga.format(i+1,*coef1))
