#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import netCDF4 as nc
import sys

file_in = sys.argv[1] # Name of input file (relative path)
file_out = sys.argv[2] # Name of input file (relative path)

src = nc.Dataset(file_in, 'r')
dst = nc.Dataset(file_out, 'w')

variab_names = src.variables.keys()
lev_names = ['level', 'lev', 'pressure', 'plev', 'plev8']
for levna in lev_names:
    if levna in variab_names:
        oklevname = levna
        levels = src.variables[levna][:]

field = src.variables[variab_names[-1]][:]
#print(field.shape)

intfi = np.trapz(field, x=levels, axis=1).squeeze()
print(intfi.shape)

for name, dimension in src.dimensions.iteritems():
    if name == oklevname:
        continue
    dst.createDimension(name, len(dimension) if not dimension.isunlimited() else None)

for name, variable in src.variables.iteritems():
    print(name, variable.datatype, variable.dimensions)
    if name != variab_names[-1] and name != oklevname:
        x = dst.createVariable(name, variable.datatype, variable.dimensions)
        dst.variables[name][:] = src.variables[name][:]
        try:
            dst.variables[name].units = src.variables[name].units
        except:
            pass

        if name == 'time':
            dst.variables[name].calendar = src.variables[name].calendar
    elif name == variab_names[-1]:
        dims = [di for di in variable.dimensions if di != oklevname]
        x = dst.createVariable(name, variable.datatype, dims)
        dst.variables[name][:] = intfi.squeeze()

src.close()
dst.close()
