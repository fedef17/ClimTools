#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os

grid_file = 'grid_25.des'
if not os.path.exists(grid_file): raise ValueError('File {} not found!'.format(grid_file))
gridtag = '25'

regrid = True
merge = True
sel_levels = [50000]
skip_existing = False
interp_style = 'bil' # 'con'

cart_ins = dict()
cart_ins['HadGEM3-GC31-HM'] = '/data-hobbes/fabiano/PRIMAVERA/hist_1950/gws/nopw/j04/primavera4/stream1/PRIMAVERA/HighResMIP/MOHC/HadGEM3-GC31-HM/hist-1950/r1i1p1f1/Primday/zg23/gn/v20180730/'
cart_ins['HadGEM3-GC31-MM'] = '/data-hobbes/fabiano/PRIMAVERA/hist_1950/gws/nopw/j04/primavera4/stream1/PRIMAVERA/HighResMIP/MOHC/HadGEM3-GC31-MM/hist-1950/r1i1p1f1/Primday/zg23/gn/v20170928/'
cart_ins['CMCC-CM2-HR4'] = '/data-hobbes/fabiano/PRIMAVERA/hist_1950/group_workspaces/jasmin2/primavera4/stream1/CMIP6/HighResMIP/CMCC/CMCC-CM2-HR4/hist-1950/r1i1p1f1/day/zg/gn/v20190105/'
cart_ins['CMCC-CM2-VHR4'] = '/data-hobbes/fabiano/PRIMAVERA/hist_1950/group_workspaces/jasmin2/primavera4/stream1/CMIP6/HighResMIP/CMCC/CMCC-CM2-VHR4/hist-1950/r1i1p1f1/day/zg/gn/v20180705/'
cart_ins['CNRM-CM6-1'] = '/data-hobbes/fabiano/PRIMAVERA/hist_1950/group_workspaces/jasmin2/primavera4/stream1/CMIP6/HighResMIP/CNRM-CERFACS/CNRM-CM6-1/hist-1950/r2i1p1f2/day/zg/gr/v20181004/'
cart_ins['ECMWF-IFS-HR'] = '/data-hobbes/fabiano/PRIMAVERA/hist_1950/group_workspaces/jasmin2/primavera4/stream1/CMIP6/HighResMIP/ECMWF/ECMWF-IFS-HR/hist-1950/r1i1p1f1/day/zg/gr/v20170915/'
cart_ins['ECMWF-IFS-LR'] = '/data-hobbes/fabiano/PRIMAVERA/hist_1950/group_workspaces/jasmin2/primavera4/stream1/CMIP6/HighResMIP/ECMWF/ECMWF-IFS-LR/hist-1950/r1i1p1f1/day/zg/gr/v20180221/'
cart_ins['MPI-ESM1-2-HR'] = '/data-hobbes/fabiano/PRIMAVERA/hist_1950/group_workspaces/jasmin2/primavera4/stream1/CMIP6/HighResMIP/MPI-M/MPI-ESM1-2-HR/hist-1950/r1i1p1f1/day/zg/gn/v20180606/'
cart_ins['MPI-ESM1-2-XR'] = '/data-hobbes/fabiano/PRIMAVERA/hist_1950/group_workspaces/jasmin2/primavera5/stream1/CMIP6/HighResMIP/MPI-M/MPI-ESM1-2-XR/hist-1950/r1i1p1f1/day/zg/gn/v20180606/'
cart_ins['EC-Earth-3-HR'] = '/data-hobbes/fabiano/PRIMAVERA/hist_1950/group_workspaces/jasmin2/primavera2/upload/EC-Earth-Consortium/EC-Earth-3-HR/incoming/s2hh/day/zg/'
cart_ins['EC-Earth-3-LR'] = '/data-hobbes/fabiano/PRIMAVERA/hist_1950/gws/nopw/j04/primavera4/upload/EC-Earth-Consortium/EC-Earth-3-LR/incoming/v20180510_144558/CMIP/EC-Earth-Consortium/EC-Earth3-LR/historical/r1i1p1f1/day/zg/gr/v20180501/'

cart_out_general = '/data-hobbes/fabiano/PRIMAVERA/hist_1950/'

for model_name in cart_ins:
    cart_in = cart_ins[model_name]
    cart_out = cart_out_general + model_name + '/'
    if skip_existing:
        if not os.path.exists(cart_out):
            print('.... processing {}\n'.format(model_name))
            os.mkdir(cart_out)
        else:
            print('{} already processed\n'.format(model_name))
            continue

    file_list = os.listdir(cart_in)
    file_list = [fi for fi in file_list if fi[-3:] == '.nc']
    file_list.sort()

    provname = cart_out+'mmm.nc'
    if regrid:
        file_in = cart_in + file_list[0]
        if sel_levels is not None:
            print('Selecting levels..\n')
            command = 'cdo sellevel'+''.join([',{}'.format(lev) for lev in sel_levels])+' {} {}'.format(file_in, provname)
            print(command)
            os.system(command)
            file_in = provname
        print('Calculating weights for interpolation....\n')
        command = 'cdo gen{},{} {} {}remapweights.nc'.format(interp_style, grid_file, file_in, cart_out)
        print(command)
        os.system(command)
        for filenam in file_list:
            file_in = cart_in + filenam
            indpo = filenam.index('.nc')
            file_out = cart_out + filenam[:indpo] + '_remap{}.nc'.format(gridtag)
            if sel_levels is not None:
                print('Selecting levels..\n')
                command = 'cdo sellevel'+''.join([',{}'.format(lev) for lev in sel_levels])+' {} {}'.format(file_in, provname)
                print(command)
                os.system(command)
                file_in = provname
            command = 'cdo remap,{},{} {} {}'.format(grid_file, cart_out+'remapweights.nc', file_in, file_out)
            print(command)
            os.system(command)

        if merge:
            command = 'cdo mergetime {}*_remap{}.nc {}{}_allyears_remap{}.nc'.format(cart_out, gridtag, cart_out, model_name, gridtag)
            print(command)
            os.system(command)
    else:
        if merge:
            command = 'cdo mergetime {}*_remap{}.nc {}{}_allyears_remap{}.nc'.format(cart_out, gridtag, cart_out, model_name, gridtag)
            print(command)
            os.system(command)
