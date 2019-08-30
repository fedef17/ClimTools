#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os

import ctool, ctp

import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
import matplotlib.animation as animation
from matplotlib.animation import ImageMagickFileWriter
import seaborn as sns

from shapely.geometry.polygon import LinearRing

import netCDF4 as nc
import cartopy.crs as ccrs
import cartopy.util as cutil
import pandas as pd

from numpy import linalg as LA
from eofs.standard import Eof
from scipy import stats, optimize, signal
import itertools as itt
import math

from sklearn.cluster import KMeans



from datetime import datetime
import pickle
from copy import deepcopy as dcopy

import iris
from cf_units import Unit

mpl.rcParams['hatch.linewidth'] = 0.1
#mpl.rcParams['hatch.color'] = 'black'

#######################################################
#
###     INPUTs reading
#
#######################################################

def isclose(a, b, rtol = 1.e-9, atol = 0.0):
    return np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=False)

def printsep(ofile = None):
    if ofile is None:
        print('\n--------------------------------------------------------\n')
    else:
        ofile.write('\n--------------------------------------------------------\n')
    return

def newline(ofile = None):
    if ofile is None:
        print('\n\n')
    else:
        ofile.write('\n\n')
    return

def datestamp():
    tempo = datetime.now().isoformat()
    tempo = tempo.split('.')[0]
    return tempo

def str_to_bool(s):
    if s == 'True' or s == 'T':
         return True
    elif s == 'False' or s == 'F':
         return False
    else:
         raise ValueError('Not a boolean value')

def openlog(cart_out, tag = None, redirect_stderr = True):
    if tag is None:
        tag = datetime.strftime(datetime.now(), format = '%d%m%y_h%H%M')
    # open our log file
    logname = 'log_{}.log'.format(tag)
    logfile = open(cart_out+logname,'w') #self.name, 'w', 0)

    # re-open stdout without buffering
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

    # redirect stdout and stderr to the log file opened above
    os.dup2(logfile.fileno(), sys.stdout.fileno())
    if redirect_stderr:
        os.dup2(logfile.fileno(), sys.stderr.fileno())

    return

def read_inputs(nomefile, key_strings, n_lines = None, itype = None, defaults = None, verbose=False):
    """
    Standard reading for input files. Searches for the keys in the input file and assigns the value to variable. Returns a dictionary with all keys and variables assigned.

    :param key_strings: List of strings to be searched in the input file.
    :param itype: List of types for the variables to be read. Must be same length as key_strings.
    :param defaults: Dict with default values for the variables.
    :param n_lines: Dict. Number of lines to be read after key.
    """

    keys = ['['+key+']' for key in key_strings]

    if n_lines is None:
        n_lines = np.ones(len(keys))
        n_lines = dict(zip(key_strings,n_lines))
    elif len(n_lines.keys()) < len(key_strings):
        for key in key_strings:
            if not key in n_lines.keys():
                n_lines[key] = 1

    if itype is None:
        itype = len(keys)*[None]
        itype = dict(zip(key_strings,itype))
    elif len(itype.keys()) < len(key_strings):
        for key in key_strings:
            if not key in itype.keys():
                warnings.warn('No type set for {}. Setting string as default..'.format(key))
                itype[key] = str
        warnings.warn('Not all input types (str, int, float, ...) have been specified')

    if defaults is None:
        warnings.warn('No defaults are set. Setting None as default value.')
        defaults = len(key_strings)*[None]
        defaults = dict(zip(key_strings,defaults))
    elif len(defaults.keys()) < len(key_strings):
        for key in key_strings:
            if not key in defaults.keys():
                defaults[key] = None

    variables = []
    is_defaults = []
    with open(nomefile, 'r') as infile:
        lines = infile.readlines()
        # Skips commented lines:
        lines = [line for line in lines if not line.lstrip()[:1] == '#']

        for key, keystr in zip(keys, key_strings):
            deflt = defaults[keystr]
            nli = n_lines[keystr]
            typ = itype[keystr]
            is_key = np.array([key in line for line in lines])
            if np.sum(is_key) == 0:
                print('Key {} not found, setting default value {}\n'.format(key,deflt))
                variables.append(deflt)
                is_defaults.append(True)
            elif np.sum(is_key) > 1:
                raise KeyError('Key {} appears {} times, should appear only once.'.format(key,np.sum(is_key)))
            else:
                num_0 = np.argwhere(is_key)[0][0]
                try:
                    if typ == list:
                        cose = lines[num_0+1].rstrip().split(',')
                        coseok = [cos.strip() for cos in cose]
                        variables.append(coseok)
                    elif typ == dict:
                        # reads subsequent lines. Stops when finds empty line.
                        iko = 1
                        line = lines[num_0+iko]
                        nuvar = dict()
                        while len(line.strip()) > 0:
                            dictkey = line.split(':')[0].strip()
                            allvals = [cos.strip() for cos in line.split(':')[1].split(',')]
                            # if len(allvals) == 1:
                            #     allvals = allvals[0]
                            nuvar[dictkey] = allvals
                            iko += 1
                            line = lines[num_0+iko]
                        variables.append(nuvar)
                    elif nli == 1:
                        cose = lines[num_0+1].rstrip().split()
                        #print(key, cose)
                        if typ == bool: cose = [str_to_bool(lines[num_0+1].rstrip().split()[0])]
                        if typ == str: cose = [lines[num_0+1].rstrip()]
                        if len(cose) == 1:
                            if cose[0] is None:
                                variables.append(None)
                            else:
                                variables.append([typ(co) for co in cose][0])
                        else:
                            variables.append([typ(co) for co in cose])
                    else:
                        cose = []
                        for li in range(nli):
                            cos = lines[num_0+1+li].rstrip().split()
                            if typ == str: cos = [lines[num_0+1+li].rstrip()]
                            if len(cos) == 1:
                                if cos[0] is None:
                                    cose.append(None)
                                else:
                                    cose.append([typ(co) for co in cos][0])
                            else:
                                cose.append([typ(co) for co in cos])
                        variables.append(cose)
                    is_defaults.append(False)
                except Exception as problemha:
                    print('Unable to read value of key {}'.format(key))
                    raise problemha

    if verbose:
        for key, var, deflt in zip(keys,variables,is_defaults):
            print('----------------------------------------------\n')
            if deflt:
                print('Key: {} ---> Default Value: {}\n'.format(key,var))
            else:
                print('Key: {} ---> Value Read: {}\n'.format(key,var))

    return dict(zip(key_strings,variables))


def get_size(obj, seen=None):
    """
    Recursively finds size of objects.

    After https://gist.github.com/bosswissam/
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])

    return size

#######################################################
#
###     Data reading/writing and pre-treatment
#
#######################################################


def check_increasing_latlon(var, lat, lon):
    """
    Checks that the latitude and longitude are in increasing order. Returns ordered arrays.

    Assumes that lat and lon are the second-last and last dimensions of the array var.
    """
    lat = np.array(lat)
    lon = np.array(lon)
    var = np.array(var)

    revlat = False
    revlon = False
    if lat[1] < lat[0]:
        revlat = True
        print('Latitude is in reverse order! Ordering..\n')
    if lon[1] < lon[0]:
        revlon = True
        print('Longitude is in reverse order! Ordering..\n')

    if revlat and not revlon:
        var = var[..., ::-1, :]
        lat = lat[::-1]
    elif revlon and not revlat:
        var = var[..., :, ::-1]
        lon = lon[::-1]
    elif revlat and revlon:
        var = var[..., ::-1, ::-1]
        lat = lat[::-1]
        lon = lon[::-1]

    return var, lat, lon


def regrid_cube(cube, ref_cube, regrid_scheme = 'linear'):
    """
    Regrids cube according to ref_cube grid. Default scheme is linear (cdo remapbil). Other scheme available: nearest and conservative (cdo remapcon).
    """
    if regrid_scheme == 'linear':
        schema = iris.analysis.Linear()
    elif regrid_scheme == 'conservative':
        schema = iris.analysis.AreaWeighted()
    elif regrid_scheme == 'nearest':
        schema = iris.analysis.Nearest()

    (nlat, nlon) = (len(cube.coord('latitude').points), len(cube.coord('longitude').points))
    (nlat_ref, nlon_ref) = (len(ref_cube.coord('latitude').points), len(ref_cube.coord('longitude').points))

    if nlat*nlon < nlat_ref*nlon_ref:
        raise ValueError('cube size {}x{} is smaller than the reference cube {}x{}!\n'.format(nlat, nlon, nlat_ref, nlon_ref))

    if nlat == nlat_ref and nlon == nlon_ref:
        lat_check = cube.coord('latitude').points == ref_cube.coord('latitude').points
        lon_check = cube.coord('longitude').points == ref_cube.coord('longitude').points

        if np.all(lat_check) and np.all(lon_check):
            print('Grid check OK\n')
        else:
            print('Same nlat, nlon, but different grid. Regridding to reference..\n')
            nucube = cube.regrid(ref_cube, schema)
    else:
        print('Different nlat, nlon. Regridding to reference..\n')
        nucube = cube.regrid(ref_cube, schema)

    return nucube


def transform_iris_cube(cube, regrid_to_reference = None, convert_units_to = None, extract_level_hPa = None, regrid_scheme = 'linear', adjust_nonstd_dates = True):
    """
    Transforms an iris cube in a variable and a set of coordinates.
    Optionally selects a level (given in hPa).
    TODO: cube regridding to a given lat/lon grid.

    < extract_level_hPa > : float. If set, only the corresponding level is extracted. Level units are converted to hPa before the selection.
    < force_level_units > : str. Sometimes level units are not set inside the netcdf file. Set units of levels to avoid errors in reading. To be used with caution, always check the level output to ensure that the units are correct.
    """
    print('INIZIO')
    ndim = cube.ndim
    datacoords = dict()
    aux_info = dict()
    ax_coord = dict()

    print(datetime.now())

    if regrid_to_reference is not None:
        cube = regrid_cube(cube, regrid_to_reference, regrid_scheme = regrid_scheme)

    print(datetime.now())

    if convert_units_to:
        if cube.units.name != convert_units_to:
            print('Converting data from {} to {}\n'.format(cube.units.name, convert_units_to))
            if cube.units.name == 'm**2 s**-2' and convert_units_to == 'm':
                cu = cu/9.80665
                cube.units = 'm'
            else:
                cube.convert_units(convert_units_to)

    print(datetime.now())
    data = cube.data
    print(datetime.now())
    aux_info['var_units'] = cube.units.name

    coord_names = [cord.name() for cord in cube.coords()]

    allco = ['lat', 'lon', 'level']
    allconames = dict()
    allconames['lat'] = np.array(['latitude', 'lat'])
    allconames['lon'] = np.array(['longitude', 'lon'])
    allconames['level'] = np.array(['level', 'lev', 'pressure', 'plev', 'plev8', 'air_pressure'])

    print(datetime.now())

    for i, nam in enumerate(coord_names):
        found = False
        if nam == 'time': continue
        for std_nam in allconames:
            if nam in allconames[std_nam]:
                coor = cube.coord(nam)
                if std_nam == 'level':
                    coor.convert_units('hPa')
                datacoords[std_nam] = coor.points
                ax_coord[std_nam] = i
                found = True
        if not found:
            print('# WARNING: coordinate {} in cube not recognized.\n'.format(nam))

    print(datetime.now())
    if 'level' in datacoords.keys() and extract_level_hPa is not None:
        okind = datacoords['level'] == extract_level_hPa
        if np.any(okind):
            datacoords['level'] = datacoords['level'][okind]
            data = data.take(first(okind), axis = ax_coord['level'])
        elif len(datacoords['level']) == 1:
            data = data.squeeze()
        else:
            raise ValueError('Level {} hPa not found among: '.format(extract_level_hPa)+(len(datacoords['level'])*'{}, ').format(*datacoords['level']))

    print(datetime.now())
    if 'time' in coord_names:
        time = cube.coord('time').points
        time_units = cube.coord('time').units
        dates = time_units.num2date(time) # this is a set of cftime._cftime.real_datetime objects
        time_cal = time_units.calendar

        if adjust_nonstd_dates:
            if dates[0].year < 1677 or dates[-1].year > 2256:
                print('WARNING!!! Dates outside pandas range: 1677-2256\n')
                dates = adjust_outofbound_dates(dates)

            if time_cal == '365_day' or time_cal == 'noleap':
                dates = adjust_noleap_dates(dates)
            elif time_cal == '360_day':
                dates = adjust_360day_dates(dates)

        datacoords['dates'] = dates
        aux_info['time_units'] = time_units.name
        aux_info['time_calendar'] = time_cal

    print(datetime.now())
    data, lat, lon = check_increasing_latlon(data, datacoords['lat'], datacoords['lon'])
    datacoords['lat'] = lat
    datacoords['lon'] = lon
    print('FINE')

    return data, datacoords, aux_info


def read_iris_nc(ifile, extract_level_hPa = None, select_var = None, regrid_to_reference = None, regrid_scheme = 'linear', convert_units_to = None, adjust_nonstd_dates = True, verbose = True, keep_only_maxdim_vars = True):
    """
    Read a netCDF file using the iris library.

    < extract_level_hPa > : float. If set, only the corresponding level is extracted. Level units are converted to hPa before the selection.
    < select_var > : str or list. For a multi variable file, only variable names corresponding to those listed in select_var are read. Redundant definition are treated safely: variable is extracted only one time.

    < keep_only_maxdim_vars > : keeps only variables with maximum size (excludes variables like time_bnds, lat_bnds, ..)
    """

    print('Reading {}\n'.format(ifile))
    is_ensemble = False
    if type(ifile) in [list, np.ndarray]:
        is_ensemble = True
        print('WARNING!!!! Reading an ENSEMBLE of input files instead than a single file! Is this desired?\n')

    fh = iris.load(ifile)

    if len(fh) == 0:
        raise ValueError('ERROR! Empty file: '+ifile)

    cudimax = np.argmax([cu.ndim for cu in fh])
    ndim = np.max([cu.ndim for cu in fh])

    dimensions = [cord.name() for cord in fh[cudimax].coords()]

    if verbose: print('Dimensions: {}\n'.format(dimensions))

    if keep_only_maxdim_vars:
        fh = [cu for cu in fh if cu.ndim == ndim]

    variab_names = [cu.name() for cu in fh]
    if verbose: print('Variables: {}\n'.format(variab_names))
    nvars = len(variab_names)
    print('Field as {} dimensions and {} vars. All vars: {}'.format(ndim, nvars, variab_names))

    all_vars = dict()
    if not is_ensemble:
        for cu in fh:
            all_vars[cu.name()] = transform_iris_cube(cu, regrid_to_reference = regrid_to_reference, convert_units_to = convert_units_to, extract_level_hPa = extract_level_hPa, regrid_scheme = regrid_scheme, adjust_nonstd_dates = adjust_nonstd_dates)
        if select_var is not None:
            print('Read variable: {}\n'.format(select_var))
            return all_vars[select_var]
    else:
        ens_id = 0
        if select_var is not None:
            print('Read variable: {}\n'.format(select_var))
        for cu in fh:
            if select_var is not None:
                if cu.name() != select_var: continue
            all_vars[cu.name()+'_{}'.format(ens_id)] = transform_iris_cube(cu, regrid_to_reference = regrid_to_reference, convert_units_to = convert_units_to, extract_level_hPa = extract_level_hPa, regrid_scheme = regrid_scheme, adjust_nonstd_dates = adjust_nonstd_dates)
            ens_id += 1

    if len(all_vars.keys()) == 1:
        print('Read variable: {}\n'.format(list(all_vars.keys())[0]))
        return list(all_vars.values())[0]
    else:
        print('Read all variables: {}\n'.format(all_vars.keys()))
        return all_vars


def readxDncfield(ifile, extract_level = None, select_var = None, pressure_in_Pa = True, force_level_units = None, verbose = True, keep_only_Ndim_vars = True):
    """
    Read a netCDF file as it is, preserving all dimensions and multiple variables.

    < extract_level > : float. If set, only the corresponding level is extracted.
    < select_var > : str or list. For a multi variable file, only variable names corresponding to those listed in select_var are read. Redundant definition are treated safely: variable is extracted only one time.

    < pressure_in_Pa > : bool. If True (default) pressure levels are converted to Pa.
    < force_level_units > : str. Set units of levels to avoid errors in reading. To be used with caution, always check the level output to ensure that the units are correct.
    < keep_only_Ndim_vars > : keeps only variables with correct size (excludes variables like time_bnds, lat_bnds, ..)
    """

    print('Reading {}\n'.format(ifile))

    fh = nc.Dataset(ifile)
    dimensions = list(fh.dimensions.keys())
    if verbose: print('Dimensions: {}\n'.format(dimensions))
    ndim = len(dimensions)

    all_variabs = list(fh.variables.keys())

    variab_names = list(fh.variables.keys())
    for nam in dimensions:
        if nam in variab_names: variab_names.remove(nam)
    if verbose: print('Variables: {}\n'.format(variab_names))
    nvars = len(variab_names)
    print('Field as {} dimensions and {} vars. All keys: {}'.format(ndim, nvars, fh.variables.keys()))

    try:
        lat_o         = fh.variables['lat'][:]
        lon_o         = fh.variables['lon'][:]
    except KeyError as ke:
        #print(repr(ke))
        lat_o         = fh.variables['latitude'][:]
        lon_o         = fh.variables['longitude'][:]
    true_dim = 2

    vars = dict()
    if select_var is None:
        for varna in variab_names:
            var = fh.variables[varna][:]
            var, lat, lon = check_increasing_latlon(var, lat_o, lon_o)
            vars[varna] = var
    else:
        print('Extracting {}\n'.format(select_var))
        for varna in variab_names:
            if varna in select_var:
                var = fh.variables[varna][:]
                var, lat, lon = check_increasing_latlon(var, lat_o, lon_o)
                vars[varna] = var
        if len(vars.keys()) == 0:
            raise KeyError('No variable corresponds to names: {}. All variabs: {}'.format(select_var, variab_names))


    if 'time' in all_variabs:
        true_dim += 1
        time        = fh.variables['time'][:]
        time_units  = fh.variables['time'].units
        time_cal    = fh.variables['time'].calendar

        time = list(time)
        dates = nc.num2date(time,time_units,time_cal)

        if dates[0].year < 1677 or dates[-1].year > 2256:
            print('WARNING!!! Dates outside pandas range: 1677-2256\n')
            dates = adjust_outofbound_dates(dates)

        if time_cal == '365_day' or time_cal == 'noleap':
            dates = adjust_noleap_dates(dates)
        elif time_cal == '360_day':
            dates = adjust_360day_dates(dates)

        print('calendar: {0}, time units: {1}'.format(time_cal,time_units))

    if true_dim == 3 and ndim > 3:
        lev_names = ['level', 'lev', 'pressure', 'plev', 'plev8']
        found = False
        for levna in lev_names:
            if levna in all_variabs:
                oklevname = levna
                level = fh.variables[levna][:]
                try:
                    nlevs = len(level)
                    found = True
                except:
                    found = False
                break

        if not found:
            print('Level name not found among: {}\n'.format(lev_names))
            print('Does the variable have levels?')
        else:
            true_dim += 1
            try:
                level_units = fh.variables[oklevname].units
            except AttributeError as atara:
                print('level units not found in file {}\n'.format(ifile))
                if force_level_units is not None:
                    level_units = force_level_units
                    print('setting level units to {}\n'.format(force_level_units))
                    print('levels are {}\n'.format(level))
                else:
                    raise atara

            print('level units are {}\n'.format(level_units))
            if pressure_in_Pa:
                if level_units in ['millibar', 'millibars','hPa']:
                    level = 100.*level
                    level_units = 'Pa'
                    print('Converting level units from hPa to Pa\n')

    print('Dimension of variables is {}\n'.format(true_dim))
    if keep_only_Ndim_vars:
        for varna in vars:
            if len(vars[varna].shape) < true_dim:
                print('Erasing variable {}\n'.format(varna))
                vars.pop(varna)

    if true_dim == 4:
        if extract_level is not None:
            lvel = extract_level
            if nlevs > 1:
                if level_units=='millibar' or level_units=='hPa':
                    l_sel=int(np.where(level==lvel)[0])
                    print('Selecting level {0} millibar'.format(lvel))
                elif level_units=='Pa':
                    l_sel=int(np.where(level==lvel*100)[0])
                    print('Selecting level {0} Pa'.format(lvel*100))
                level = lvel
            else:
                level = level[0]
                l_sel = 0

            for varna in vars:
                vars[varna] = vars[varna][:,l_sel, ...].squeeze()
            true_dim = true_dim - 1
        else:
            levord = level.argsort()
            level = level[levord]
            for varna in vars:
                vars[varna] = vars[varna][:, levord, ...]

    var_units = dict()
    for varna in vars:
        try:
            var_units[varna] = fh.variables[varna].units
        except:
            var_units[varna] = None

        if var_units[varna] == 'm**2 s**-2':
            print('From geopotential (m**2 s**-2) to geopotential height (m)')   # g0=9.80665 m/s2
            vars[varna] = vars[varna]/9.80665
            var_units[varna] = 'm'

    print('Returned variables are: {}'.format(vars.keys()))
    if len(vars.keys()) == 1:
        vars = list(vars.values())[0]
        var_units = list(var_units.values())[0]

    if true_dim == 2:
        return vars, lat, lon, var_units
    elif true_dim == 3:
        return vars, lat, lon, dates, time_units, var_units, time_cal
    elif true_dim == 4:
        return vars, level, lat, lon, dates, time_units, var_units, time_cal


def read4Dncfield(ifile, extract_level = None, compress_dummy_dim = True, increasing_plev = True):
    '''
    GOAL
        Read netCDF file of 4Dfield, optionally selecting a level.
    USAGE
        var, dates = read4Dncfield(ifile, extract_level = level)
        ifile: filename
        extract_level: level to be selected in hPa
    '''
    #----------------------------------------------------------------------------------------
    print('__________________________________________________________')
    print('Reading the 4D field [time x level x lat x lon]: \n{0}'.format(ifile))
    #----------------------------------------------------------------------------------------
    fh = nc.Dataset(ifile, mode='r')
    variabs=[]
    for variab in fh.variables:
        variabs.append(variab)
    #print('The variables in the nc file are: ', variabs)
    lev_names = ['level', 'lev', 'pressure', 'plev', 'plev8']
    for levna in lev_names:
        if levna in variabs:
            oklevname = levna
            level = fh.variables[levna][:]
            nlevs = len(level)
            break

    try:
        lat         = fh.variables['lat'][:]
        lon         = fh.variables['lon'][:]
    except KeyError as ke:
        print(repr(ke))
        lat         = fh.variables['latitude'][:]
        lon         = fh.variables['longitude'][:]

    time        = fh.variables['time'][:]
    time_units  = fh.variables['time'].units
    time_cal    = fh.variables['time'].calendar

    try:
        var_units   = fh.variables[variabs[-1]].units
    except:
        var_units = None

    if extract_level is not None:
        lvel = extract_level
        if nlevs > 1:
            level_units = fh.variables[levna].units
            if level_units=='millibar' or level_units=='hPa':
                l_sel=int(np.where(level==lvel)[0])
                print('Selecting level {0} millibar'.format(lvel))
            elif level_units=='Pa':
                l_sel=int(np.where(level==lvel*100)[0])
                print('Selecting level {0} Pa'.format(lvel*100))
            level = lvel
        else:
            level = level[0]
            l_sel = 0

        var         = fh.variables[variabs[-1]][:,l_sel,:,:]
        txt='{0}{1} dimension for a single ensemble member [time x lat x lon]: {2}'.format(variabs[-1],lvel,var.shape)
    else:
        var         = fh.variables[variabs[-1]][:,:,:,:]
        txt='{0} dimension for a single ensemble member [time x lat x lon]: {1}'.format(variabs[-1],var.shape)
        if increasing_plev:
            levord = level.argsort()
            level = level[levord]
            var = var[:, levord, ...]

    print(txt)
    #print(fh.variables)
    if var_units == 'm**2 s**-2':
        print('From geopotential (m**2 s**-2) to geopotential height (m)')   # g0=9.80665 m/s2
        var=var/9.80665
        var_units='m'
    print('calendar: {0}, time units: {1}'.format(time_cal,time_units))

    time = list(time)
    dates = nc.num2date(time,time_units,time_cal)
    fh.close()

    if time_cal == '365_day' or time_cal == 'noleap':
        dates = adjust_noleap_dates(dates)
    elif time_cal == '360_day':
        dates = adjust_360day_dates(dates)

    if compress_dummy_dim:
        var = var.squeeze()

    var, lat, lon = check_increasing_latlon(var, lat, lon)

    return var, level, lat, lon, dates, time_units, var_units, time_cal


def adjust_noleap_dates(dates):
    """
    When the time_calendar is 365_day or noleap, nc.num2date() returns a cftime array which is not convertible to datetime (and to pandas DatetimeIndex). This fixes this problem, returning the usual datetime array.
    """
    dates_ok = []
    #for ci in dates: dates_ok.append(datetime.strptime(ci.strftime(), '%Y-%m-%d %H:%M:%S'))
    # diffs = []
    for ci in dates:
        # coso = ci.isoformat()
        coso = ci.strftime()
        nudat = pd.Timestamp(coso).to_pydatetime()
        # print(coso, nudat)
        # if ci-nudat >= pd.Timedelta('1 days'):
        #     raise ValueError
        # diffs.append(ci-nudat)
        dates_ok.append(nudat)

    dates_ok = np.array(dates_ok)
    # print(diffs)

    return dates_ok


def adjust_outofbound_dates(dates):
    """
    Pandas datetime index is limited to 1677-2256.
    This temporary fix allows to handle with pandas outside that range, simply adding 1700 years to the dates.
    Still this will give problems with longer integrations... planned migration from pandas datetime to Datetime.datetime.
    """
    dates_ok = []
    diff = 2000

    for ci in dates:
        coso = ci.isoformat()
        listasp = coso.split('-')
        listasp[0] = '{:04d}'.format(int(listasp[0])+diff)
        coso = '-'.join(listasp)

        nudat = pd.Timestamp(coso).to_pydatetime()
        dates_ok.append(nudat)

    dates_ok = np.array(dates_ok)

    return dates_ok


def adjust_360day_dates(dates):
    """
    When the time_calendar is 360_day (please not!), nc.num2date() returns a cftime array which is not convertible to datetime (obviously)(and to pandas DatetimeIndex). This fixes this problem in a completely arbitrary way, missing one day each two months. Returns the usual datetime array.
    """
    dates_ok = []
    #for ci in dates: dates_ok.append(datetime.strptime(ci.strftime(), '%Y-%m-%d %H:%M:%S'))
    strindata = '{:4d}-{:02d}-{:02d} 12:00:00'

    for ci in dates:
        firstday = strindata.format(ci.year, 1, 1)
        num = ci.dayofyr-1
        add_day = num//72 # salto un giorno ogni 72
        okday = pd.Timestamp(firstday)+pd.Timedelta('{} days'.format(num+add_day))
        dates_ok.append(okday.to_pydatetime())

    dates_ok = np.array(dates_ok)

    return dates_ok


def read3Dncfield(ifile, compress_dummy_dim = True):
    '''
    GOAL
        Read netCDF file of 3Dfield
    USAGE
        var, dates = read3Dncfield(fname)
        fname: filname
    '''
    #----------------------------------------------------------------------------------------
    print('__________________________________________________________')
    print('Reading the 3D field [time x lat x lon]: \n{0}'.format(ifile))
    #----------------------------------------------------------------------------------------
    fh = nc.Dataset(ifile, mode='r')
    variabs=[]
    for variab in fh.variables:
        variabs.append(variab)
    #print('The variables in the nc file are: ', variabs)

    try:
        lat         = fh.variables['lat'][:]
        lon         = fh.variables['lon'][:]
    except KeyError as ke:
        print(repr(ke))
        lat         = fh.variables['latitude'][:]
        lon         = fh.variables['longitude'][:]

    time        = fh.variables['time'][:]
    time_units  = fh.variables['time'].units
    time_cal    = fh.variables['time'].calendar

    try:
        var_units   = fh.variables[variabs[-1]].units
    except:
        var_units = None

    var         = fh.variables[variabs[-1]][:,:,:]
    txt='{0} dimension [time x lat x lon]: {1}'.format(variabs[-1],var.shape)

    if compress_dummy_dim and var.ndim > 3:
        var = var.squeeze()
    #print(fh.variables)
    time = list(time)
    dates = nc.num2date(time, time_units, time_cal)
    fh.close()

    if time_cal == '365_day' or time_cal == 'noleap':
        dates = adjust_noleap_dates(dates)

    var, lat, lon = check_increasing_latlon(var, lat, lon)

    print(txt)

    return var, lat, lon, dates, time_units, var_units


def read2Dncfield(ifile):
    '''
    GOAL
        Read netCDF file of 2Dfield
    USAGE
        var = read2Dncfield(fname)
        fname: filename
    '''
    #----------------------------------------------------------------------------------------
    print('__________________________________________________________')
    print('Reading the 2D field [lat x lon]: \n{0}'.format(ifile))
    #----------------------------------------------------------------------------------------
    fh = nc.Dataset(ifile, mode='r')
    variabs=[]
    for variab in fh.variables:
        variabs.append(variab)
    #print('The variables in the nc file are: ', variabs)

    try:
        lat         = fh.variables['lat'][:]
        lon         = fh.variables['lon'][:]
    except KeyError as ke:
        print(repr(ke))
        lat         = fh.variables['latitude'][:]
        lon         = fh.variables['longitude'][:]

    #var_units   = fh.variables[variabs[2]].units
    var         = fh.variables[variabs[2]][:,:]
    try:
        var_units   = fh.variables[variabs[-1]].units
    except:
        var_units = None

    txt='{0} dimension [lat x lon]: {1}'.format(variabs[2],var.shape)
    #print(fh.variables)
    fh.close()
    var, lat, lon = check_increasing_latlon(var, lat, lon)

    #print('\n'+txt)

    return var, var_units, lat, lon


def read_N_2Dfields(ifile):
    '''
    GOAL
        Read var in ofile netCDF file
    USAGE
        read a number N of 2D fields [latxlon]
        fname: output filname
    '''
    fh = nc.Dataset(ifile, mode='r')
    variabs=[]
    for variab in fh.variables:
        variabs.append(variab)
    #print('The variables in the nc file are: ', variabs)

    num         = fh.variables['num'][:]

    try:
        lat         = fh.variables['lat'][:]
        lon         = fh.variables['lon'][:]
    except KeyError as ke:
        print(repr(ke))
        lat         = fh.variables['latitude'][:]
        lon         = fh.variables['longitude'][:]

    var         = fh.variables[variabs[3]][:,:,:]
    var_units   = fh.variables[variabs[3]].units
    txt='{0} dimension [num x lat x lon]: {1}'.format(variabs[3],var.shape)
    #print(fh.variables)
    fh.close()
    var, lat, lon = check_increasing_latlon(var, lat, lon)
    #print('\n'+txt)

    return var, var_units, lat, lon


def create_iris_cube(data, varname, varunits, iris_coords_list, long_name = None):
    """
    Creates an iris.cube.Cube instance.

    < iris_coords_list > : list of iris.coords.DimCoord objects (use routine create_iris_coord_list for standard coordinates).
    """
    # class iris.cube.Cube(data, standard_name=None, long_name=None, var_name=None, units=None, attributes=None, cell_methods=None, dim_coords_and_dims=None, aux_coords_and_dims=None, aux_factories=None)

    allcoords = []
    if not isinstance(iris_coords_list[0], iris.coords.DimCoord):
        raise ValueError('coords not in iris format')

    allcoords = [(cor, i) for i, cor in enumerate(iris_coords_list)]

    cube = iris.cube.Cube(data, standard_name = varname, units = varunits, dim_coords_and_dims = allcoords, long_name = long_name)

    return cube


def create_iris_coord_list(coords_points, coords_names, time_units = None, time_calendar = None, level_units = None):
    """
    Creates a list of coords in iris format for standard (lat, lon, time, level) coordinates.
    """

    coords_list = []
    for i, (cordpo, nam) in enumerate(zip(coords_points, coords_names)):
        cal = None
        circ = False
        if nam in ['latitude', 'longitude', 'lat', 'lon']:
            units = 'degrees'
            if 'lon' in nam: circ = True
        if nam == 'time':
            units = Unit(time_units, calendar = time_calendar)
        if nam in ['lev', 'level', 'plev']:
            units = level_units

        cord = create_iris_coord(cordpo, std_name = nam, units = units, circular = circ)
        coords_list.append(cord)

    return coords_list


def create_iris_coord(points, std_name, long_name = None, units = None, circular = False, calendar = None):
    """
    Creates an iris.coords.DimCoord instance.
    """
    # class iris.coords.DimCoord(points, standard_name=None, long_name=None, var_name=None, units='1', bounds=None, attributes=None, coord_system=None, circular=False)

    if std_name == 'longitude' or std_name == 'lon':
        circular = True
    if std_name == 'time' and (calendar is None or units is None):
        raise ValueError('No calendar/units given for time!')
        units = Unit(units, calendar = calendar)

    coord = iris.coords.DimCoord(points, standard_name = std_name, long_name = long_name, units = units, circular = circular)

    return coord


def save2Dncfield(lats,lons,variab,varname,ofile):
    '''
    GOAL
        Save var in ofile netCDF file
    USAGE
        save2Dncfield(var,ofile)
        fname: output filname
    '''
    try:
        os.remove(ofile) # Remove the outputfile
    except OSError:
        pass
    dataset = nc.Dataset(ofile, 'w', format='NETCDF4_CLASSIC')
    #print(dataset.file_format)

    lat = dataset.createDimension('lat', variab.shape[0])
    lon = dataset.createDimension('lon', variab.shape[1])

    # Create coordinate variables for 2-dimensions
    lat = dataset.createVariable('lat', np.float32, ('lat',))
    lon = dataset.createVariable('lon', np.float32, ('lon',))
    # Create the actual 2-d variable
    var = dataset.createVariable(varname, np.float64,('lat','lon'))

    #print('variable:', dataset.variables[varname])

    #for varn in dataset.variables:
    #    print(varn)
    # Variable Attributes
    lat.units='degree_north'
    lon.units='degree_east'
    #var.units = varunits

    lat[:]=lats
    lon[:]=lons
    var[:,:]=variab

    dataset.close()

    #----------------------------------------------------------------------------------------
    print('The 2D field [lat x lon] is saved as \n{0}'.format(ofile))
    print('__________________________________________________________')
    #----------------------------------------------------------------------------------------
    return


def save3Dncfield(lats, lons, variab, varname, varunits, dates, timeunits, time_cal, ofile):
    '''
    GOAL
        Save var in ofile netCDF file
    USAGE
        save3Dncfield(var,ofile)
        fname: output filname
    '''
    try:
        os.remove(ofile) # Remove the outputfile
    except OSError:
        pass
    dataset = nc.Dataset(ofile, 'w', format='NETCDF4_CLASSIC')
    #print(dataset.file_format)

    time = dataset.createDimension('time', None)
    lat = dataset.createDimension('lat', variab.shape[1])
    lon = dataset.createDimension('lon', variab.shape[2])

    # Create coordinate variables for 3-dimensions
    time = dataset.createVariable('time', np.float64, ('time',))
    lat = dataset.createVariable('lat', np.float32, ('lat',))
    lon = dataset.createVariable('lon', np.float32, ('lon',))
    # Create the actual 3-d variable
    var = dataset.createVariable(varname, np.float64,('time','lat','lon'))

    #print('variable:', dataset.variables[varname])

    #for varn in dataset.variables:
    #    print(varn)
    # Variable Attributes
    time.units=timeunits
    time.calendar=time_cal
    lat.units='degree_north'
    lon.units='degree_east'
    var.units = varunits

    # Fill in times.
    time[:] = date2num(dates, units = timeunits, calendar = time_cal)#, calendar = times.calendar)
    print(time_cal)
    print('time values (in units {0}): {1}'.format(timeunits,time[:]))
    print(dates)

    #print('time values (in units %s): ' % time)

    lat[:]=lats
    lon[:]=lons
    var[:,:,:]=variab

    dataset.close()

    #----------------------------------------------------------------------------------------
    print('The 3D field [time x lat x lon] is saved as \n{0}'.format(ofile))
    print('__________________________________________________________')
    #----------------------------------------------------------------------------------------
    return


def save_N_2Dfields(lats,lons,variab,varname,varunits,ofile):
    '''
    GOAL
        Save var in ofile netCDF file
    USAGE
        save a number N of 2D fields [latxlon]
        fname: output filname
    '''
    try:
        os.remove(ofile) # Remove the outputfile
    except OSError:
        pass
    dataset = nc.Dataset(ofile, 'w', format='NETCDF4_CLASSIC')
    #print(dataset.file_format)

    num = dataset.createDimension('num', variab.shape[0])
    lat = dataset.createDimension('lat', variab.shape[1])
    lon = dataset.createDimension('lon', variab.shape[2])

    # Create coordinate variables for 3-dimensions
    num = dataset.createVariable('num', np.int32, ('num',))
    lat = dataset.createVariable('lat', np.float32, ('lat',))
    lon = dataset.createVariable('lon', np.float32, ('lon',))
    # Create the actual 3-d variable
    var = dataset.createVariable(varname, np.float64,('num','lat','lon'))

    #print('variable:', dataset.variables[varname])

    #for varn in dataset.variables:
    #    print(varn)
    # Variable Attributes
    lat.units='degree_north'
    lon.units='degree_east'
    var.units = varunits

    num[:]=np.arange(variab.shape[0])
    lat[:]=lats
    lon[:]=lons
    var[:,:,:]=variab

    dataset.close()

    #----------------------------------------------------------------------------------------
    print('The {0} 2D fields [num x lat x lon] are saved as \n{1}'.format(variab.shape[0], ofile))
    print('__________________________________________________________')
    #----------------------------------------------------------------------------------------
    return

###### Selecting part of the dataset

def sel_area_translate(area):
    if area=='EAT':
        printarea='Euro-Atlantic'
        latN = 87.5
        latS = 30.0
        lonW =-80.0
        lonE = 40.0
    elif area=='PNA':
        printarea='Pacific North American'
        latN = 87.5
        latS = 30.0
        lonW = 140.0
        lonE = 280.0
    elif area=='NH':
        printarea='Northern Hemisphere'
        latN = 90.0
        latS = 0.0
        lonW = -180.
        lonE = 180.
    elif area=='Eu':
        printarea='Europe'
        latN = 72.0
        latS = 27.0
        lonW = -22.0
        lonE = 45.0
    elif area=='Med':
        printarea='Mediterranean'
        latN = 50.0
        latS = 25.0
        lonW = -10.0
        lonE = 40.0
    elif (type(area) in [list, tuple, np.ndarray]) and len(area) == 4:
        lonW, lonE, latS, latN = area
    else:
        raise ValueError('area {} not recognised'.format(area))

    return [lonW, lonE, latS, latN]

def sel_area(lat, lon, var, area):
    '''
    GOAL
        Selecting the area of interest from a nc dataset.
    USAGE
        var_area, lat_area, lon_area = sel_area(lat,lon,var,area)

    :param area: str or list. If str: 'EAT', 'PNA', 'NH', 'Eu' or 'Med'. If list: a custom set can be defined. Order is (latS, latN, lonW, lonE).
    '''
    if area=='EAT':
        printarea='Euro-Atlantic'
        latN = 87.5
        latS = 30.0
        lonW =-80.0     #280
        lonE = 40.0     #40
        # lat and lon are extracted from the netcdf file, assumed to be 1D
        #If 0<lon<360, convert to -180<lon<180
        if lon.min() >= 0:
            lon_new=lon-180
            var_roll=np.roll(var,int(len(lon)//2),axis=-1)
        else:
            var_roll=var
            lon_new=lon

    elif area=='PNA':
        printarea='Pacific North American'
        latN = 87.5
        latS = 30.0
        lonW = 140.0
        lonE = 280.0
        # lat and lon are extracted from the netcdf file, assumed to be 1D
        #If -180<lon<180, convert to 0<lon<360
        if lon.min() < 0:
            lon_new=lon+180
            var_roll=np.roll(var,int(len(lon)//2),axis=-1)
        else:
            var_roll=var
            lon_new=lon

    elif area=='NH':
        printarea='Northern Hemisphere'
        latN = 90.0
        latS = 0.0
        lonW = lon.min()
        lonE = lon.max()
        var_roll=var
        lon_new=lon

    elif area=='Eu':
        printarea='Europe'
        latN = 72.0
        latS = 27.0
        lonW = -22.0
        lonE = 45.0
        # lat and lon are extracted from the netcdf file, assumed to be 1D
        #If 0<lon<360, convert to -180<lon<180
        if lon.min() >= 0:
            lon_new=lon-180
            var_roll=np.roll(var,int(len(lon)//2),axis=-1)
        else:
            var_roll=var
            lon_new=lon
    elif area=='Med':
        printarea='Mediterranean'
        latN = 50.0
        latS = 25.0
        lonW = -10.0
        lonE = 40.0
        # lat and lon are extracted from the netcdf file, assumed to be 1D
        #If 0<lon<360, convert to -180<lon<180
        if lon.min() >= 0:
            lon_new=lon-180
            print(var.shape)
            var_roll=np.roll(var,int(len(lon)//2),axis=-1)
        else:
            var_roll=var
            lon_new=lon
    elif (type(area) == list) or (type(area) == tuple) and len(area) == 4:
        lonW, lonE, latS, latN = area
        print('custom lat {}-{} lon {}-{}'.format(latS, latN, lonW, lonE))
        if lon.min() >= 0:
            lon_new=lon-180
            var_roll=np.roll(var,int(len(lon)//2),axis=-1)
        else:
            var_roll=var
            lon_new=lon
    else:
        raise ValueError('area {} not recognised'.format(area))

    latidx = (lat >= latS) & (lat <= latN)
    lonidx = (lon_new >= lonW) & (lon_new <= lonE)

    print('Area: ', lonW, lonE, latS, latN)
    # print(lat, lon_new)
    # print(latidx, lonidx)
    # print(var_roll.shape, len(latidx), len(lonidx))
    if var.ndim == 3:
        var_area = var_roll[:, latidx][..., lonidx]
    elif var.ndim == 2:
        var_area = var_roll[latidx, ...][..., lonidx]
    else:
        raise ValueError('Variable has {} dimensions, should have 2 or 3.'.format(var.ndim))

    return var_area, lat[latidx], lon_new[lonidx]


def sel_season(var, dates, season, cut = True):
    """
    Selects the desired seasons from the dataset.

    :param var: the variable matrix

    :param dates: the dates as extracted from the nc file.

    :param season: the season to be extracted.
    Formats accepted for season:
        - any sequence of at least 2 months with just the first month capital letter: JJA, ND, DJFM, ...
        - a single month with its short 3-letters name (First letter is capital): Jan, Feb, Mar, ...

    :param cut: bool. If True eliminates partial seasons.

    """

    dates_pdh = pd.to_datetime(dates)
    # day = pd.Timedelta('1 days')
    # dates_pdh_day[1]-dates_pdh_day[0] == day

    mesi_short = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_seq = 2*'JFMAMJJASOND'

    if season in month_seq and len(season) > 1:
        ind1 = month_seq.find(season)
        ind2 = ind1 + len(season)
        indxs = np.arange(ind1,ind2)
        indxs = indxs % 12 + 1

        mask = dates_pdh.month == indxs[0]
        for ind in indxs[1:]:
            mask = (mask) | (dates_pdh.month == ind)
    elif season in mesi_short:
        mask = (dates_pdh.month == mesi_short.index(season)+1)
    else:
        raise ValueError('season not understood, should be in DJF, JJA, ND,... format or the short 3 letters name of a month (Jan, Feb, ...)')

    var_season = var[mask, ...]
    dates_season = dates[mask]
    dates_season_pdh = pd.to_datetime(dates_season)

    #print(var_season.shape)

    if np.sum(mask) == 1:
        var_season = var_season[np.newaxis, :]

    #print(var_season.shape)

    if season in mesi_short or len(dates) <= 12:
        cut = False

    if cut:
        if (12 in indxs) and (1 in indxs):
            #REMOVING THE FIRST MONTHS (for the first year) because there is no previuos december
            start_cond = (dates_season_pdh.year == dates_pdh.year[0]) & (dates_season_pdh.month == indxs[0])
            if np.sum(start_cond):
                start = np.argmax(start_cond)
            else:
                start = 0

            #REMOVING THE LAST MONTHS (for the last year) because there is no following january
            end_cond = (dates_season_pdh.year == dates_pdh.year[-1]) & (dates_season_pdh.month == indxs[0])
            if np.sum(end_cond):
                end = np.argmax(end_cond)
            else:
                end = None

            var_season = var_season[start:end, ...]
            dates_season = dates_season[start:end]

    #print(var_season.shape)

    return var_season, dates_season


def daily_climatology(var, dates, window, refyear = 2001):
    """
    Performs a daily climatological mean of the dataset using a window of n days around the day considered (specified in input). Example: with window = 5, the result for day 15/02 is done averaging days from 13/02 to 17/02 in the full dataset.

    var has to be a daily dataset.

    Window has to be an odd integer number. If n is even, the odd number n+1 is taken as window.
    If window == 1, computes the simple day-by-day mean.

    Dates of the climatology are referred to year <refyear>, has no effect on the calculation.
    """

    if not check_daily(dates):
        raise ValueError('Not a daily dataset\n')

    dates_pdh = pd.to_datetime(dates)

    months = np.unique(dates_pdh.month)
    daysok = []
    for mon in months:
        mask = dates_pdh.month == mon
        okday = np.unique(dates_pdh[mask].day)
        if mon == 2 and okday[-1] == 29:
            daysok.append(okday[:-1])
        else:
            daysok.append(okday)

    delta = window//2

    filt_mean = []
    filt_std = []
    dates_filt = []

    for mon, daymon in zip(months, daysok):
        for day in daymon:
            data = pd.to_datetime('{:4d}{:02d}{:02d}'.format(refyear, mon,day), format='%Y%m%d')
            if window > 2:
                doy = data.dayofyear
                okdates = ((dates_pdh.dayofyear - doy) % 365 <= delta) | ((doy - dates_pdh.dayofyear) % 365 <= delta)
            else:
                okdates = (dates_pdh.month == mon) & (dates_pdh.day == day)
            #print(mon,day,doy,np.sum(okdates))
            filt_mean.append(np.mean(var[okdates, ...], axis = 0))
            filt_std.append(np.std(var[okdates, ...], axis = 0))
            dates_filt.append(data)

    dates_ok = pd.to_datetime(dates_filt).to_pydatetime()

    filt_mean = np.stack(filt_mean)
    filt_std = np.stack(filt_std)

    return filt_mean, dates_ok, filt_std


def trend_daily_climat(var, dates, window_days = 5, window_years = 20, step_year = 5):
    """
    Performs daily_climatology on a running window of n years, in order to take into account possible trends in the mean state.
    """
    dates_pdh = pd.to_datetime(dates)
    climat_mean = []
    dates_climate_mean = []
    allyears = np.unique(dates_pdh.year)
    years = allyears[::step_year]

    for yea in years:
        okye = (dates_pdh.year >= yea - window_years//2) & (dates_pdh.year <= yea + window_years//2)
        numyea = np.sum((allyears >= yea - window_years//2) & (allyears <= yea + window_years//2))
        print(yea, numyea)
        if numyea < window_years:
            print('skipped')
            continue
        clm, dtclm, _ = daily_climatology(var[okye], dates[okye], window_days, refyear = yea)
        climat_mean.append(clm)
        dates_climate_mean.append(dtclm)

    return climat_mean, dates_climate_mean


def trend_monthly_climat(var, dates, window_years = 20, step_year = 5):
    """
    Performs monthly_climatology on a running window of n years, in order to take into account possible trends in the mean state.
    """
    dates_pdh = pd.to_datetime(dates)
    climat_mean = []
    dates_climate_mean = []

    allyears = np.unique(dates_pdh.year)
    years = allyears[::step_year]

    for yea in years:
        okye = (dates_pdh.year >= yea - window_years//2) & (dates_pdh.year <= yea + window_years//2)
        numyea = np.sum((allyears >= yea - window_years//2) & (allyears <= yea + window_years//2))
        print(yea, numyea)
        if numyea < window_years:
            print('skipped')
            continue
        clm, dtclm, _ = monthly_climatology(var[okye], dates[okye], refyear = yea)
        climat_mean.append(clm)
        dates_climate_mean.append(dtclm)

    return climat_mean, dates_climate_mean


def monthly_climatology(var, dates, refyear = 2001, dates_range = None):
    """
    Performs a monthly climatological mean of the dataset.
    Works both on monthly and daily datasets.

    Dates of the climatology are referred to year <refyear>, has no effect on the calculation.

    < dates_range > : list, tuple. first and last dates to be considered in datetime format. If years, use range_years() function.
    """

    if dates_range is not None:
        var, dates = sel_time_range(var, dates, dates_range)
        refyear = dates_range[1].year

    dates_pdh = pd.to_datetime(dates)

    months = np.unique(dates_pdh.month)
    dayref = np.unique(dates_pdh.day)[0]

    filt_mean = []
    filt_std = []
    dates_filt = []

    for mon in months:
        data = pd.to_datetime('2001{:02d}{:02d}'.format(mon,dayref), format='%Y%m%d')
        okdates = (dates_pdh.month == mon)
        filt_mean.append(np.mean(var[okdates,...], axis = 0))
        filt_std.append(np.std(var[okdates,...], axis = 0))
        dates_filt.append(data)

    dates_ok = pd.to_datetime(dates_filt).to_pydatetime()

    filt_mean = np.stack(filt_mean)
    filt_std = np.stack(filt_std)

    return filt_mean, dates_ok, filt_std


def seasonal_climatology(var, dates, season, dates_range = None, cut = True):
    """
    Performs a seasonal climatological mean of the dataset.
    Works both on monthly and daily datasets.

    Dates of the climatology are referred to year <refyear>, has no effect on the calculation.

    < dates_range > : list, tuple. first and last dates to be considered in datetime format. If years, use range_years() function.
    """

    if season != 'year':
        all_var_seas, _ = seasonal_set(var, dates, season, dates_range = dates_range, cut = cut)
        all_seas = [np.mean(varse, axis = 0) for varse in all_var_seas]
    else:
        all_seas = yearly_average(var, dates, dates_range = dates_range, cut = cut)[0]

    seas_mean = np.mean(all_seas, axis = 0)
    seas_std = np.std(all_seas, axis = 0)

    return seas_mean, seas_std


def seasonal_set(var, dates, season, dates_range = None, cut = True):
    """
    Cuts var and dates, creating a list of variables relative to each season and a list of dates.
    Works both on monthly and daily datasets.

    < dates_range > : list, tuple. first and last dates to be considered in datetime format. If years, use range_years() function.
    """

    if dates_range is not None:
        var, dates = sel_time_range(var, dates, dates_range)

    if len(var) <= len(season):
        cut = False

    dates_pdh = pd.to_datetime(dates)

    var_season, dates_season = sel_season(var, dates, season, cut = cut)

    if check_daily(dates):
        dates_diff = dates_season[1:] - dates_season[:-1]
        jump = dates_diff > pd.Timedelta('2 days')
        okju = np.where(jump)[0] + 1
        okju = np.append([0], okju)
        okju = np.append(okju, [len(dates_season)])
        n_seas = len(okju) - 1

        all_dates_seas = [dates_season[okju[i]:okju[i+1]] for i in range(n_seas)]
        all_var_seas = [var_season[okju[i]:okju[i+1], ...] for i in range(n_seas)]
    else:
        n_seas = len(var_season)//len(season)
        all_dates_seas = [dates_season[len(season)*i:len(season)*(i+1)] for i in range(n_seas)]
        all_var_seas = [var_season[len(season)*i:len(season)*(i+1), ...] for i in range(n_seas)]

    return np.array(all_var_seas), np.array(all_dates_seas)


def range_years(year1, year2):
    data1 = pd.to_datetime('{}0101'.format(year1), format='%Y%m%d')
    data2 = pd.to_datetime('{}1231'.format(year2), format='%Y%m%d')
    return data1, data2


def sel_time_range(var, dates, dates_range, ignore_HHMM = True):
    """
    Extracts a subset in time.
    < ignore_HHMM > : if True, considers only day, mon and year.
    """

    if ignore_HHMM:
        okdates = np.array([(da.date() >= dates_range[0].date()) & (da.date() <= dates_range[1].date()) for da in dates])
    else:
        dates_pdh = pd.to_datetime(dates)
        okdates = (dates_pdh >= dates_range[0]) & (dates_pdh <= dates_range[1])

    return var[okdates, ...], dates[okdates]


def range_dates_monthly(first, last, monday = 15):
    """
    Creates a sequence of monthly dates between first and last datetime objects.
    """

    strindata = '{:4d}-{:02d}-{:02d} 12:00:00'
    ye0 = first.year
    mo0 = first.month
    ye1 = last.year
    mo1 = last.month

    datesmon = []
    for ye in range(ye0, ye1+1):
        mouno = 1
        modue = 13
        if ye == ye0:
            mouno = mo0
        elif ye == ye1:
            modue = mo1+1
        for mo in range(mouno,modue):
            datesmon.append(pd.Timestamp(strindata.format(ye,mo,monday)).to_pydatetime())

    datesmon = np.array(datesmon)

    return datesmon


def complete_time_range(var_season, dates_season, dates_all = None):
    """
    Completes a time series with missing dates. Returns a masked numpy array.
    """
    if dates_all is None:
        if check_daily(dates_season):
            freq = 'D'
            dates_all_pdh = pd.date_range(dates_season[0], dates_season[-1], freq = freq)
            dates_all = np.array([da.to_pydatetime() for da in dates_all_pdh])
        else:
            freq = 'M'
            dates_all = range_dates_monthly(dates_season[0], dates_season[-1])

    vals, okinds, okinds_seas = np.intersect1d(dates_all, dates_season, assume_unique = True, return_indices = True)

    # shapea = list(var_season.shape)
    # shapea[0] = len(dates_all)
    # np.tile(maska, (3, 7, 1)).T.shape
    var_all = np.ma.empty(len(dates_all))
    var_all.mask = ~okinds
    var_all[okinds] = var_season

    return var_all, dates_all


def date_series(init_dat, end_dat, freq = 'day', calendar = 'proleptic_gregorian'):
    """
    Creates a complete date_series between init_dat and end_dat.

    < freq > : 'day' or 'mon'
    """
    dates = pd.date_range(init_dat, end_dat, freq = freq[0]).to_pydatetime()

    return dates


def check_daily(dates, allow_diff_HH = True):
    """
    Checks if the dataset is a daily dataset.

    < allow_diff_HH > : if set to True (default), allows for a difference up to 12 hours.
    """
    daydelta = pd.Timedelta('1 days')
    delta = dates[1]-dates[0]

    if delta == daydelta:
        return True
    elif allow_diff_HH and (daydelta >= pd.Timedelta('12 hours') and daydelta <= pd.Timedelta('36 hours')):
        return True
    else:
        return False


def running_mean(var, wnd):
    """
    Performs a running mean (if multidim, the mean is done on the first axis).

    < wnd > : is the window length.
    """
    if var.ndim == 1:
        tempser = pd.Series(var)
        rollpi_temp = tempser.rolling(wnd, center = True).mean()
    else:
        rollpi_temp = []
        for i in range(len(var)):
            if i-wnd//2 < 0 or i + wnd//2 > len(var)-1:
                rollpi_temp.append(np.nan*np.ones(var[0].shape))
            else:
                rollpi_temp.append(np.mean(var[i-wnd//2:i+wnd//2+1, ...], axis = 0))

        rollpi_temp = np.stack(rollpi_temp)

    return rollpi_temp


def anomalies_daily_detrended(var, dates, climat_mean = None, dates_climate_mean = None, window_days = 5, window_years = 20, step_year = 5):
    """
    Calculates the daily anomalies wrt a trending climatology. climat_mean and dates_climate_mean are the output of trend_daily_climat().
    """
    dates_pdh = pd.to_datetime(dates)

    print('DENTRO ANOMALIES DETRENDED: {} {}\n'.format(len(dates), len(dates_pdh)))

    if climat_mean is None or dates_climate_mean is None:
        climat_mean, dates_climate_mean = trend_daily_climat(var, dates, window_days = window_days, window_years = window_years, step_year = step_year)

    if len(climat_mean) == 0:
        raise ValueError('ERROR in calculation of detrended climatology. Too few years? Try lowering wnd_years or set detrending to False')

    var_anom_tot = []
    year_ref_all = np.array([dat[0].year for dat in dates_climate_mean])
    #year_steps = np.unique(dates_pdh.year)[::step_year]
    #okyetot = np.zeros(len(var), dtype=bool)

    for yea in np.unique(dates_pdh.year):
        yearef = np.argmin(abs(year_ref_all - yea))
        okye = dates_pdh.year == yea
        # if np.sum(okye) <= 1:
        #     print('This year {} has only one day.. If you really want to take it in, change this line\n')
        #     continue

        var_anom = anomalies_daily(var[okye], dates[okye], climat_mean = climat_mean[yearef], dates_climate_mean = dates_climate_mean[yearef], window = window_days)
        var_anom_tot.append(var_anom)

    # for yea, clm, dtclm in zip(year_steps, climat_mean, dates_climate_mean):
    #     okye = (dates_pdh.year >= yea - step_year/2.) & (dates_pdh.year < yea + step_year/2.)
    #     print(dates[okye][0], dates[okye][-1])
    #     var_anom = anomalies_daily(var[okye], dates[okye], climat_mean = clm, dates_climate_mean = dtclm, window = window_days)
    #
    #     #okyetot = okyetot | okye
    #     var_anom_tot.append(var_anom)

    var_anom_tot = np.concatenate(var_anom_tot)
    if len(var_anom_tot) != len(var):
        raise ValueError('{} and {} differ'.format(len(var_anom_tot), len(var)))

    return var_anom_tot


def anomalies_daily(var, dates, climat_mean = None, dates_climate_mean = None, window = 5):
    """
    Computes anomalies for a field with respect to the climatological mean (climat_mean). If climat_mean is not set, it is calculated making a daily or monthly climatology on var with the specified window.

    Works only for daily datasets.

    :param climat_mean: the climatological mean to be subtracted.
    :param dates_climate_mean: the
    :param window: int. Calculate the running mean on N day window.
    """

    if len(dates) > 1: # Stupid datasets...
        if not check_daily(dates):
            raise ValueError('Not a daily dataset')

    if climat_mean is None or dates_climate_mean is None:
        climat_mean, dates_climate_mean, _ = daily_climatology(var, dates, window)

    dates_pdh = pd.to_datetime(dates)
    dates_climate_mean_pdh = pd.to_datetime(dates_climate_mean)
    var_anom = np.empty_like(var)

    for el, dat in zip(climat_mean, dates_climate_mean_pdh):
        mask = (dates_pdh.month == dat.month) & (dates_pdh.day == dat.day)
        var_anom[mask, ...] = var[mask, ...] - el

    mask = (dates_pdh.month == 2) & (dates_pdh.day == 29)
    okel = (dates_climate_mean_pdh.month == 2) & (dates_climate_mean_pdh.day == 28)

    var_anom[mask, ...] = var[mask, ...] - climat_mean[okel, ...]

    return var_anom


def anomalies_monthly_detrended(var, dates, climat_mean = None, dates_climate_mean = None, window_years = 20, step_year = 5):
    """
    Calculates the monthly anomalies wrt a trending climatology. climat_mean and dates_climate_mean are the output of trend_monthly_climat().
    """
    dates_pdh = pd.to_datetime(dates)
    if climat_mean is None or dates_climate_mean is None:
        climat_mean, dates_climate_mean = trend_monthly_climat(var, dates, window_years = window_years, step_year = step_year)

    var_anom_tot = []
    #year_steps = np.unique(dates_pdh.year)[::step_year]
    year_ref_all = np.array([dat[0].year for dat in dates_climate_mean])

    for yea in np.unique(dates_pdh.year):
        #yearef = np.argmin(abs(year_steps - yea))
        yearef = np.argmin(abs(year_ref_all - yea))
        okye = dates_pdh.year == yea
        var_anom = anomalies_monthly(var[okye], dates[okye], climat_mean = climat_mean[yearef], dates_climate_mean = dates_climate_mean[yearef])
        var_anom_tot.append(var_anom)

    var_anom_tot = np.concatenate(var_anom_tot)
    if len(var_anom_tot) != len(var):
        raise ValueError('{} and {} differ'.format(len(var_anom_tot), len(var)))

    return var_anom_tot


def anomalies_monthly(var, dates, climat_mean = None, dates_climate_mean = None):
    """
    Computes anomalies for a field with respect to the climatological mean (climat_mean). If climat_mean is not set, it is calculated making a monthly climatology on var.

    Works only for monthly datasets.

    :param climat_mean: the climatological mean to be subtracted.
    :param dates_climate_mean: the
    """

    if check_daily(dates):
        raise ValueError('Not a monthly dataset')

    if climat_mean is None or dates_climate_mean is None:
        climat_mean, dates_climate_mean, _ = monthly_climatology(var, dates)

    dates_pdh = pd.to_datetime(dates)
    dates_climate_mean_pdh = pd.to_datetime(dates_climate_mean)
    var_anom = np.empty_like(var)

    for el, dat in zip(climat_mean, dates_climate_mean_pdh):
        mask = (dates_pdh.month == dat.month)
        var_anom[mask, ...] = var[mask, ...] - el

    return var_anom


def anomalies_ensemble(var_ens, extreme = 'mean'):
    """
    Calculates mean and anomaly on an ensemble. Optionally calculates other statistical quantities rather than the mean, set by the key "extreme".

    Possible quantities to calculate are:
    - mean: the time mean.
    - {}th_perc: the nth percentile of each member in time. i.e. for the 90th percentile, the key '90th_perc' has to be set
    - maximum: the maximum of each member in time.
    - std: the standard deviation of each member in time.
    - trend: the trend of each member in time.

    The "extreme_ensemble_mean" is calculated averaging on the full ensemble the quantities above. Anomalies are calculated with respect to this "ensemble mean".

    """
    numens = len(var_ens)

    if extreme=='mean':
        #Compute the time mean over the entire period, for each ensemble member
        varextreme_ens=[np.mean(var_ens[i],axis=0) for i in range(numens)]
    elif len(extreme.split("_"))==2:
        #Compute the chosen percentile over the period, for each ensemble member
        q=int(extreme.partition("th")[0])
        varextreme_ens=[np.percentile(var_ens[i],q,axis=0) for i in range(numens)]
    elif extreme=='maximum':
        #Compute the maximum value over the period, for each ensemble member
        varextreme_ens=[np.max(var_ens[i],axis=0) for i in range(numens)]
    elif extreme=='std':
        #Compute the standard deviation over the period, for each ensemble member
        varextreme_ens=[np.std(var_ens[i],axis=0) for i in range(numens)]
    elif extreme=='trend':
        #Compute the linear trend over the period, for each ensemble member
        trendmap=np.empty((var_ens[0].shape[1],var_ens[0].shape[2]))
        trendmap_ens=[]
        for i in range(numens):
            for la in range(var_ens[0].shape[1]):
                for lo in range(var_ens[0].shape[2]):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(list(range(var_ens[0].shape[0])),var_ens[i][:,la,lo])
                    trendmap[la,lo]=slope
            trendmap_ens.append(trendmap)
        varextreme_ens = trendmap_ens

    print('\n------------------------------------------------------------')
    print('Anomalies and ensemble mean are computed with respect to the {0}'.format(extreme))
    print('------------------------------------------------------------\n')

    extreme_ensemble_mean = np.mean(varextreme_ens_np, axis = 0)
    extreme_ens_anomalies = varextreme_ens_np - extreme_ensemble_mean

    return extreme_ens_anomalies, extreme_ensemble_mean


def yearly_average(var, dates, dates_range = None, cut = True):
    """
    Averages year per year.
    """
    if dates_range is not None:
        var, dates = sel_time_range(var, dates, dates_range)

    dates_pdh = pd.to_datetime(dates)

    nuvar = []
    nudates = []
    for year in np.unique(dates_pdh.year):
        data = pd.to_datetime('{}0101'.format(year), format='%Y%m%d')
        okdates = (dates_pdh.year == year)
        if cut and len(np.unique(dates_pdh[okdates].month)) < 12:
            continue
        nuvar.append(np.mean(var[okdates, ...], axis = 0))
        nudates.append(data)

    nuvar = np.stack(nuvar)

    return nuvar, nudates


def global_mean(field, latitude, mask = None, skip_nan = True):
    """
    Calculates a global mean of field, weighting with the cosine of latitude.

    Accepts 3D (time, lat, lon) and 2D (lat, lon) input arrays.
    """
    weights_array = abs(np.cos(np.deg2rad(latitude)))

    if mask is not None:
        if field.ndim == 3 and mask.ndim == 2:
            mask = np.tile(mask, (field.shape[0],1,1))

    zonal_field = zonal_mean(field, mask = mask, skip_nan = skip_nan)
    #print(zonal_field.shape)
    if np.any(np.isnan(zonal_field)):
        if zonal_field.ndim == 2:
            indexes = np.isnan(zonal_field)[0,:]
        else:
            indexes = np.isnan(zonal_field)
        weights_array[indexes] = 0
        zonal_field[np.isnan(zonal_field)] = 0.0
    mea = np.average(zonal_field, weights = weights_array, axis = -1)

    if np.any(np.isnan(mea)):
        print('non dovrebbe essere NaN')
        raise ValueError

    return mea


def band_mean_from_zonal(zonal_field, latitude, latmin, latmax):
    """
    Calculates a global mean of field, weighting with the cosine of latitude.

    Accepts 3D (time, lat, lon) and 2D (lat, lon) input arrays.
    """
    okpo = (latitude >= latmin) & (latitude <= latmax)
    weights_array = abs(np.cos(np.deg2rad(latitude[okpo])))

    mea = np.average(zonal_field[..., okpo], weights = weights_array, axis = -1)

    return mea


def zonal_mean(field, mask = None, skip_nan = True, skip_inf = True):
    """
    Calculates a zonal mean of field.

    Accepts 3D (time, lat, lon) and 2D (lat, lon) input arrays.
    """

    if skip_inf:
        skip_nan = True
        field[np.isinf(field)] = np.nan

    if mask is not None:
        zonal_mask = np.any(mask, axis = -1)
        if np.all(zonal_mask):
            mea = np.average(field, axis = -1, weights = mask)
        else:
            mask[~ zonal_mask, 0] = True
            mea = np.average(field, axis = -1, weights = mask)
            mea[~ zonal_mask] = np.nan
    elif skip_nan:
        mea = np.nanmean(field, axis = -1)
    else:
        mea = np.average(field, axis = -1)

    return mea

#######################################################
#
###     EOF computation / Clustering / algebra
#
#######################################################

def first(condition):
    """
    Returns the first index for which condition is True. Works only for 1-dim arrays.
    """
    #ind = np.asarray(condition).nonzero()[0][0]
    ind = np.where(condition.flatten())[0][0]

    return ind


def Rcorr(x, y, latitude = None):
    """
    Returns correlation coefficient between two array of arbitrary shape.
    """
    if latitude is not None:
        weights_u = abs(np.cos(np.deg2rad(latitude)))
        weights = np.tile(weights_u, (x.shape[1], 1)).T.flatten()
        corrcoef = np.cov(x.flatten(), y.flatten(), aweights = weights)/np.sqrt(np.cov(x.flatten(), x.flatten(), aweights = weights) * np.cov(y.flatten(), y.flatten(), aweights = weights))
    else:
        corrcoef = np.corrcoef(x.flatten(), y.flatten())

    return corrcoef[1,0]


def distance(x, y, latitude = None):
    """
    L2 distance.
    """
    if latitude is not None:
        weights = abs(np.cos(np.deg2rad(latitude)))
        x2 = np.empty_like(x)
        y2 = np.empty_like(y)
        for j in range(x.shape[1]):
            x2[:,j] = x[:,j]*weights
            y2[:,j] = y[:,j]*weights

        return LA.norm(x2-y2)
    else:
        return LA.norm(x-y)


def E_rms(x, y, latitude = None):
    """
    Returns root mean square deviation: sqrt(1/N sum (xn-yn)**2). This is consistent with the Dawson (2015) paper.
    < latitude > : if given, weights with the cosine of latitude (FIRST axis of the array).
    """
    n = x.size
    n_lons = x.shape[1]

    if latitude is not None:
        weights = abs(np.cos(np.deg2rad(latitude)))
        n_norm = 0.
        sum_norm = 0.
        for i, w in enumerate(weights):
            sum_norm += w*np.sum((x[i,:]-y[i,:])**2)
            n_norm += n_lons*w

        E = np.sqrt(sum_norm/n_norm)
    else:
        E = 1/np.sqrt(n) * LA.norm(x-y)

    return E


def E_rms_cp(x, y, latitude = None):
    """
    Returns centered-pattern root mean square, e.g. first subtracts the mean to the two series and then computes E_rms.
    """
    x1 = x - x.mean()
    y1 = y - y.mean()

    E = E_rms(x1, y1, latitude = latitude)

    return E


def cosine(x, y):
    """
    Calculates the cosine of the angle between x and y. If x and y are 2D, the scalar product is taken using the np.vdot() function.
    """

    if x.ndim != y.ndim:
        raise ValueError('x and y have different dimension')
    elif x.shape != y.shape:
        raise ValueError('x and y have different shapes')

    if x.ndim == 1:
        return np.dot(x,y)/(LA.norm(x)*LA.norm(y))
    elif x.ndim == 2:
        return np.vdot(x,y)/(LA.norm(x)*LA.norm(y))
    else:
        return np.dot(x.flatten(),y.flatten())/(LA.norm(x)*LA.norm(y))


def cosine_cp(x, y):
    """
    Before calculating the cosine, subtracts the mean to both x and y. This is exactly the same as calculating the correlation coefficient R.
    """

    x1 = x - x.mean()
    y1 = y - y.mean()

    return cosine(x1,y1)


def linear_regre(x, y, return_resids = False):
    """
    Makes a linear regression of dataset y in function of x. Returns the coefficient m and c: y = mx + c.
    """

    xord = np.argsort(x)
    x = x[xord]
    y = y[xord]

    A = np.vstack([x,np.ones(len(x))]).T  # A = [x.T|1.T] dove 1 = [1,1,1,1,1,..]
    res = np.linalg.lstsq(A,y)
    m,c = res[0]
    resid = res[1]

    if return_resids:
        return m, c, resid
    else:
        return m, c


def linear_regre_witherr(x, y):
    """
    Makes a linear regression of dataset y in function of x using numpy.polyfit. Returns the coefficient m and c: y = mx + c. And their estimated error.
    """

    xord = np.argsort(x)
    x = x[xord]
    y = y[xord]

    res = np.polyfit(x, y, deg = 1, cov = True)
    m,c = res[0]
    covmat = res[1]

    err_m = np.sqrt(covmat[0,0])
    err_c = np.sqrt(covmat[1,1])

    return m, c, err_m, err_c


def cutline_fit(x, y, n_cut = 1, approx_cut = None):
    """
    Makes a linear regression with a series of n_cut+1 segments. Finds the points where to "cut" by minimizing the residuals of the least squares fit.

    < n_cut > : number of cuts of the polygonal chain.
    < approx_cut > : list of tuples (x1, x2). The cut point is in this region. Accurate approx_cut speeds up the (stupid) calculation.
    """

    xord = np.argsort(x)
    x = x[xord]
    y = y[xord]

    if approx_cut is None:
        approx_cut = n_cut*[(x[0], x[-1])]

    if n_cut != 1:
        raise ValueError('Not developed yet for n_cut > 1')

    finished = 0
    x1 = approx_cut[0][0]
    x2 = approx_cut[0][1]
    oks = np.where(x > x1)[0]
    icut_ini = oks[0]
    icut_fin = oks[1]
    print('INTERVAL ', x[icut_ini], x[icut_fin])

    r1s = []
    r2s = []
    resids = []
    for icut in range(icut_ini, icut_fin):
        xfit1 = x[:icut]
        yfit1 = y[:icut]
        m1, c1, resid1 = linear_regre(xfit1, yfit1, return_resids = True)
        xfit2 = x[icut:]
        yfit2 = y[icut:]
        m2, c2, resid2 = linear_regre(xfit2, yfit2, return_resids = True, fix_intercept = c1)

        r1s.append((m1, c1))
        r2s.append((m2, c2))
        resids.append(np.sum(resid1)+np.sum(resid2))


    best_fit = np.argmin(resids)
    r1_best = r1s[best_fit]
    r2_best = r2s[best_fit]

    xcut = (x[best_fit+icut_ini]+x[best_fit+icut_ini-1])/2.

    return xcut, r1_best, r2_best


def cutline2_fit(x, y, n_cut = 1, approx_par = None):
    """
    Makes a linear regression with a series of n_cut+1 segments. Finds the points where to "cut" by minimizing the residuals of the least squares fit.

    < n_cut > : number of cuts of the polygonal chain.
    < approx_cut > : list of tuples (x1, x2). The cut point is in this region. Accurate approx_cut speeds up the (stupid) calculation.
    """

    xord = np.argsort(x)
    x = x[xord]
    y = y[xord]

    if n_cut > 2:
        raise ValueError('Not developed yet for n_cut > 2')

    if n_cut == 0:
        def func(x, m1, c1):
            y = m1*x+c1
            return y

        result = optimize.curve_fit(func, x, y, p0 = approx_par)
        m1, c1 = result[0]

        xcuts = []
        lines = [(m1,c1)]
    elif n_cut == 1:
        def func(x, m1, m2, c1, xcut):
            c2 = xcut*(m1-m2)+c1
            x1 = x[x < xcut]
            y1 = m1*x1+c1
            x2 = x[x >= xcut]
            y2 = m2*x2+c2

            y = np.concatenate([y1,y2])
            return y

        result = optimize.curve_fit(func, x, y, p0 = approx_par)
        m1, m2, c1, xcut = result[0]
        c2 = xcut*(m1-m2)+c1

        xcuts = [xcut]
        lines = [(m1,c1), (m2,c2)]
    elif n_cut == 2:
        def func(x, m1, m2, m3, c1, xcut1, xcut2):
            print(xcut1, xcut2)
            if xcut2 < xcut1:
                y = m1*x + c1
                return y
            c2 = xcut1*(m1-m2)+c1
            c3 = xcut2*(m2-m3)+c2
            x1 = x[x < xcut1]
            y1 = m1*x1+c1

            x2 = x[(x >= xcut1) & (x < xcut2)]
            y2 = m2*x2+c2

            x3 = x[x >= xcut2]
            y3 = m3*x3+c3

            y = np.concatenate([y1,y2,y3])
            return y

        print(x.shape, y.shape)
        result = optimize.curve_fit(func, x, y, p0 = approx_par)
        m1, m2, m3, c1, xcut1, xcut2 = result[0]
        c2 = xcut1*(m1-m2)+c1
        c3 = xcut2*(m2-m3)+c2

        xcuts = [xcut1, xcut2]
        lines = [(m1,c1), (m2,c2), (m3,c3)]

    return xcuts, lines


def genlatlon(n_lat, n_lon, lon_limits = (-180., 180.), lat_limits = (-90., 90.)):
    """
    Generates lat and lon arrays, using the number of points n_lat,n_lon and the full range (-90,90), (-180,180).
    """
    lat = np.linspace(lat_limits[0], lat_limits[1], n_lat)
    lon = np.linspace(lon_limits[0], lon_limits[1], n_lon)

    return lat, lon


def eof_computation_bkp(var, varunits, lat, lon):
    """
    Compatibility version.

    Computes the EOFs of a given variable. In the first dimension there has to be different time or ensemble realizations of variable.

    The data are weighted with respect to the cosine of latitude.
    """
    # The data array is dimensioned (ntime, nlat, nlon) and in order for the latitude weights to be broadcastable to this shape, an extra length-1 dimension is added to the end:
    weights_array = np.sqrt(np.cos(np.deg2rad(lat)))[:, np.newaxis]

    start = datetime.now()
    solver = Eof(var, weights=weights_array)
    end = datetime.now()
    print('EOF computation took me {:7.2f} seconds'.format((end-start).total_seconds()))

    #ALL VARIANCE FRACTIONS
    varfrac = solver.varianceFraction()
    acc = np.cumsum(varfrac*100)

    #PCs unscaled  (case 0 of scaling)
    pcs_unscal0 = solver.pcs()
    #EOFs unscaled  (case 0 of scaling)
    eofs_unscal0 = solver.eofs()

    #PCs scaled  (case 1 of scaling)
    pcs_scal1 = solver.pcs(pcscaling=1)

    #EOFs scaled (case 2 of scaling)
    eofs_scal2 = solver.eofs(eofscaling=2)

    return solver, pcs_scal1, eofs_scal2, pcs_unscal0, eofs_unscal0, varfrac


def eof_computation(var, latitude, weight = True):
    """
    Computes the EOFs of a given variable. In the first dimension there has to be different time or ensemble realizations of variable.

    If weigth is True, the data are weighted with respect to the cosine of latitude.
    """
    # The data array is dimensioned (ntime, nlat, nlon) and in order for the latitude weights to be broadcastable to this shape, an extra length-1 dimension is added to the end:
    if weight:
        weights_array = np.sqrt(np.cos(np.deg2rad(latitude)))[:, np.newaxis]
    else:
        weights_array = None

    start = datetime.now()
    solver = Eof(var, weights=weights_array)
    end = datetime.now()
    print('EOF computation took me {:7.2f} seconds'.format((end-start).total_seconds()))

    return solver


def Kmeans_clustering_from_solver(eof_solver, numclus, numpcs, **kwargs):
    """
    Wrapper to Kmeans_clustering starting from eof_solver, the output of func eof_computation.

    numpcs is the dimension of the Eof space to be considered.
    """
    PCs = eof_solver.pcs()[:,:numpcs]

    centroids, labels = Kmeans_clustering(PCs, numclus, **kwargs)

    return centroids, labels


def Kmeans_clustering(PCs, numclus, order_by_frequency = True, algorithm = 'sklearn', n_init_sk = 600,  max_iter_sk = 1000, npart_molt = 1000):
    """
    Computes the Kmeans clustering on the given pcs. Two algorithms can be used:
    - the sklearn KMeans algorithm (algorithm = 'sklearn')
    - the fortran algorithm developed by Molteni (algorithm = 'molteni')

    < param PCs > : the unscaled PCs of the EOF decomposition. The dimension should be already limited to the desired numpcs: PCs.shape = (numpoints, numpcs)
    < param numclus > : number of clusters.
    """

    start = datetime.now()
    if algorithm == 'sklearn':
        clus = KMeans(n_clusters=numclus, n_init = n_init_sk, max_iter = max_iter_sk)

        clus.fit(PCs)
        centroids = clus.cluster_centers_
        labels = clus.labels_
    elif algorithm == 'molteni':
        pc = np.transpose(PCs)
        nfcl, labels, centroids, varopt, iseed = ctool.cluster_toolkit.clus_opt(numclus, npart_molt, pc)
        centroids = np.array(centroids)
        centroids = centroids.T # because of fortran lines/columns ordering
        labels = np.array(labels) - 1 # because of fortran numbering
    else:
        raise ValueError('algorithm {} not recognised'.format(algorithm))

    # print(algorithm)
    # print(centroids)
    end = datetime.now()
    print('k-means algorithm took me {:7.2f} seconds'.format((end-start).total_seconds()))

    ## Ordering clusters for number of members
    centroids = np.array(centroids)
    labels = np.array(labels).astype(int)

    if order_by_frequency:
        centroids, labels = clus_order_by_frequency(centroids, labels)

    return centroids, labels


def clusters_sig(pcs, centroids, labels, dates, nrsamp = 1000, npart_molt = 100):
    """
    H_0: There are no regimes ---> multi-normal distribution PDF
    Synthetic datasets modelled on the PCs of the original data are computed (synthetic PCs have the same lag-1, mean and standard deviation of the original PCs)
    SIGNIFICANCE = % of times that the optimal variance ratio found by clustering the real dataset exceeds the optimal ratio found by clustering the syntetic dataset (is our distribution more clustered than a multinormal distribution?)

    < pcs > : the series of principal components. pcs.shape = (n_days, numpcs)
    < dates > : the dates.
    < nrsamp > : the number of synthetic datasets used to calculate the significance.
    """

    #PCunscal = solver.pcs()[:,:numpcs]
    pc_trans = np.transpose(pcs)

    numclus = centroids.shape[0]
    varopt = calc_varopt_molt(pcs, centroids, labels)

    dates = pd.to_datetime(dates)
    deltas = dates[1:]-dates[:-1]
    deltauno = dates[1]-dates[0]
    ndis = np.sum(abs(deltas) > 2*deltauno) # Finds number of divisions in the dataset (n_seasons - 1)

    print('check: number of seasons = {}\n'.format(ndis+1))

    #=======parallel=======
    start = datetime.now()
    significance = ctp.cluster_toolkit_parallel.clus_sig_p_ncl(nrsamp, numclus, npart_molt, ndis, pc_trans, varopt)
    end = datetime.now()
    print('significance computation took me {:6.2f} seconds'.format((end-start).seconds))
    print('significance for {} clusters = {:6.2f}'.format(numclus, significance))

    return significance


def calc_varopt_molt(pcs, centroids, labels):
    """
    Calculates the variance ratio of the partition, as defined in Molteni's cluster_sig.
    In Molteni this is defined as: media pesata sulla frequenza del quadrato della norma di ogni cluster centroid Sum(centroid**2) DIVISO media della varianza interna su tutti gli elementi Sum(pc-centroid)**2.
    < pcs > : the sequence of pcs.
    < centroids > : the cluster centroids coordinates.
    < labels > : the cluster labels for each element.
    """

    numpcs = centroids.shape[1]
    freq_clus_abs = calc_clus_freq(labels)/100.

    varopt = np.sum(freq_clus_abs*np.sum(centroids**2, axis = 1))

    varint = np.sum([np.sum((pc-centroids[lab])**2) for pc, lab in zip(pcs, labels)])/len(labels)

    varopt = varopt/varint

    return varopt


def calc_autocorr_wlag(pcs, dates = None, maxlag = 40, out_lag1 = False):
    """
    Calculates the variance ratio of the partition, as defined in Molteni's cluster_sig.
    In Molteni this is defined as: media pesata sulla frequenza del quadrato della norma di ogni cluster centroid Sum(centroid**2) DIVISO media della varianza interna su tutti gli elementi Sum(pc-centroid)**2.
    < pcs > : the sequence of pcs.
    < centroids > : the cluster centroids coordinates.
    < labels > : the cluster labels for each element.
    """

    if dates is not None:
        dates_diff = dates[1:] - dates[:-1]
        jump = dates_diff > pd.Timedelta('2 days')

        okju = np.where(jump)[0] + 1
        okju = np.append([0], okju)
        okju = np.append(okju, [len(dates)])
        n_seas = len(okju) - 1
    else:
        n_seas = 0

    if n_seas > 0:
        all_dates_seas = [dates[okju[i]:okju[i+1]] for i in range(n_seas)]
        all_var_seas = [pcs[okju[i]:okju[i+1], ...] for i in range(n_seas)]
        len_sea = np.array([len(da) for da in all_dates_seas])
        if np.any(len_sea < maxlag):
            print('too short season, calc the total autocorr')
            n_seas = 0
        else:
            res_seas = []
            for pcs_seas in all_var_seas:
                res = signal.correlate(pcs_seas, pcs_seas)
                pio = np.argmax(res[:,pcs.shape[1]-1])
                ini = pio - maxlag
                if ini < 0: ini = 0
                fin = pio + maxlag
                if fin >= len(res): fin = None
                #print(pio, ini, fin)
                res_seas.append(res[ini:fin, :])
            results = np.mean(res_seas, axis = 0)

    if n_seas == 0:
        res = signal.correlate(pcs, pcs)
        pio = np.argmax(res[:,pcs.shape[1]-1])
        ini = pio - maxlag
        if ini < 0: ini = 0
        fin = pio + maxlag
        #print(pio, ini, fin)
        if fin > len(res): fin = None
        results = res[ini:fin, :]

    results = results/np.max(results)

    if out_lag1:
        res_ok = results[results.shape[0]//2-1, results.shape[1]//2]
    else:
        res_ok = results[:, results.shape[1]//2]
    #res_ok = np.max(results[np.argmax(results, axis = 0)-1])

    return res_ok


def calc_trend_climatevar(global_tas, var, var_units = None):
    """
    Calculates the trend in some variable with warming. So the trend is in var_units/K.
    : global_tas : timeseries of the global temperature
    : var : the variable. The first axis is assumed to be the time axis: var.shape[0] == len(global_tas)
    """

    if len(global_tas) != var.shape[0]:
        raise ValueError('Shapes of global_tas and var dont match')

    var_trend = np.zeros(var.shape[1:], dtype = float)
    var_trend_err = np.zeros(var.shape[1:], dtype = float)

    # for dim in range(1, var.ndim):
    if var.ndim == 2:
        dim = 1
        for j in range(var.shape[dim]):
            m, c, err_m, err_c = linear_regre_witherr(global_tas, var[:, j])
            var_trend[j] = m
            var_trend_err[j] = err_m
    elif var.ndim == 3:
        for i in range(var.shape[1]):
            for j in range(var.shape[2]):
                m, c, err_m, err_c = linear_regre_witherr(global_tas, var[:, i, j])
                var_trend[i, j] = m
                var_trend_err[i, j] = err_m
    elif var.ndim == 4:
        for k in range(var.shape[1]):
            for i in range(var.shape[2]):
                for j in range(var.shape[3]):
                    m, c, err_m, err_c = linear_regre_witherr(global_tas, var[:, k, i, j])
                    var_trend[k, i, j] = m
                    var_trend_err[k, i, j] = err_m

    return var_trend, var_trend_err


def calc_clus_freq(labels, numclus = None):
    """
    Calculates clusters frequency.
    """
    if numclus is None:
        numclus = int(np.max(labels)+1)
    #print('yo',labels.shape)

    num_mem = []
    for i in range(numclus):
        num_mem.append(np.sum(labels == i))
    num_mem = np.array(num_mem)

    freq_clus = 100.*num_mem/len(labels)

    return freq_clus


def calc_seasonal_clus_freq(labels, dates, nmonths_season = 3):
    """
    Calculates cluster frequency season by season.
    """
    numclus = int(np.max(labels)+1)

    dates_pdh = pd.to_datetime(dates)
    dates_init = dates_pdh[0]
    season_range = pd.Timedelta('{} days 00:00:00'.format(nmonths_season*31))

    freqs = []
    while dates_init < dates_pdh[-1]:
        dateok = (dates_pdh >= dates_init) & (dates_pdh < dates_init+season_range)
        freqs.append(calc_clus_freq(labels[dateok], numclus = numclus))
        nextdat = dates_pdh > dates_init+season_range
        if np.sum(nextdat) > 0:
            dates_init = dates_pdh[nextdat][0]
        else:
            break

    freqs = np.stack(freqs)

    return freqs.T


def calc_monthly_clus_freq(labels, dates):
    """
    Calculates monthly cluster frequency.
    """
    numclus = int(np.max(labels)+1)

    dates_pdh = pd.to_datetime(dates)
    years = dates_pdh.year
    months = dates_pdh.month
    yemon = np.unique([(ye,mo) for ye,mo in zip(years, months)], axis = 0)

    strindata = '{:4d}-{:02d}-{:02d} 12:00:00'

    freqs = []
    datesmon = []
    for (ye, mo) in yemon:
        dateok = (dates_pdh.year == ye) & (dates_pdh.month == mo)
        freqs.append(calc_clus_freq(labels[dateok], numclus = numclus))
        datesmon.append(pd.Timestamp(strindata.format(ye,mo,15)).to_pydatetime())

    freqs = np.stack(freqs)
    datesmon = np.stack(datesmon)

    return freqs.T, datesmon


def change_clus_order(centroids, labels, new_ord):
    """
    Changes order of cluster centroids and labels according to new_order.
    """
    numclus = int(np.max(labels)+1)
    #print('yo',labels.shape, new_ord)

    labels_new = np.array(labels)
    for nu, i in zip(range(numclus), new_ord):
        labels_new[labels == i] = nu
    labels = labels_new

    centroids = centroids[new_ord, ...]

    return centroids, labels


def clus_order_by_frequency(centroids, labels):
    """
    Orders the clusters in decreasing frequency. Returns new labels and ordered centroids.
    """
    numclus = int(np.max(labels)+1)
    #print('yo',labels.shape)

    freq_clus = calc_clus_freq(labels)
    new_ord = freq_clus.argsort()[::-1]

    centroids, labels = change_clus_order(centroids, labels, new_ord)

    return centroids, labels


def clus_compare_projected(centroids, labels, cluspattern_AREA, cluspattern_ref_AREA, solver_ref, numpcs):
    """
    Compares a set of patterns with a reference set, after projecting on the reference base. This is done to calculate the differences in a reduced space.
    Returns the patterns ordered in the best match to the reference ones and the RMS distance and the pattern correlation between the two sets.
    """

    pcs_ref = []
    pcs = []
    for clu_ref, clu in zip(cluspattern_ref_AREA, cluspattern_AREA):
        pcs_ref.append(solver_ref.projectField(clu_ref, neofs=numpcs, eofscaling=0, weighted=True))
        pcs.append(solver_ref.projectField(clu, neofs=numpcs, eofscaling=0, weighted=True))
    pcs_ref = np.stack(pcs_ref)
    pcs = np.stack(pcs)

    perm = match_pc_sets(pcs_ref, pcs)
    centroids, labels = change_clus_order(centroids, labels, perm)

    et, patcor = calc_RMS_and_patcor(pcs_ref, pcs[perm, ...])

    return perm, centroids, labels, et, patcor


def clus_compare_patternsonly(centroids, labels, cluspattern_AREA, cluspattern_ref_AREA):
    """
    Compares a set of patterns with a reference set.
    Returns the patterns ordered in the best match to the reference ones and the RMS distance and the pattern correlation between the two sets.
    """

    perm = match_pc_sets(cluspattern_ref_AREA, cluspattern_AREA)
    centroids, labels = change_clus_order(centroids, labels, perm)

    et, patcor = calc_RMS_and_patcor(cluspattern_ref_AREA, cluspattern_AREA[perm, ...])

    return perm, centroids, labels, et, patcor


def match_pc_sets(pcset_ref, pcset, verbose = False):
    """
    Find the best possible match between two sets of PCs.

    Given two sets of PCs, finds the combination of PCs that minimizes the mean total squared error. If the input PCs represent cluster centroids, the results then correspond to the best unique match between the two sets of cluster centroids.

    The first set of PCs is left in the input order and the second set of PCs is re-arranged.
    Output:
    - new_ord, the permutation of the second set that best matches the first.
    """
    pcset_ref = np.array(pcset_ref)
    pcset = np.array(pcset)
    if pcset_ref.shape != pcset.shape:
        raise ValueError('the PC sets must have the same dimensionality')

    numclus = pcset_ref.shape[0]

    perms = list(itt.permutations(list(range(numclus))))
    nperms = len(perms)

    mean_rms = []
    mean_patcor = []
    for p in perms:
        all_rms = [LA.norm(pcset_ref[i] - pcset[p[i]]) for i in range(numclus)]
        all_patcor = [Rcorr(pcset_ref[i], pcset[p[i]]) for i in range(numclus)]
        if verbose:
            print('Permutation: ', p)
            print(all_rms)
            print(all_patcor)
        mean_patcor.append(np.mean(all_patcor))
        mean_rms.append(np.mean(all_rms))

    mean_rms = np.array(mean_rms)
    jmin = mean_rms.argmin()
    mean_patcor = np.array(mean_patcor)
    jmin2 = mean_patcor.argmax()

    if jmin != jmin2:
        print('WARNING: bad matching. Best permutation with RMS is {}, with patcor is {}'.format(perms[jmin], perms[jmin2]))

    return np.array(perms[jmin])


def calc_RMS_and_patcor(clusters_1, clusters_2):
    """
    Computes the distance and cosine of the angle between two sets of clusters.
    It is assumed the clusters are in matrix/vector form.

    Works both in the physical space (with cluster patterns) and with the cluster PCs.

    IMPORTANT!! It assumes the clusters are already ordered. (compares 1 with 1, ecc..)
    """

    et = []
    patcor = []

    for c1, c2 in zip(clusters_1, clusters_2):
        dist = LA.norm(c1-c2)
        et.append(dist)
        cosin = cosine(c1,c2)
        patcor.append(cosin)

    return np.array(et), np.array(patcor)


def find_cluster_minmax(PCs, centroids, labels):
    """
    Finds closer and further points from the centroid in each cluster.
    """

    ens_mindist = []
    ens_maxdist = []
    for nclus in range(numclus):
        for ens in range(numens):
            normens = centroids[nclus,:] - PCs[ens,:]
            norm[nclus,ens] = math.sqrt(sum(normens**2))

        ens_mindist.append((np.argmin(norm[nclus,:]), norm[nclus].min()))
        ens_maxdist.append((np.argmax(norm[nclus,:]), norm[nclus].max()))

    return ens_mindist, ens_maxdist


def calc_composite_map(var, mask):
    """
    Calculates the composite, according to mask (boolean array). Var is assumed to be 3D.
    """

    pattern = np.mean(var[mask, ...], axis = 0)
    # pattern_std = np.std(var[mask, ...], axis = 0)

    return pattern #, pattern_std


def calc_regime_residtimes(indices, dates = None, count_incomplete = True, skip_singleday_pause = True):
    """
    Calculates residence times given a set of indices indicating cluster numbers.

    For each cluster, the observed set of residence times is given and a transition probability is calculated.

    < dates > : list of datetime objects (or datetime array). If set, the code eliminates spourious transitions between states belonging to two different seasons.

    < count_incomplete > : counts also residence periods that terminate with the end of the season and not with a transition to another cluster.
    < skip_singleday_pause > : If a regime stops for a single day and then starts again, the two periods will be summed on. The single day is counted as another regime's day, to preserve the total number.
    """
    indices = np.array(indices)
    numclus = int(indices.max() + 1)
    numseq = np.arange(len(indices))

    resid_times = []
    regime_nums = []
    if dates is None:
        for clu in range(numclus):
            clu_resids = []
            clu_num_reg = []
            okclu = indices == clu
            init = False

            for el, num in zip(okclu, numseq):
                if el:
                    if not init:
                        init = True
                        num_days = 1
                        num_reg = [num]
                    else:
                        num_days += 1
                        num_reg.append(num)
                else:
                    if init:
                        clu_resids.append(num_days)
                        clu_num_reg.append((num_reg[0], num_reg[-1]))
                        init = False

            resid_times.append(np.array(clu_resids))
            regime_nums.append(np.array(clu_num_reg))
    else:
        dates = pd.to_datetime(dates)
        duday = pd.Timedelta('2 days 00:00:00')
        regime_dates = []
        for clu in range(numclus):
            clu_resids = []
            clu_dates_reg = []
            clu_num_reg = []
            okclu = indices == clu
            init = False
            pause = False

            old_date = dates[0]
            for el, dat, num in zip(okclu, dates, numseq):
                #print(dat)
                if dat - old_date > duday:
                    #print('new season\n')
                    if init and count_incomplete:
                        #print('count incompleeeete \n')
                        clu_resids.append(num_days)
                        clu_dates_reg.append((date_reg[0], date_reg[-1]))
                        clu_num_reg.append((num_reg[0], num_reg[-1]))
                    init = False

                if el:
                    if not init:
                        #print('new regime period\n')
                        init = True
                        num_days = 1
                        date_reg = [dat]
                        num_reg = [num]
                        #clu_dates_reg.append(dat)
                    else:
                        #print('+1\n')
                        num_days += 1
                        date_reg.append(dat)
                        num_reg.append(num)
                        if pause:
                            # regime started again after one day pause
                            pause = False
                else:
                    if skip_singleday_pause:
                        if init and not pause:
                            # first day of pause
                            pause = True
                        elif init and pause:
                            # this is the second day of pause, closing regime period
                            clu_resids.append(num_days)
                            clu_dates_reg.append((date_reg[0], date_reg[-1]))
                            clu_num_reg.append((num_reg[0], num_reg[-1]))
                            init = False
                            pause = False
                    else:
                        if init:
                            clu_resids.append(num_days)
                            clu_dates_reg.append((date_reg[0], date_reg[-1]))
                            clu_num_reg.append((num_reg[0], num_reg[-1]))
                            init = False

                old_date = dat

            resid_times.append(np.array(clu_resids))
            regime_dates.append(np.array(clu_dates_reg))
            regime_nums.append(np.array(clu_num_reg))

    if dates is None:
        return np.array(resid_times)
    else:
        return np.array(resid_times), np.array(regime_dates), np.array(regime_nums)


def calc_days_event(labels, resid_times, regime_nums):
    """
    Returns two quantities. For each point, the number of days from the beginning of the cluster event (first day, second day, ...) and the total number of days of the event the point belongs to.
    """
    days_event = np.zeros(len(labels))
    length_event = np.zeros(len(labels))

    numclus = len(resid_times)

    for clu in range(numclus):
        ok_nums = regime_nums[clu]
        ok_times = resid_times[clu]

        okclu = labels == clu
        indici = np.arange(len(labels))[okclu]
        for ind in indici:
            imi1 = np.argmin(abs(ok_nums[:,0]-ind))
            val1 = abs(ok_nums[imi1,0]-ind)
            imi2 = np.argmin(abs(ok_nums[:,1]-ind))
            val2 = abs(ok_nums[imi2,1]-ind)
            if val1 <= val2:
                imi = imi1
            else:
                imi = imi2
            #print('p', ind, ok_nums[imi, :])
            if ok_nums[imi,0] <= ind and ok_nums[imi,1] > ind:
                print(ok_nums[imi,0], ind, ok_nums[imi,1])
                days_event[ind] = ind-ok_nums[imi,0]+1
                length_event[ind] = ok_times[imi]

    return days_event, length_event


def regime_filter_long(labels, dates, days_thres = 4):
    """
    Filters the regime label series keeping only events lasting at least days_event days.
    The other labels are set to -1.
    """

    resid_times, regime_dates, regime_nums = calc_regime_residtimes(labels, dates)
    days_event, length_event = calc_days_event(labels, resid_times, regime_nums)

    oklabs = length_event >= days_thres
    filt_labels = dcopy(labels)
    filt_labels[~oklabs] = -1

    return filt_labels


def calc_regime_transmatrix(n_ens, indices_set, dates_set, max_days_between = 3, filter_longer_than = 1, filter_shorter_than = None):
    """
    This calculates the probability for the regime transitions to other regimes. A matrix is produced with residence probability on the diagonal. Works with multimember ensemble.

    < n_ens > : number of ensemble members.
    < indices_set > : indices array or list of indices array, one per each ensemble member.
    < dates_set > : dates array or list of dates array, one per each ensemble member.

    < filter_longer_than > : excludes residende periods shorter than # days. If set to 0, single day transition are considered. Default is 1, so that single day regime periods are skipped.
    < filter_shorter_than > : excludes residende periods longer than # days.
    < max_days_between > : maximum number of days before the next regime to consider the transition.
    """

    if n_ens > 1:
        for ens in range(n_ens):
            trans_matrix_ens, _ = count_regime_transitions(indices_set[ens], dates_set[ens], max_days_between = max_days_between, filter_longer_than = filter_longer_than, filter_shorter_than = filter_shorter_than)
            if ens == 0:
                trans_matrix = trans_matrix_ens
            else:
                trans_matrix += trans_matrix_ens
    else:
        trans_matrix, _ = count_regime_transitions(indices_set, dates_set, max_days_between = max_days_between, filter_longer_than = filter_longer_than, filter_shorter_than = filter_shorter_than)

    for n in range(trans_matrix.shape[0]):
        trans_matrix[n, :] = trans_matrix[n, :]/np.sum(trans_matrix[n, :])

    return trans_matrix


def find_transition_pcs(n_ens, indices_set, dates_set, pcs_set, fix_length = 2, filter_longer_than = 1, filter_shorter_than = None, skip_persistence = False, only_direct = False):
    """
    This finds the pcs corresponding to regime transitions or to regime residences. All are saved in a matrix-like format of shape numclus x numclus. Works with multimember ensemble.

    < n_ens > : number of ensemble members.
    < indices_set > : indices array or list of indices arrays, one per each ensemble member.
    < dates_set > : dates array or list of dates arrays, one per each ensemble member.
    < pcs_set > : pcs array or list of pcs arrays, one per each ensemble member.

    < filter_longer_than > : excludes residende periods shorter than # days. If set to 0, single day transition are considered. Default is 1, so that single day regime periods are skipped.
    < filter_shorter_than > : excludes residende periods longer than # days.

    < fix_length > : all transitions are vector of pcs of fixed length. sets max_days_between to fix_length-1. Default is 2, only direct transitions are considered.
    < only_direct > : regardless of fix_length, only direct transitions are considered.
    """

    if n_ens == 1:
        dates_set = [dates_set]
        indices_set = [indices_set]
        pcs_set = [pcs_set]

    if only_direct:
        max_days_between = 1
    else:
        max_days_between = fix_length-1

    for ens in range(n_ens):
        # print('ens member {}\n'.format(ens))
        trans_matrix_ens, trans_matrix_nums_ens = count_regime_transitions(indices_set[ens], dates_set[ens], max_days_between = max_days_between, filter_longer_than = filter_longer_than, filter_shorter_than = filter_shorter_than)
        if ens == 0:
            trans_matrix = np.empty(trans_matrix_ens.shape, dtype=object)
            numclus = trans_matrix.shape[0]
            for i in range(numclus):
                for j in range(numclus):
                    trans_matrix[i,j] = []

        for i in range(numclus):
            for j in range(numclus):
                # if i == j and skip_persistence:
                #     print((i,j), '--> skipping..')
                #     continue
                # else:
                #     print((i,j))
                numsok = trans_matrix_nums_ens[i,j]
                for (okin, okou) in numsok:
                    while okou-okin+1 < fix_length:
                        okin -= 1
                        okou += 1
                    if okou-okin+1 > fix_length:
                        okou -= 1
                    trans_matrix[i,j].append(pcs_set[ens][okin:okou+1, :])

    return trans_matrix


def count_regime_transitions(indices, dates, filter_longer_than = 1, filter_shorter_than = None, max_days_between = 3):
    """
    This calculates the probability for the regime transitions to other regimes. A matrix is produced with residence probability on the diagonal. count_incomplete is defaulted to True, but the last season' regime is only used for considering the transition from the previous one.

    < filter_longer_than > : excludes residende periods shorter than # days. If set to 0, single day transition are considered. Default is 1, so that single day regime periods are skipped.
    < filter_shorter_than > : excludes residende periods longer than # days.
    < max_days_between > : maximum number of days before the next regime to consider the transition.
    """

    if filter_longer_than < 1:
        skip_singleday_pause = False
    else:
        skip_singleday_pause = True

    resid_times, dates_reg, num_reg = calc_regime_residtimes(indices, dates = dates, count_incomplete = True, skip_singleday_pause = skip_singleday_pause)

    numclus = len(resid_times)

    # Filtering day length
    filt_resid_times = []
    #filt_dates_init = []
    #filt_dates_fin = []
    filt_num_init = []
    filt_num_fin = []
    for reg, rsd, dat, nu in zip(range(numclus), resid_times, dates_reg, num_reg):
        oks = rsd > filter_longer_than
        if filter_shorter_than is not None:
            oks = oks & (rsd < filter_shorter_than)
        filt_resid_times.append(rsd[oks])
        #filt_dates_init.append([da[0] for da in dat[oks]])
        #filt_dates_fin.append([da[1] for da in dat[oks]])
        filt_num_init.append([da[0] for da in nu[oks]])
        filt_num_fin.append([da[1] for da in nu[oks]])

    # metto in ordine i periodi e le date
    #all_dats_ini = np.concatenate(filt_dates_init)
    #all_dats_fin = np.concatenate(filt_dates_fin)
    all_nums_ini = np.concatenate(filt_num_init)
    all_nums_fin = np.concatenate(filt_num_fin)
    all_rsd = np.concatenate(filt_resid_times)
    all_clus = np.concatenate([num*np.ones(len(filt_resid_times[num]), dtype = int) for num in range(numclus)])

    #sortdat = all_dats_ini.argsort()
    sortdat = all_nums_ini.argsort()
    #all_dats_ini = all_dats_ini[sortdat]
    #all_dats_fin = all_dats_fin[sortdat]
    all_nums_ini = all_nums_ini[sortdat]
    all_nums_fin = all_nums_fin[sortdat]
    all_rsd = all_rsd[sortdat]
    all_clus = all_clus[sortdat]

    trans_matrix = np.zeros((numclus, numclus))
    #trans_matrix_dates = np.empty((numclus, numclus), dtype = object)
    trans_matrix_nums = np.empty((numclus, numclus), dtype = object)
    for i in range(numclus):
        for j in range(numclus):
            #trans_matrix_dates[i,j] = []
            trans_matrix_nums[i,j] = []

    #timedel = pd.Timedelta('{} days'.format(max_days_between))

    n_tot = len(all_rsd)
    #for num, day_in, day_out, lenday, reg in zip(range(n_tot), all_dats_ini, all_dats_fin, all_rsd, all_clus):
    for num, day_in, day_out, lenday, reg in zip(range(n_tot), all_nums_ini, all_nums_fin, all_rsd, all_clus):
        i = num + 1
        if i >= n_tot: break

        #day_new = all_dats_ini[i]
        #if day_new - day_out > timedel: continue
        day_new = all_nums_ini[i]
        if day_new - day_out > max_days_between: continue

        #trans_matrix[reg,reg] += len(pd.date_range(day_in, day_out))-1-filter_longer_than
        trans_matrix[reg,reg] += day_out - day_in - filter_longer_than # devo fare anche -1?
        #trans_matrix_dates[reg,reg].append((day_in, day_out))
        trans_matrix_nums[reg,reg].append((day_in, day_out))
        #print('perm', reg, len(pd.date_range(day_in, day_out))-1-filter_longer_than, (day_in-all_dats_ini[0]).days)

        reg_new = all_clus[num+1]
        #print('trans', reg, reg_new, (day_new-all_dats_ini[0]).days)
        trans_matrix[reg,reg_new] += 1
        #trans_matrix_dates[reg,reg_new].append((day_out, day_new))
        trans_matrix_nums[reg,reg_new].append((day_out, day_new))

    return trans_matrix, trans_matrix_nums


def rotate_space_interclus_section_3D(centroids, clusA, clusB, pcs, transitions = None):
    """
    Rotates the basis so as to have one axis match the intercluster vector linking clusA and clusB. This is done to have the space which maximizes the division between the two clusters and better visualize the dynamics.

    Returns the projection of the new basis set onto the old one and the new set of rotated PCs.

    All is done in 3D geometry. The pcs are returned in 3D as well.
    """

    icvec = centroids[clusB][:3]-centroids[clusA][:3]

    #Find closest axis to interclus vector
    iax = np.argmin(abs(icvec))
    orig = np.zeros(3)
    orig[iax] = 1

    # Find the rotation that moves orig to icvec
    rot = get_rotation_3D(orig, icvec)
    invrot = rot.T

    new_base = new_basis_set_3D(rot)
    new_pcs = np.array([np.dot(invrot, pc[:3]) for pc in pcs])

    if transitions is not None:
        new_trans = []
        for trans in transitions:
            new_trans.append(np.array([np.dot(invrot, pc[:3]) for pc in trans]))
        return new_base, new_pcs, new_trans
    else:
        return new_base, new_pcs


def calc_pdf(data, bnd_width = None):
    """
    Calculates pdf using Gaussian kernel on a set of data (scipy gaussian_kde). Returns a function that can be evaluated on a grid for plotting purposes.
    """

    k = stats.kde.gaussian_kde(data, bw_method = bnd_width)

    return k

def count_occurrences(data, num_range = None, convert_to_frequency = True):
    """
    Counts occurrences of each integer value inside data. num_range defaults to (min, max). If num_range is specified, the last count is the sum of all longer occurrences.
    """
    if num_range is None:
        num_range = (np.min(data), np.max(data))

    numarr = np.arange(num_range[0], num_range[1]+1)
    unique, caun = np.unique(data, return_counts=True)
    cosi = dict(zip(unique, caun))

    counts = []
    for el in numarr:
        if el in cosi.keys():
            counts.append(cosi[el])
        else:
            counts.append(0)

    kiavi = np.array(list(cosi.keys()))
    pius_keys = kiavi[kiavi > num_range[1]]
    vals_pius = np.sum(np.array([cosi[k] for k in pius_keys]))
    counts.append(vals_pius)
    numarr = np.append(numarr, num_range[1]+1)

    counts = np.array(counts)
    if convert_to_frequency:
        counts = 1.0*counts/np.sum(counts)

    return numarr, counts


def compute_centroid_distance(PCs, centroids, labels):
    """
    Calculates distances of pcs to centroid assigned by label.
    """
    distances = []
    for pc, lab in zip(PCs, labels):
        distances.append(distance(pc, centroids[lab]))

    return np.array(distances)


def compute_clusterpatterns(var, labels):
    """
    Calculates the cluster patterns.
    """

    labels = np.array(labels)
    numclus = int(np.max(labels)+1)
    #print('yo',labels.shape)

    cluspatt = []
    freqs = []
    for nclus in range(numclus):
        mask = labels == nclus
        freq_perc = 100.0*np.sum(mask)/len(labels)
        freqs.append(freq_perc)
        print('CLUSTER {} ---> {:4.1f}%\n'.format(nclus, freq_perc))
        cluspattern = np.mean(var[mask, ...], axis=0)
        cluspatt.append(cluspattern)

    cluspatt = np.stack(cluspatt)

    return cluspatt


def clus_eval_indexes(PCs, centroids, labels):
    """
    Computes clustering evaluation indexes, as the Davies-Bouldin Index, the Dunn Index, the optimal variance ratio and the Silhouette value. Also computes cluster sigmas and distances.
    """
    ### Computing clustering evaluation Indexes
    numclus = len(centroids)
    inertia_i = np.empty(numclus)
    for i in range(numclus):
        lab_clus = labels == i
        inertia_i[i] = np.sum([np.sum((pcok-centroids[i])**2) for pcok in PCs[lab_clus]])

    clus_eval = dict()
    clus_eval['Indexes'] = dict()

    # Optimal ratio

    n_clus = np.empty(numclus)
    for i in range(numclus):
        n_clus[i] = np.sum(labels == i)

    mean_intra_clus_variance = np.sum(inertia_i)/len(labels)

    dist_couples = dict()
    coppie = list(itt.combinations(list(range(numclus)), 2))
    for (i,j) in coppie:
        dist_couples[(i,j)] = LA.norm(centroids[i]-centroids[j])

    mean_inter_clus_variance = np.sum(np.array(dist_couples.values())**2)/len(coppie)

    clus_eval['Indexes']['Inter-Intra Variance ratio'] = mean_inter_clus_variance/mean_intra_clus_variance

    sigma_clusters = np.sqrt(inertia_i/n_clus)
    clus_eval['Indexes']['Inter-Intra Distance ratio'] = np.mean(dist_couples.values())/np.mean(sigma_clusters)

    # Davies-Bouldin Index
    R_couples = dict()
    for (i,j) in coppie:
        R_couples[(i,j)] = (sigma_clusters[i]+sigma_clusters[j])/dist_couples[(i,j)]

    DBI = 0.
    for i in range(numclus):
        coppie_i = [coup for coup in coppie if i in coup]
        Di = np.max([R_couples[cop] for cop in coppie_i])
        DBI += Di

    DBI /= numclus
    clus_eval['Indexes']['Davies-Bouldin'] = DBI

    # Dunn Index

    Delta_clus = np.empty(numclus)
    for i in range(numclus):
        lab_clus = labels == i
        distances = [LA.norm(pcok-centroids[i]) for pcok in PCs[lab_clus]]
        Delta_clus[i] = np.sum(distances)/n_clus[i]

    clus_eval['Indexes']['Dunn'] = np.min(dist_couples.values())/np.max(Delta_clus)

    clus_eval['Indexes']['Dunn 2'] = np.min(dist_couples.values())/np.max(sigma_clusters)

    # Silhouette
    sils = []
    for ind, el, lab in zip(range(len(PCs)), PCs, labels):
        lab_clus = labels == lab
        lab_clus[ind] = False
        ok_Pcs = PCs[lab_clus]
        a = np.sum([LA.norm(okpc - el) for okpc in ok_Pcs])/n_clus[lab]

        bs = []
        others = list(range(numclus))
        others.remove(lab)
        for lab_b in others:
            lab_clus = labels == lab_b
            ok_Pcs = PCs[lab_clus]
            b = np.sum([LA.norm(okpc - el) for okpc in ok_Pcs])/n_clus[lab_b]
            bs.append(b)

        b = np.min(bs)
        sils.append((b-a)/max([a,b]))

    sils = np.array(sils)
    sil_clus = []
    for i in range(numclus):
        lab_clus = labels == i
        popo = np.sum(sils[lab_clus])/n_clus[i]
        sil_clus.append(popo)

    siltot = np.sum(sil_clus)/numclus

    clus_eval['Indexes']['Silhouette'] = siltot
    clus_eval['clus_silhouettes'] = sil_clus

    clus_eval['Indexes']['Dunn2/DB'] = clus_eval['Indexes']['Dunn 2']/clus_eval['Indexes']['Davies-Bouldin']

    clus_eval['R couples'] = R_couples
    clus_eval['Inter cluster distances'] = dist_couples
    clus_eval['Sigma clusters'] = sigma_clusters

    return clus_eval

#######################################################
#
###     Geometrical functions
#
#######################################################

def normalize(vector):
    norm_vector = vector/LA.norm(vector)
    return norm_vector


def orthogonal_plane(vector, point):
    """
    Finds the plane orthogonal to vector and passing from point. Returns a function plane(x,y). If the vector is in the (x,y) plane, returns a function plane(x,z).
    """
    line = normalize(vector)

    if line[2] != 0:
        #print('Returning plane as function of (x,y) couples')
        def plane(x,y):
            z = point[2] - ( line[0]*(x-point[0]) + line[1]*(y-point[1]) ) / line[2]
            return np.array([x,y,z])
    else:
        print('vector is horizontal. Returning plane as function of (x,z) couples\n')
        def plane(x,z):
            y = point[1] - line[0] * (x-point[0]) / line[1]
            return np.array([x,y,z])

    return plane


def skewsymmat(v):
    """
    Returns the skew symmetric matric produced by vector.
    """
    matr = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    return matr


def rotation_matrix_3D(axis, angle):
    """
    Calculates the rotation matrix for rotation of angle around axis, using the Euler-Rodriguez formula.
    angle in radians.
    """
    idm = np.array([[1,0,0],[0,1,0],[0,0,1]])
    kn = normalize(axis)

    kx = skewsymmat(kn)
    kxkx = np.dot(kx,kx)

    rot = idm + np.sin(angle)*kx + (1-np.cos(angle))*kxkx

    return rot


def get_rotation_3D(a, b):
    """
    Finds the rotation matrix that transforms vector a in vector b. (the simplest)
    """

    k = np.cross(a,b)
    angle = np.arccos(np.dot(a,b)/(LA.norm(a)*LA.norm(b)))

    rot = rotation_matrix_3D(k, angle)

    return rot


def new_basis_set_3D(rot):
    """
    Components of the new basis set obtained through a 3D rotation in the original eof base.
    """
    basis = []
    for el in range(3):
        basis.append(rot[:,el])

    return basis


def project_vector_on_basis(vector, basis):
    """
    Gives a set of components of vector in the specified basis.
    """
    vec_comp = []
    for bavec in basis:
        vec_comp.append(np.dot(vector.flatten(), bavec.flatten()))
    vec_comp = np.array(vec_comp)

    return vec_comp


def project_set_on_new_basis(old_components_set, old_basis, new_basis):
    """
    Projects the old_components_set onto the new_basis.
    < old_basis > : the old set of basis orthogonal arrays (numpcs).
    < new_basis > : the new set of basis orthogonal arrays (numpcs).

    < old_components_set > : the set of components to be projected (n_points, numpcs).
    """
    change_basis_mat = change_of_base_matrix(old_basis, new_basis)

    new_components_set = []
    for vec in old_components_set:
        new_components_set.append(change_basis_mat.dot(vec))

    new_components_set = np.array(new_components_set)

    return new_components_set


def change_of_base_matrix(old_basis, new_basis):
    """
    Finds the matrix for change of base to the new_basis.
    < old_basis > : the old set of basis orthogonal arrays (numpcs).
    < new_basis > : the new set of basis orthogonal arrays (numpcs).
    """
    # find the components of new basis vectors onto the old basis
    new_basis_comp = []
    for bavec in new_basis:
        new_basis_comp.append(project_vector_on_basis(bavec, old_basis))
    new_basis_comp = np.array(new_basis_comp).T # le cui COLONNE sono i vettori della base nuova nelle coordinate della vecchia

    # find the change of basis matrix
    change_basis_mat = LA.inv(new_basis_comp)

    return change_basis_mat

# def base_set(plane):
#     """
#     Finds an arbitrary base to project onto inside plane.
#     """


#######################################################
#
###     Plots and visualization
#
#######################################################

def color_brightness(color):
    return (color[0] * 299 + color[1] * 587 + color[2] * 114)/1000

def makeRectangle(area, npo = 20):
    """
    Produces a shapely polygon to properly plot a lat/lon rectangle on a map.
    lonW, lonE, latS, latN = area
    """
    lonW, lonE, latS, latN = area
    rlons = np.concatenate([np.tile(lonW, npo), np.linspace(lonW, lonE, npo), np.tile(lonE, npo), np.linspace(lonE, lonW, npo)])
    rlats = np.concatenate([np.linspace(latS,latN,npo), np.tile(latN, npo), np.linspace(latN,latS,npo), np.tile(latS, npo)])
    ring = LinearRing(list(zip(rlons, rlats)))

    return ring

def color_set(n, cmap = 'nipy_spectral', bright_thres = None, full_cb_range = False, only_darker_colors = False, use_seaborn = True, sns_palette = 'hls'):
    """
    Gives a set of n well chosen (hopefully) colors, darker than bright_thres. bright_thres ranges from 0 (darker) to 1 (brighter).

    < full_cb_range > : if True, takes all cb values. If false takes the portion 0.05/0.95.
    """
    if not use_seaborn:
        if bright_thres is None:
            if only_darker_colors:
                bright_thres = 0.6
            else:
                bright_thres = 1.0

        cmappa = cm.get_cmap(cmap)
        colors = []

        if full_cb_range:
            valori = np.linspace(0.0,1.0,n)
        else:
            valori = np.linspace(0.05,0.95,n)

        for cos in valori:
            colors.append(cmappa(cos))

        for i, (col,val) in enumerate(zip(colors, valori)):
            if color_brightness(col) > bright_thres:
                # Looking for darker color
                col2 = cmappa(val+1.0/(3*n))
                col3 = cmappa(val-1.0/(3*n))
                colori = [col, col2, col3]
                brighti = np.array([color_brightness(co) for co in colori]).argmin()
                colors[i] = colori[brighti]
    else:
        # Calling the default seaborn palette
        colors = sns.color_palette(sns_palette, n)
        if sns_palette == 'Paired':
            max_sns_paired = 10
            other_colors = ['burlywood', 'saddlebrown', 'palegoldenrod', 'goldenrod', 'lightslategray', 'darkslategray', 'orchid', 'darkmagenta']
            if n > max_sns_paired and n <= max_sns_paired + len(other_colors):
                for i in range(n - max_sns_paired):
                    colors[max_sns_paired+i] = cm.colors.ColorConverter.to_rgb(other_colors[i])

    return colors


def plot_mapc_on_ax(ax, data, lat, lon, proj, cmappa, cbar_range, n_color_levels = 21, draw_contour_lines = False, n_lines = 5, bounding_lat = None, plot_margins = None, add_hatching = None, hatch_styles = ['', '', '...'], hatch_levels = [0.2, 0.8], colors = None, clevels = None, add_rectangles = None, draw_grid = False):
    """
    Plots field contours on the axis of a figure.

    < data >: the field to plot
    < lat, lon >: latitude and longitude

    < proj >: The ccrs transform used for the plot.
    < cmappa >: color map.
    < cbar_range >: limits of the color bar.

    < n_color_levels >: number of color levels.
    < draw_contour_lines >: draw lines in addition to the color levels?
    < n_lines >: number of lines to draw.

    < add_hatching > : bool mask. Where True the map is hatched.
    """

    if clevels is None:
        clevels = np.linspace(cbar_range[0], cbar_range[1], n_color_levels)
    print(clevels)
    print(np.min(data), np.max(data))

    ax.set_global()
    ax.coastlines(linewidth = 2)
    if draw_grid:
        gl = ax.gridlines(crs = ccrs.PlateCarree(), draw_labels = False, linewidth = 1, color = 'gray', alpha = 0.5, linestyle = '-')

    cyclic = False
    if max(lon)+10. > 360 and min(lon)-10. < 0.:
        print('Adding cyclic point\n')
        cyclic = True
        #lon = np.append(lon, 360)
        #data = np.c_[data,data[:,0]]
        lon_o = lon
        data, lon = cutil.add_cyclic_point(data, coord = lon)

        if add_hatching is not None:
            add_hatching, lon = cutil.add_cyclic_point(add_hatching, coord = lon_o)

    xi,yi = np.meshgrid(lon,lat)

    map_plot = ax.contourf(xi, yi, data, clevels, cmap = cmappa, transform = ccrs.PlateCarree(), extend = 'both', corner_mask = False, colors = colors)
    if add_hatching is not None:
        print('adding hatching')
        #pickle.dump([lat, lon, add_hatching], open('hatchdimerda.p','wb'))
        hatch = ax.contourf(xi, yi, add_hatching, levels = hatch_levels, transform = ccrs.PlateCarree(), hatches = hatch_styles, colors = 'none')

    nskip = len(clevels)//n_lines - 1
    if draw_contour_lines:
        map_plot_lines = ax.contour(xi, yi, data, clevels[::nskip], colors = 'k', transform = ccrs.PlateCarree(), linewidth = 0.5)

    if isinstance(proj, ccrs.PlateCarree):
        if plot_margins is not None:
            map_set_extent(ax, proj, bnd_box = plot_margins)
    else:
        map_set_extent(ax, proj, bounding_lat = bounding_lat)

    if add_rectangles is not None:
        if type(add_rectangles[0]) in [list, tuple, np.ndarray]:
            rectangle_list = add_rectangles
        else:
            rectangle_list = [add_rectangles]
        colors = color_set(len(rectangle_list))
        if len(rectangle_list) == 1: colors = ['white']
        for rect, col in zip(rectangle_list, colors):
            # ax.add_patch(mpatches.Rectangle(xy = [rect[0], rect[2]], width = rect[1]-rect[0], height = rect[3]-rect[2], facecolor = 'none', edgecolor = col, alpha = 1.0, transform = proj, linewidth = 2.0, zorder = 20))
            ring = makeRectangle(rect)
            ax.add_geometries([ring], ccrs.PlateCarree(), facecolor='none', edgecolor=col, linewidth = 2.0)

    return map_plot


def get_cbar_range(data, symmetrical = False, percentiles = (0,100), n_color_levels = None):
    mi = np.nanpercentile(data, percentiles[0])
    ma = np.nanpercentile(data, percentiles[1])
    if symmetrical:
        oko = max(abs(mi), abs(ma))
        if n_color_levels is not None:
            spi = 2*oko/(n_color_levels-1)
            spi_ok = np.ceil(spi*100)/100
            oko2 = spi_ok*(n_color_levels-1)/2
        else:
            oko2 = oko
        oko1 = -oko2
    else:
        oko1 = mi
        oko2 = ma

    return (oko1, oko2)


def adjust_ax_scale(axes, sel_axis = 'both'):
    """
    Given a set of axes, uniformizes the scales.
    < sel_axis > : 'x', 'y' or 'both' (default)
    """

    if sel_axis == 'x' or sel_axis == 'both':
        limits_min = []
        limits_max = []
        for ax in axes:
            limits_min.append(ax.get_xlim()[0])
            limits_max.append(ax.get_xlim()[1])

        maxlim = np.max(limits_max)
        minlim = np.min(limits_min)
        for ax in axes:
            ax.set_xlim((minlim,maxlim))

    if sel_axis == 'y' or sel_axis == 'both':
        limits_min = []
        limits_max = []
        for ax in axes:
            limits_min.append(ax.get_ylim()[0])
            limits_max.append(ax.get_ylim()[1])

        maxlim = np.max(limits_max)
        minlim = np.min(limits_min)
        for ax in axes:
            ax.set_ylim((minlim,maxlim))

    return


def adjust_color_scale(color_maps):
    """
    Given a set of color_maps, uniformizes the color scales.
    """

    limits_min = []
    limits_max = []
    for co in color_maps:
        limits_min.append(co.get_clim()[0])
        limits_max.append(co.get_clim()[1])

    maxlim = np.max(limits_max)
    minlim = np.min(limits_min)
    for co in color_maps:
        co.set_clim((minlim,maxlim))

    return

def def_projection(visualization, central_lat_lon):
    """
    Defines projection for the map plot.
    """
    if central_lat_lon is not None:
        (clat, clon) = central_lat_lon
    else:
        clon = 0.

    if visualization == 'standard' or visualization == 'Standard':
        proj = ccrs.PlateCarree()
    elif visualization == 'polar' or visualization == 'Npolar' or visualization == 'npolar':
        proj = ccrs.Orthographic(central_longitude = clon, central_latitude = 90)
    elif visualization == 'Spolar' or visualization == 'spolar':
        proj = ccrs.Orthographic(central_longitude = clon, central_latitude = -90)
    elif visualization == 'Nstereo' or visualization == 'stereo' or visualization == 'nstereo':
        proj = ccrs.NorthPolarStereo()#central_longitude=clon)
    elif visualization == 'Sstereo' or visualization == 'sstereo':
        proj = ccrs.SouthPolarStereo()#central_longitude=clon)
    elif visualization == 'Robinson' or visualization == 'robinson':
        proj = ccrs.Robinson(central_longitude = clon)
    else:
        raise ValueError('visualization {} not recognised. Only standard, Npolar (or polar), Spolar, Nstereo (or stereo), Sstereo accepted'.format(visualization))

    return proj


def map_set_extent(ax, proj, bnd_box = None, bounding_lat = None):
    """
    Reduces ax to the required lat/lon box.

    < bnd_box >: 4-element tuple or list. (Wlon, Elon, Slat, Nlat) for standard visualization.
    < bounding_lat >: for polar plots, the equatorward boundary latitude.
    """

    if bnd_box is not None:
        if isinstance(proj, ccrs.PlateCarree):
            print(bnd_box)
            ax.set_extent(bnd_box, crs = ccrs.PlateCarree())

    if bounding_lat is not None:
        if (isinstance(proj, ccrs.Orthographic) or isinstance(proj, ccrs.Stereographic)):
            if bounding_lat > 0:
                ax.set_extent((-180, 180, bounding_lat, 90), crs = ccrs.PlateCarree())
            else:
                ax.set_extent((-180, 180, -90, bounding_lat), crs = ccrs.PlateCarree())

    # theta = np.linspace(0, 2*np.pi, 100)
    # center, radius = [0.5, 0.5], 0.2
    # verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    # circle = mpath.Path(verts * radius + center)
    #
    # ax2.set_boundary(circle, transform=ax2.transAxes)

    return


def plot_map_contour(data, lat, lon, filename = None, visualization = 'standard', central_lat_lon = None, cmap = 'RdBu_r', title = None, xlabel = None, ylabel = None, cb_label = None, cbar_range = None, plot_anomalies = True, n_color_levels = 21, draw_contour_lines = False, n_lines = 5, color_percentiles = (0,100), figsize = (8,6), bounding_lat = 30, plot_margins = None, add_rectangles = None, draw_grid = False):
    """
    Plots a single map to a figure.

    < data >: the field to plot
    < lat, lon >: latitude and longitude
    < filename >: name of the file to save the figure to. If None, the figure is just shown.

    < visualization >: 'standard' calls PlateCarree cartopy map, 'polar' calls Orthographic map, 'stereo' calls NorthPolarStereo. Also available Spolar and Sstereo.
    < central_lat_lon >: Tuple, (clat, clon). Is needed only for Orthographic plots. If not given, the mean lat and lon are taken.
    < cmap >: name of the color map.
    < cbar_range >: limits of the color bar.

    < plot_anomalies >: if True, the colorbar is symmetrical, so that zero corresponds to white. If cbar_range is set, plot_anomalies is set to False.
    < n_color_levels >: number of color levels.
    < draw_contour_lines >: draw lines in addition to the color levels?
    < n_lines >: number of lines to draw.


    """
    #if filename is None:
        #plt.ion()

    proj = def_projection(visualization, central_lat_lon)

    # Plotting figure
    fig4 = plt.figure(figsize = figsize)
    ax = plt.subplot(projection = proj)

    # Determining color levels
    cmappa = cm.get_cmap(cmap)

    if cbar_range is None:
        mi = np.percentile(data, color_percentiles[0])
        ma = np.percentile(data, color_percentiles[1])
        if plot_anomalies:
            # making a symmetrical color axis
            oko = max(abs(mi), abs(ma))
            spi = 2*oko/(n_color_levels-1)
            spi_ok = np.ceil(spi*100)/100
            oko2 = spi_ok*(n_color_levels-1)/2
            oko1 = -oko2
        else:
            oko1 = mi
            oko2 = ma
        cbar_range = (oko1, oko2)

    clevels = np.linspace(cbar_range[0], cbar_range[1], n_color_levels)

    map_plot = plot_mapc_on_ax(ax, data, lat, lon, proj, cmappa, cbar_range, n_color_levels = n_color_levels, draw_contour_lines = draw_contour_lines, n_lines = n_lines, bounding_lat = bounding_lat, plot_margins = plot_margins, add_rectangles = add_rectangles, draw_grid = draw_grid)

    title_obj = plt.title(title, fontsize=20, fontweight='bold')
    title_obj.set_position([.5, 1.05])

    cax = plt.axes([0.1, 0.11, 0.8, 0.05]) #horizontal
    cb = plt.colorbar(map_plot, cax=cax, orientation='horizontal')#, labelsize=18)
    cb.ax.tick_params(labelsize=14)
    cb.set_label(cb_label, fontsize=16)

    top    = 0.88  # the top of the subplots
    bottom = 0.20    # the bottom of the subplots
    left   = 0.02    # the left side
    right  = 0.98  # the right side
    hspace = 0.20   # height reserved for white space
    wspace = 0.05    # width reserved for blank space
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    # save the figure or show it
    if filename is not None:
        fig4.savefig(filename)
        # plt.close(fig4)

    return fig4


def plot_double_sidebyside(data1, data2, lat, lon, filename = None, visualization = 'standard', central_lat_lon = None, cmap = 'RdBu_r', title = None, xlabel = None, ylabel = None, cb_label = None, stitle_1 = 'data1', stitle_2 = 'data2', cbar_range = None, plot_anomalies = True, n_color_levels = 21, draw_contour_lines = False, n_lines = 5, color_percentiles = (0,100), use_different_grids = False, bounding_lat = 30, plot_margins = None, add_rectangles = None):
    """
    Plots multiple maps on a single figure (or more figures if needed).

    < data1, data2 >: the fields to plot
    < lat, lon >: latitude and longitude
    < filename >: name of the file to save the figure to. If more figures are needed, the others are named as filename_1.pdf, filename_2.pdf, ...

    < visualization >: 'standard' calls PlateCarree cartopy map, 'polar' calls Orthographic map.
    < central_lat_lon >: Tuple, (clat, clon). Is needed only for Orthographic plots. If not given, the mean lat and lon are taken.
    < cmap >: name of the color map.
    < cbar_range >: limits of the color bar.

    < plot_anomalies >: if True, the colorbar is symmetrical, so that zero corresponds to white. If cbar_range is set, plot_anomalies is set to False.
    < n_color_levels >: number of color levels.
    < draw_contour_lines >: draw lines in addition to the color levels?
    < n_lines >: number of lines to draw.
    < color_percentiles > : define the range of data to be represented in the color bar. e.g. (0,100) to full range, (5,95) to enhance features.

    < use_different_grids > : if True, lat and lon are read as 2-element lists [lat1, lat2] [lon1, lon2] which specify separately latitude and longitude of the two datasets. To be used if the datasets dimensions do not match.

    """

    #if filename is None:
    #    plt.ion()

    proj = def_projection(visualization, central_lat_lon)

    # Determining color levels
    cmappa = cm.get_cmap(cmap)


    if cbar_range is None:
        data = np.concatenate([data1.flatten(),data2.flatten()])
        mi = np.percentile(data, color_percentiles[0])
        ma = np.percentile(data, color_percentiles[1])
        if plot_anomalies:
            # making a symmetrical color axis
            oko = max(abs(mi), abs(ma))
            spi = 2*oko/(n_color_levels-1)
            spi_ok = np.ceil(spi*100)/100
            oko2 = spi_ok*(n_color_levels-1)/2
            oko1 = -oko2
        else:
            oko1 = mi
            oko2 = ma
        cbar_range = (oko1, oko2)

    clevels = np.linspace(cbar_range[0], cbar_range[1], n_color_levels)

    fig = plt.figure(figsize=(24,14))

    if use_different_grids:
        lat1, lat2 = lat
        lon1, lon2 = lon
    else:
        lat1 = lat
        lon1 = lon
        lat2 = lat
        lon2 = lon

    ax = plt.subplot(1, 2, 1, projection=proj)

    map_plot = plot_mapc_on_ax(ax, data1, lat1, lon1, proj, cmappa, cbar_range, n_color_levels = n_color_levels, draw_contour_lines = draw_contour_lines, n_lines = n_lines, bounding_lat = bounding_lat, plot_margins = plot_margins, add_rectangles = add_rectangles)
    ax.set_title(stitle_1, fontsize = 25)

    ax = plt.subplot(1, 2, 2, projection=proj)

    map_plot = plot_mapc_on_ax(ax, data2, lat2, lon2, proj, cmappa, cbar_range, n_color_levels = n_color_levels, draw_contour_lines = draw_contour_lines, n_lines = n_lines, bounding_lat = bounding_lat, plot_margins = plot_margins, add_rectangles = add_rectangles)
    ax.set_title(stitle_2, fontsize = 25)

    cax = plt.axes([0.1, 0.06, 0.8, 0.03])
    cb = plt.colorbar(map_plot,cax=cax, orientation='horizontal')
    cb.ax.tick_params(labelsize=18)
    cb.set_label(cb_label, fontsize=20)

    plt.suptitle(title, fontsize=35, fontweight='bold')

    plt.subplots_adjust(top=0.85)
    top    = 0.90  # the top of the subplots
    bottom = 0.13    # the bottom
    left   = 0.02    # the left side
    right  = 0.98  # the right side
    hspace = 0.20   # the amount of height reserved for white space between subplots
    wspace = 0.05    # the amount of width reserved for blank space between subplots
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    # save the figure or show it
    if filename is not None:
        fig.savefig(filename)
        # plt.close(fig)

    return fig


def plot_triple_sidebyside(data1, data2, lat, lon, filename = None, visualization = 'standard', central_lat_lon = None, cmap = 'RdBu_r', title = None, xlabel = None, ylabel = None, cb_label = None, stitle_1 = 'data1', stitle_2 = 'data2', cbar_range = None, plot_anomalies = True, n_color_levels = 21, draw_contour_lines = False, n_lines = 5, color_percentiles = (0,100), use_different_grids = False, bounding_lat = 30, plot_margins = None, add_rectangles = None):
    """
    Plots multiple maps on a single figure (or more figures if needed).

    < data1, data2 >: the fields to plot
    < lat, lon >: latitude and longitude
    < filename >: name of the file to save the figure to. If more figures are needed, the others are named as filename_1.pdf, filename_2.pdf, ...

    < visualization >: 'standard' calls PlateCarree cartopy map, 'polar' calls Orthographic map.
    < central_lat_lon >: Tuple, (clat, clon). Is needed only for Orthographic plots. If not given, the mean lat and lon are taken.
    < cmap >: name of the color map.
    < cbar_range >: limits of the color bar.

    < plot_anomalies >: if True, the colorbar is symmetrical, so that zero corresponds to white. If cbar_range is set, plot_anomalies is set to False.
    < n_color_levels >: number of color levels.
    < draw_contour_lines >: draw lines in addition to the color levels?
    < n_lines >: number of lines to draw.
    < color_percentiles > : define the range of data to be represented in the color bar. e.g. (0,100) to full range, (5,95) to enhance features.

    < use_different_grids > : if True, lat and lon are read as 2-element lists [lat1, lat2] [lon1, lon2] which specify separately latitude and longitude of the two datasets. To be used if the datasets dimensions do not match.

    """

    #if filename is None:
    #    plt.ion()

    proj = def_projection(visualization, central_lat_lon)

    # Determining color levels
    cmappa = cm.get_cmap(cmap)


    if cbar_range is None:
        data = np.concatenate([data1.flatten(),data2.flatten()])
        mi = np.percentile(data, color_percentiles[0])
        ma = np.percentile(data, color_percentiles[1])
        if plot_anomalies:
            # making a symmetrical color axis
            oko = max(abs(mi), abs(ma))
            spi = 2*oko/(n_color_levels-1)
            spi_ok = np.ceil(spi*100)/100
            oko2 = spi_ok*(n_color_levels-1)/2
            oko1 = -oko2
        else:
            oko1 = mi
            oko2 = ma
        cbar_range = (oko1, oko2)

    clevels = np.linspace(cbar_range[0], cbar_range[1], n_color_levels)

    fig = plt.figure(figsize=(36,14))

    if use_different_grids:
        lat1, lat2 = lat
        lon1, lon2 = lon
    else:
        lat1 = lat
        lon1 = lon
        lat2 = lat
        lon2 = lon

    ax = plt.subplot(1, 3, 1, projection=proj)

    map_plot = plot_mapc_on_ax(ax, data1, lat1, lon1, proj, cmappa, cbar_range, n_color_levels = n_color_levels, draw_contour_lines = draw_contour_lines, n_lines = n_lines, bounding_lat = bounding_lat, plot_margins = plot_margins, add_rectangles = add_rectangles)
    ax.set_title(stitle_1, fontsize = 25)

    ax = plt.subplot(1, 3, 2, projection=proj)

    map_plot = plot_mapc_on_ax(ax, data2, lat2, lon2, proj, cmappa, cbar_range, n_color_levels = n_color_levels, draw_contour_lines = draw_contour_lines, n_lines = n_lines, bounding_lat = bounding_lat, plot_margins = plot_margins, add_rectangles = add_rectangles)
    ax.set_title(stitle_2, fontsize = 25)

    if use_different_grids:
        raise ValueError('To be implemented')
    else:
        diff = data1-data2

    ax = plt.subplot(1, 3, 3, projection=proj)

    map_plot = plot_mapc_on_ax(ax, diff, lat1, lon1, proj, cmappa, cbar_range, n_color_levels = n_color_levels, draw_contour_lines = draw_contour_lines, n_lines = n_lines, bounding_lat = bounding_lat, plot_margins = plot_margins, add_rectangles = add_rectangles)
    ax.set_title('Diff', fontsize = 25)

    cax = plt.axes([0.1, 0.06, 0.8, 0.03])
    cb = plt.colorbar(map_plot,cax=cax, orientation='horizontal')
    cb.ax.tick_params(labelsize=18)
    cb.set_label(cb_label, fontsize=20)

    plt.suptitle(title, fontsize=35, fontweight='bold')

    plt.subplots_adjust(top=0.85)
    top    = 0.90  # the top of the subplots
    bottom = 0.13    # the bottom
    left   = 0.02    # the left side
    right  = 0.98  # the right side
    hspace = 0.20   # the amount of height reserved for white space between subplots
    wspace = 0.05    # the amount of width reserved for blank space between subplots
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    # save the figure or show it
    if filename is not None:
        fig.savefig(filename)
        # plt.close(fig)

    return fig


def plot_multimap_contour(dataset, lat, lon, filename, max_ax_in_fig = 30, number_subplots = False, cluster_labels = None, cluster_colors = None, repr_cluster = None, visualization = 'standard', central_lat_lon = None, cmap = 'RdBu_r', title = None, xlabel = None, ylabel = None, cb_label = None, cbar_range = None, plot_anomalies = True, n_color_levels = 21, draw_contour_lines = False, n_lines = 5, subtitles = None, color_percentiles = (5,95), fix_subplots_shape = None, figsize = (15,12), bounding_lat = 30, plot_margins = None, add_rectangles = None, draw_grid = False, reference_abs_field = None):
    """
    Plots multiple maps on a single figure (or more figures if needed).

    < data >: list, the fields to plot
    < lat, lon >: latitude and longitude
    < filename >: name of the file to save the figure to. If more figures are needed, the others are named as filename_1.pdf, filename_2.pdf, ...

    < max_ax_in_fig >: maximum number of figure panels inside a single figure. More figures are produced if dataset is longer.
    < number_subplots > : show the absolute number of each member on top of the subplot.

    If the fields to plot belong to different clusters, other functions are available:
    < cluster_labels > : labels that assign each member of dataset to a cluster.
    < cluster_colors > : colors for the different clusters. Cluster numbers have this color background.
    < repr_cluster > : number of the member representative of each cluster. A colored rectangle is drawn around them.

    < visualization >: 'standard' calls PlateCarree cartopy map, 'polar' calls Orthographic map.
    < central_lat_lon >: Tuple, (clat, clon). Is needed only for Orthographic plots. If not given, the mean lat and lon are taken.
    < cmap >: name of the color map.
    < cbar_range >: limits of the color bar.

    < plot_anomalies >: if True, the colorbar is symmetrical, so that zero corresponds to white. If cbar_range is set, plot_anomalies is set to False.
    < n_color_levels >: number of color levels.
    < draw_contour_lines >: draw lines in addition to the color levels?
    < n_lines >: number of lines to draw.

    < subtitles > : list of subtitles for all subplots.
    < color_percentiles > : define the range of data to be represented in the color bar. e.g. (0,100) to full range, (5,95) to enhance features.
    < fix_subplots_shape > : Fixes the number of subplots in rows and columns.

    """

    proj = def_projection(visualization, central_lat_lon)

    # Determining color levels
    cmappa = cm.get_cmap(cmap)

    if cbar_range is None:
        mi = np.percentile(dataset, color_percentiles[0])
        ma = np.percentile(dataset, color_percentiles[1])
        if plot_anomalies:
            # making a symmetrical color axis
            oko = max(abs(mi), abs(ma))
            spi = 2*oko/(n_color_levels-1)
            spi_ok = np.ceil(spi*100)/100
            oko2 = spi_ok*(n_color_levels-1)/2
            oko1 = -oko2
        else:
            oko1 = mi
            oko2 = ma
        cbar_range = (oko1, oko2)

    clevels = np.linspace(cbar_range[0], cbar_range[1], n_color_levels)

    # Begin plotting
    numens = len(dataset)

    if fix_subplots_shape is None:
        num_figs = int(np.ceil(1.0*numens/max_ax_in_fig))
        numens_ok = int(np.ceil(numens//num_figs))
        side1 = int(np.ceil(np.sqrt(numens_ok)))
        side2 = int(np.ceil(numens_ok/float(side1)))
    else:
        (side1, side2) = fix_subplots_shape
        numens_ok = side1*side2
        num_figs = int(np.ceil(1.0*numens/numens_ok))

    if filename is not None:
        namef = []
        namef.append(filename)
        if num_figs > 1:
            indp = filename.rfind('.')
            figform = filename[indp:]
            basename = filename[:indp]
            for i in range(num_figs)[1:]:
                namef.append(basename+'_{}'.format(i)+figform)

    if cluster_labels is not None:
        numclus = len(np.unique(cluster_labels))
        if cluster_colors is None:
            cluster_colors = color_set(numclus)

    all_figures = []
    for i in range(num_figs):
        fig = plt.figure(figsize = figsize)#(24,14)
        #fig, axs = plt.subplots(side1, side2, figsize = figsize)
        axes_for_colorbar = []
        for nens in range(numens_ok*i, numens_ok*(i+1)):
            if nens >= numens:
                print('no more panels here')
                break
            nens_rel = nens - numens_ok*i
            ax = plt.subplot(side1, side2, nens_rel+1, projection=proj)

            map_plot = plot_mapc_on_ax(ax, dataset[nens], lat, lon, proj, cmappa, cbar_range, n_color_levels = n_color_levels, draw_contour_lines = draw_contour_lines, n_lines = n_lines, bounding_lat = bounding_lat, plot_margins = plot_margins, add_rectangles = add_rectangles, draw_grid = draw_grid)

            if number_subplots:
                subtit = nens
                title_obj = plt.text(-0.05, 1.05, subtit, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=20, fontweight='bold', zorder = 20)
            if subtitles is not None:
                fonsiz = 25
                if numens_ok > 6:
                    fonsiz = 15
                elif numens_ok > 12:
                    fonsiz = 10
                ax.set_title(subtitles[nens], fontsize = fonsiz)
            if cluster_labels is not None:
                for nclus in range(numclus):
                    if nens in np.where(cluster_labels == nclus)[0]:
                        okclus = nclus
                        bbox=dict(facecolor=cluster_colors[nclus], alpha = 0.7, edgecolor='black', boxstyle='round,pad=0.2')
                        title_obj.set_bbox(bbox)

                if repr_cluster is not None:
                    if nens == repr_cluster[okclus]:
                        rect = plt.Rectangle((-0.01,-0.01), 1.02, 1.02, fill = False, transform = ax.transAxes, clip_on = False, zorder = 10)
                        rect.set_edgecolor(cluster_colors[okclus])
                        rect.set_linewidth(6.0)
                        ax.add_artist(rect)

        #rect = [left, bottom, width, height]
        cax = plt.axes([0.1, 0.08, 0.8, 0.03])
        #cax = plt.axes([0.1, 0.1, 0.8, 0.05])

        # cb = plt.colorbar(map_plot, ax = axs[0, :2], shrink=0.6, location='bottom')
        cb = plt.colorbar(map_plot,cax=cax, orientation='horizontal')
        cb.ax.tick_params(labelsize=18)
        cb.set_label(cb_label, fontsize=20)

        plt.suptitle(title, fontsize=35, fontweight='bold')

        top    = 0.92  # the top of the subplots
        bottom = 0.13    # the bottom
        left   = 0.02    # the left side
        right  = 0.98  # the right side
        hspace = 0.20   # the amount of height reserved for white space between subplots
        wspace = 0.05    # the amount of width reserved for blank space between subplots
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

        if filename is not None:
            fig.savefig(namef[i])
        all_figures.append(fig)
        # plt.close(fig)

    return all_figures


def plot_pdfpages(filename, figs, save_single_figs = True, fig_names = None):
    """
    Saves a list of figures to a pdf file.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    pdf = PdfPages(filename)
    for fig in figs:
        pdf.savefig(fig)
    pdf.close()

    if save_single_figs:
        indp = filename.index('.')
        cartnam = filename[:indp]+'_figures/'
        if not os.path.exists(cartnam):
            os.mkdir(cartnam)
        if fig_names is None:
            fig_names = ['pag_{}'.format(i+1) for i in range(len(figs))]
        for fig,nam in zip(figs, fig_names):
            fig.savefig(cartnam+nam+'.pdf')

    return


def plot_lat_crosssection(data, lat, levels, filename = None, cmap = 'RdBu_r', title = None, xlabel = None, ylabel = None, cb_label = None, cbar_range = None, plot_anomalies = False, n_color_levels = 21, draw_contour_lines = False, n_lines = 5, color_percentiles = (0,100), figsize = (10,6), pressure_levels = True, set_logscale_levels = False, return_ax = False):
    """
    Plots a latitudinal cross section map.

    < data >: the field to plot
    < lat, levels >: latitude and levels
    < filename >: name of the file to save the figure to. If None, the figure is just shown.

    < cmap >: name of the color map.
    < cbar_range >: limits of the color bar.

    < plot_anomalies >: if True, the colorbar is symmetrical, so that zero corresponds to white. If cbar_range is set,  plot_anomalies is set to False.
    < n_color_levels >: number of color levels.
    < draw_contour_lines >: draw lines in addition to the color levels?
    < n_lines >: number of lines to draw.

    """

    # Determining color levels
    cmappa = cm.get_cmap(cmap)

    if cbar_range is None:
        mi = np.percentile(data, color_percentiles[0])
        ma = np.percentile(data, color_percentiles[1])
        if plot_anomalies:
            # making a symmetrical color axis
            oko = max(abs(mi), abs(ma))
            spi = 2*oko/(n_color_levels-1)
            spi_ok = np.ceil(spi*100)/100
            oko2 = spi_ok*(n_color_levels-1)/2
            oko1 = -oko2
        else:
            oko1 = mi
            oko2 = ma
        cbar_range = (oko1, oko2)

    clevels = np.linspace(cbar_range[0], cbar_range[1], n_color_levels)

    # Plotting figure
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111)

    xi,yi = np.meshgrid(lat, levels)
    if pressure_levels == True:
        plt.gca().invert_yaxis()
        if set_logscale_levels:
            ax.set_yscale('log')

    map_plot = ax.contourf(xi, yi, data, clevels, cmap = cmappa, extend = 'both', corner_mask = False)
    if draw_contour_lines:
        map_plot_lines = ax.contour(xi, yi, data, n_lines, colors = 'k', linewidth = 0.5)

    plt.grid()

    title_obj = plt.title(title, fontsize=20, fontweight='bold')
    title_obj.set_position([.5, 1.05])

    cax = plt.axes([0.1, 0.11, 0.8, 0.05]) #horizontal
    cb = plt.colorbar(map_plot, cax=cax, orientation='horizontal')#, labelsize=18)
    cb.ax.tick_params(labelsize=14)
    cb.set_label(cb_label, fontsize=16)

    top    = 0.88  # the top of the subplots
    bottom = 0.20    # the bottom of the subplots
    left   = 0.1    # the left side
    right  = 0.98  # the right side
    hspace = 0.20   # height reserved for white space
    wspace = 0.05    # width reserved for blank space
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    # save the figure or show it
    if filename is not None:
        fig.savefig(filename)
        plt.close(fig)

    if return_ax:
        return fig, ax
    else:
        return fig


def plot_animation_map(maps, lat, lon, labels = None, fps_anim = 5, title = None, filename = None, visualization = 'standard', central_lat_lon = None, cmap = 'RdBu_r', xlabel = None, ylabel = None, cb_label = None, cbar_range = None, plot_anomalies = True, n_color_levels = 21, draw_contour_lines = False, n_lines = 5, color_percentiles = (0,100), figsize = (8,6)):
    """
    Shows animation of a sequence of maps or saves it to a gif file.
    < maps > : list, the sequence of maps to be plotted.
    < labels > : title of each map.

    < filename > : if None the animation is run and shown, instead it is saved to filename.
    < **kwargs > : all other kwargs as in plot_map_contour()
    """
    if filename is None:
        plt.ion()
    if labels is None:
        labels = np.arange(len(maps))

    proj = def_projection(visualization, central_lat_lon)

    # Determining color levels
    cmappa = cm.get_cmap(cmap)

    if cbar_range is None:
        mi = np.percentile(maps, color_percentiles[0])
        ma = np.percentile(maps, color_percentiles[1])
        if plot_anomalies:
            # making a symmetrical color axis
            oko = max(abs(mi), abs(ma))
            spi = 2*oko/(n_color_levels-1)
            spi_ok = np.ceil(spi*100)/100
            oko2 = spi_ok*(n_color_levels-1)/2
            oko1 = -oko2
        else:
            oko1 = mi
            oko2 = ma
        cbar_range = (oko1, oko2)

    clevels = np.linspace(cbar_range[0], cbar_range[1], n_color_levels)

    # Plotting figure
    fig = plt.figure(figsize = figsize)
    ax = plt.subplot(projection = proj)

    plt.title(title)

    def update_lines(num):
        print(num)
        lab = labels[num]
        mapa = maps[num]

        plot_mapc_on_ax(ax, mapa, lat, lon, proj, cmappa, cbar_range, n_color_levels = n_color_levels, draw_contour_lines = draw_contour_lines, n_lines = n_lines, bounding_lat = bounding_lat, plot_margins = plot_margins, add_rectangles = add_rectangles)
        showdate.set_text('{}'.format(lab))

        return

    mapa = maps[0]
    map_plot = plot_mapc_on_ax(ax, mapa, lat, lon, proj, cmappa, cbar_range, n_color_levels = n_color_levels, draw_contour_lines = draw_contour_lines, n_lines = n_lines, bounding_lat = bounding_lat, plot_margins = plot_margins, add_rectangles = add_rectangles)

    showdate = ax.text(0.5, 0.9, labels[0], transform=fig.transFigure, fontweight = 'bold', color = 'black', bbox=dict(facecolor='lightsteelblue', edgecolor='black', boxstyle='round,pad=1'))

    #cax = plt.axes([0.1, 0.11, 0.8, 0.05]) #horizontal
    cb = plt.colorbar(map_plot, orientation='horizontal') #cax=cax, labelsize=18)
    cb.ax.tick_params(labelsize=14)
    cb.set_label(cb_label, fontsize=16)

    if filename is not None:
        metadata = dict(title=title, artist='climtools_lib')
        writer = ImageMagickFileWriter(fps = fps_anim, metadata = metadata)#, frame_size = (1200, 900))
        with writer.saving(fig, filename, 100):
            for i, lab in enumerate(labels):
                print(lab)
                update_lines(i)
                writer.grab_frame()

        return writer
    else:
        print('vai?')
        print(len(maps))
        line_ani = animation.FuncAnimation(fig, update_lines, len(maps), interval = 100./fps_anim, blit=False)

        return line_ani


def Taylor_plot_EnsClus(models, observation, filename, title = None, label_bias_axis = None, label_ERMS_axis = None, show_numbers = True, only_numbers = False, colors_all = None, cluster_labels = None, cluster_colors = None, repr_cluster = None, verbose = True):
    """
    Produces two figures:
    - a Taylor diagram
    - a bias/E_rms plot

    < models > : a set of patterns (2D matrices or pc vectors) corresponding to different simulation results/model behaviours.
    < observation > : the corresponding observed pattern. observation.shape must be the same as each of the models' shape.

    < show_numbers > : print a number given by the order of the models sequence at each point in the plot.
    < only_numbers > : print just numbers in the plot instead of points.

    If the fields to plot belong to different clusters, other functions are available:
    < cluster_labels > : labels that assign each member of dataset to a cluster.
    < cluster_colors > : colors for the different clusters. Cluster numbers have this color background.
    < repr_cluster > : number of the member representative of each cluster. A colored rectangle is drawn around them.
    """
    fig6 = plt.figure(figsize=(8,6))
    ax = fig6.add_subplot(111, polar = True)
    plt.title(title)
    #ax.set_facecolor(bgcol)

    ax.set_thetamin(0)
    ax.set_thetamax(180)

    sigmas_pred = np.array([np.std(var) for var in models])
    sigma_obs = np.std(observation)
    corrs_pred = np.array([Rcorr(observation, var) for var in models])

    if cluster_labels is not None:
        numclus = len(np.unique(cluster_labels))
        if cluster_colors is None:
            cluster_colors = color_set(numclus)

        colors_all = [cluster_colors[clu] for clu in cluster_labels]

    angles = np.arccos(corrs_pred)

    ax.scatter([0.], [sigma_obs], color = 'black', s = 40, clip_on=False)

    if not show_numbers:
        ax.scatter(angles, sigmas_pred, s = 10, color = colors_all)
        if repr_cluster is not None:
            ax.scatter(angles[repr_cluster], sigmas_pred[repr_cluster], color = cluster_colors, edgecolor = 'black', s = 40)
    else:
        if only_numbers:
            ax.scatter(angles, sigmas_pred, s = 0, color = colors_all)
            for i, (ang, sig, col) in enumerate(zip(angles, sigmas_pred, colors_all)):
                zord = 5
                siz = 3
                if i in repr_ens:
                    zord = 21
                    siz = 4
                gigi = ax.text(ang, sig, i, ha="center", va="center", color = col, fontsize = siz, zorder = zord, weight = 'bold')
        else:
            for i, (ang, sig, col) in enumerate(zip(angles, sigmas_pred, colors_all)):
                zord = i + 1
                siz = 4
                if repr_cluster is not None:
                    if i in repr_cluster:
                        zord = zord + numens
                        siz = 5
                        ax.scatter(ang, sig, color = col, alpha = 0.7, s = 60, zorder = zord, edgecolor = col)
                    else:
                        ax.scatter(ang, sig, s = 30, color = col, zorder = zord, alpha = 0.7)
                else:
                    ax.scatter(ang, sig, s = 30, color = col, zorder = zord, alpha = 0.7)
                gigi = ax.text(ang, sig, i, ha="center", va="center", color = 'white', fontsize = siz, zorder = zord, WEIGHT = 'bold')

    ok_cos = np.array([-0.99, -0.95, -0.9, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99])
    labgr = ['{:4.2f}'.format(co) for co in ok_cos]
    anggr = np.rad2deg(np.arccos(ok_cos))

    plt.thetagrids(anggr, labels=labgr, zorder = 0)

    for sig in [1., 2., 3.]:
        circle = plt.Circle((sigma_obs, 0.), sig*sigma_obs, transform=ax.transData._b, fill = False, edgecolor = 'black', linestyle = '--')
        ax.add_artist(circle)

    top    = 0.88  # the top of the subplots
    bottom = 0.02    # the bottom
    left   = 0.02    # the left side
    right  = 0.98  # the right side
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    fig6.savefig(filename)


    indp = filename.rfind('.')
    figform = filename[indp:]
    basename = filename[:indp]
    nuname = basename+'_biases'+figform

    fig7 = plt.figure(figsize=(8,6))
    ax = fig7.add_subplot(111)
    plt.title(title)

    biases = np.array([np.mean(var) for var in models])
    ctr_patt_RMS = np.array([E_rms_cp(var, observation) for var in models])
    RMS = np.array([E_rms(var, observation) for var in models])

    if verbose and cluster_labels is not None:
        print('----------------------------\n')
        min_cprms = ctr_patt_RMS.argmin()
        print('The member with smallest centered-pattern RMS is member {} of cluster {}\n'.format(min_cprms, cluster_labels[min_cprms]))
        print('----------------------------\n')
        min_rms = RMS.argmin()
        print('The member with smallest absolute RMS is member {} of cluster {}\n'.format(min_rms, cluster_labels[min_rms]))
        print('----------------------------\n')
        min_bias = np.abs((biases - np.mean(observation))).argmin()
        print('The member with closest mean anomaly is member {} of cluster {}\n'.format(min_bias, cluster_labels[min_bias]))
        print('----------------------------\n')
        max_corr = corrs_pred.argmax()
        print('The member with largest correlation coefficient is member {} of cluster {}\n'.format(max_corr, cluster_labels[max_corr]))

    if not show_numbers:
        ax.scatter(biases, ctr_patt_RMS, color = colors_all, s =10)
        if repr_cluster is not None:
            ax.scatter(biases[repr_cluster], ctr_patt_RMS[repr_cluster], color = cluster_colors, edgecolor = 'black', s = 40)
    else:
        if only_numbers:
            ax.scatter(biases, ctr_patt_RMS, s = 0, color = colors_all)
            for i, (ang, sig, col) in enumerate(zip(biases, ctr_patt_RMS, colors_all)):
                zord = 5
                siz = 7
                if repr_cluster is not None:
                    if i in repr_cluster:
                        zord = 21
                        siz = 9
                gigi = ax.text(ang, sig, i, ha="center", va="center", color = col, fontsize = siz, zorder = zord, weight = 'bold')
        else:
            for i, (ang, sig, col) in enumerate(zip(biases, ctr_patt_RMS, colors_all)):
                zord = i + 1
                siz = 7
                if repr_cluster is not None:
                    if i in repr_cluster:
                        zord = zord + numens
                        siz = 9
                        ax.scatter(ang, sig, color = col, alpha = 0.7, s = 200, zorder = zord, edgecolor = col)
                    else:
                        ax.scatter(ang, sig, s = 120, color = col, zorder = zord, alpha = 0.7)
                else:
                    ax.scatter(ang, sig, s = 120, color = col, zorder = zord, alpha = 0.7)
                gigi = ax.text(ang, sig, i, ha="center", va="center", color = 'white', fontsize = siz, zorder = zord, WEIGHT = 'bold')

    plt.xlabel(label_bias_axis)
    plt.ylabel(label_ERMS_axis)

    for sig in [1., 2., 3.]:
        circle = plt.Circle((np.mean(observation), 0.), sig*sigma_obs, fill = False, edgecolor = 'black', linestyle = '--')
        ax.add_artist(circle)

    plt.scatter(np.mean(observation), 0., color = 'black', s = 120, zorder = 5)
    plt.grid()

    fig7.savefig(nuname)

    return


def ellipse_plot(x, y, errx, erry, labels = None, ax = None, filename = None, polar = False, colors = None, alpha = 0.5, legendfontsize = 18):
    """
    Produces a plot with ellipse patches indicating error bars.

    If polar set to True, x is the polar axis (angle in radians) and y is the radial axis.
    """
    if ax is None:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, polar = polar)
    else:
        filename = None

    if colors is None:
        colors = color_set(len(x))

    all_artists = []
    for i, (xu, yu, errxu, erryu) in enumerate(zip(x, y, errx, erry)):
        ell = mpl.patches.Ellipse(xy = (xu, yu), width = 2*errxu, height = 2*erryu, angle = 0.0)
        ax.add_artist(ell)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(alpha)
        ell.set_facecolor(colors[i])
        if labels is not None:
            ell.set_label(labels[i])
        all_artists.append(ell)

    if not polar:
        ax.set_xlim(min(x)-abs(max(errx)), max(x)+abs(max(errx)))
        ax.set_ylim(min(y)-abs(max(erry)), max(y)+abs(max(erry)))

    if labels is not None:
        ax.legend(handles = all_artists, fontsize = legendfontsize)

    if filename is not None:
        fig.savefig(filename)

    return


def Taylor_plot(models, observation, filename = None, ax = None, title = None, label_bias_axis = None, label_ERMS_axis = None, colors = None, markers = None, only_first_quarter = False, legend = True, marker_edge = None, labels = None, obs_label = None, mod_points_size = 60, obs_points_size = 90, enlarge_rmargin = True, relative_std = True, max_val_sd = None):
    """
    Produces two figures:
    - a Taylor diagram
    - a bias/E_rms plot

    < models > : a set of patterns (2D matrices or pc vectors) corresponding to different simulation results/model behaviours.
    < observation > : the corresponding observed pattern. observation.shape must be the same as each of the models' shape.

    < colors > : list of colors for each model point
    < markers > : list of markers for each model point
    """
    # if ax is None and filename is None:
    #     raise ValueError('Where do I plot this? specify ax or filename')
    # elif filename is not None and ax is not None:
    #     raise ValueError('Where do I plot this? specify ax OR filename, not BOTH')

    if ax is None:
        fig6 = plt.figure(figsize=(8,6))
        ax = fig6.add_subplot(111, polar = True)
        ax_specified = False
    else:
        ax_specified = True

    if title is not None:
        ax.set_title(title)
    #ax.set_facecolor(bgcol)

    ax.set_thetamin(0)
    if only_first_quarter:
        ax.set_thetamax(90)
        ok_cos = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99])
    else:
        ax.set_thetamax(180)
        ok_cos = np.array([-0.99, -0.95, -0.9, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99])

    labgr = ['{:4.2f}'.format(co) for co in ok_cos]
    anggr = np.rad2deg(np.arccos(ok_cos))

    ax.set_thetagrids(anggr, labels=labgr)

    if relative_std:
        sigma_obs_abs = np.std(observation)
        sigmas_pred = np.array([np.std(var)/sigma_obs_abs for var in models])
        sigma_obs = 1.0
    else:
        sigmas_pred = np.array([np.std(var) for var in models])
        sigma_obs = np.std(observation)
    corrs_pred = np.array([Rcorr(observation, var) for var in models])

    if colors is None:
        colors = color_set(len(models))

    angles = np.arccos(corrs_pred)

    if enlarge_rmargin:
        ax.set_ymargin(0.2)

    if markers is None:
        markers = ['o']*len(angles)
    if labels is None:
        labels = [None]*len(angles)
    for ang, sig, col, sym, lab in zip(angles, sigmas_pred, colors, markers, labels):
        ax.scatter(ang, sig, s = mod_points_size, color = col, marker = sym, edgecolor = marker_edge, label = lab, clip_on=False)

    ax.scatter([0.], [sigma_obs], color = 'black', s = obs_points_size, marker = 'D', clip_on=False, label = obs_label)

    if max_val_sd is None:
        max_val_sd = 1.1 * np.max(sigmas_pred)
    ax.set_ylim(0., max_val_sd)

    for sig in [1., 2., 3.]:
        circle = plt.Circle((sigma_obs, 0.), sig*sigma_obs, transform=ax.transData._b, fill = False, edgecolor = 'black', linestyle = '--')
        ax.add_artist(circle)

    if legend:
        ax.legend(fontsize = 'small')

    if ax_specified:
        return

    top    = 0.88  # the top of the subplots
    bottom = 0.1    # the bottom
    left   = 0.02    # the left side
    right  = 0.98  # the right side
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)

    if filename is not None:
        fig6.savefig(filename)

        indp = filename.rfind('.')
        figform = filename[indp:]
        basename = filename[:indp]
        nuname = basename+'_biases'+figform

    fig7 = plt.figure(figsize=(8,6))
    ax = fig7.add_subplot(111)
    plt.title(title)

    biases = np.array([np.mean(var) for var in models])
    ctr_patt_RMS = np.array([E_rms_cp(var, observation) for var in models])
    RMS = np.array([E_rms(var, observation) for var in models])

    if markers is None:
        markers = ['o']*len(angles)
    if labels is None:
        labels = [None]*len(angles)
    for bia, ctpr, col, sym, lab in zip(biases, ctr_patt_RMS, colors, markers, labels):
        ax.scatter(bia, ctpr, s = mod_points_size+20, color = col, marker = sym, edgecolor = marker_edge, label = lab, clip_on=False)

    plt.xlabel(label_bias_axis)
    plt.ylabel(label_ERMS_axis)

    for sig in [1., 2., 3.]:
        circle = plt.Circle((np.mean(observation), 0.), sig*sigma_obs_abs, fill = False, edgecolor = 'black', linestyle = '--', clip_on=True)
        ax.add_artist(circle)

    plt.scatter(np.mean(observation), 0., color = 'black', s = obs_points_size+20, marker = 'D', zorder = 5, label = obs_label)

    if legend:
        plt.legend(fontsize = 'small')
    plt.grid()

    if filename is not None:
        fig7.savefig(nuname)

    return fig6, fig7


def plotcorr(x, y, filename, xlabel = 'x', ylabel = 'y', xlim = None, ylim = None, format = 'pdf'):
    """
    Plots correlation graph between x and y, fitting a line and calculating Pearson's R coeff.
    :param filename: abs. path of the graph
    :params xlabel, ylabel: labels for x and y axes
    """
    pearR = np.corrcoef(x,y)[1,0]
    A = np.vstack([x,np.ones(len(x))]).T  # A = [x.T|1.T] dove 1 = [1,1,1,1,1,..]
    m,c = np.linalg.lstsq(A,y)[0]
    xlin = np.linspace(min(x)-0.05*(max(x)-min(x)),max(x)+0.05*(max(x)-min(x)),11)

    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(111)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.scatter(x, y, label='Data', color='blue', s=4, zorder=3)
    if xlim is not None:
        if np.isnan(xlim[1]):
            plt.xlim(xlim[0],plt.xlim()[1])
        elif np.isnan(xlim[0]):
            plt.xlim(plt.xlim()[0],xlim[1])
        else:
            plt.xlim(xlim[0],xlim[1])
    if ylim is not None:
        if np.isnan(ylim[1]):
            plt.ylim(ylim[0],plt.ylim()[1])
        elif np.isnan(ylim[0]):
            plt.ylim(plt.ylim()[0],ylim[1])
        else:
            plt.ylim(ylim[0],ylim[1])
    plt.plot(xlin, xlin*m+c, color='red', label='y = {:8.2f} x + {:8.2f}'.format(m,c))
    plt.title("Pearson's R = {:5.2f}".format(pearR))
    plt.legend(loc=4,fancybox =1)
    fig.savefig(filename, format=format, dpi=150)
    plt.close()

    return


def plot_regime_pdf_onax(ax, labels, pcs, reg, eof_proj = (0,1), color = None, fig_label = None, xi_grid = None, yi_grid = None, n_grid_points = 100, levels = None, normalize_pdf = True, plot_centroid = False, eof_axis_lim = None, lw = 1):
    """
    Plots the 2D projection of the regime pdf on the two chosen eof axes (eof_proj).
    """

    if fig_label is None:
        fig_label = 'regime {}'.format(reg)

    if xi_grid is None:
        (x0, x1) = (np.percentile(pcs, 1), np.percentile(pcs, 99))
        xss = np.linspace(x0, x1, n_grid_points)
        xi_grid, yi_grid = np.meshgrid(xss, xss)

    if color is None:
        cmappa = 'Blues'
    else:
        cmappa = custom_alphagradient_cmap(color)

    okclu = labels == reg
    okpc = pcs[okclu, :]
    kufu = calc_pdf(okpc[:,eof_proj].T)
    zi = kufu(np.vstack([xi_grid.flatten(), yi_grid.flatten()]))
    #print(min(zi), max(zi))
    if normalize_pdf:
        zi = zi/np.max(zi)
    elif levels is not None:
        if type(levels) is not int:
            levels = np.array(levels)*np.max(zi)
    cont = ax.contour(xi_grid, yi_grid, zi.reshape(xi_grid.shape), levels, cmap = cm.get_cmap(cmappa), linewidths = lw)

    if plot_centroid:
        cent = np.mean(okpc[:,eof_proj], axis = 0)
        ax.scatter(cent[0], cent[1], color = color, marker = 'x', s = 20)

    if eof_axis_lim is not None:
        ax.set_xlim(eof_axis_lim)
        ax.set_ylim(eof_axis_lim)
    #ax.clabel(cont, inline=1, fontsize=10)

    return


def plot_allregime_pdfs(labels, pcs, eof_proj = [(0,1), (0,2), (1,2)], all_regimes_together = True, n_grid_points = 100, filename = None, centroids = None, levels = [0.1, 0.5]):
    """
    Plots the 2D projection of the regime pdf on the two chosen eof axes (eof_proj).
    """
    n_clus = np.max(labels)+1

    fig = plt.figure(figsize = (16,12))
    if not all_regimes_together:
        nrow = n_clus
        ncol = len(eof_proj)
    else:
        nrow = 1
        ncol = len(eof_proj)

    colors = color_set(n_clus)

    (x0, x1) = (np.percentile(pcs, 1), np.percentile(pcs, 99))
    xss = np.linspace(x0, x1, n_grid_points)
    xi_grid, yi_grid = np.meshgrid(xss, xss)

    for reg, col in zip(range(n_clus), colors):
        for i, proj in enumerate(eof_proj):
            if not all_regimes_together:
                ind = ncol*reg + i+1
            else:
                ind = i+1
            ax = plt.subplot(nrow, ncol, ind)
            plot_regime_pdf_onax(ax, labels, pcs, reg, eof_proj = proj, color = col, levels = levels)
            ax.set_xlabel('EOF {}'.format(proj[0]))
            ax.set_ylabel('EOF {}'.format(proj[1]))

            if centroids is not None:
                print(ind, centroids[reg][proj[0]], centroids[reg][proj[1]])
                ax.scatter(centroids[reg][proj[0]], centroids[reg][proj[1]], color = col, marker = 'x', s = 100)

    plt.tight_layout()
    if filename is not None:
        fig.savefig(filename)

    return fig


def custom_alphagradient_cmap(color):
    # define custom colormap with fixed colour and alpha gradient
    # use simple linear interpolation in the entire scale
    if type(color) is str:
        color = cm.colors.ColorConverter.to_rgb(color)

    cm.register_cmap(name='custom', data={'red': [(0.,color[0],color[0]), (1.,color[0],color[0])], 'green': [(0.,color[1],color[1]), (1.,color[1],color[1])], 'blue':  [(0.,color[2],color[2]), (1.,color[2],color[2])], 'alpha': [(0.,0,0), (1.,1,1)]})

    cmap = cm.get_cmap('custom')

    return cmap


#def plot_multimodel_regime_pdfs(model_names, labels_set, pcs_set, eof_proj = [(0,1), (0,2), (1,2)], n_grid_points = 100, filename = None, colors = None, levels = [0.1, 0.5], centroids_set = None):
def plot_multimodel_regime_pdfs(results, model_names = None, eof_proj = [(0,1), (0,2), (1,2)], n_grid_points = 100, filename = None, colors = None, levels = [0.1, 0.5], plot_centroids = True, figsize = (16,12), reference = None, eof_axis_lim = None, nolegend = False, check_for_eofs = True, fix_subplots_shape = None, fac = 1.5):
    """
    Plots the 2D projection of the regime pdf on the two chosen eof axes (eof_proj).

    One figure per each regime with all model pdfs.
    """
    if model_names is None:
        model_names = list(results.keys())

    if colors is None:
        colors = color_set(len(model_names))
        if reference is not None:
            colors[first(np.array(model_names) == reference)] = 'black'
        else:
            colors[-1] = 'black'

    n_clus = np.max(list(results.values())[0]['labels'])+1

    fig = plt.figure(figsize = figsize)

    if fix_subplots_shape is None:
        nrow = n_clus
        ncol = len(eof_proj)
    else:
        nrow, ncol = fix_subplots_shape
        if nrow*ncol < n_clus*len(eof_proj):
            raise ValueError('{} not enough for {} subplots needed'.format(fix_subplots_shape, (n_clus, len(eof_proj))))

    x0s = []
    x1s = []

    if reference is not None and check_for_eofs:
        if 'model_eofs' not in results[model_names[0]].keys():
            eofkey = 'eofs'
        else:
            eofkey = 'model_eofs'
        check_all = [cosine(results[mod][eofkey], results[reference][eofkey]) for mod in model_names]
        if not np.all([isclose(ck, 1.0) for ck in check_all]):
            print(check_all)
            print('EOFs are different! PCs have to be projected on reference space!\n')
            nu_pcs_set = dict()
            for mod in model_names:
                if mod == reference:
                    nu_pcs_set[mod] = results[mod]['pcs']
                else:
                    nu_pcs_set[mod] = project_set_on_new_basis(results[mod]['pcs'], results[mod][eofkey], results[reference][eofkey])
        else:
            nu_pcs_set = {mod: results[mod]['pcs'] for mod in model_names}
    else:
        nu_pcs_set = {mod: results[mod]['pcs'] for mod in model_names}

    for pcs in nu_pcs_set.values():
        (x0, x1) = (np.min(pcs), np.max(pcs))
        x0s.append(x0)
        x1s.append(x1)

    if np.min(x0s) < 0 and np.max(x1s) > 0:
        xss = np.linspace(fac*np.min(x0s), fac*np.max(x1s), n_grid_points)
    elif np.min(x0s) > 0:
        xss = np.linspace(0., fac*np.max(x1s), n_grid_points)
    elif np.max(x1s) < 0:
        xss = np.linspace(fac*np.min(x0s), 0., n_grid_points)

    xi_grid, yi_grid = np.meshgrid(xss, xss)

    ind = 0
    for reg in range(n_clus):
        for i, proj in enumerate(eof_proj):
            ind += 1
            #ind = ncol*reg + i+1
            ax = plt.subplot(nrow, ncol, ind)
            ax.axhline(0, color = 'grey')
            ax.axvline(0, color = 'grey')
            for mod, col in zip(model_names, colors):
                labels = results[mod]['labels']
                #pcs = results[mod]['pcs']
                pcs = nu_pcs_set[mod]
                lw = 0.8
                if mod == reference: lw = 2
                plot_regime_pdf_onax(ax, labels, pcs, reg, eof_proj = proj, color = col, fig_label = mod, levels = levels, plot_centroid = plot_centroids, eof_axis_lim = eof_axis_lim, lw = lw, xi_grid = xi_grid, yi_grid = yi_grid)
            ax.set_xlabel('EOF {}'.format(proj[0]))
            ax.set_ylabel('EOF {}'.format(proj[1]))

            # if plot_centroids is not None:
            #     for mod, col in zip(model_names, colors):
            #         cent = results[mod]['centroids']
            #         ax.scatter(cent[reg][proj[0]], cent[reg][proj[1]], color = col, marker = 'x', s = 20)

    plt.tight_layout()
    if not nolegend:
        #plt.subplots_adjust(bottom = 0.12, top = 0.93)
        fig = custom_legend(fig, colors, model_names)

    if filename is not None:
        fig.savefig(filename)

    return fig


def custom_legend(fig, colors, labels, loc = 'lower center', ncol = None, fontsize = 10, bottom_margin = 0.08):
    plt.subplots_adjust(bottom = bottom_margin)
    if ncol is None:
        ncol = int(np.ceil(len(labels)/2.0))
    proxy = [plt.Rectangle((0,0),1,1, fc = col) for col in colors]
    fig.legend(proxy, labels, loc = loc, ncol = ncol, fontsize = fontsize)
    return fig


def clus_visualize_2D():
    """
    Makes a 2D plot using the coordinates in the first 2 EOFs of single realizations and clusters.
    """



    return


def clus_visualize_3D():
    """
    Makes a 3D plot using the coordinates in the first 3 EOFs of single realizations and clusters.
    """

    return
