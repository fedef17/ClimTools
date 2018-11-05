#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os

from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as PathEffects
import matplotlib.animation as animation
from matplotlib.animation import ImageMagickFileWriter

import netCDF4 as nc
import cartopy.crs as ccrs
import cartopy.util as cutil
import pandas as pd

from numpy import linalg as LA
from eofs.standard import Eof
from scipy import stats
import itertools as itt

from sklearn.cluster import KMeans
import ctool
import ctp

from datetime import datetime
import pickle


#######################################################
#
###     INPUTs reading
#
#######################################################

def printsep(ofile = None):
    if ofile is None:
        print('\n--------------------------------------------------------\n')
    else:
        ofile.write('\n--------------------------------------------------------\n')
    return

def str_to_bool(s):
    if s == 'True' or s == 'T':
         return True
    elif s == 'False' or s == 'F':
         return False
    else:
         raise ValueError('Not a boolean value')

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


def readxDncfield(ifile):
    """
    Read a netCDF file as it is, preserving all dimensions.
    """

    fh = nc.Dataset(ifile)
    ndim = len(fh.variables.keys()) - 1
    print('Field as {} dimensions. All keys: {}'.format(ndim, fh.variables.keys()))
    del fh

    if ndim == 2:
        out = read2Dncfield(ifile)
    elif ndim == 3:
        out = read3Dncfield(ifile)
    elif ndim == 4:
        out = read4Dncfield(ifile)

    return out


def read4Dncfield(ifile, extract_level = None, compress_dummy_dim = True):
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
    var_units   = fh.variables[variabs[-1]].units
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
    for ci in dates:
        dates_ok.append(pd.Timestamp(ci.strftime()).to_pydatetime())

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
        add_day = num/72 # salto un giorno ogni 72
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
    var_units   = fh.variables[variabs[-1]].units
    var         = fh.variables[variabs[-1]][:,:,:]
    txt='{0} dimension [time x lat x lon]: {1}'.format(variabs[-1],var.shape)

    if compress_dummy_dim and var.ndim > 3:
        var = var.squeeze()
    #print(fh.variables)
    time = list(time)
    dates=nc.num2date(time,time_units)
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
    var_units   = fh.variables[variabs[3]].units

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

    #for varn in dataset.variables.keys():
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

    #for varn in dataset.variables.keys():
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

    #for varn in dataset.variables.keys():
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

def sel_area(lat,lon,var,area):
    '''
    GOAL
        Selecting the area of interest from a nc dataset.
    USAGE
        var_area, lat_area, lon_area = sel_area(lat,lon,var,area)

    :param area: can be 'EAT', 'PNA', 'NH', 'Eu' or 'Med'
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
            var_roll=np.roll(var,int(len(lon)/2),axis=-1)
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
            var_roll=np.roll(var,int(len(lon)/2),axis=-1)
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
            var_roll=np.roll(var,int(len(lon)/2),axis=-1)
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
            var_roll=np.roll(var,int(len(lon)/2),axis=-1)
        else:
            var_roll=var
            lon_new=lon

    latidx = (lat >= latS) & (lat <= latN)
    lonidx = (lon_new >= lonW) & (lon_new <= lonE)

    print(var_roll.shape, len(latidx), len(lonidx))
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

    var_season = var[mask,:,:]
    dates_season = dates[mask]
    dates_season_pdh = pd.to_datetime(dates_season)

    if var_season.ndim == 2:
        var_season = var_season[np.newaxis, :]

    if season in mesi_short:
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

            var_season = var_season[start:end,:,:]
            dates_season = dates_season[start:end]

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

    delta = window/2

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
            filt_mean.append(np.mean(var[okdates,:,:], axis = 0))
            filt_std.append(np.std(var[okdates,:,:], axis = 0))
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
    years = np.unique(dates_pdh.year)[::step_year]

    for yea in years:
        okye = (dates_pdh.year >= yea - window_years/2) & (dates_pdh.year <= yea + window_years/2)
        clm, dtclm, _ = daily_climatology(var[okye], dates[okye], window_days, refyear = yea)
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
        filt_mean.append(np.mean(var[okdates,:,:], axis = 0))
        filt_std.append(np.std(var[okdates,:,:], axis = 0))
        dates_filt.append(data)

    dates_ok = pd.to_datetime(dates_filt).to_pydatetime()

    filt_mean = np.stack(filt_mean)
    filt_std = np.stack(filt_std)

    return filt_mean, dates_ok, filt_std


def range_years(year1, year2):
    data1 = pd.to_datetime('{}0101'.format(year1), format='%Y%m%d')
    data2 = pd.to_datetime('{}1231'.format(year2), format='%Y%m%d')
    return data1, data2


def sel_time_range(var, dates, dates_range):
    """
    Extracts a subset in time.
    """
    dates_pdh = pd.to_datetime(dates)
    okdates = (dates_pdh >= dates_range[0]) & (dates_pdh <= dates_range[1])

    return var[okdates, ...], dates[okdates]


def check_daily(dates):
    """
    Checks if the dataset is a daily dataset.
    """
    daydelta = pd.Timedelta('1 days')
    delta = dates[1]-dates[0]

    if delta == daydelta:
        return True
    else:
        return False


def running_mean(var, wnd):
    """
    Performs a running mean.
    """
    tempser = pd.Series(var)
    rollpi_temp = tempser.rolling(wnd, center = True).mean()

    return rollpi_temp


def anomalies_daily_detrended(var, dates, climat_mean = None, dates_climate_mean = None, window_days = 5, window_years = 20, step_year = 5):
    """
    Calculates the daily anomalies wrt a trending climatology. climat_mean and dates_climate_mean are the output of trend_daily_climat().
    """
    dates_pdh = pd.to_datetime(dates)
    if climat_mean is None or dates_climate_mean is None:
        climat_mean, dates_climate_mean = trend_daily_climat(var, dates, window_days = window_days, window_years = window_years, step_year = step_year)

    var_anom_tot = []
    year_steps = np.unique(dates_pdh.year)[::step_year]
    #okyetot = np.zeros(len(var), dtype=bool)

    for yea in np.unique(dates_pdh.year):
        yearef = np.argmin(abs(year_steps - yea))
        okye = dates_pdh.year == yea
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

    if not check_daily(dates):
        raise ValueError('Not a daily dataset')

    if climat_mean is None or dates_climate_mean is None:
        climat_mean, dates_climate_mean, _ = daily_climatology(var, dates, window)

    dates_pdh = pd.to_datetime(dates)
    dates_climate_mean_pdh = pd.to_datetime(dates_climate_mean)
    var_anom = np.empty_like(var)

    for el, dat in zip(climat_mean, dates_climate_mean_pdh):
        mask = (dates_pdh.month == dat.month) & (dates_pdh.day == dat.day)
        var_anom[mask,:,:] = var[mask,:,:] - el

    mask = (dates_pdh.month == 2) & (dates_pdh.day == 29)
    okel = (dates_climate_mean_pdh.month == 2) & (dates_climate_mean_pdh.day == 28)

    var_anom[mask,:,:] = var[mask,:,:] - climat_mean[okel,:,:]

    return var_anom


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
        var_anom[mask,:,:] = var[mask,:,:] - el

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
                    slope, intercept, r_value, p_value, std_err = stats.linregress(range(var_ens[0].shape[0]),var_ens[i][:,la,lo])
                    trendmap[la,lo]=slope
            trendmap_ens.append(trendmap)
        varextreme_ens = trendmap_ens

    print('\n------------------------------------------------------------')
    print('Anomalies and ensemble mean are computed with respect to the {0}'.format(extreme))
    print('------------------------------------------------------------\n')

    extreme_ensemble_mean = np.mean(varextreme_ens_np, axis = 0)
    extreme_ens_anomalies = varextreme_ens_np - extreme_ensemble_mean

    return extreme_ens_anomalies, extreme_ensemble_mean


def yearly_average(var, dates):
    """
    Averages year per year.
    """

    dates_pdh = pd.to_datetime(dates)

    nuvar = []
    nudates = []
    for year in np.unique(dates_pdh.year):
        data = pd.to_datetime('{}0101'.format(year), format='%Y%m%d')
        okdates = (dates_pdh.year == year)
        nuvar.append(np.mean(var[okdates, ...], axis = 0))
        nudates.append(data)

    nuvar = np.stack(nuvar)

    return nuvar, nudates


def global_mean(field, latitude):
    """
    Calculates a global mean of field, weighting with the cosine of latitude.

    Accepts 3D (time, lat, lon) and 2D (lat, lon) input arrays.
    """
    weights_array = abs(np.cos(np.deg2rad(latitude)))

    zonal_field = zonal_mean(field)
    mea = np.average(zonal_field, weights = weights_array, axis = -1)

    return mea


def zonal_mean(field):
    """
    Calculates a zonal mean of field.

    Accepts 3D (time, lat, lon) and 2D (lat, lon) input arrays.
    """

    mea = np.mean(field, axis = -1)

    return mea

#######################################################
#
###     EOF computation / Clustering / algebra
#
#######################################################

def Rcorr(x,y):
    """
    Returns correlation coefficient between two array of arbitrary shape.
    """
    return np.corrcoef(x.flatten(), y.flatten())[1,0]


def distance(x, y):
    """
    L2 distance.
    """
    return LA.norm(x-y)


def E_rms(x,y):
    """
    Returns root mean square deviation: sqrt(1/N sum (xn-yn)**2).
    """
    n = x.size
    #E = np.sqrt(1.0/n * np.sum((x.flatten()-y.flatten())**2))
    E = 1/np.sqrt(n) * LA.norm(x-y)

    return E


def E_rms_cp(x,y):
    """
    Returns centered-pattern root mean square, e.g. first subtracts the mean to the two series and then computes E_rms.
    """
    x1 = x - x.mean()
    y1 = y - y.mean()

    E = E_rms(x1, y1)

    return E


def cosine(x,y):
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
        raise ValueError('Too many dimensions')


def cosine_cp(x,y):
    """
    Before calculating the cosine, subtracts the mean to both x and y. This is exactly the same as calculating the correlation coefficient R.
    """

    x1 = x - x.mean()
    y1 = y - y.mean()

    return cosine(x1,y1)


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
    ndis = np.sum(deltas > 3*deltauno) # Finds number of divisions in the dataset (n_seasons - 1)

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
    freq_mem_abs = calc_clus_freq(labels)/100.

    varopt = np.sum(freq_mem_abs*np.sum(centroids**2, axis = 1))

    varint = np.sum([np.sum((pc-centroids[lab])**2) for pc, lab in zip(pcs, labels)])/len(labels)

    varopt = varopt/varint

    return varopt


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

    freq_mem = 100.*num_mem/len(labels)

    return freq_mem


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

    return freqs


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

    freq_mem = calc_clus_freq(labels)
    new_ord = freq_mem.argsort()[::-1]

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

    perms = list(itt.permutations(range(numclus)))
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

    return et, patcor


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

    pattern = np.mean(var[mask,:,:], axis = 0)
    # pattern_std = np.std(var[mask,:,:], axis = 0)

    return pattern #, pattern_std


def calc_residence_times(indices, dates = None, count_incomplete = True, skip_singleday_pause = True):
    """
    Calculates residence times given a set of indices indicating cluster numbers.

    For each cluster, the observed set of residence times is given and a transition probability is calculated.

    < dates > : list of datetime objects (or datetime array). If set, the code eliminates spourious transitions between states belonging to two different seasons.

    < count_incomplete > : counts also residence periods that terminate with the end of the season and not with a transition to another cluster.
    < skip_singleday_pause > : If a regime stops for a single day and then starts again, the two periods will be summed on. The single day is counted as another regime's day, to preserve the total number.
    """
    indices = np.array(indices)
    numclus = int(indices.max() + 1)

    resid_times = []
    if dates is None:
        for clu in range(numclus):
            clu_resids = []
            okclu = indices == clu
            init = False
            pause = False

            for el in okclu:
                if el:
                    if not init:
                        init = True
                        num_days = 1
                    else:
                        num_days += 1
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
                            init = False
                            pause = False
                    else:
                        if init:
                            clu_resids.append(num_days)
                            init = False


            resid_times.append(np.array(clu_resids))
    else:
        dates = pd.to_datetime(dates)
        duday = pd.Timedelta('2 days 00:00:00')
        dates_init = []
        for clu in range(numclus):
            clu_resids = []
            clu_dates_init = []
            okclu = indices == clu
            init = False
            pause = False

            old_date = dates[0]
            for el, dat in zip(okclu, dates):
                if dat - old_date > duday:
                    init = False
                    if count_incomplete:
                        clu_resids.append(num_days)
                    else:
                        clu_dates_init.pop()
                if el:
                    if not init:
                        init = True
                        num_days = 1
                        clu_dates_init.append(dat)
                    else:
                        num_days += 1
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
                            init = False
                            pause = False
                    else:
                        if init:
                            clu_resids.append(num_days)
                            init = False

                old_date = dat

            resid_times.append(np.array(clu_resids))
            dates_init.append(np.array(clu_dates_init))

    if dates is None:
        return np.array(resid_times)
    else:
        return np.array(resid_times), np.array(dates_init)


def compute_centroid_distance(PCs, centroids, labels):
    """
    Calculates distances of pcs to centroid assigned by label.
    """
    distances = []
    for pc, lab in zip(PCs, labels):
        distances.append( distance(pc, centroids[lab]) )

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
        cluspattern = np.mean(var[mask,:,:], axis=0)
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
    coppie = list(itt.combinations(range(numclus), 2))
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
        others = range(numclus)
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
###     Plots and visualization
#
#######################################################

def color_brightness(color):
    return (color[0] * 299 + color[1] * 587 + color[2] * 114)/1000

def color_set(n, cmap = 'nipy_spectral', bright_thres = 0.6, full_cb_range = False):
    """
    Gives a set of n well chosen (hopefully) colors, darker than bright_thres. bright_thres ranges from 0 (darker) to 1 (brighter).

    < full_cb_range > : if True, takes all cb values. If false takes the portion 0.05/0.95.
    """
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

    return colors


def plot_mapc_on_ax(ax, data, lat, lon, proj, cmappa, cbar_range, n_color_levels = 21, draw_contour_lines = False, n_lines = 5, clip_to_box = True):
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

    """

    clevels = np.linspace(cbar_range[0], cbar_range[1], n_color_levels)
    print(clevels)
    print(np.min(data), np.max(data))

    ax.set_global()
    ax.coastlines(linewidth = 2)

    cyclc = False
    if max(lon)+10. % 360 > min(lon):
        print('Adding cyclic point\n')
        cyclic = True
        #lon = np.append(lon, 360)
        #data = np.c_[data,data[:,0]]
        data, lon = cutil.add_cyclic_point(data, coord = lon)

        # if np.min(lat) >= 0:
        #     latpiu = np.sort(-1.0*lat[lat > 0])
        #     lat = np.concatenate([latpiu, lat])
        #     data = np.concatenate([np.zeros((len(latpiu), len(lon))), data])

    xi,yi = np.meshgrid(lon,lat)

    map_plot = ax.contourf(xi, yi, data, clevels, cmap = cmappa, transform = ccrs.PlateCarree(), extend = 'both', corner_mask = False)
    if draw_contour_lines:
        map_plot_lines = ax.contour(xi, yi, data, n_lines, colors = 'k', transform = ccrs.PlateCarree(), linewidth = 0.5)

    if clip_to_box:
        if cyclic:
            latlonlim = [-180, 180, lat.min(), lat.max()]
        else:
            latlonlim = [lon.min(), lon.max(), lat.min(), lat.max()]
        ax.set_extent(latlonlim, crs = proj)

    return map_plot


def get_cbar_range(data, symmetrical = False, percentiles = (5,95), n_color_levels = None):
    mi = np.percentile(data, percentiles[0])
    ma = np.percentile(data, percentiles[1])
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


def plot_map_contour(data, lat, lon, filename = None, visualization = 'standard', central_lat_lon = None, cmap = 'RdBu_r', title = None, xlabel = None, ylabel = None, cb_label = None, cbar_range = None, plot_anomalies = True, n_color_levels = 21, draw_contour_lines = False, n_lines = 5, color_percentiles = (2,98), figsize = (8,6)):
    """
    Plots a single map to a figure.

    < data >: the field to plot
    < lat, lon >: latitude and longitude
    < filename >: name of the file to save the figure to. If None, the figure is just shown.

    < visualization >: 'standard' calls PlateCarree cartopy map, 'polar' calls Orthographic map.
    < central_lat_lon >: Tuple, (clat, clon). Is needed only for Orthographic plots. If not given, the mean lat and lon are taken.
    < cmap >: name of the color map.
    < cbar_range >: limits of the color bar.

    < plot_anomalies >: if True, the colorbar is symmetrical, so that zero corresponds to white. If cbar_range is set, plot_anomalies is set to False.
    < n_color_levels >: number of color levels.
    < draw_contour_lines >: draw lines in addition to the color levels?
    < n_lines >: number of lines to draw.

    """

    if visualization == 'standard':
        proj = ccrs.PlateCarree()
    elif visualization == 'polar':
        if central_lat_lon is not None:
            (clat, clon) = central_lat_lon
        else:
            clat = lat.min() + (lat.max()-lat.min())/2
            clon = lon.min() + (lon.max()-lon.min())/2
        proj = ccrs.Orthographic(central_longitude=clon, central_latitude=clat)
    else:
        raise ValueError('visualization {} not recognised. Only "standard" or "polar" accepted'.format(visualization))

    # Determining color levels
    cmappa = cm.get_cmap(cmap)

    if cbar_range is None:
        mi = np.percentile(data, 5)
        ma = np.percentile(data, 95)
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
    fig4 = plt.figure(figsize = figsize)
    ax = plt.subplot(projection = proj)

    map_plot = plot_mapc_on_ax(ax, data, lat, lon, proj, cmappa, cbar_range, n_color_levels = n_color_levels, draw_contour_lines = draw_contour_lines, n_lines = n_lines)

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
        plt.close(fig4)
    else:
        plt.show(fig4)

    return


def plot_double_sidebyside(data1, data2, lat, lon, filename = None, visualization = 'standard', central_lat_lon = None, cmap = 'RdBu_r', title = None, xlabel = None, ylabel = None, cb_label = None, stitle_1 = None, stitle_2 = None, cbar_range = None, plot_anomalies = True, n_color_levels = 21, draw_contour_lines = False, n_lines = 5, color_percentiles = (2,98)):
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

    """

    if visualization == 'standard':
        proj = ccrs.PlateCarree()
    elif visualization == 'polar':
        if central_lat_lon is not None:
            (clat, clon) = central_lat_lon
        else:
            clat = lat.min() + (lat.max()-lat.min())/2
            clon = lon.min() + (lon.max()-lon.min())/2
        proj = ccrs.Orthographic(central_longitude=clon, central_latitude=clat)
    else:
        raise ValueError('visualization {} not recognised. Only "standard" or "polar" accepted'.format(visualization))

    # Determining color levels
    cmappa = cm.get_cmap(cmap)

    data = np.stack([data1,data2])

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

    fig = plt.figure(figsize=(24,14))

    ax = plt.subplot(1, 2, 1, projection=proj)
    map_plot = plot_mapc_on_ax(ax, data1, lat, lon, proj, cmappa, cbar_range, n_color_levels = n_color_levels, draw_contour_lines = draw_contour_lines, n_lines = n_lines)
    ax.set_title(stitle_1, fontsize = 25)
    ax = plt.subplot(1, 2, 2, projection=proj)
    map_plot = plot_mapc_on_ax(ax, data2, lat, lon, proj, cmappa, cbar_range, n_color_levels = n_color_levels, draw_contour_lines = draw_contour_lines, n_lines = n_lines)
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
        plt.close(fig)
    else:
        plt.show(fig)

    return


def plot_multimap_contour(dataset, lat, lon, filename, max_ax_in_fig = 30, number_subplots = True, cluster_labels = None, cluster_colors = None, repr_cluster = None, visualization = 'standard', central_lat_lon = None, cmap = 'RdBu_r', title = None, xlabel = None, ylabel = None, cb_label = None, cbar_range = None, plot_anomalies = True, n_color_levels = 21, draw_contour_lines = False, n_lines = 5, subtitles = None, color_percentiles = (5,95), fix_subplots_shape = None, figsize = (15,12)):
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

    if visualization == 'standard':
        proj = ccrs.PlateCarree()
    elif visualization == 'polar':
        if central_lat_lon is not None:
            (clat, clon) = central_lat_lon
        else:
            clat = lat.min() + (lat.max()-lat.min())/2
            clon = lon.min() + (lon.max()-lon.min())/2
        proj = ccrs.Orthographic(central_longitude=clon, central_latitude=clat)
    else:
        raise ValueError('visualization {} not recognised. Only "standard" or "polar" accepted'.format(visualization))

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
        numens_ok = int(np.ceil(numens/num_figs))
        side1 = int(np.ceil(np.sqrt(numens_ok)))
        side2 = int(np.ceil(numens_ok/float(side1)))
    else:
        (side1, side2) = fix_subplots_shape
        numens_ok = side1*side2
        num_figs = int(np.ceil(1.0*numens/numens_ok))

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

    for i in range(num_figs):
        fig = plt.figure(figsize = figsize)#(24,14)
        for nens in range(numens_ok*i, numens_ok*(i+1)):
            nens_rel = nens - numens_ok*i
            ax = plt.subplot(side1, side2, nens_rel+1, projection=proj)

            map_plot = plot_mapc_on_ax(ax, dataset[nens], lat, lon, proj, cmappa, cbar_range, n_color_levels = n_color_levels, draw_contour_lines = draw_contour_lines, n_lines = n_lines)

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

        cax = plt.axes([0.1, 0.06, 0.8, 0.03])
        #cax = plt.axes([0.1, 0.1, 0.8, 0.05])
        cb = plt.colorbar(map_plot,cax=cax, orientation='horizontal')
        cb.ax.tick_params(labelsize=18)
        cb.set_label(cb_label, fontsize=20)

        plt.suptitle(title, fontsize=35, fontweight='bold')

        top    = 0.90  # the top of the subplots
        bottom = 0.13    # the bottom
        left   = 0.02    # the left side
        right  = 0.98  # the right side
        hspace = 0.20   # the amount of height reserved for white space between subplots
        wspace = 0.05    # the amount of width reserved for blank space between subplots
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

        fig.savefig(namef[i])
        plt.close(fig)

    return


def plot_animation_map(maps, lat, lon, labels = None, fps_anim = 5, title = None, filename = None, visualization = 'standard', central_lat_lon = None, cmap = 'RdBu_r', xlabel = None, ylabel = None, cb_label = None, cbar_range = None, plot_anomalies = True, n_color_levels = 21, draw_contour_lines = False, n_lines = 5, color_percentiles = (2,98), figsize = (8,6)):
    """
    Shows animation of a sequence of maps or saves it to a gif file.
    < maps > : list, the sequence of maps to be plotted.
    < labels > : title of each map.

    < filename > : if None the animation is run and shown, instead it is saved to filename.
    < **kwargs > : all other kwargs as in plot_map_contour()
    """

    if labels is None:
        labels = np.arange(len(maps))

    if visualization == 'standard':
        proj = ccrs.PlateCarree()
    elif visualization == 'polar':
        if central_lat_lon is not None:
            (clat, clon) = central_lat_lon
        else:
            clat = lat.min() + (lat.max()-lat.min())/2
            clon = lon.min() + (lon.max()-lon.min())/2
        proj = ccrs.Orthographic(central_longitude=clon, central_latitude=clat)
    else:
        raise ValueError('visualization {} not recognised. Only "standard" or "polar" accepted'.format(visualization))

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

        plot_mapc_on_ax(ax, mapa, lat, lon, proj, cmappa, cbar_range, n_color_levels = n_color_levels, draw_contour_lines = draw_contour_lines, n_lines = n_lines)
        showdate.set_text('{}'.format(lab))

        return

    mapa = maps[0]
    map_plot = plot_mapc_on_ax(ax, mapa, lat, lon, proj, cmappa, cbar_range, n_color_levels = n_color_levels, draw_contour_lines = draw_contour_lines, n_lines = n_lines)

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

    plt.thetagrids(anggr, labels=labgr, frac = 1.1, zorder = 0)

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


def Taylor_plot(models, observation, filename = None, ax = None, title = None, label_bias_axis = None, label_ERMS_axis = None, colors = None, markers = None, only_first_quarter = False, legend = True, marker_edge = None, labels = None, obs_label = None, mod_points_size = 35, obs_points_size = 50, enlarge_rmargin = True):
    """
    Produces two figures:
    - a Taylor diagram
    - a bias/E_rms plot

    < models > : a set of patterns (2D matrices or pc vectors) corresponding to different simulation results/model behaviours.
    < observation > : the corresponding observed pattern. observation.shape must be the same as each of the models' shape.

    < colors > : list of colors for each model point
    < markers > : list of markers for each model point
    """
    if ax is None and filename is None:
        raise ValueError('Where do I plot this? specify ax or filename')
    elif filename is not None and ax is not None:
        raise ValueError('Where do I plot this? specify ax OR filename, not BOTH')

    if ax is None:
        fig6 = plt.figure(figsize=(8,6))
        ax = fig6.add_subplot(111, polar = True)

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

    ax.set_thetagrids(anggr, labels=labgr, frac = 1.1)

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
        ax.scatter(ang, sig, s = mod_points_size, color = col, marker = sym, edgecolor = marker_edge, label = lab)

    ax.scatter([0.], [sigma_obs], color = 'black', s = obs_points_size, marker = 'D', clip_on=False, label = obs_label)

    for sig in [1., 2., 3.]:
        circle = plt.Circle((sigma_obs, 0.), sig*sigma_obs, transform=ax.transData._b, fill = False, edgecolor = 'black', linestyle = '--')
        ax.add_artist(circle)

    if legend:
        ax.legend(fontsize = 'small', loc = 1)

    if filename is None:
        return

    top    = 0.88  # the top of the subplots
    bottom = 0.1    # the bottom
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

    if markers is None:
        markers = ['o']*len(angles)
    if labels is None:
        labels = [None]*len(angles)
    for bia, ctpr, col, sym, lab in zip(biases, ctr_patt_RMS, colors, markers, labels):
        ax.scatter(bia, ctpr, s = mod_points_size+20, color = col, marker = sym, edgecolor = marker_edge, label = lab)

    plt.xlabel(label_bias_axis)
    plt.ylabel(label_ERMS_axis)

    for sig in [1., 2., 3.]:
        circle = plt.Circle((np.mean(observation), 0.), sig*sigma_obs, fill = False, edgecolor = 'black', linestyle = '--')
        ax.add_artist(circle)

    plt.scatter(np.mean(observation), 0., color = 'black', s = obs_points_size+20, marker = 'D', zorder = 5, label = obs_label)

    if legend:
        plt.legend(fontsize = 'small', loc = 1)
    plt.grid()

    fig7.savefig(nuname)

    return


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
