#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os

from matplotlib import pyplot as plt

import netCDF4 as nc
import cartopy.crs as ccrs
import pandas as pd

from numpy import linalg as LA
from eofs.standard import Eof
from scipy import stats

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

def read4Dncfield(ifile, extract_level = None):
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
    if ('level' in variabs):
        level       = fh.variables['level'][:]
        level_units = fh.variables['level'].units
    elif ('lev' in variabs):
        level       = fh.variables['lev'][:]
        level_units = fh.variables['lev'].units
    elif ('pressure' in variabs):
        level       = fh.variables['pressure'][:]
        level_units = fh.variables['pressure'].units
    elif ('plev' in variabs):
        level       = fh.variables['plev'][:]
        level_units = fh.variables['plev'].units
    elif ('plev8' in variabs):
        level       = fh.variables['plev8'][:]
        level_units = fh.variables['plev8'].units

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
        if level_units=='millibar' or level_units=='hPa':
            l_sel=int(np.where(level==lvel)[0])
            print('Selecting level {0} millibar'.format(lvel))
        elif level_units=='Pa':
            l_sel=int(np.where(level==lvel*100)[0])
            print('Selecting level {0} Pa'.format(lvel*100))
        del level
        level=lvel
        var         = fh.variables[variabs[-1]][:,l_sel,:,:]
        txt='{0}{1} dimension for a single ensemble member [time x lat x lon]: {2}'.format(variabs[-1],lvel,var.shape)
    else:
        var         = fh.variables[variabs[-1]][:,:,:,:]
        txt='{0} dimension for a single ensemble member [time x lat x lon]: {1}'.format(variabs[-1],var.shape)
    #print(fh.variables)
    if var_units == 'm**2 s**-2':
        print('From geopotential (m**2 s**-2) to geopotential height (m)')   # g0=9.80665 m/s2
        var=var/9.80665
        var_units='m'
    print('calendar: {0}, time units: {1}'.format(time_cal,time_units))
    dates = nc.num2date(time,time_units,time_cal)
    fh.close()

    print(txt)

    return var, level, lat, lon, dates, time_units, var_units, time_cal

def read3Dncfield(ifile):
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
    #print(fh.variables)
    dates=nc.num2date(time,time_units)
    fh.close()

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

    return var_area,lat[latidx],lon_new[lonidx]


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
            data = pd.to_datetime('2001{:02d}{:02d}'.format(mon,day), format='%Y%m%d')
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

    filt_mean = np.concatenate(filt_mean)
    filt_std = np.concatenate(filt_std)

    return filt_mean, dates_ok, filt_std


def monthly_climatology(var, dates, refyear = 2001):
    """
    Performs a monthly climatological mean of the dataset.
    Works both on monthly and daily datasets.

    Dates of the climatology are referred to year <refyear>, has no effect on the calculation.
    """

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

    filt_mean = np.concatenate(filt_mean)
    filt_std = np.concatenate(filt_std)

    return filt_mean, dates_ok, filt_std


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

#######################################################
#
###     EOF computation / Clustering / algebra
#
#######################################################

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


def eof_computation_bkp(var, varunits, lat, lon):
    """
    Compatibility version.

    Computes the EOFs of a given variable. In the first dimension there has to be different time or ensemble realizations of variable.

    The data are weighted with respect to the cosine of latitude.
    """
    # The data array is dimensioned (ntime, nlat, nlon) and in order for the latitude weights to be broadcastable to this shape, an extra length-1 dimension is added to the end:
    weights_array = np.sqrt(np.cos(np.deg2rad(lat)))[:, np.newaxis]

    start = datetime.datetime.now()
    solver = Eof(var, weights=weights_array)
    end = datetime.datetime.now()
    print('EOF computation took me {:6.1f} seconds'.format(end-start))

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
        weights_array = np.sqrt(np.cos(np.deg2rad(lat)))[:, np.newaxis]
    else:
        weights_array = None

    start = datetime.datetime.now()
    solver = Eof(var, weights=weights_array)
    end = datetime.datetime.now()
    print('EOF computation took me {:6.1f} seconds'.format(end-start))

    return solver


def Kmeans_clustering(solver, numclus, numpcs, order_by_frequency = True, algorithm = 'sklearn', n_init_sk = 600,  max_iter_sk = 1000, npart_molt = 100):
    """
    Computes the Kmeans clustering on the given pcs. Two algorithms can be used:
    - the sklearn KMeans algorithm (algorithm = 'sklearn')
    - the fortran algorithm developed by Molteni (algorithm = 'molteni')
    """
    from sklearn.cluster import KMeans
    import ctool

    PCs = solver.pcs()[:,:numpcs]

    start = datetime.datetime.now()
    if algorithm == 'sklearn':
        clus = KMeans(n_clusters=numclus, n_init = n_init_sk, max_iter = max_iter_sk)

        clus.fit(PCs)
        centroids = clus.cluster_centers_
        labels = clus.labels_
    elif algorithm == 'molteni':
        pc = np.transpose(PCs)
        nfcl, labels, centroids, varopt, iseed = ctool.cluster_toolkit.clus_opt(numclus, npart_molt, pc)
        centroids = np.array(centroids)
        labels = np.array(labels) - 1 # because of fortran numbering
    else:
        raise ValueError('algorithm {} not recognised'.format(algorithm))

    end = datetime.datetime.now()
    print('k-means algorithm took me {:6.1f} seconds'.format(end-start))


    ## Ordering clusters for number of members
    centroids = np.array(centroids)
    labels = np.array(labels)

    if order_by_frequency:
        num_mem = []
        for i in range(numclus):
            num_mem.append(np.sum(labels == i))
        num_mem = np.array(num_mem)

        new_ord = num_mem.argsort()[::-1]
        centroids = centroids[new_ord]

        labels_new = np.array(labels)
        for nu, i in zip(range(numclus), new_ord):
            labels_new[labels == i] = nu
        labels = labels_new

    return centroids, labels


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
    coppie = list(combinations(range(numclus), 2))
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
