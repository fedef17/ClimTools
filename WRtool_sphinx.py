#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os

from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as PathEffects

import netCDF4 as nc
import cartopy.crs as ccrs
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

import climtools_lib as ctl
import climdiags as cd

#######################################

"""
# load the base and stoc files

# for each one run wrtool for:
- the historic period/ in chunks of 30 years? / for the future?
- for NAO/PNA/NH

# compute residence times and monthly population indices

# check:
- differences among single members/different periods/stoc vs base
- correlation with virna ocean indices

# run on the ensemble
"""

cart_in = '/home/federico/work/SPHINX/'
cart_out = '/home/federico/work/SPHINX/Lisboa/'

base_files = ['lcb0-1850-2100-NDJFM_zg500_NH.nc', 'lcb1-1850-2100-NDJFM_zg500_NH.nc', 'lcb2-1850-2100-NDJFM_zg500_NH.nc']
base_tags = ['lcb0', 'lcb1', 'lcb2']

stoc_files = ['lcs0-1850-2100-NDJFM_zg500_NH.nc', 'lcs1-1850-2100-NDJFM_zg500_NH.nc', 'lcs2-1850-2100-NDJFM_zg500_NH.nc']
stoc_tags = ['lcs0', 'lcs1', 'lcs2']

tot_files = base_files+stoc_files
tot_tags = base_tags+stoc_tags

years1 = np.arange(1850,2071,10)
years2 = np.arange(1880,2101,10)

yr_ranges = []
yr_ranges.append((1850,2005))
yr_ranges.append((2005,2100))
for y1, y2 in zip(years1, years2):
    yr_ranges.append((y1,y2))

results = dict()
# Beginning the analysis
for fil,tag in zip(tot_files, tot_tags):
    print('Analyzing {}\n'.format(tag))
    var, level, lat, lon, dates, time_units, var_units, time_cal = ctl.read4Dncfield(cart_in+fil, extract_level = 50000.)

    var_season, dates_season = ctl.sel_season(var, dates, 'DJF')
    dates_season_pdh = pd.to_datetime(dates_season)

    for ran in yr_ranges:
        print('analyzing range {}\n'.format(ran))
        okdat = (dates_season_pdh.year >= sel_range[0]) & (dates_season_pdh.year <= sel_range[1])
        var_season = var_season[okdat, ...]
        dates_season = dates_season[okdat, ...]

        area = 'EAT'
        resu = cd.WRtool_core(var_season, lat, lon, dates_season, area)
        results[(tag, area, ran)] = resu
        area = 'PNA'
        resu = cd.WRtool_core(var_season, lat, lon, dates_season, area, numclus = 3)
        results[(tag, area, ran)] = resu


pickle.dump(results, open(cart_out+'results_SPHINX.p','w'))
