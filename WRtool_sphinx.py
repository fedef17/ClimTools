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

base_files = ['lcb0-1850-2100-NDJFM_zg500_NH_14473.nc', 'lcb1-1850-2100-NDJFM_zg500_NH_14473.nc', 'lcb2-1850-2100-NDJFM_zg500_NH_14473.nc']
base_tags = ['lcb0', 'lcb1', 'lcb2']

stoc_files = ['lcs0-1850-2100-NDJFM_zg500_NH_14473.nc', 'lcs1-1850-2100-NDJFM_zg500_NH_14473.nc', 'lcs2-1850-2100-NDJFM_zg500_NH_14473.nc']
stoc_tags = ['lcs0', 'lcs1', 'lcs2']

tot_files = base_files+stoc_files
tot_tags = base_tags+stoc_tags

#years1 = np.arange(1855,2066,10)
#years2 = np.arange(1885,2096,10)
years1 = np.arange(1850,2071,5)
years2 = np.arange(1880,2101,5)

yr_ranges = []
yr_ranges.append((1850,2005))
yr_ranges.append((2006,2100))
for y1, y2 in zip(years1, years2):
    yr_ranges.append((y1,y2))

yr_ranges = [(1850,2100)]

erafile = cart_in + 'era_1979-2014_nh.nc'
ERA_ref_EAT = cd.WRtool_from_file(erafile, 'DJF', 'EAT', extract_level_4D = 50000., numclus = 4, output_results_only = False)
ERA_ref_PNA = cd.WRtool_from_file(erafile, 'DJF', 'PNA', extract_level_4D = 50000., numclus = 4, output_results_only = False)

results = dict()
# Beginning the analysis
for fil,tag in zip(tot_files, tot_tags):
    print('Analyzing {}\n'.format(tag))
    var, level, lat, lon, dates, time_units, var_units, time_cal = ctl.read4Dncfield(cart_in+fil, extract_level = 50000.)

    var_season, dates_season = ctl.sel_season(var, dates, 'DJF')
    dates_season_pdh = pd.to_datetime(dates_season)

    for ran in yr_ranges:
        runsig = True
        if ran[1]-ran[0] > 50:
            runsig = False
        print('analyzing range {}\n'.format(ran))
        okdat = (dates_season_pdh.year >= ran[0]) & (dates_season_pdh.year <= ran[1])
        var_season_range = var_season[okdat, ...]
        dates_season_range = dates_season[okdat, ...]

        area = 'EAT'
        ref_solver = ERA_ref_EAT['solver']
        ref_patterns_area = ERA_ref_EAT['cluspattern_area']
        resu = cd.WRtool_core(var_season_range, lat, lon, dates_season_range, area, run_significance_calc = runsig, ref_solver = ref_solver, ref_patterns_area = ref_patterns_area)
        results[(tag, area, ran)] = resu

        area = 'PNA'
        ref_solver = ERA_ref_PNA['solver']
        ref_patterns_area = ERA_ref_PNA['cluspattern_area']
        resu = cd.WRtool_core(var_season_range, lat, lon, dates_season_range, area, numclus = 4, run_significance_calc = runsig, ref_solver = ref_solver, ref_patterns_area = ref_patterns_area)
        results[(tag, area, ran)] = resu


pickle.dump(results, open(cart_out+'results_SPHINX_fullperiod.p','w'))
