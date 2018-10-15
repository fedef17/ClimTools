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

##############################################
############ INPUTS #########

season = 'DJF'
area = 'EAT'
wnd = 5

numpcs = 4
numclus = 4

#inputs = 'all_models'
inputs = 'ecmwf_ensemble_full'

if inputs == 'all_models':
    cart_in = '/home/federico/work/Primavera/DATA_stream1/'
    cart_out = '/home/federico/work/Primavera/Results_WP2/'
    filenames = ['zg500_Aday_CMCC-CM2-HR4_288x192_regrid25_1979-2014.nc', 'zg500_Aday_CMCC-CM2-VHR4_1152x768_regrid25_1979-2014.nc', 'zg500_Aday_CNRM-CM6-1-HR_TL359_regrid25_1979-2014.nc', 'zg500_Aday_CNRM-CM6-1_TL127_regrid25_1979-2014.nc', 'zg500_Aday_EC-Earth3-HR_T511_regrid25_1979-2014.nc', 'zg500_Aday_EC-Earth3_T255_regrid25_1979-2014.nc', 'zg500_Aday_ECMWF-IFS-HR_1442x1021_regrid25_1979-2014.nc', 'zg500_Aday_ECMWF-IFS-LR_360x181_regrid25_1979-2014.nc', 'zg500_Aday_MPIESM-1-2-HR_384x192_regrid25_1979-2014.nc', 'zg500_Aday_MPIESM-1-2-XR_768x384_regrid25_1979-2014.nc', 'zg500_APrimday_HadGEM3-GC31-HM_N512_regrid25_1979-2014.nc', 'zg500_APrimday_HadGEM3-GC31-MM_N216_regrid25_1979-2014.nc']
    tags = ['CMCC_LR','CMCC_HR', 'CNRM_HR', 'CNRM_LR', 'ECE_HR', 'ECE_LR', 'ECMWF_HR', 'ECMWF_LR', 'MPI_LR', 'MPI_HR', 'HadGEM_HR', 'HadGEM_LR']
elif inputs == 'ecmwf_ensemble':
    cart_in = '/home/federico/work/Primavera/DATA_stream1/Primavera_runs_chrisrob/'
    cart_out = '/home/federico/work/Primavera/Results_WP2/ECMWF_ensemble/'
    filenames = ['rid_HR_1979-2014_ens1.nc', 'rid_HR_1979-2014_ens2.nc', 'rid_HR_1979-2014_ens3.nc', 'rid_HR_1979-2014_ens4.nc', 'rid_LR_1979-2014_ens1.nc', 'rid_LR_1979-2014_ens2.nc', 'rid_LR_1979-2014_ens3.nc', 'rid_LR_1979-2014_ens4.nc', 'rid_LR_1979-2014_ens5.nc', 'rid_LR_1979-2014_ens6.nc']
    tags = ['HR_1', 'HR_2', 'HR_3', 'HR_4', 'LR_1', 'LR_2', 'LR_3', 'LR_4', 'LR_5', 'LR_6']
elif inputs == 'ecmwf_ensemble_full':
    cart_in = '/home/federico/work/Primavera/DATA_stream1/Primavera_runs_chrisrob/'
    cart_out = '/home/federico/work/Primavera/Results_WP2/ECMWF_ensemble_full/'
    filenames = ['AMIP_HR_ens1.nc','AMIP_HR_ens2.nc','AMIP_HR_ens3.nc','AMIP_HR_ens4.nc','AMIP_LR_ens1.nc','AMIP_LR_ens2.nc','AMIP_LR_ens3.nc','AMIP_LR_ens4.nc','AMIP_LR_ens5.nc','AMIP_LR_ens6.nc']
    tags = ['HR_1_full', 'HR_2_full', 'HR_3_full', 'HR_4_full', 'LR_1_full', 'LR_2_full', 'LR_3_full', 'LR_4_full', 'LR_5_full', 'LR_6_full']

ifile_ERA = 'zg500_ERAInterim_1979-2014.nc'

ifile_NCEP = 'zg500_Aday_NCEPNCAR_2deg_1979-2014.nc'
#############################

def compute(ifile):
    ## PRECOMPUTE
    print(ifile)
    print('Running precompute\n')
    var, level, lat, lon, dates, time_units, var_units, time_cal = ctl.read4Dncfield(ifile, extract_level = 50000.)

    var_season, dates_season = ctl.sel_season(var, dates, season, wnd)

    climat_mean, dates_climat, climat_std = ctl.daily_climatology(var_season, dates_season, wnd)

    var_anom = ctl.anomalies_daily(var_season, dates_season, climat_mean = climat_mean, dates_climate_mean = dates_climat)

    var_area, lat_area, lon_area = ctl.sel_area(lat, lon, var_anom, area)
    print(var_area.shape)

    print('Running compute\n')
    #### EOF COMPUTATION
    eof_solver = ctl.eof_computation(var_area, lat_area)
    PCs = eof_solver.pcs()[:, :numpcs]

    print('Running clustering\n')
    #### CLUSTERING
    centroids, labels = ctl.Kmeans_clustering(PCs, numclus, algorithm = 'molteni')

    cluspattern = ctl.compute_clusterpatterns(var_anom, labels)
    cluspatt_area = []
    for clu in cluspattern:
        cluarea, _, _ = ctl.sel_area(lat, lon, clu, area)
        cluspatt_area.append(cluarea)
    cluspatt_area = np.stack(cluspatt_area)

    varopt = ctl.calc_varopt_molt(PCs, centroids, labels)
    print('varopt: {:8.4f}\n'.format(varopt))
    freq_mem = ctl.calc_clus_freq(labels)

    print('Running clus sig\n')
    significance = ctl.clusters_sig(PCs, centroids, labels, dates_season, nrsamp = 5000)
    # significance_2 = ctl.clusters_sig(PCs, centroids, labels, dates_season, nrsamp = 5000)
    # print('Significances: {:7.3f} vs {:7.3f}\n'.format(significance, significance_2))

    return lat, lon, var_anom, eof_solver, centroids, labels, cluspattern, cluspatt_area, freq_mem, significance

#############################################
results = dict()
results['significance'] = dict()
results['freq_mem'] = dict()
results['cluspattern'] = dict()
results['cluspattern_area'] = dict()
results['labels'] = dict()
results['et'] = dict()
results['patcor'] = dict()

### ERA reference
lat, lon, var_ERA, solver_ERA, centroids_ERA, labels_ERA, cluspattern_ERA, cluspatt_area_ERA, freq_mem_ERA, significance_ERA = compute(cart_in+ifile_ERA)

tag = 'ERA'
print('\n ----------------------\n')
print('Results for {}\n'.format(tag))
results['significance'][tag] = significance_ERA
print('Significance: {:6.3f}\n'.format(significance_ERA))
results['freq_mem'][tag] = freq_mem_ERA
print('frequency: {}\n'.format(freq_mem_ERA))
results['cluspattern'][tag] = cluspattern_ERA
results['cluspattern_area'][tag] = cluspatt_area_ERA
results['labels'][tag] = labels_ERA
print('----------------------\n')

### NCEP reference
lat, lon, var_NCEP, solver_NCEP, centroids_NCEP, labels_NCEP, cluspattern_NCEP, cluspatt_area_NCEP, freq_mem_NCEP, significance_NCEP = compute(cart_in+ifile_NCEP)

tag = 'NCEP'
print('\n ----------------------\n')
print('Results for {}\n'.format(tag))
results['significance'][tag] = significance_NCEP
print('Significance: {:6.3f}\n'.format(significance_NCEP))
results['freq_mem'][tag] = freq_mem_NCEP
print('frequency: {}\n'.format(freq_mem_NCEP))
results['cluspattern'][tag] = cluspattern_NCEP
results['cluspattern_area'][tag] = cluspatt_area_NCEP
results['labels'][tag] = labels_NCEP
print('----------------------\n')

for fil, tag in zip(filenames, tags):
    print('\n analyzing: {} --> {} \n'.format(fil, tag))
    lat, lon, var_anom, solver, centroids, labels, cluspattern, cluspatt_area, freq_mem, significance = compute(cart_in+fil)

    perm, centroids, labels, et, patcor = ctl.clus_compare_projected(centroids, labels, cluspatt_area, cluspatt_area_ERA, solver_ERA, numpcs)

    cluspattern = cluspattern[perm, ...]
    cluspatt_area = cluspatt_area[perm, ...]
    print(freq_mem, perm)
    freq_mem = np.array(freq_mem)
    freq_mem = freq_mem[perm]

    namef=cart_out+'regime_indices_{}.txt'.format(tag)
    np.savetxt(namef, labels, fmt='%d')

    print('\n ----------------------\n')
    print('Results for {}\n'.format(tag))
    results['significance'][tag] = significance
    print('Significance: {:6.3f}\n'.format(significance))
    results['freq_mem'][tag] = freq_mem
    print('frequency: {}\n'.format(freq_mem))
    results['cluspattern'][tag] = cluspattern
    results['cluspattern_area'][tag] = cluspatt_area
    results['labels'][tag] = labels
    results['et'][tag] = et
    print('et: {}\n'.format(et))
    results['patcor'][tag] = patcor
    print('patcor: {}\n'.format(patcor))
    print('----------------------\n')
    # e qui i plotssss

pickle.dump(results, open(cart_out+'res_primavera.p', 'w'))
