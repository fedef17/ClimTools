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

####################################################

"""
Diagnostics for standard climate outputs. Contains higher level tools that make use of climtools_lib.
Tools contained:
- WRtool
"""

####################################################
def WRtool_from_file(ifile, season, area, sel_range = None, extract_level_4D = None, **kwargs):
    """
    Wrapper for inputing a filename.

    < ifile > : str or list. The input netCDF file (if more input files are given, these are concatenated one after the other.)

    < extract_level_4D > : float or None. Level to be extracted from a multi-level nc file.

    < season > : string, can be any group of months identified by the first letter (e.g. DJF, JJAS, ..) or a three-letter single month name (Mar, Jun, ..)

    < area > : string. Restricts the input field to this region. (EAT, PNA, NH, Med, Eu)
    """

    print('Running precompute\n')
    if type(ifile) is not list:
        var, level, lat, lon, dates, time_units, var_units, time_cal = ctl.read4Dncfield(ifile, extract_level = extract_level_4D)

        var_season, dates_season = ctl.sel_season(var, dates, season)
    else:
        print('Concatenating {} input files..\n'.format(len(ifile)))
        var = []
        dates = []
        for fil in ifile:
            var, level, lat, lon, dates, time_units, var_units, time_cal = ctl.read4Dncfield(fil, extract_level = extract_level_4D)

            var_season, dates_season = ctl.sel_season(var, dates, season)
            var.append(var_season)
            dates.append(dates_season)

        var_season = np.concatenate(var)
        dates_season = np.concatenate(dates)


    if sel_range is not None:
        print('Selecting date range {}\n'.format(sel_range))
        dates_season_pdh = pd.to_datetime(dates_season)
        okdat = (dates_season_pdh.year >= sel_range[0]) & (dates_season_pdh.year <= sel_range[1])
        var_season = var_season[okdat, ...]
        dates_season = dates_season[okdat, ...]


    results = WRtool_core(var_season, lat, lon, dates_season, area, **kwargs)

    return results


def WRtool_from_ensset(ensset, dates_set, lat, lon, season, area, **kwargs):
    """
    Wrapper for inputing an ensemble to be analyzed together.

    < ensset > : list. list of members to be concatenated for the analysis
    < dates_set > : dates of each member.

    < season > : string, can be any group of months identified by the first letter (e.g. DJF, JJAS, ..) or a three-letter single month name (Mar, Jun, ..)

    """

    var = []
    dates = []
    for ens, dat in zip(ensset, dates_set):
        var_season, dates_season = ctl.sel_season(var, dates, season, wnd)
        var.append(var_season)
        dates.append(dates_season)

    var = np.concatenate(var)
    dates = np.concatenate(dates)

    results = WRtool_core(var, lat, lon, dates, area, **kwargs)

    return results


def WRtool_core(var_season, lat, lon, dates_season, area, wnd = 5, numpcs = 4, numclus = 4, ref_solver = None, ref_patterns_area = None, clus_algorhitm = 'molteni', nrsamp_sig = 5000, heavy_output = False, run_significance_calc = True, detrended_eof_calculation = False, detrended_anom_for_clustering = False):
    """
    Tools for calculating Weather Regimes clusters. The clusters are found through Kmeans_clustering.
    This is the core: works on a set of variables already filtered for the season.

    < numpcs > : int. Number of Principal Components to be retained in the reduced phase space.
    < numclus > : int. Number of clusters.
    < wnd > : int. Number of days of the averaging window to calculate the climatology.

    < clus_algorhitm > : 'molteni' or 'sklearn', algorithm to be used for clustering.
    < nrsamp_sig > : number of samples to be used for significance calculation.

    < heavy_output > : bool. Output only the main results: cluspatterns, significance, patcor, et, labels. Instead outputs the eof solver and the local and global anomalies fields as well.

    < ref_solver >, < ref_patterns_area > : reference solver (ERA) and reference cluster pattern for cluster comparison.

    < detrended_eof_calculation > : Calculates a 20-year running mean for the geopotential before calculating the eofs.
    < detrended_anom_for_clustering > : Calculates the anomalies for clustering using the same detrended running mean.
    """
    ## PRECOMPUTE
    if detrended_anom_for_clustering and not detrended_eof_calculation:
        detrended_eof_calculation = True
        print('Setting detrended_eof_calculation = True\n')

    if detrended_eof_calculation:
        # Detrending
        print('Detrended eof calculation\n')
        climat_mean_dtr, dates_climat_dtr = ctl.trend_daily_climat(var_season, dates_season, window_days = wnd)
        var_anom_dtr = ctl.anomalies_daily_detrended(var_season, dates_season, climat_mean = climat_mean_dtr, dates_climate_mean = dates_climat_dtr)
        var_area_dtr, lat_area, lon_area = ctl.sel_area(lat, lon, var_anom_dtr, area)

        print('Running compute\n')
        #### EOF COMPUTATION
        eof_solver = ctl.eof_computation(var_area_dtr, lat_area)

        if detrended_anom_for_clustering:
            # Use detrended anomalies for clustering calculations
            PCs = eof_solver.pcs()[:, :numpcs]
        else:
            # Use anomalies wrt total time mean for clustering calculations
            climat_mean, dates_climat, climat_std = ctl.daily_climatology(var_season, dates_season, wnd)
            var_anom = ctl.anomalies_daily(var_season, dates_season, climat_mean = climat_mean, dates_climate_mean = dates_climat)
            var_area, lat_area, lon_area = ctl.sel_area(lat, lon, var_anom, area)

            PCs = eof_solver.projectField(var_area, neofs=numpcs, eofscaling=0, weighted=True)
    else:
        climat_mean, dates_climat, climat_std = ctl.daily_climatology(var_season, dates_season, wnd)
        var_anom = ctl.anomalies_daily(var_season, dates_season, climat_mean = climat_mean, dates_climate_mean = dates_climat)
        var_area, lat_area, lon_area = ctl.sel_area(lat, lon, var_anom, area)

        print('Running compute\n')
        #### EOF COMPUTATION
        eof_solver = ctl.eof_computation(var_area, lat_area)
        PCs = eof_solver.pcs()[:, :numpcs]

    print('Running clustering\n')
    #### CLUSTERING
    centroids, labels = ctl.Kmeans_clustering(PCs, numclus, algorithm = clus_algorhitm)

    dist_centroid = ctl.compute_centroid_distance(PCs, centroids, labels)

    if detrended_anom_for_clustering:
        cluspattern = ctl.compute_clusterpatterns(var_anom_dtr, labels)
    else:
        cluspattern = ctl.compute_clusterpatterns(var_anom, labels)

    cluspatt_area = []
    for clu in cluspattern:
        cluarea, _, _ = ctl.sel_area(lat, lon, clu, area)
        cluspatt_area.append(cluarea)
    cluspatt_area = np.stack(cluspatt_area)

    varopt = ctl.calc_varopt_molt(PCs, centroids, labels)
    print('varopt: {:8.4f}\n'.format(varopt))
    freq_clus = ctl.calc_clus_freq(labels)

    results = dict()

    if run_significance_calc:
        print('Running clus sig\n')
        significance = ctl.clusters_sig(PCs, centroids, labels, dates_season, nrsamp = nrsamp_sig)
        results['significance'] = significance

    if ref_solver is not None and ref_patterns_area is not None:
        print('Running compare\n')
        perm, centroids, labels, et, patcor = ctl.clus_compare_projected(centroids, labels, cluspatt_area, ref_patterns_area, ref_solver, numpcs)

        print('Optimal permutation: {}\n'.format(perm))
        cluspattern = cluspattern[perm, ...]
        cluspatt_area = cluspatt_area[perm, ...]
        freq_clus = freq_clus[perm]

        results['RMS'] = et
        results['patcor'] = patcor

    results['freq_clus'] = freq_clus
    results['cluspattern'] = cluspattern
    results['cluspattern_area'] = cluspatt_area
    results['lat'] = lat
    results['lat_area'] = lat_area
    results['lon'] = lon
    results['lon_area'] = lon_area
    results['labels'] = labels
    results['centroids'] = centroids
    results['dist_centroid'] = dist_centroid

    results['pcs'] = PCs
    results['eofs'] = eof_solver.eofs()[:numpcs]
    results['eofs_eigenvalues'] = eof_solver.eigenvalues()[:numpcs]
    results['eofs_varfrac'] = eof_solver.varianceFraction()[:numpcs]

    results['trans_matrix'] = ctl.calc_regime_transmatrix(1, labels, dates_season)
    results['dates'] = dates_season

    if heavy_output:
        results['regime_transition_pcs'] = ctl.find_transition_pcs(1, labels, dates_season, PCs, filter_longer_than = 3)
        if detrended_anom_for_clustering:
            results['var_area'] = var_area_dtr
            results['var_glob'] = var_anom_dtr
        else:
            results['var_area'] = var_area
            results['var_glob'] = var_anom
        results['solver'] = eof_solver

    return results


def WRtool_core_ensemble(n_ens, var_season_set, lat, lon, dates_season_set, area, ens_names = None, wnd = 5, numpcs = 4, numclus = 4, ref_solver = None, ref_patterns_area = None, clus_algorhitm = 'molteni', nrsamp_sig = 5000, heavy_output = False, run_significance_calc = True, detrended_eof_calculation = False, detrended_anom_for_clustering = False):
    """
    Tools for calculating Weather Regimes clusters. The clusters are found through Kmeans_clustering.
    This is the core: works on a set of variables already filtered for the season.

    < numpcs > : int. Number of Principal Components to be retained in the reduced phase space.
    < numclus > : int. Number of clusters.
    < wnd > : int. Number of days of the averaging window to calculate the climatology.

    < clus_algorhitm > : 'molteni' or 'sklearn', algorithm to be used for clustering.
    < nrsamp_sig > : number of samples to be used for significance calculation.

    < heavy_output > : bool. Output only the main results: cluspatterns, significance, patcor, et, labels. Instead outputs the eof solver and the local and global anomalies fields as well.

    < ref_solver >, < ref_patterns_area > : reference solver (ERA) and reference cluster pattern for cluster comparison.

    < detrended_eof_calculation > : Calculates a 20-year running mean for the geopotential before calculating the eofs.
    < detrended_anom_for_clustering > : Calculates the anomalies for clustering using the same detrended running mean.
    """
    ## PRECOMPUTE
    if detrended_anom_for_clustering and not detrended_eof_calculation:
        detrended_eof_calculation = True
        print('Setting detrended_eof_calculation = True\n')

    results = dict()
    results['all'] = dict()
    if ens_names is None:
        ens_names = ['ens{}'.format(i) for i in range(n_ens)]
    for ennam in ens_names:
        results[ennam] = dict()

    var_anom = []
    var_anom_dtr = []
    if detrended_eof_calculation:
        trace_ens = []
        for ens in range(n_ens):
            # Detrending
            print('Detrended eof calculation\n')
            climat_mean_dtr, dates_climat_dtr = ctl.trend_daily_climat(var_season_set[ens], dates_season_set[ens], window_days = wnd)
            trace_ens.append(len(var_season_set[ens]))
            if heavy_output:
                results[ens_names[ens]]['climate_mean_dtr'] = np.mean(np.stack(climat_mean_dtr), axis = 1)
                results[ens_names[ens]]['climate_mean_dtr_dates'] = pd.to_datetime(dates_climat_dtr).year
            var_anom_dtr.append(ctl.anomalies_daily_detrended(var_season_set[ens], dates_season_set[ens], climat_mean = climat_mean_dtr, dates_climate_mean = dates_climat_dtr))

        var_anom_dtr = np.concatenate(var_anom_dtr)

        var_area_dtr, lat_area, lon_area = ctl.sel_area(lat, lon, var_anom_dtr, area)

        print('Running compute\n')
        #### EOF COMPUTATION
        eof_solver = ctl.eof_computation(var_area_dtr, lat_area)

        if detrended_anom_for_clustering:
            # Use detrended anomalies for clustering calculations
            PCs = eof_solver.pcs()[:, :numpcs]
        else:
            # Use anomalies wrt total time mean for clustering calculations
            for ens in range(n_ens):
                # Detrending
                climat_mean, dates_climat, climat_std = ctl.daily_climatology(var_season_set[ens], dates_season_set[ens], wnd)
                if heavy_output:
                    results[ens_names[ens]]['climate_mean'] = np.mean(climat_mean, axis = 1)
                    results[ens_names[ens]]['climate_mean_dates'] = dates_climat
                var_anom.append(ctl.anomalies_daily(var_season_set[ens], dates_season_set[ens], climat_mean = climat_mean, dates_climate_mean = dates_climat))

            var_anom = np.concatenate(var_anom)
            var_area, lat_area, lon_area = ctl.sel_area(lat, lon, var_anom, area)

            PCs = eof_solver.projectField(var_area, neofs=numpcs, eofscaling=0, weighted=True)
    else:
        trace_ens = []
        for ens in range(n_ens):
            trace_ens.append(len(var_season_set[ens]))
            climat_mean, dates_climat, climat_std = ctl.daily_climatology(var_season_set[ens], dates_season_set[ens], wnd)
            if heavy_output:
                results[ens_names[ens]]['climate_mean'] = np.mean(climat_mean, axis = 1)
                results[ens_names[ens]]['climate_mean_dates'] = dates_climat
            var_anom.append(ctl.anomalies_daily(var_season_set[ens], dates_season_set[ens], climat_mean = climat_mean, dates_climate_mean = dates_climat))

        var_anom = np.concatenate(var_anom)
        var_area, lat_area, lon_area = ctl.sel_area(lat, lon, var_anom, area)

        print('Running compute\n')
        #### EOF COMPUTATION
        eof_solver = ctl.eof_computation(var_area, lat_area)
        PCs = eof_solver.pcs()[:, :numpcs]

    print('Running clustering\n')
    #### CLUSTERING
    centroids, labels = ctl.Kmeans_clustering(PCs, numclus, algorithm = clus_algorhitm)

    dist_centroid = ctl.compute_centroid_distance(PCs, centroids, labels)

    if detrended_anom_for_clustering:
        cluspattern = ctl.compute_clusterpatterns(var_anom_dtr, labels)
    else:
        cluspattern = ctl.compute_clusterpatterns(var_anom, labels)

    cluspatt_area = []
    for clu in cluspattern:
        cluarea, _, _ = ctl.sel_area(lat, lon, clu, area)
        cluspatt_area.append(cluarea)
    cluspatt_area = np.stack(cluspatt_area)

    varopt = ctl.calc_varopt_molt(PCs, centroids, labels)
    print('varopt: {:8.4f}\n'.format(varopt))
    freq_clus = ctl.calc_clus_freq(labels)

    if run_significance_calc:
        print('Running clus sig\n')
        significance = ctl.clusters_sig(PCs, centroids, labels, dates_season, nrsamp = nrsamp_sig)
        results['all']['significance'] = significance

    if ref_solver is not None and ref_patterns_area is not None:
        print('Running compare\n')
        perm, centroids, labels, et, patcor = ctl.clus_compare_projected(centroids, labels, cluspatt_area, ref_patterns_area, ref_solver, numpcs)

        print('Optimal permutation: {}\n'.format(perm))
        cluspattern = cluspattern[perm, ...]
        cluspatt_area = cluspatt_area[perm, ...]
        freq_clus = freq_clus[perm]

        results['all']['RMS'] = et
        results['all']['patcor'] = patcor

    results['all']['freq_clus'] = freq_clus
    results['all']['cluspattern'] = cluspattern
    results['all']['cluspattern_area'] = cluspatt_area
    results['all']['lat'] = lat
    results['all']['lat_area'] = lat_area
    results['all']['lon'] = lon
    results['all']['lon_area'] = lon_area

    results['all']['centroids'] = centroids
    results['all']['eofs'] = eof_solver.eofs()[:numpcs]
    results['all']['eofs_eigenvalues'] = eof_solver.eigenvalues()[:numpcs]
    results['all']['eofs_varfrac'] = eof_solver.varianceFraction()[:numpcs]

    if heavy_output:
        results['all']['solver'] = eof_solver

    for ens, ennam in enumerate(ens_names):
        ind1 = int(np.sum(trace_ens[:ens]))
        ind2 = ind1 + trace_ens[ens]
        results[ennam]['labels'] = labels[ind1:ind2]
        results[ennam]['dist_centroid'] = dist_centroid[ind1:ind2]
        results[ennam]['pcs'] = PCs[ind1:ind2]
        results[ennam]['dates'] = dates_season_set[ens]

        results[ennam]['freq_clus'] = ctl.calc_clus_freq(labels[ind1:ind2])
        results[ennam]['trans_matrix'] = ctl.calc_regime_transmatrix(1, labels[ind1:ind2], dates_season_set[ens])

        if heavy_output:
            if detrended_anom_for_clustering:
                results[ennam]['var_area'] = var_area_dtr[ind1:ind2]
                results[ennam]['var_glob'] = var_anom_dtr[ind1:ind2]
            else:
                results[ennam]['var_area'] = var_area[ind1:ind2]
                results[ennam]['var_glob'] = var_anom[ind1:ind2]

    results['all']['trans_matrix'] = ctl.calc_regime_transmatrix(n_ens, [results[ennam]['labels'] for ennam in ens_names], dates_season_set)
    if heavy_output:
        results['all']['regime_transition_pcs'] = ctl.find_transition_pcs(n_ens, [results[ennam]['labels'] for ennam in ens_names], dates_season_set, [results[ennam]['pcs'] for ennam in ens_names], filter_longer_than = 3)

    return results
