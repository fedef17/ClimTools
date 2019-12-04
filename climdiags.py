#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os

from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as PathEffects
from matplotlib.colors import LogNorm

import iris
import netCDF4 as nc
import pandas as pd
from datetime import datetime
import pickle
from copy import deepcopy as copy

import climtools_lib as ctl

###############################################################################

"""
Diagnostics for standard climate outputs. Contains higher level tools that make use of climtools_lib.
Tools contained:
- WRtool
- heat_flux_calc
"""
###############################################################################
### constants
cp = 1005.0 # specific enthalpy dry air - J kg-1 K-1
cpw = 1840.0 # specific enthalpy water vapor - J kg-1 K-1
# per la moist air sarebbe cpm = cp + q*cpw, ma conta al massimo per un 3 %
L = 2501000.0 # J kg-1
Lsub = 2835000.0 # J kg-1
g = 9.81 # m s-2
Rearth = 6371.0e3 # mean radius

###############################################################################

#############################################################################
#############################################################################

##########            Weather Regimes and Transitions              ##########

#############################################################################
#############################################################################


def WRtool_from_file(ifile, season, area, regrid_to_reference_cube = None, sel_yr_range = None, extract_level_hPa = None, netcdf4_read = False, **kwargs):
    """
    Wrapper for inputing a filename.

    < ifile > : str or list. The input netCDF file (if more input files are given, these are concatenated one after the other.)

    < extract_level_hPa > : float or None. Level to be extracted from a multi-level nc file: units are hPa.

    < sel_yr_range > : tuple, (start_year, end_year). Selects the given range of years.
    < regrid_to_reference_cube > : iris.cube.Cube instance. The data read from ifile are regridded to this resolution.

    < season > : string, can be any group of months identified by the first letter (e.g. DJF, JJAS, ..) or a three-letter single month name (Mar, Jun, ..)

    < area > : string. Restricts the input field to this region. (EAT, PNA, NH, Med, Eu)
    """
    # print(kwargs.keys())
    # print(numpcs)
    # print(netcdf4_read)

    print('Running precompute\n')
    if type(ifile) not in [list, np.ndarray]:
        if netcdf4_read:
            #var, lat, lon, dates, time_units, var_units, time_cal = ctl.readxDncfield(ifile, extract_level = extract_level_hPa)
            var, coords, aux_info = ctl.readxDncfield(ifile, extract_level = extract_level_hPa)
            lat = coords['lat']
            lon = coords['lon']
            dates = coords['dates']
        else:
            var, coords, aux_info = ctl.read_iris_nc(ifile, extract_level_hPa = extract_level_hPa, regrid_to_reference = regrid_to_reference_cube)
            lat = coords['lat']
            lon = coords['lon']
            dates = coords['dates']

        var_season, dates_season = ctl.sel_season(var, dates, season)
    else:
        print('Concatenating {} input files..\n'.format(len(ifile)))
        var_sel = []
        var_full = []
        dates_sel = []
        dates_full = []
        for fil in ifile:
            # var, lat, lon, dates, time_units, var_units, time_cal = ctl.readxDncfield(fil, extract_level = extract_level_hPa)
            if netcdf4_read:
                var, coords, aux_info = ctl.readxDncfield(ifile, extract_level = extract_level_hPa)
                lat = coords['lat']
                lon = coords['lon']
                dates = coords['dates']
                # var, lat, lon, dates, time_units, var_units, time_cal = ctl.readxDncfield(ifile, extract_level = extract_level_hPa)
            else:
                var, coords, aux_info = ctl.read_iris_nc(fil, extract_level_hPa = extract_level_hPa, regrid_to_reference = regrid_to_reference_cube)
                lat = coords['lat']
                lon = coords['lon']
                dates = coords['dates']

            dates_full.append(dates)
            var_full.append(var)

            var_season, dates_season = ctl.sel_season(var, dates, season, cut = False)
            var_sel.append(var_season)
            dates_sel.append(dates_season)

        var_season = np.concatenate(var_sel)
        dates_season = np.concatenate(dates_sel)
        var = np.concatenate(var_full)
        dates = np.concatenate(dates_full)

    #HERE calculate climat_mean
    print('Calculating mean climatology\n')
    if ctl.check_daily(dates):
        climat_mean, dates_climate_mean, _ = ctl.daily_climatology(var, dates, window = kwargs['wnd_days'])

        climat_mean_dtr, dates_climate_mean_dtr = None, None
        if kwargs['detrended_eof_calculation']:
            print('Calculating detrended climatology')
            climat_mean_dtr, dates_climate_mean_dtr = ctl.trend_daily_climat(var, dates, window_days = kwargs['wnd_days'], window_years = kwargs['wnd_years'])
    else:
        climat_mean, dates_climate_mean, _ = ctl.monthly_climatology(var, dates)

        climat_mean_dtr, dates_climate_mean_dtr = None, None
        if kwargs['detrended_eof_calculation']:
            print('Calculating detrended climatology')
            climat_mean_dtr, dates_climate_mean_dtr = ctl.trend_monthly_climat(var, dates, window_years = kwargs['wnd_years'])

    if sel_yr_range is not None:
        print('Selecting date range {}\n'.format(sel_yr_range))
        dates_season_pdh = pd.to_datetime(dates_season)
        okdat = (dates_season_pdh.year >= sel_yr_range[0]) & (dates_season_pdh.year <= sel_yr_range[1])
        var_season = var_season[okdat, ...]
        dates_season = dates_season[okdat]

        dates_pdh = pd.to_datetime(dates)
        okdat = (dates_pdh.year >= sel_yr_range[0]) & (dates_pdh.year <= sel_yr_range[1])
        dates = dates[okdat]

    results = WRtool_core(var_season, lat, lon, dates_season, area, climat_mean = climat_mean, dates_climate_mean = dates_climate_mean, climat_mean_dtr = climat_mean_dtr, dates_climate_mean_dtr = dates_climate_mean_dtr, **kwargs)

    results['dates_allyear'] = dates
    if netcdf4_read:
        results['time_cal'] = time_cal
        results['time_units'] = time_units
    else:
        results['time_cal'] = aux_info['time_calendar']
        results['time_units'] = aux_info['time_units']

    var, datesmon = ctl.calc_monthly_clus_freq(results['labels'], dates_season)
    results['monthly_freq'] = dict()
    results['monthly_freq']['freq'] = var
    results['monthly_freq']['dates'] = pd.to_datetime(datesmon)

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


def WRtool_core(var_season, lat, lon, dates_season, area, wnd_days = 20, wnd_years = 30, numpcs = 4, perc = None, numclus = 4, ref_solver = None, ref_patterns_area = None, clus_algorhitm = 'molteni', nrsamp_sig = 5000, heavy_output = False, run_significance_calc = True, significance_calc_routine = 'BootStrap25', detrended_eof_calculation = False, detrended_anom_for_clustering = False, use_reference_eofs = False, use_reference_clusters = False, ref_clusters_centers = None, climat_mean = None, dates_climate_mean = None, climat_mean_dtr = None, dates_climate_mean_dtr = None, bad_matching_rule = 'rms_mean', matching_hierarchy = None):
    """
    Tools for calculating Weather Regimes clusters. The clusters are found through Kmeans_clustering.
    This is the core: works on a set of variables already filtered for the season.

    < numpcs > : int. Number of Principal Components to be retained in the reduced phase space.
    < numclus > : int. Number of clusters.
    < wnd_days > : int. Number of days of the averaging window to calculate the climatology.

    < clus_algorithm > : 'molteni' or 'sklearn', algorithm to be used for clustering.
    < nrsamp_sig > : number of samples to be used for significance calculation.

    < heavy_output > : bool. Output only the main results: cluspatterns, significance, patcor, et, labels. Instead outputs the eof solver and the local and global anomalies fields as well.

    < ref_solver >, < ref_patterns_area > : reference solver (ERA) and reference cluster pattern for cluster comparison.

    < detrended_eof_calculation > : Calculates a 20-year running mean for the geopotential before calculating the eofs.
    < detrended_anom_for_clustering > : Calculates the anomalies for clustering using the same detrended running mean.

    Note on the anomaly calculation: it is suggested to calculate the climatological mean (climat_mean or climate_mean_dtr if detrending is active) before the season selection, so outside WRtool_core. This is especially suggested when using a large wnd_days (like 15, 20 days).
    """
    is_daily = ctl.check_daily(dates_season)
    if is_daily:
        print('Analyzing a set of daily data..\n')
    else:
        print('Analyzing a set of monthly data..\n')


    if use_reference_clusters:
        print('\n\n <<<<< Using reference cluster centers! No KMeans is run on the models. (use_reference_clusters set to True)>>>>> \n\n\n')
        use_reference_eofs = True
        if ref_clusters_centers is None:
            raise ValueError('reference cluster centers is None!')

    if use_reference_eofs:
        print('\n\n <<<<< Using reference EOF space for the whole analysis!! (use_reference_eofs set to True)>>>>> \n\n\n')
        if ref_solver is None:
            raise ValueError('reference solver is None!')

    ## PRECOMPUTE
    if detrended_anom_for_clustering and not detrended_eof_calculation:
        detrended_eof_calculation = True
        print('Setting detrended_eof_calculation = True\n')

    if detrended_eof_calculation:
        # Detrending
        print('Detrended eof calculation\n')
        if is_daily:
            if climat_mean_dtr is None:
                climat_mean_dtr, dates_climate_mean_dtr = ctl.trend_daily_climat(var_season, dates_season, window_days = wnd_days, window_years = wnd_years)
            var_anom_dtr = ctl.anomalies_daily_detrended(var_season, dates_season, climat_mean = climat_mean_dtr, dates_climate_mean = dates_climate_mean_dtr)
        else:
            if climat_mean_dtr is None:
                climat_mean_dtr, dates_climate_mean_dtr = ctl.trend_monthly_climat(var_season, dates_season, window_years = wnd_years)
            var_anom_dtr = ctl.anomalies_monthly_detrended(var_season, dates_season, climat_mean = climat_mean_dtr, dates_climate_mean = dates_climate_mean_dtr)

        var_area_dtr, lat_area, lon_area = ctl.sel_area(lat, lon, var_anom_dtr, area)

        print('Running compute\n')
        #### EOF COMPUTATION
        eof_solver = ctl.eof_computation(var_area_dtr, lat_area)

        if perc is not None:
            varfrac = eof_solver.varianceFraction()
            acc = np.cumsum(varfrac*100)
            numpcs = ctl.first(acc >= perc)
            print('Selected numpcs = {}, accounting for {}% of the total variance.\n'.format(numpcs, perc))

        if detrended_anom_for_clustering:
            # Use detrended anomalies for clustering calculations
            if not use_reference_eofs:
                PCs = eof_solver.pcs()[:, :numpcs]
            else:
                PCs = ref_solver.projectField(var_area_dtr, neofs = numpcs, eofscaling = 0, weighted = True)
        else:
            # Use anomalies wrt total time mean for clustering calculations
            if is_daily:
                if climat_mean is None:
                    climat_mean, dates_climate_mean, climat_std = ctl.daily_climatology(var_season, dates_season, wnd_days)
                var_anom = ctl.anomalies_daily(var_season, dates_season, climat_mean = climat_mean, dates_climate_mean = dates_climate_mean)
            else:
                if climat_mean is None:
                    climat_mean, dates_climate_mean, climat_std = ctl.monthly_climatology(var_season, dates_season)
                var_anom = ctl.anomalies_monthly(var_season, dates_season, climat_mean = climat_mean, dates_climate_mean = dates_climate_mean)

            var_area, lat_area, lon_area = ctl.sel_area(lat, lon, var_anom, area)

            if not use_reference_eofs:
                PCs = eof_solver.projectField(var_area, neofs=numpcs, eofscaling=0, weighted=True)
            else:
                PCs = ref_solver.projectField(var_area, neofs=numpcs, eofscaling=0, weighted=True)
    else:
        # No detrending
        if is_daily:
            if climat_mean is None:
                climat_mean, dates_climate_mean, climat_std = ctl.daily_climatology(var_season, dates_season, wnd_days)
            var_anom = ctl.anomalies_daily(var_season, dates_season, climat_mean = climat_mean, dates_climate_mean = dates_climate_mean)
        else:
            if climat_mean is None:
                climat_mean, dates_climate_mean, climat_std = ctl.monthly_climatology(var_season, dates_season)
            var_anom = ctl.anomalies_monthly(var_season, dates_season, climat_mean = climat_mean, dates_climate_mean = dates_climate_mean)

        var_area, lat_area, lon_area = ctl.sel_area(lat, lon, var_anom, area)

        print('Running compute\n')
        #### EOF COMPUTATION
        eof_solver = ctl.eof_computation(var_area, lat_area)
        if perc is not None:
            varfrac = eof_solver.varianceFraction()
            acc = np.cumsum(varfrac*100)
            numpcs = ctl.first(acc >= perc)
            print('Selected numpcs = {}, accounting for {}% of the total variance.\n'.format(numpcs, perc))

        if not use_reference_eofs:
            PCs = eof_solver.pcs()[:, :numpcs]
        else:
            PCs = ref_solver.projectField(var_area, neofs=numpcs, eofscaling=0, weighted=True)

    if not use_reference_clusters:
        print('Running clustering\n')
        #### CLUSTERING
        centroids, labels = ctl.Kmeans_clustering(PCs, numclus, algorithm = clus_algorhitm)
        dist_centroid = ctl.compute_centroid_distance(PCs, centroids, labels)
    else:
        print('Assigning pcs to closest reference cluster center\n')
        centroids = ref_clusters_centers
        labels = []
        dist_centroid = []
        for el in PCs:
            distcen = [ctl.distance(el, centr) for centr in centroids]
            labels.append(np.argmin(distcen))
            dist_centroid.append(np.min(distcen))
        labels = np.array(labels)
        dist_centroid = np.array(dist_centroid)

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
        perm, centroids, labels, et, patcor = ctl.clus_compare_projected(centroids, labels, cluspatt_area, ref_patterns_area, ref_solver, numpcs, bad_matching_rule = bad_matching_rule, matching_hierarchy = matching_hierarchy)

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

    effcen = []
    for clus in range(numclus):
        oklabs = labels == clus
        effcen.append(np.mean(PCs[oklabs], axis = 0))

    results['eff_centroids'] = np.stack(effcen) # mean of the PCs for each cluster

    if 'use_reference_eofs' and ref_solver is not None:
        results['eofs_ref_pcs'] = ref_solver.eofs()[:numpcs]
    else:
        results['eofs_ref_pcs'] = eof_solver.eofs()[:numpcs]
    results['model_eofs'] = eof_solver.eofs()[:numpcs]
    results['model_eofs_eigenvalues'] = eof_solver.eigenvalues()[:numpcs]
    results['model_eofs_varfrac'] = eof_solver.varianceFraction()[:numpcs]

    results['resid_times'] = ctl.calc_regime_residtimes(labels, dates = dates_season)[0]
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


def WRtool_core_ensemble(n_ens, var_season_set, lat, lon, dates_season_set, area, ens_names = None, wnd = 5, numpcs = 4, numclus = 4, ref_solver = None, ref_patterns_area = None, clus_algorhitm = 'sklearn', nrsamp_sig = 5000, heavy_output = False, run_significance_calc = True, detrended_eof_calculation = False, detrended_anom_for_clustering = False):
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
    results['all']['model_eofs'] = eof_solver.eofs()[:numpcs]
    results['all']['model_eofs_eigenvalues'] = eof_solver.eigenvalues()[:numpcs]
    results['all']['model_eofs_varfrac'] = eof_solver.varianceFraction()[:numpcs]

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
        results[ennam]['resid_times'] = ctl.calc_regime_residtimes(labels[ind1:ind2], dates = dates_season_set[ens])[0]
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


#############################################################################
#############################################################################

##########          Energy balance, heat fluxes                    ##########

#############################################################################
#############################################################################


def quant_flux_calc(va, lat, lon, levels, dates_6hrs, ps, dates_ps, quantity = None, seasons = ['DJF', 'JJA']):
    """
    Calculates meridional fluxes of quantity quant.

    The following fields are needed at least at 6hrs frequency (actually only for the SH flux, the others should work with lower frequencies as well):
    < va > : meridional wind (3d)
    < quantity > : transported quantity (3d)

    If < quantity > is not given, va is considered as the 3d flux. So, for example, for SH flux the v*T product may be given in input instead of va with quantity left None.

    < lat, lon, levels > : the latitude, longitude and levels of the 3d grid.
    < dates_6hrs > : the dates of the 6hrs quantities.

    < ps > : surface pressure (same lat/lon grid)
    < dates_ps > : dates of the surface pressure (daily or even monthly data are ok)

    < seasons > : returns mean fluxes for each season.
    """

    print('Levels: {}\n'.format(levels))
    if np.max(levels) < np.mean(ps)/2:
        raise ValueError('Level units are not in Pa?')

    press0 = dict()
    press0['year'] = np.mean(ps, axis = 0)
    for seas in seasons:
        press0[seas] = np.mean(ctl.sel_season(ps, dates_ps, seas, cut = False)[0], axis = 0)

    fluxes_levels = dict()
    fluxes_cross = dict()
    fluxes_maps = dict()
    fluxes_zonal = dict()

    if quantity is not None:
        va = va*quantity
        del quantity

    for seas in seasons:
        fluxes_levels[seas] = np.mean(ctl.sel_season(va, dates_6hrs, seas, cut = False)[0], axis = 0)
    fluxes_levels['year'] = np.mean(va, axis = 0)
    del va

    zonal_factor = 2*np.pi*Rearth*np.cos(np.deg2rad(lat))
    for flun in fluxes_levels:
        fluxes_cross[flun] = np.mean(fluxes_levels[flun], axis = -1)*zonal_factor
        fluxes_maps[flun] = np.zeros(fluxes_levels[flun].shape[1:])

    print('Starting vertical integration\n')
    for seas in press0:
        for ila in range(len(lat)):
            for ilo in range(len(lon)):
                #print(lat[ila], lon[ilo])
                p0 = press0[seas][ila, ilo]

                # Selecting pressure levels lower than surface pressure
                exclude_pres = levels > p0
                if np.any(exclude_pres):
                    lev0 = np.argmax(exclude_pres)
                else:
                    lev0 = None
                levels_ok = np.append(levels[:lev0], p0)

                if not np.all(np.diff(levels_ok) > 0):
                    print(levels_ok)
                    raise ValueError('levels not increasing')

                coso = fluxes_levels[seas][:, ila, ilo]
                coso = np.append(coso[:lev0], np.interp(p0, levels, coso))
                fluxes_maps[seas][ila, ilo] = np.trapz(coso, x = levels_ok)

    print('done\n')

    zonal_factor = 2*np.pi*Rearth*np.cos(np.deg2rad(lat))
    for fu in fluxes_maps:
        fluxes_zonal[fu] = np.mean(fluxes_maps[fu], axis = -1)*zonal_factor

    results = dict()
    results['zonal'] = fluxes_zonal
    results['maps'] = fluxes_maps
    results['cross'] = fluxes_cross

    return results


def heat_flux_calc(file_list, file_ps, cart_out, tag, full_calculation = False, zg_in_ERA_units = False, seasons = ['DJF', 'JJA'], netcdf_level_units = None):
    """
    Calculates meridional heat fluxes: SH (sensible), LH (latent) and PE (potential energy).

    The following fields are needed at least at 6hrs frequency (actually only for the SH flux, the others should work with lower frequencies as well): < va > meridional wind (3d), < ta > temperature (3d), < zg > geopotential (3d), < q > specific humidity (3d). These are read from the netcdf files in < file_list >. This can be either a single file with all variables in input or a single file for each variable. In the second case, the order matters and is v,t,z,q.

    Due to memory errors, new option added:
    < full_calculation > : if True, the product between variables are performed inside the routine (easier to overload the memory). Instead, precalculated <vt>, <vz> and <vq> products are considered (strictly in this order).

    < file_list > : str or list. Read above for description.

    < file_ps > : str, netcdf file containing surface pressure (same lat/lon grid, daily or even monthly data are ok)

    < seasons > : list, returns mean fluxes for all the specified seasons and for the annual mean.

    < zg_in_ERA_units > : the geopotential is in m**2 s**-2 units. If full_calculation is True, this is autodetected. If the products have already been performed this has to be specified by hand.
    """
    ##############################################

    factors = {'SH': cp/g, 'PE': 1., 'LH': L/g}

    press0row, latdad, londsad, datespress, time_units, var_units = ctl.read3Dncfield(file_ps)

    figure_file_exp = cart_out+'hf_{}.pdf'.format(tag)
    figures_exp = []
    figure_file_exp_maps = cart_out+'hf_{}_maps.pdf'.format(tag)
    figures_exp_maps = []

    fluxes_levels = dict()
    fluxes_cross = dict()
    fluxes_maps = dict()
    fluxes_zonal = dict()

    varnames = [['v','va'], ['t','ta'], ['z','zg'], ['q','hus']]
    fluxnames = ['SH', 'PE', 'LH']
    results = dict()

    if full_calculation:
        vars = dict()
        # leggo v
        if type(file_list) is str:
            # varuna, level, lat, lon, dates, time_units, var_units, time_cal = ctl.readxDncfield(file_list, select_var = varnames[0])
            varuna, coords, aux_info = ctl.readxDncfield(ifile, select_var = varnames[0])
            lat = coords['lat']
            lon = coords['lon']
            dates = coords['dates']
        else:
            # varuna, level, lat, lon, dates, time_units, var_units, time_cal = ctl.readxDncfield(file_list[0])
            varuna, coords, aux_info = ctl.readxDncfield(file_list[0])
            lat = coords['lat']
            lon = coords['lon']
            dates = coords['dates']
            if list(varuna.keys())[0] not in varnames[0]:
                raise ValueError('{} is not v. Please give input files in order v,t,z,q.'.format(list(varuna.keys())[0]))

        # Changing to standard names
        for varna in varuna:
            for varnamok in varnames:
                if varna in varnamok: vars[varnamok[0]] = varuna.pop(varna)

        print(varuna.keys(), vars.keys())

        # leggo t
        for i, (varnamok, flun) in enumerate(zip(varnames[1:], fluxnames)):
            varname = varnamok[0]
            print('Extracting {}\n'.format(varname))

            if type(file_list) is str:
                # varuna, level, lat, lon, dates, time_units, var_units, time_cal = ctl.readxDncfield(file_list, select_var = varnamok)
                varuna, coords, aux_info = ctl.readxDncfield(file_list, select_var = varnamok)
                lat = coords['lat']
                lon = coords['lon']
                dates = coords['dates']
            else:
                varuna, coords, aux_info = ctl.readxDncfield(file_list[i+1])
                lat = coords['lat']
                lon = coords['lon']
                dates = coords['dates']
                # varuna, level, lat, lon, dates, time_units, var_units, time_cal = ctl.readxDncfield(file_list[i+1])
                if list(varuna.keys())[0] not in varnamok:
                    raise ValueError('{} is not t. Please give input files in order v,t,z,q.'.format(list(varuna.keys())[0]))

            vars[varname] = varuna.pop(list(varuna.keys())[0])
            del varuna

            if flun == 'PE' and list(var_units.values())[0] == 'm**2 s**-2':
                fact = factors['PE']/g
            else:
                fact = factors[flun]

            print('Calculating {}\n'.format(flun))
            results[flun] = quant_flux_calc(vars['v'], lat, lon, level, dates, press0row, datespress, quantity = factors[flun]*vars[varname], seasons = seasons)
            print('OK\n')
            del vars[varname]

        del vars
    else:
        for i, flun in enumerate(fluxnames):
            varuna, coords, aux_info = ctl.readxDncfield(file_list[i])
            lat = coords['lat']
            lon = coords['lon']
            dates = coords['dates']
            # varuna, level, lat, lon, dates, time_units, var_units, time_cal = ctl.readxDncfield(file_list[i])
            varuna = varuna[list(varuna.keys())[0]]

            if flun == 'PE' and zg_in_ERA_units:
                fact = factors['PE']/g
            else:
                fact = factors[flun]
            varuna *= fact

            print('Calculating {}\n'.format(flun))
            results[flun] = quant_flux_calc(varuna, lat, lon, level, dates, press0row, datespress, seasons = seasons)
            print('OK\n')
            del varuna

    if not os.path.exists(cart_out): os.mkdir(cart_out)

    for fu in results:
        filena = cart_out + '{}flux_map_{}_{}seasons.nc'.format(fu, tag, len(seasons))
        vals = [results[fu]['maps'][key] for key in seasons]
        vals.append(results[fu]['maps']['year'])
        ctl.save_N_2Dfields(lat,lon,np.stack(vals), fu, 'W/m', filena)

    # for fu in era_fluxes_zonal:
    #     #print(lat, era_lat, era_fluxes_zonal[fu])
    #     era_fluxes_zonal_itrp[fu] = np.interp(lat, era_lat[::-1], era_fluxes_zonal[fu][::-1])
    #     print('{:12.3e} - {:12.3e}\n'.format(era_fluxes_zonal_itrp[fu].min(), era_fluxes_zonal_itrp[fu].max()))
    results['tot'] = dict()
    for taw in ['zonal', 'maps', 'cross']:
        results['tot'][taw] = dict()

    for seas in seasons+['year']:
        for taw in ['zonal', 'maps', 'cross']:
            total = np.sum([results[flun][taw][seas] for flun in fluxnames], axis = 0)
            results['tot'][taw][seas] = total

    margins = dict()
    for taw in ['zonal', 'maps', 'cross']:
        for flu in results:
            margins[(flu, taw)] = (1.1*np.min([results[flu][taw][seas] for seas in seasons]), 1.1*np.max([results[flu][taw][seas] for seas in seasons]))

    fluxnames = ['SH', 'PE', 'LH']
    # Single exps, single season figures:
    for seas in seasons+['year']:
        fig = plt.figure()
        plt.title('{} - Meridional heat fluxes - {}'.format(tag, seas))
        for flun in fluxnames:
            plt.plot(lat, results[flun]['zonal'][seas], label = flun)
        plt.plot(lat, results['tot']['zonal'][seas], label = 'Total')
        plt.legend()
        plt.grid()
        plt.ylim(margins[('tot', 'zonal')])
        plt.xlabel('Latitude')
        plt.ylabel('Integrated Net Heat Flux (W)')
        figures_exp.append(fig)

        for flun in fluxnames:
            fig = ctl.plot_map_contour(results[flun]['maps'][seas], lat, lon, title = 'SH - {} - {}'.format(tag, seas), cbar_range = margins[(flun, 'maps')])
            figures_exp_maps.append(fig)

        fig = ctl.plot_map_contour(results['tot']['maps'][seas], lat, lon, title = 'Total meridional flux - {} - {}'.format(tag, seas), cbar_range = margins[('tot', 'maps')])
        figures_exp_maps.append(fig)

    for flun in fluxnames+['tot']:
        fig = plt.figure()
        plt.title('{} fluxes - {}'.format(flun, tag))
        cset = ctl.color_set(len(seasons)+1, only_darker_colors = False)
        for seas, col in zip(seasons+['year'], cset):
            plt.plot(lat, results[flun]['zonal'][seas], label = seas, color = col, linewidth = 2.)
            #plt.plot(lat, era_fluxes_zonal_itrp[(seas, flun)], label = 'ERA '+seas, color = col, linewidth = 0.7, linestyle = '--')
        plt.legend()
        plt.grid()
        plt.ylim(margins[(flun, 'zonal')])
        plt.xlabel('Latitude')
        plt.ylabel('Integrated Net Heat Flux (W)')
        figures_exp.append(fig)

    pickle.dump(results, open(cart_out+'out_hfc_{}_.p'.format(tag), 'wb'))

    print('Saving figures...\n')
    ctl.plot_pdfpages(figure_file_exp, figures_exp)
    ctl.plot_pdfpages(figure_file_exp_maps, figures_exp_maps)

    return results

#############################################################################
#############################################################################

##########           Calculate indices: NAM, SAM, PDO, AMV, ...     ##########

#############################################################################
#############################################################################


def calc_index_climatology(clim_psl, index_name):
    """
    Calculates various climatological indices.

    < clim_psl > : the sea level pressure.
    < index_name > : can be one among 'NAM', 'SAM', 'PDO', 'AMV'.
    """




    return



#############################################################################
#############################################################################

##########              Input/output                    ##########

#############################################################################
#############################################################################

def out_WRtool_netcdf(cart_out, models, obs, inputs):
    """
    Output in netcdf format.
    """
    print('Saving netcdf output to {}\n'.format(cart_out))
    # Salvo EOFs, cluspattern_area, clus_pattern_global
    long_name = 'geopotential height anomaly at 500 hPa'
    std_name = 'geopotential_height_anomaly'
    units = 'm'

    # print('obs: ', obs.keys())
    print('models: ', list(models.values())[0].keys())
    for nam in ['model_eofs', 'cluspattern', 'cluspattern_area']:
        obsnam = nam
        modelnam = nam
        # if nam == 'model_eofs':
        #     if 'model_eofs' not in list(models.values())[0].keys():
        #         modelnam = 'eofs' # backward Compatibility
        #     if 'model_eofs' not in obs.keys():
        #         obsnam = 'eofs' # backward Compatibility

        if obs is not None:
            outfil = cart_out + '{}_ref.nc'.format(nam)
            var = obs[obsnam]
            lat = obs['lat_area']
            lon = obs['lon_area']
            if nam == 'cluspattern':
                lat = obs['lat']
                lon = obs['lon']
            index = ctl.create_iris_coord(np.arange(len(var)), None, long_name = 'index')
            colist = ctl.create_iris_coord_list([lat, lon], ['latitude', 'longitude'])
            colist = [index] + colist
            cubo = ctl.create_iris_cube(var, std_name, units, colist, long_name = long_name)
            iris.save(cubo, outfil)

        if inputs['use_reference_eofs'] and nam == 'model_eofs': continue

        # for each model
        for mod in models:
            outfil = cart_out + '{}_{}.nc'.format(nam, mod)

            var = models[mod][modelnam]
            lat = models[mod]['lat_area']
            lon = models[mod]['lon_area']
            if nam == 'cluspattern':
                lat = models[mod]['lat']
                lon = models[mod]['lon']
            index = ctl.create_iris_coord(np.arange(len(var)), None, long_name = 'index')
            colist = ctl.create_iris_coord_list([lat, lon], ['latitude', 'longitude'])
            colist = [index] + colist
            cubo = ctl.create_iris_cube(var, std_name, units, colist, long_name = long_name)

            iris.save(cubo, outfil)

    # Salvo time series: labels, pcs, dist_centroid?
    long_name = 'cluster index (ranges from 0 to {})'.format(inputs['numclus']-1)
    std_name = None
    units = '1'
    if obs is not None:
        outfil = cart_out + 'regime_index_ref.nc'

        var = obs['labels']
        dates_all = obs['dates_allyear']
        dates_season = obs['dates']

        if not inputs['is_ensemble'] or (inputs['is_ensemble'] and inputs['ens_option'] == 'member'):
            var_all, da = ctl.complete_time_range(var, dates_season, dates_all = dates_all)
        else:
            var_all = var
            da = dates_season

        time = nc.date2num(da, units = obs['time_units'], calendar = obs['time_cal'])
        time_index = ctl.create_iris_coord(time, 'time', units = obs['time_units'], calendar = obs['time_cal'])

        cubo = ctl.create_iris_cube(var_all, std_name, units, [time_index], long_name = long_name)
        iris.save(cubo, outfil)

    for mod in models:
        outfil = cart_out + 'regime_index_{}.nc'.format(mod)

        var = models[mod]['labels']
        dates_all = models[mod]['dates_allyear']
        dates_season = models[mod]['dates']

        if not inputs['is_ensemble'] or (inputs['is_ensemble'] and inputs['ens_option'] == 'member'):
            var_all, da = ctl.complete_time_range(var, dates_season, dates_all = dates_all)
        else:
            var_all = var
            da = dates_season

        time = nc.date2num(da, units = models[mod]['time_units'], calendar = models[mod]['time_cal'])
        tim_dif = time[1:]-time[:-1]
        okpo = tim_dif < 0

        print(tim_dif)
        print(okpo)
        print(np.sum(okpo))
        if np.any(okpo):
            allinds = np.arange(len(tim_dif))[okpo]+1
            allinds = [0] + list(allinds) + [None]
            #check
            if len(inputs['ensemble_members'][mod]) != len(allinds[:-1]):
                raise ValueError('num of ens mems dont match {} vs {}\n'.format(len(inputs['ensemble_members'][mod]),len(allinds[:-1])))

            for ens_id, i1, i2 in zip(inputs['ensemble_members'][mod], allinds[:-1], allinds[1:]):
                timeok = time[i1:i2]
                varok = var_all[i1:i2]
                print(ens_id, i1, i2)

                time_index = ctl.create_iris_coord(timeok, 'time', units = models[mod]['time_units'], calendar = models[mod]['time_cal'])

                cubo = ctl.create_iris_cube(varok, std_name, units, [time_index], long_name = long_name)
                outfil_ens = outfil[:-3] + '_{}.nc'.format(ens_id)
                iris.save(cubo, outfil_ens)
        else:
            outfil2 = cart_out + 'regime_index_{}_compressed.npy'.format(mod)
            if type(var_all) is np.ma.core.MaskedArray:
                np.save(outfil2, var_all.compressed())
            else:
                np.save(outfil2, var_all)

            try:
                time_index = ctl.create_iris_coord(time, 'time', units = models[mod]['time_units'], calendar = models[mod]['time_cal'])
                cubo = ctl.create_iris_cube(var_all, std_name, units, [time_index], long_name = long_name)
                iris.save(cubo, outfil)
            except Exception as errrr:
                print('Error with model {}\n'.format(mod))
                pickle.dump([mod, models[mod]['time_units'], models[mod]['time_cal'], time], open('error_iris_monotonic_{}.p'.format(mod), 'wb'))
                #raise errrr

    # monthly clus frequency
    std_name = None
    units = '1'
    if obs is not None:
        outfil = cart_out + 'clus_freq_monthly_ref.nc'

        var, datesmon = ctl.calc_monthly_clus_freq(obs['labels'], obs['dates'])

        cubolis = []
        for i, fre in enumerate(var):
            long_name = 'clus {} frequency'.format(i)
            var_all, datesall = ctl.complete_time_range(fre, datesmon)

            time = nc.date2num(datesall, units = obs['time_units'], calendar = obs['time_cal'])
            time_index = ctl.create_iris_coord(time, 'time', units = obs['time_units'], calendar = obs['time_cal'])

            cubo = ctl.create_iris_cube(var_all, std_name, units, [time_index], long_name = long_name)
            cubolis.append(cubo)

        cubolis = iris.cube.CubeList(cubolis)
        iris.save(cubolis, outfil)

    for mod in models:
        outfil = cart_out + 'clus_freq_monthly_{}.nc'.format(mod)
        var, datesmon = ctl.calc_monthly_clus_freq(models[mod]['labels'], models[mod]['dates'])

        if not inputs['is_ensemble'] or (inputs['is_ensemble'] and inputs['ens_option'] == 'member'):
            _, datesall = ctl.complete_time_range(var[0], datesmon)
        else:
            datesall = datesmon
        time = nc.date2num(datesall, units = models[mod]['time_units'], calendar = models[mod]['time_cal'])

        tim_dif = time[1:]-time[:-1]
        okpo = tim_dif < 0

        if np.any(okpo):
            allinds = np.arange(len(tim_dif))[okpo]+1
            allinds = [0] + list(allinds) + [None]
            #check
            if len(inputs['ensemble_members'][mod]) != len(allinds[:-1]):
                raise ValueError('num of ens mems dont match {} vs {}\n'.format(len(inputs['ensemble_members'][mod]),len(allinds[:-1])))

            for ens_id, i1, i2 in zip(inputs['ensemble_members'][mod], allinds[:-1], allinds[1:]):
                timeok = time[i1:i2]
                time_index = ctl.create_iris_coord(timeok, 'time', units = models[mod]['time_units'], calendar = models[mod]['time_cal'])

                cubolis = []
                for i, fre in enumerate(var):
                    varok = fre[i1:i2]
                    cubo = ctl.create_iris_cube(varok, std_name, units, [time_index], long_name = long_name)
                    cubolis.append(cubo)

                cubolis = iris.cube.CubeList(cubolis)
                outfil_ens = outfil[:-3] + '_{}.nc'.format(ens_id)
                iris.save(cubolis, outfil_ens)
        else:
            cubolis = []
            for i, fre in enumerate(var):
                if not inputs['is_ensemble'] or (inputs['is_ensemble'] and inputs['ens_option'] == 'member'):
                    var_all, datesall = ctl.complete_time_range(fre, datesmon)
                else:
                    var_all = fre
                    datesall = datesmon

                time = nc.date2num(datesall, units = models[mod]['time_units'], calendar = models[mod]['time_cal'])
                time_index = ctl.create_iris_coord(time, 'time', units = models[mod]['time_units'], calendar = models[mod]['time_cal'])

                cubo = ctl.create_iris_cube(var_all, std_name, units, [time_index], long_name = long_name)
                cubolis.append(cubo)

            cubolis = iris.cube.CubeList(cubolis)
            iris.save(cubolis, outfil)

    # pcs
    std_name = None
    units = 'm'
    if obs is not None:
        outfil = cart_out + 'pcs_timeseries_ref.nc'

        var = obs['pcs'].T
        dates_all = obs['dates_allyear']
        dates_season = obs['dates']

        cubolis = []
        for i, fre in enumerate(var):
            long_name = 'pcs {}'.format(i)
            if not inputs['is_ensemble'] or (inputs['is_ensemble'] and inputs['ens_option'] == 'member'):
                var_all, da = ctl.complete_time_range(fre, dates_season, dates_all = dates_all)
            else:
                var_all = fre
                da = dates_season

            time = nc.date2num(da, units = obs['time_units'], calendar = obs['time_cal'])
            time_index = ctl.create_iris_coord(time, 'time', units = obs['time_units'], calendar = obs['time_cal'])

            cubo = ctl.create_iris_cube(var_all, std_name, units, [time_index], long_name = long_name)
            cubolis.append(cubo)

        cubolis = iris.cube.CubeList(cubolis)
        iris.save(cubolis, outfil)

    for mod in models:
        outfil = cart_out + 'pcs_timeseries_{}.nc'.format(mod)

        var = models[mod]['pcs'].T
        dates_all = models[mod]['dates_allyear']
        dates_season = models[mod]['dates']
        print(var.shape)
        print(dates_season.shape)

        if not inputs['is_ensemble'] or (inputs['is_ensemble'] and inputs['ens_option'] == 'member'):
            _, da = ctl.complete_time_range(var[0], dates_season, dates_all = dates_all)
        else:
            da = dates_season
        time = nc.date2num(da, units = models[mod]['time_units'], calendar = models[mod]['time_cal'])

        tim_dif = time[1:]-time[:-1]
        okpo = tim_dif < 0
        if np.any(okpo):
            allinds = np.arange(len(tim_dif))[okpo]+1
            allinds = [0] + list(allinds) + [None]
            #check
            if len(inputs['ensemble_members'][mod]) != len(allinds[:-1]):
                raise ValueError('num of ens mems dont match {} vs {}\n'.format(len(inputs['ensemble_members'][mod]),len(allinds[:-1])))

            for ens_id, i1, i2 in zip(inputs['ensemble_members'][mod], allinds[:-1], allinds[1:]):
                timeok = time[i1:i2]
                time_index = ctl.create_iris_coord(timeok, 'time', units = models[mod]['time_units'], calendar = models[mod]['time_cal'])

                cubolis = []
                for i, fre in enumerate(var):
                    varok = fre[i1:i2]
                    cubo = ctl.create_iris_cube(varok, std_name, units, [time_index], long_name = long_name)
                    cubolis.append(cubo)

                cubolis = iris.cube.CubeList(cubolis)
                outfil_ens = outfil[:-3] + '_{}.nc'.format(ens_id)
                iris.save(cubolis, outfil_ens)
        else:
            cubolis = []
            for i, fre in enumerate(var):
                if not inputs['is_ensemble'] or (inputs['is_ensemble'] and inputs['ens_option'] == 'member'):
                    var_all, da = ctl.complete_time_range(fre, dates_season, dates_all = dates_all)
                else:
                    var_all = fre
                    da = dates_season

                time = nc.date2num(da, units = models[mod]['time_units'], calendar = models[mod]['time_cal'])
                time_index = ctl.create_iris_coord(time, 'time', units = models[mod]['time_units'], calendar = models[mod]['time_cal'])

                cubo = ctl.create_iris_cube(var_all, std_name, units, [time_index], long_name = long_name)
                cubolis.append(cubo)

            cubolis = iris.cube.CubeList(cubolis)
            iris.save(cubolis, outfil)

    return


def out_WRtool_mainres(outfile, models, obs, inputs):
    """
    Output of results in synthetic text format.
    """
    print('Writing main results to {}\n'.format(outfile))

    filos = open(outfile, 'w')
    ctl.printsep(filos)
    ctl.newline(filos)
    filos.write('Results of WRtool - {}\n'.format(ctl.datestamp()))
    filos.write('----> Area: {}, Season: {}, year_range: {}\n'.format(inputs['area'], inputs['season'], inputs['year_range']))
    ctl.newline(filos)
    ctl.printsep(filos)

    nsqr = np.sqrt(models[inputs['model_names'][0]]['cluspattern_area'].size)

    if obs is not None:
        # OUT for the observations
        filos.write('----> observed: {}\n'.format(inputs['obs_name']))

        if 'significance' in obs.keys():
            ctl.newline(filos)
            filos.write('---- Sharpness of regime structure ----\n')
            filos.write('{:8.3f}'.format(obs['significance']))

        ctl.newline(filos)
        filos.write('---- Optimal ratio of regime structure ----\n')
        varopt = ctl.calc_varopt_molt(obs['pcs'], obs['centroids'], obs['labels'])
        filos.write('{:10.4f}'.format(varopt))

        ctl.newline(filos)
        filos.write('---- Regimes frequencies ----\n')
        stringa = inputs['numclus']*'{:8.2f}'+'\n'
        filos.write(stringa.format(*obs['freq_clus']))

        if inputs['show_transitions']:
            ctl.newline(filos)
            filos.write('---- Transition matrix ----\n')
            stringa = (inputs['numclus']+1)*'{:11s}' + '\n'
            filos.write(stringa.format(*(['']+inputs['patnames_short'])))
            for i, cen in enumerate(obs['trans_matrix']):
                stringa = len(cen)*'{:11.2e}' + '\n'
                filos.write('{:11s}'.format(inputs['patnames_short'][i])+stringa.format(*cen))

        ctl.newline(filos)
        filos.write('---- Centroids coordinates (in pc space) ----\n')
        for i, cen in enumerate(obs['centroids']):
            stringa = len(cen)*'{:10.2f}' + '\n'
            filos.write('cluster {}: '.format(i) + stringa.format(*cen))

        ctl.newline(filos)
        if 'model_eofs_varfrac' in obs.keys():
            oks = np.sqrt(obs['model_eofs_eigenvalues'][:10])
            filos.write('---- sqrt eigenvalues of first {} model EOFs ----\n'.format(len(oks)))
        else:
            oks = np.sqrt(obs['eofs_eigenvalues'][:10])
            filos.write('---- sqrt eigenvalues of first {} EOFs ----\n'.format(len(oks)))
        stringa = len(oks)*'{:10.3e}'+'\n'
        filos.write(stringa.format(*oks))

        ctl.newline(filos)
        if 'model_eofs_varfrac' in obs.keys():
            oks = np.cumsum(obs['model_eofs_varfrac'][:10])
            filos.write('---- cumulative varfrac explained by first {} model EOFs ----\n'.format(len(oks)))
        else:
            oks = np.cumsum(obs['eofs_varfrac'][:10])
            filos.write('---- cumulative varfrac explained by first {} EOFs ----\n'.format(len(oks)))
        stringa = len(oks)*'{:8.2f}'+'\n'
        filos.write(stringa.format(*oks))
        ctl.newline(filos)
        ctl.printsep(filos)

    # NOw for each model

    for mod in inputs['model_names']:
        if 'RMS' in models[mod].keys():
            filos.write('----> model: {}\n'.format(mod))
            ctl.newline(filos)
            filos.write('---- RMS and pattern correlation wrt observed patterns ----\n')
            stringa = 'RMS:     '+inputs['numclus']*'{:8.2f}'+'\n'
            filos.write(stringa.format(*(models[mod]['RMS']/nsqr)))
            stringa = 'patcor:  '+inputs['numclus']*'{:8.2f}'+'\n'
            filos.write(stringa.format(*models[mod]['patcor']))
        else:
            filos.write('----> observed: {}\n'.format(mod))

        if 'significance' in models[mod].keys():
            ctl.newline(filos)
            filos.write('---- Sharpness of regime structure ----\n')
            filos.write('{:8.3f}'.format(models[mod]['significance']))

        ctl.newline(filos)
        filos.write('---- Optimal ratio of regime structure ----\n')
        varopt = ctl.calc_varopt_molt(models[mod]['pcs'], models[mod]['centroids'], models[mod]['labels'])
        filos.write('{:10.4f}'.format(varopt))

        ctl.newline(filos)
        filos.write('---- Regimes frequencies ----\n')
        stringa = inputs['numclus']*'{:8.2f}'+'\n'
        filos.write(stringa.format(*models[mod]['freq_clus']))

        if inputs['show_transitions']:
            ctl.newline(filos)
            filos.write('---- Transition matrix ----\n')
            stringa = (inputs['numclus']+1)*'{:11s}' + '\n'
            filos.write(stringa.format(*(['']+inputs['patnames_short'])))
            for i, cen in enumerate(models[mod]['trans_matrix']):
                stringa = len(cen)*'{:11.2e}' + '\n'
                filos.write('{:11s}'.format(inputs['patnames_short'][i])+stringa.format(*cen))

        ctl.newline(filos)
        filos.write('---- Centroids coordinates (in pc space) ----\n')
        for i, cen in enumerate(models[mod]['centroids']):
            stringa = len(cen)*'{:10.2f}' + '\n'
            filos.write('cluster {}: '.format(i) + stringa.format(*cen))

        ctl.newline(filos)
        if 'model_eofs_eigenvalues' in models[mod].keys():
            oks = np.sqrt(models[mod]['model_eofs_eigenvalues'][:10])
            filos.write('---- sqrt eigenvalues of first {} model EOFs ----\n'.format(len(oks)))
        else:
            oks = np.sqrt(models[mod]['eofs_eigenvalues'][:10])
            filos.write('---- sqrt eigenvalues of first {} EOFs ----\n'.format(len(oks)))
        stringa = len(oks)*'{:10.3e}'+'\n'
        filos.write(stringa.format(*oks))

        ctl.newline(filos)
        if 'model_eofs_varfrac' in models[mod].keys():
            oks = np.cumsum(models[mod]['model_eofs_varfrac'][:10])
            filos.write('---- cumulative varfrac explained by first {} model EOFs ----\n'.format(len(oks)))
        else:
            oks = np.cumsum(models[mod]['eofs_varfrac'][:10])
            filos.write('---- cumulative varfrac explained by first {} EOFs ----\n'.format(len(oks)))
        stringa = len(oks)*'{:8.2f}'+'\n'
        filos.write(stringa.format(*oks))
        ctl.newline(filos)
        ctl.printsep(filos)

    if inputs['groups'] is not None:
        ctl.newline(filos)
        ctl.printsep(filos)
        for gru in inputs['groups']:
            if 'RMS' in list(models.values())[0].keys():
                filos.write('----> group: {}\n'.format(gru))
                ctl.newline(filos)
                filos.write('---- RMS and pattern correlation wrt observed patterns ----\n')
                stringa = 'RMS:     '+inputs['numclus']*'{:8.2f}'+'\n'
                coso = np.sqrt(np.mean(np.array([models[mod]['RMS'] for mod in inputs['groups'][gru]])**2, axis = 0))/nsqr
                cosoerr = np.std([models[mod]['RMS'] for mod in inputs['groups'][gru]], axis = 0)/nsqr
                filos.write(stringa.format(*coso))
                stringa = '+/-      '+inputs['numclus']*'{:8.2f}'+'\n'
                filos.write(stringa.format(*cosoerr))
                stringa = 'patcor:  '+inputs['numclus']*'{:8.2f}'+'\n'
                coso = np.mean([models[mod]['patcor'] for mod in inputs['groups'][gru]], axis = 0)
                cosoerr = np.std([models[mod]['patcor'] for mod in inputs['groups'][gru]], axis = 0)
                filos.write(stringa.format(*coso))
                stringa = '+/-      '+inputs['numclus']*'{:8.2f}'+'\n'
                filos.write(stringa.format(*cosoerr))

            if 'significance' in list(models.values())[0].keys():
                ctl.newline(filos)
                filos.write('---- Sharpness of regime structure ----\n')
                sig = np.mean([models[mod]['significance'] for mod in inputs['groups'][gru]])
                std = np.std([models[mod]['significance'] for mod in inputs['groups'][gru]])
                filos.write('{:8.3f} +/- {:8.3f}'.format(sig, std))

            ctl.newline(filos)
            filos.write('---- Optimal ratio of regime structure ----\n')
            varopts = []
            for mod in inputs['groups'][gru]:
                varopts.append(ctl.calc_varopt_molt(models[mod]['pcs'], models[mod]['centroids'], models[mod]['labels']))
            sig = np.mean(varopts)
            std = np.std(varopts)
            filos.write('{:10.4f} +/- {:10.4f}'.format(sig, std))

            ctl.newline(filos)
            filos.write('---- Regimes frequencies ----\n')
            stringa = '    '+inputs['numclus']*'{:8.2f}'+'\n'
            coso = np.mean([models[mod]['freq_clus'] for mod in inputs['groups'][gru]], axis = 0)
            std = np.std([models[mod]['freq_clus'] for mod in inputs['groups'][gru]], axis = 0)
            filos.write(stringa.format(*coso))
            stringa = '+/- '+inputs['numclus']*'{:8.2f}'+'\n'
            filos.write(stringa.format(*std))

            if inputs['show_transitions']:
                ctl.newline(filos)
                filos.write('---- Transition matrix ----\n')
                stringa = (inputs['numclus']+1)*'{:11s}' + '\n'
                filos.write(stringa.format(*(['']+inputs['patnames_short'])))

                coso = np.mean([models[mod]['trans_matrix'] for mod in inputs['groups'][gru]], axis = 0)
                std = np.std([models[mod]['trans_matrix'] for mod in inputs['groups'][gru]], axis = 0)
                for i, cen in enumerate(coso):
                    stringa = len(cen)*'{:11.2e}' + '\n'
                    filos.write('{:11s}'.format(inputs['patnames_short'][i])+stringa.format(*cen))

                filos.write('---- Std dev of transition matrix ----\n')
                for i, cen in enumerate(std):
                    stringa = len(cen)*'{:11.2e}' + '\n'
                    filos.write('{:11s}'.format(inputs['patnames_short'][i])+stringa.format(*cen))

            ctl.newline(filos)
            coso = np.mean([models[mod]['centroids'] for mod in inputs['groups'][gru]], axis = 0)
            std = np.std([models[mod]['centroids'] for mod in inputs['groups'][gru]], axis = 0)
            filos.write('---- Centroids coordinates (in pc space) ----\n')
            for i, cen in enumerate(coso):
                stringa = len(cen)*'{:10.2f}' + '\n'
                filos.write('cluster {}: '.format(i) + stringa.format(*cen))

            filos.write('---- Std dev of centroids coordinates (in pc space) ----\n')
            for i, cen in enumerate(std):
                stringa = len(cen)*'{:10.2f}' + '\n'
                filos.write('cluster {}: '.format(i) + stringa.format(*cen))

            ctl.newline(filos)
            ctl.printsep(filos)

    filos.close()

    return


#############################################################################
#############################################################################

##########              Plots and visualization                    ##########

#############################################################################
#############################################################################

def plot_WRtool_results(cart_out, tag, n_ens, result_models, result_obs, model_names = None, obs_name = None, patnames = None, patnames_short = None, custom_model_colors = None, compare_models = None, central_lat_lon = (70, 0), visualization = 'Nstereo', groups = None, group_symbols = None, reference_group = None, bounding_lat = 30, plot_margins = None, draw_rectangle_area = None, taylor_mark_dim = 100, out_only_main_figs = True, use_seaborn = True, color_palette = 'hls', show_transitions = False, draw_grid = False):
    """
    Plot the results of WRtool.

    < n_ens > : int, number of ensemble members
    < result_models > : dict, output of WRtool, either for single or multiple member analysis
    < result_obs > : dict, output of WRtool for a single reference observation

    < model_names > : list. If None, the names are taken from results.keys().
    < groups > : dict, only needed for multimember. Each entry contains a list of results.keys() belonging to that group. Group names are the group dict keys().
    < custom_model_colors > : len(models)+1 colors for the models.
    < compare_models > : list of tuples. Each tuple (model_1, model_2) is compared directly (regime statistics, patterns, ecc.)

    < central_lat_lon > : tuple. Latitude and longitude of the central point in the maps. Usually (70,0) for EAT, (70,-90) per PNA.
    """
    symbols = ['o', 'd', 'v', '*', 'P', 'h', 'X', 'p', '1']
    cart_out = cart_out + tag + '/'
    if not os.path.exists(cart_out): os.mkdir(cart_out)

    n_clus = len(result_obs['cluspattern'])

    if model_names is None:
        if n_ens == 1:
            model_names = ['model']
        else:
            model_names = list(result_models.keys())
    if obs_name is None:
        obs_name = 'Obs'

    if n_ens == 1 and len(result_models.keys()) > 1:
        print('entro', result_models.keys())
        resultooo = copy(result_models)
        result_models = dict()
        result_models[model_names[0]] = resultooo

    all_figures = []

    compare_models = None
    if groups is not None:
        compare_models = []
        # labels = []
        for ll in range(len(list(groups.values())[0])):
            for k in groups:
                if len(groups[k]) < len(list(groups.values())[0]): continue # groups have unequal lengths
                #labels.append(groups[k][ll])
                if k != reference_group:
                    compare_models.append((groups[k][ll], groups[reference_group][ll]))

        labels = model_names

        for k in groups:
            if k != reference_group:
                compare_models.append((k, reference_group))
    else:
        #labels = result_models.keys()
        labels = model_names
        groups = dict()
        groups['all'] = labels
        group_symbols = dict()
        group_symbols['all'] = 'o'

    print('COMPARE:', compare_models)

    if custom_model_colors is None:
        colors = ctl.color_set(len(labels)+1, only_darker_colors = False, use_seaborn = use_seaborn, sns_palette = color_palette)
        color_dict = dict(zip(labels, colors))
    else:
        if len(custom_model_colors) != len(labels)+1:
            raise ValueError('Need {} custom_model_colors, {} given.'.format(len(result_models)+1, len(custom_model_colors)))
        colors = custom_model_colors
        color_dict = dict(zip(labels, colors))

    nuko = ctl.color_set(len(groups.keys()), only_darker_colors = False, use_seaborn = use_seaborn, sns_palette = 'Set2')
    for k, col in zip(groups.keys(), nuko):
        color_dict[k] = col

    if 'significance' in list(result_models.values())[0].keys():
        wi = 0.6
        fig = plt.figure()
        ax = plt.subplot(111)
        i = 0
        for k in groups:
            for mod in groups[k]:
                col = color_dict[mod]
                ax.bar(i, result_models[mod]['significance'], width = wi, color = col, label = mod)
                i+=0.7
            i+=0.5

        if 'significance' in result_obs.keys():
            ax.bar(i, result_obs['significance'], width = wi,  color = 'black', label = obs_name)
        if len(labels) > 4:
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 'small')
        else:
            ax.legend(fontsize = 'small', loc = 4)
        ax.set_title('Significance of regime structure')
        ax.set_xticks([])
        #ax.set_xticks(range(len(labels+[obs_name])), minor = False)
        #ax.set_xticklabels(labels+[obs_name], size='small')
        ax.set_ylabel('Significance')
        fig.savefig(cart_out+'Significance_all_{}.pdf'.format(tag))
        all_figures.append(fig)

        wi = 0.6
        fig = plt.figure()
        ax = plt.subplot(111)
        i = 0
        for k in groups:
            sig = np.mean([result_models[mod]['significance'] for mod in groups[k]])
            stddev = np.std([result_models[mod]['significance'] for mod in groups[k]])
            col = color_dict[k]
            ax.bar(i, sig, yerr = stddev, width = wi, color = col, ecolor = 'black', label = k, capsize = 5)
            i+=1.2

        if 'significance' in result_obs.keys():
            ax.bar(i, result_obs['significance'], width = wi,  color = 'black', label = obs_name)
        if len(groups.keys()) > 4:
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 'small')
        else:
            ax.legend(fontsize = 'small', loc = 4)
        ax.set_title('Significance of regime structure')
        ax.set_xticks([])
        #ax.set_xticks(range(len(labels+[obs_name])), minor = False)
        #ax.set_xticklabels(labels+[obs_name], size='small')
        ax.set_ylabel('Significance')
        fig.savefig(cart_out+'Significance_groups_{}.pdf'.format(tag))
        all_figures.append(fig)

        wi = 0.6
        fig = plt.figure()
        ax = plt.subplot(111)
        i = 0
        for ll in range(len(list(groups.values())[0])):
            for k in groups:
                if len(groups[k]) < len(list(groups.values())[0]): continue # groups have unequal lengths
                mod = groups[k][ll]
                col = color_dict[mod]
                ax.bar(i, result_models[mod]['significance'], width = wi, color = col, label = mod)
                i+=0.7
            i+=0.5

        if 'significance' in result_obs.keys():
            ax.bar(i, result_obs['significance'], width = wi,  color = 'black', label = obs_name)
        if len(labels) > 4:
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 'small')
        else:
            ax.legend(fontsize = 'small', loc = 4)
        ax.set_title('Significance of regime structure')
        ax.set_xticks([])
        #ax.set_xticks(range(len(labels+[obs_name])), minor = False)
        #ax.set_xticklabels(labels+[obs_name], size='small')
        ax.set_ylabel('Significance')
        fig.savefig(cart_out+'Significance_1vs1_{}.pdf'.format(tag))
        all_figures.append(fig)

    nsqr = np.sqrt(result_obs['cluspattern_area'].size)
    if 'RMS' in list(result_models.values())[0].keys():
        wi = 0.6
        fig = plt.figure()
        ax = plt.subplot(111)
        i = 0
        for k in groups:
            for mod in groups[k]:
                col = color_dict[mod]
                rms = np.sqrt(np.mean(np.array(result_models[mod]['RMS'])**2))/nsqr
                ax.bar(i, rms, width = wi, color = col, label = mod)
                i+=0.7
            i+=0.5

        if len(labels) > 4:
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 'small')
        else:
            ax.legend(fontsize = 'small', loc = 4)
        ax.set_title('Total RMS vs observations')
        ax.set_xticks([])
        #ax.set_xticks(range(len(labels+[obs_name])), minor = False)
        #ax.set_xticklabels(labels+[obs_name], size='small')
        ax.set_ylabel('RMS (m)')
        fig.savefig(cart_out+'RMS_all_{}.pdf'.format(tag))
        all_figures.append(fig)

        wi = 0.6
        fig = plt.figure()
        ax = plt.subplot(111)
        i = 0
        for k in groups:
            rmss = [np.sqrt(np.mean(np.array(result_models[mod]['RMS'])**2))/nsqr for mod in groups[k]]
            sig = np.mean(rmss)
            stddev = np.std(rmss)
            # sig = np.mean([result_models[mod]['RMS'] for mod in groups[k]])
            # stddev = np.std([result_models[mod]['RMS'] for mod in groups[k]])
            col = color_dict[k]
            ax.bar(i, sig, yerr = stddev, width = wi, color = col, ecolor = 'black', label = k, capsize = 5)
            i+=1.2

        if len(groups.keys()) > 4:
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 'small')
        else:
            ax.legend(fontsize = 'small', loc = 4)
        ax.set_title('Total RMS vs observations')
        ax.set_xticks([])
        #ax.set_xticks(range(len(labels+[obs_name])), minor = False)
        #ax.set_xticklabels(labels+[obs_name], size='small')
        ax.set_ylabel('RMS (m)')
        fig.savefig(cart_out+'RMS_groups_{}.pdf'.format(tag))
        all_figures.append(fig)

        wi = 0.6
        fig = plt.figure()
        ax = plt.subplot(111)
        i = 0
        for ll in range(len(list(groups.values())[0])):
            for k in groups:
                if len(groups[k]) < len(list(groups.values())[0]): continue # groups have unequal lengths
                mod = groups[k][ll]
                col = color_dict[mod]
                rms = np.sqrt(np.mean(np.array(result_models[mod]['RMS'])**2))/nsqr
                ax.bar(i, rms, width = wi, color = col, label = mod)
                i+=0.7
            i+=0.5
        if len(labels) > 4:
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 'small')
        else:
            ax.legend(fontsize = 'small', loc = 4)
        ax.set_title('Total RMS vs observations')
        ax.set_xticks([])
        #ax.set_xticks(range(len(labels+[obs_name])), minor = False)
        #ax.set_xticklabels(labels+[obs_name], size='small')
        ax.set_ylabel('RMS (m)')
        fig.savefig(cart_out+'RMS_1vs1_{}.pdf'.format(tag))
        all_figures.append(fig)

    if 'patcor' in list(result_models.values())[0].keys():
        wi = 0.6
        fig = plt.figure()
        ax = plt.subplot(111)
        i = 0
        for k in groups:
            for mod in groups[k]:
                col = color_dict[mod]
                #rms = np.sqrt(np.mean(np.array(result_models[mod]['RMS'])**2))/nsqr
                rms = np.mean(result_models[mod]['patcor'])
                ax.bar(i, rms, width = wi, color = col, label = mod)
                i+=0.7
            i+=0.5

        if len(labels) > 4:
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 'small')
        else:
            ax.legend(fontsize = 'small', loc = 4)
        ax.set_title('Pattern correlation with observations')
        ax.set_xticks([])
        #ax.set_xticks(range(len(labels+[obs_name])), minor = False)
        #ax.set_xticklabels(labels+[obs_name], size='small')
        ax.set_ylabel('Correlation')
        fig.savefig(cart_out+'patcor_all_{}.pdf'.format(tag))
        all_figures.append(fig)

        wi = 0.6
        fig = plt.figure()
        ax = plt.subplot(111)
        i = 0
        for k in groups:
            rmss = [np.mean(np.array(result_models[mod]['patcor'])) for mod in groups[k]]
            sig = np.mean(rmss)
            stddev = np.std(rmss)
            # sig = np.mean([result_models[mod]['RMS'] for mod in groups[k]])
            # stddev = np.std([result_models[mod]['RMS'] for mod in groups[k]])
            col = color_dict[k]
            ax.bar(i, sig, yerr = stddev, width = wi, color = col, ecolor = 'black', label = k, capsize = 5)
            i+=1.2

        if len(groups.keys()) > 4:
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 'small')
        else:
            ax.legend(fontsize = 'small', loc = 4)
        ax.set_title('Pattern correlation with observations')
        ax.set_xticks([])
        #ax.set_xticks(range(len(labels+[obs_name])), minor = False)
        #ax.set_xticklabels(labels+[obs_name], size='small')
        ax.set_ylabel('Correlation')
        fig.savefig(cart_out+'patcor_groups_{}.pdf'.format(tag))
        all_figures.append(fig)

        wi = 0.6
        fig = plt.figure()
        ax = plt.subplot(111)
        i = 0
        for ll in range(len(list(groups.values())[0])):
            for k in groups:
                if len(groups[k]) < len(list(groups.values())[0]): continue # groups have unequal lengths
                mod = groups[k][ll]
                col = color_dict[mod]
                rms = np.mean(np.array(result_models[mod]['patcor']))
                ax.bar(i, rms, width = wi, color = col, label = mod)
                i+=0.7
            i+=0.5
        if len(labels) > 4:
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 'small')
        else:
            ax.legend(fontsize = 'small', loc = 4)
        ax.set_title('Pattern correlation with observations')
        ax.set_xticks([])
        #ax.set_xticks(range(len(labels+[obs_name])), minor = False)
        #ax.set_xticklabels(labels+[obs_name], size='small')
        ax.set_ylabel('Correlation')
        fig.savefig(cart_out+'patcor_1vs1_{}.pdf'.format(tag))
        all_figures.append(fig)

    patt_ref = result_obs['cluspattern']
    lat = result_obs['lat']
    lon = result_obs['lon']

    if patt_ref[0].shape != list(result_models.values())[0]['cluspattern'][0].shape:
        nupatt_ref = []
        _, oklats, _ = np.intersect1d(result_obs['lat'], list(result_models.values())[0]['lat'], assume_unique = True, return_indices = True)
        _, oklons, _ = np.intersect1d(result_obs['lon'], list(result_models.values())[0]['lon'], assume_unique = True, return_indices = True)
        for nu in range(n_clus):
            coso = patt_ref[nu][oklats, :]
            coso2 = coso[:, oklons]
            nupatt_ref.append(coso2)
            lat = result_obs['lat'][oklats]
            lon = result_obs['lon'][oklons]
        patt_ref = nupatt_ref

    if patnames is None:
        patnames = ['clus_{}'.format(i) for i in range(len(patt_ref))]
    if patnames_short is None:
        patnames_short = ['c{}'.format(i) for i in range(len(patt_ref))]

    # PLOTTIN the frequency histogram
    plot_diffs = True
    if 'freq_clus' in list(result_models.values())[0].keys():
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.grid(axis = 'y', zorder = 0)
        wi = 0.8
        n_tot = len(labels)+1
        for j in range(n_clus):
            central = j*(n_tot*1.5)
            for i, (mod, col) in enumerate(zip(labels, colors)):
                labelmod = None
                if j == 0: labelmod = mod
                if plot_diffs:
                    ax.bar(central-(n_tot-1)/2.+i, result_models[mod]['freq_clus'][j]-result_obs['freq_clus'][j], width = wi, color = col, label = labelmod, zorder = 5)
                else:
                    ax.bar(central-(n_tot-1)/2.+i, result_models[mod]['freq_clus'][j], width = wi, color = col, label = labelmod, zorder = 5)
            labelmod = None
            if j == 0: labelmod = obs_name
            if not plot_diffs:
                ax.bar(central-(n_tot-1)/2.+i+1, result_obs['freq_clus'][j], width = wi,  color = 'black', label = labelmod, zorder = 5)
        if len(labels) > 4:
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 'small')
        else:
            ax.legend(fontsize = 'small', loc = 1)
        ax.set_title('Regimes frequencies')
        ax.set_xticks([j*(n_tot*1.5) for j in range(n_clus)], minor = False)
        ax.set_xticklabels(patnames_short, size='small')
        ax.set_ylabel('Frequency')
        fig.savefig(cart_out+'Regime_frequency_{}.pdf'.format(tag))
        all_figures.append(fig)

        fig = plt.figure()
        ax = plt.subplot(111)
        ax.grid(axis = 'y', zorder = 0)
        ax.set_axisbelow(True)
        wi = 0.8

        n_tot = len(groups.keys())+1
        for j in range(n_clus):
            central = j*(n_tot*1.5)

            for i, k in enumerate(groups.keys()):
                sig = np.mean([result_models[mod]['freq_clus'][j] for mod in groups[k]])
                stddev = np.std([result_models[mod]['freq_clus'][j] for mod in groups[k]])
                col = color_dict[k]
                labelmod = None
                if j == 0: labelmod = k
                if plot_diffs:
                    ax.bar(central-(n_tot-1)/2.+i, sig-result_obs['freq_clus'][j], width = wi, color = col, label = labelmod, zorder = 5)
                    ax.errorbar(central-(n_tot-1)/2.+i, sig-result_obs['freq_clus'][j], yerr = stddev, color = 'black', capsize = 3, zorder = 6)
                else:
                    ax.bar(central-(n_tot-1)/2.+i, sig, yerr = stddev, width = wi, color = col, ecolor = 'black', label = labelmod, capsize = 3, zorder = 5)
                    ax.errorbar(central-(n_tot-1)/2.+i, sig, yerr = stddev, color = 'black', capsize = 3, zorder = 6)
            labelmod = None
            if j == 0: labelmod = obs_name
            if not plot_diffs:
                ax.bar(central-(n_tot-1)/2.+i+1, result_obs['freq_clus'][j], width = wi,  color = 'black', label = labelmod, zorder = 5)
        if len(groups.keys()) > 4:
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 'small')
        else:
            ax.legend(fontsize = 'small', loc = 4)
        ax.set_title('Regimes frequencies')
        ax.set_xticks([j*(n_tot*1.5) for j in range(n_clus)], minor = False)
        ax.set_xticklabels(patnames_short, size='small')
        ax.set_ylabel('Frequency')
        fig.savefig(cart_out+'Regime_frequency_groups_{}.pdf'.format(tag))
        all_figures.append(fig)

    i1 = int(np.ceil(np.sqrt(n_clus)))
    i2 = n_clus//i1
    if i2*i1 < n_clus:
        i2 = i2 + 1

    # PLOTTIN the persistence histograms
    if 'resid_times' in list(result_models.values())[0].keys():
        axes = []
        for lab in labels:
            fig = plt.figure()
            # binzzz = np.arange(0,36,5)
            for j in range(n_clus):
                ax = fig.add_subplot(i1,i2,j+1)
                ax.set_title(patnames[j])

                max_days = 29
                numarr, frek_obs = ctl.count_occurrences(result_obs['resid_times'][j], num_range = (0, max_days))
                ax.bar(numarr, frek_obs, alpha = 0.5, label = obs_name, color = 'indianred')
                coso_obs = ctl.running_mean(frek_obs[:-1], 3)
                numarr, frek_mod = ctl.count_occurrences(result_models[lab]['resid_times'][j], num_range = (0, max_days))
                ax.bar(numarr, frek_mod, alpha = 0.5, label = lab, color = 'steelblue')
                coso_mod = ctl.running_mean(frek_mod[:-1], 3)
                ax.plot(numarr[:-1], coso_obs, color = 'indianred')
                ax.plot(numarr[:-1], coso_mod, color = 'steelblue')
                ax.legend()
                ax.set_xlim(0, max_days+2)
                tics = np.arange(0,max_days+2,5)
                labs = ['{}'.format(ti) for ti in tics[:-1]]
                labs.append('>{}'.format(max_days))
                ax.set_xticks(tics, minor = False)
                ax.set_xticklabels(labs, size='small')
                ax.set_xlabel('Days')
                ax.set_ylabel('Frequency')
                axes.append(ax)


            plt.suptitle('Residence times - {}'.format(lab))
            fig.tight_layout()
            fig.subplots_adjust(top=0.88)

            all_figures.append(fig)

        ctl.adjust_ax_scale(axes)

        if compare_models is not None:
            axes = []
            for coup in compare_models:
                print(coup)
                fig = plt.figure()
                for j in range(n_clus):
                    ax = fig.add_subplot(i1,i2,j+1)
                    ax.set_title(patnames[j])

                    if coup[1] in model_names:
                        model_1 = result_models[coup[1]]['resid_times'][j]
                    elif coup[1] in groups.keys():
                        model_1 = np.concatenate([result_models[k]['resid_times'][j] for k in groups[coup[1]]])
                    else:
                        print('# WARNING: compare_models: {} not found. continue..\n'.format(coup[1]))
                        continue

                    if coup[0] in model_names:
                        model_0 = result_models[coup[0]]['resid_times'][j]
                    elif coup[0] in groups.keys():
                        model_0 = np.concatenate([result_models[k]['resid_times'][j] for k in groups[coup[0]]])
                    else:
                        print('# WARNING: compare_models: {} not found. continue..\n'.format(coup[0]))
                        continue

                    max_days = 29
                    numarr, frek_obs_ERA = ctl.count_occurrences(result_obs['resid_times'][j], num_range = (0, max_days))
                    numarr, frek_obs = ctl.count_occurrences(model_1, num_range = (0, max_days))
                    frek_obs -= frek_obs_ERA
                    ax.bar(numarr, frek_obs, alpha = 0.5, label = coup[1], color = 'indianred')
                    coso_obs = ctl.running_mean(frek_obs[:-1], 3)
                    numarr, frek_mod = ctl.count_occurrences(model_0, num_range = (0, max_days))
                    frek_mod -= frek_obs_ERA
                    ax.bar(numarr, frek_mod, alpha = 0.5, label = coup[0], color = 'steelblue')
                    coso_mod = ctl.running_mean(frek_mod[:-1], 3)
                    ax.plot(numarr[:-1], coso_obs, color = 'indianred')
                    ax.plot(numarr[:-1], coso_mod, color = 'steelblue')
                    ax.legend()
                    ax.set_xlim(0, max_days+2)
                    ax.set_ylim(-0.06,0.06)
                    tics = np.arange(0,max_days+2,5)
                    labs = ['{}'.format(ti) for ti in tics[:-1]]
                    labs.append('>{}'.format(max_days))
                    ax.set_xticks(tics, minor = False)
                    ax.set_xticklabels(labs, size='small')
                    ax.set_xlabel('Days')
                    ax.set_ylabel('Frequency diff', fontsize = 'small')
                    axes.append(ax)

                plt.suptitle('Resid. times diffs - {} vs {}'.format(coup[0], coup[1]))
                fig.tight_layout()
                fig.subplots_adjust(top=0.88)

                all_figures.append(fig)

            ctl.adjust_ax_scale(axes)

    # PLOTTIN the transition matrices
    if 'trans_matrix' in list(result_models.values())[0].keys() and show_transitions:
        cmapparu = cm.get_cmap('RdBu_r')
        mappe = []

        for lab in labels:
            fig = plt.figure(figsize=(16,6))
            ax = fig.add_subplot(121)
            gigi = ax.imshow(result_models[lab]['trans_matrix'], norm = LogNorm(vmin = 0.01, vmax = 1.0))
            ax.xaxis.tick_top()
            #ax.invert_yaxis()
            # ax.set_xticks(np.arange(n_clus)+0.5, minor=False)
            # ax.set_yticks(np.arange(n_clus)+0.5, minor=False)
            ax.set_xticks(np.arange(n_clus), minor = False)
            ax.set_xticklabels(patnames_short, size='small')
            ax.set_yticks(np.arange(n_clus), minor = False)
            ax.set_yticklabels(patnames_short, size='small')
            cb = plt.colorbar(gigi)
            cb.set_label('Transition probability')

            ax = fig.add_subplot(122)
            vmin = np.percentile(result_models[lab]['trans_matrix']-result_obs['trans_matrix'], 5)
            vmax = np.percentile(result_models[lab]['trans_matrix']-result_obs['trans_matrix'], 95)
            cos = np.max(abs(np.array([vmin,vmax])))
            gigi = ax.imshow(result_models[lab]['trans_matrix']-result_obs['trans_matrix'], vmin = -cos, vmax = cos, cmap = cmapparu)
            ax.xaxis.tick_top()
            ax.set_xticks(np.arange(n_clus), minor = False)
            ax.set_xticklabels(patnames_short, size='small')
            ax.set_yticks(np.arange(n_clus), minor = False)
            ax.set_yticklabels(patnames_short, size='small')
            cb = plt.colorbar(gigi)
            cb.set_label('Differences btw {} and {}'.format(lab, obs_name))
            mappe.append(gigi)

            fig.suptitle(lab)
            fig.tight_layout()
            fig.subplots_adjust(top=0.88)
            all_figures.append(fig)

        if compare_models is not None:
            for coup in compare_models:
                print(coup)
                if coup[1] in model_names:
                    model_1 = result_models[coup[1]]['trans_matrix']
                elif coup[1] in groups.keys():
                    model_1 = np.mean([result_models[k]['trans_matrix'] for k in groups[coup[1]]], axis = 0)
                else:
                    print('# WARNING: compare_models: {} not found. continue..\n'.format(coup[1]))
                    continue

                if coup[0] in model_names:
                    model_0 = result_models[coup[0]]['trans_matrix']
                elif coup[0] in groups.keys():
                    model_0 = np.mean([result_models[k]['trans_matrix'] for k in groups[coup[0]]], axis = 0)
                else:
                    print('# WARNING: compare_models: {} not found. continue..\n'.format(coup[0]))
                    continue

                fig = plt.figure(figsize=(24,6))
                ax = fig.add_subplot(131)
                vmin = np.percentile(model_0-result_obs['trans_matrix'], 5)
                vmax = np.percentile(model_0-result_obs['trans_matrix'], 95)
                cos = np.max(abs(np.array([vmin,vmax])))
                gigi = ax.imshow(model_0-result_obs['trans_matrix'], vmin = -cos, vmax = cos, cmap = cmapparu)
                ax.set_title('{} vs obs'.format(coup[0]))
                ax.xaxis.tick_top()
                ax.set_xticks(np.arange(n_clus), minor = False)
                ax.set_xticklabels(patnames_short, size='small')
                ax.set_yticks(np.arange(n_clus), minor = False)
                ax.set_yticklabels(patnames_short, size='small')
                mappe.append(gigi)

                ax = fig.add_subplot(132)
                vmin = np.percentile(model_1-result_obs['trans_matrix'], 5)
                vmax = np.percentile(model_1-result_obs['trans_matrix'], 95)
                cos = np.max(abs(np.array([vmin,vmax])))
                gigi = ax.imshow(model_1-result_obs['trans_matrix'], vmin = -cos, vmax = cos, cmap = cmapparu)
                ax.set_title('{} vs obs'.format(coup[1]))
                ax.xaxis.tick_top()
                ax.set_xticks(np.arange(n_clus), minor = False)
                ax.set_xticklabels(patnames_short, size='small')
                ax.set_yticks(np.arange(n_clus), minor = False)
                ax.set_yticklabels(patnames_short, size='small')
                mappe.append(gigi)
                cb = plt.colorbar(gigi)
                cb.set_label('Diffs vs observed')

                ax = fig.add_subplot(133)
                vmin = np.percentile(model_0-model_1, 5)
                vmax = np.percentile(model_0-model_1, 95)
                cos = np.max(abs(np.array([vmin,vmax])))
                gigi = ax.imshow(model_0-model_1, vmin = -cos, vmax = cos, cmap = cmapparu)
                ax.set_title('{} vs {}'.format(coup[0], coup[1]))
                ax.xaxis.tick_top()
                ax.set_xticks(np.arange(n_clus), minor = False)
                ax.set_xticklabels(patnames_short, size='small')
                ax.set_yticks(np.arange(n_clus), minor = False)
                ax.set_yticklabels(patnames_short, size='small')
                cb = plt.colorbar(gigi)
                cb.set_label('Diffs btw models')

                plt.suptitle('{} vs {}'.format(coup[0], coup[1]))
                all_figures.append(fig)

        ctl.adjust_color_scale(mappe)

    patt = result_obs['cluspattern']
    lat_ref = result_obs['lat']
    lon_ref = result_obs['lon']
    filename = cart_out+'Allclus_OBSERVED.pdf'
    figs = ctl.plot_multimap_contour(patt, lat_ref, lon_ref, filename, visualization = visualization, central_lat_lon = central_lat_lon, cmap = 'RdBu_r', title = 'Observed weather regimes', subtitles = patnames, cb_label = 'Geopotential height anomaly (m)', color_percentiles = (0.5,99.5), number_subplots = False, bounding_lat = bounding_lat, plot_margins = plot_margins, add_rectangles = draw_rectangle_area, draw_grid = draw_grid)
    all_figures += figs
    figs[0].savefig(filename)

    # PLOTTIN the cluster patterns
    for lab in labels:
        print(lab)
        lat = result_models[lab]['lat']
        lon = result_models[lab]['lon']
        patt = result_models[lab]['cluspattern']
        if np.any(np.isnan(patt)):
            print('There are {} NaNs in this patt.. replacing with zeros\n'.format(np.sum(np.isnan(patt))))
            patt[np.isnan(patt)] = 0.0
        if not out_only_main_figs:
            cartout_mod = cart_out + 'mod_{}/'.format(lab)
            if not os.path.exists(cartout_mod): os.mkdir(cartout_mod)
            filename = cartout_mod+'Allclus_'+lab+'.pdf'
        else:
            filename = None

        figs = ctl.plot_multimap_contour(patt, lat, lon, filename, visualization = visualization, central_lat_lon = central_lat_lon, cmap = 'RdBu_r', title = 'Simulated weather regimes - {}'.format(lab), subtitles = patnames, cb_label = 'Geopotential height anomaly (m)', color_percentiles = (0.5,99.5), number_subplots = False, bounding_lat = bounding_lat, plot_margins = plot_margins, add_rectangles = draw_rectangle_area, draw_grid = draw_grid)
        all_figures += figs
        if not out_only_main_figs:
            for patuno, patuno_ref, pp, pps in zip(patt, patt_ref, patnames, patnames_short):
                nunam = cartout_mod+'clus_'+pps+'_'+lab+'.pdf'
                print(nunam)
                fig = ctl.plot_triple_sidebyside(patuno, patuno_ref, [lat, lat_ref], [lon, lon_ref], filename = nunam, visualization = visualization, central_lat_lon = central_lat_lon, title = pp+' ({})'.format(lab), cb_label = 'Geopotential height anomaly (m)', stitle_1 = lab, stitle_2 = 'ERA', color_percentiles = (0.5,99.5), draw_contour_lines = True, bounding_lat = bounding_lat, plot_margins = plot_margins, add_rectangles = draw_rectangle_area, draw_grid = draw_grid, use_different_grids = True)
                all_figures.append(fig)

    if compare_models is not None and not out_only_main_figs:
        for coup in compare_models:
            if coup[0] not in model_names:
                continue
            patt = result_models[coup[0]]['cluspattern']
            patt2 = result_models[coup[1]]['cluspattern']
            if np.any(np.isnan(patt+patt2)):
                print('There are {} NaNs in this patt.. replacing with zeros\n'.format(np.sum(np.isnan(patt+patt2))))
                patt[np.isnan(patt)] = 0.0
                patt2[np.isnan(patt2)] = 0.0

            for patuno, patuno_ref, pp, pps in zip(patt, patt2, patnames, patnames_short):
                nunam = cart_out+'compare_clus_'+pps+'_'+coup[0]+'_vs_'+coup[1]+'.pdf'
                fig = ctl.plot_triple_sidebyside(patuno, patuno_ref, lat, lon, filename = nunam, visualization = visualization, central_lat_lon = central_lat_lon, title = pp, cb_label = 'Geopotential height anomaly (m)', stitle_1 = coup[0], stitle_2 = coup[1], color_percentiles = (0.5,99.5), draw_contour_lines = True, bounding_lat = bounding_lat, plot_margins = plot_margins, add_rectangles = draw_rectangle_area, draw_grid = draw_grid)
                all_figures.append(fig)


    markers = None
    if group_symbols is not None:
        markers = []
        for lab in labels:
            for k in groups:
                if lab in groups[k]:
                    print(lab, k, group_symbols[k])
                    markers.append(group_symbols[k])

    group_colors = [color_dict[k] for k in groups.keys()]
    print(markers)

    obs = result_obs['cluspattern_area']
    modpats = [result_models[lab]['cluspattern_area'] for lab in labels]
    max_val_sd = 1.1*np.max([np.max([np.std(pat[i])/np.std(obs[i]) for pat in modpats]) for i in range(len(patnames))])
    # Taylor plots
    if not out_only_main_figs:
        for num, patt in enumerate(patnames):
            obs = result_obs['cluspattern_area'][num, ...]
            modpats = [result_models[lab]['cluspattern_area'][num, ...] for lab in labels]

            filename = cart_out + 'TaylorPlot_{}.pdf'.format(patnames_short[num])
            label_ERMS_axis = 'Total RMS error (m)'
            label_bias_axis = 'Pattern mean (m)'

            figs = ctl.Taylor_plot(modpats, obs, filename, title = patt, label_bias_axis = label_bias_axis, label_ERMS_axis = label_ERMS_axis, colors = colors, markers = markers, only_first_quarter = False, legend = True, marker_edge = None, labels = labels, obs_label = obs_name, mod_points_size = taylor_mark_dim, obs_points_size = int(1.1*taylor_mark_dim), max_val_sd = max_val_sd)
            all_figures += figs

    numens_ok = int(np.ceil(n_clus))
    side1 = int(np.ceil(np.sqrt(numens_ok)))
    side2 = int(np.ceil(numens_ok/float(side1)))
    if side1 > side2:
        cos = side1
        side1 = side2
        side2 = cos

    fig = plt.figure(figsize=(16,12))
    for num, patt in enumerate(patnames):
        ax = plt.subplot(side1, side2, num+1, polar = True)

        obs = result_obs['cluspattern_area'][num, ...]
        modpats = [result_models[lab]['cluspattern_area'][num, ...] for lab in labels]

        legok = False
        ctl.Taylor_plot(modpats, obs, ax = ax, title = None, colors = colors, markers = markers, only_first_quarter = True, legend = legok, labels = labels, obs_label = obs_name, mod_points_size = taylor_mark_dim, obs_points_size = int(1.1*taylor_mark_dim), max_val_sd = max_val_sd)


    #Custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    if markers is not None:
        mark_all = markers
    else:
        mark_all = ['o']*len(labels)
    legend_elements = []
    for col, lab, mark in zip(colors, labels, mark_all):
        if lab is None:
            break
        legend_elements.append(Line2D([0], [0], marker=mark, color=col, label=lab, linestyle = ''))

    legend_elements.append(Line2D([0], [0], marker='D', color='black', label='ERA', linestyle = ''))
    fig.legend(handles=legend_elements, loc=1, fontsize = 'large')

    fig.tight_layout()
    if len(patnames) == 4:
        n1 = plt.text(0.15,0.6,patnames[0],transform=fig.transFigure, fontsize = 20)
        n3 = plt.text(0.6,0.6,patnames[1],transform=fig.transFigure, fontsize = 20)
        n2 = plt.text(0.15,0.1,patnames[2],transform=fig.transFigure, fontsize = 20)
        n4 = plt.text(0.6,0.1,patnames[3],transform=fig.transFigure, fontsize = 20)
        bbox=dict(facecolor = 'lightsteelblue', alpha = 0.7, edgecolor='black', boxstyle='round,pad=0.2')
        n1.set_bbox(bbox)
        n2.set_bbox(bbox)
        n3.set_bbox(bbox)
        n4.set_bbox(bbox)

    fig.savefig(cart_out + 'TaylorPlot.pdf')
    all_figures.append(fig)

    # Ellipse plot
    if 'RMS' in list(result_models.values())[0].keys():
        data = dict()
        for cos in ['RMS', 'patcor']:
            fac = 1.
            if cos == 'RMS': fac = nsqr
            data[cos] = []
            data[cos+'_err'] = []
            data[cos+'_errlarge'] = []
            for grp in groups:
                data[cos].append(np.mean([result_models[mod][cos] for mod in groups[grp]], axis = 0)/fac)
                data[cos+'_err'].append(np.std([result_models[mod][cos] for mod in groups[grp]], axis = 0)/(fac * np.sqrt(len(groups[grp])-1)))
                data[cos+'_errlarge'].append(np.std([result_models[mod][cos] for mod in groups[grp]], axis = 0)/fac)
            data[cos] = np.stack(data[cos])
            data[cos+'_err'] = np.stack(data[cos+'_err'])
            data[cos+'_errlarge'] = np.stack(data[cos+'_errlarge'])

        if not np.any([len(groups[k]) == 1 for k in groups.keys()]):
            fig = plt.figure(figsize = (16,12))

            for j in range(n_clus):
                ax = fig.add_subplot(i1,i2,j+1)
                ax.set_title(patnames[j], fontsize = 18, fontweight = 'bold')

                ctl.ellipse_plot(data['patcor'][:,j], data['RMS'][:,j], data['patcor_err'][:,j], data['RMS_err'][:,j], labels = list(groups.keys()), ax = ax, colors = group_colors, alpha = 0.7)

                for grp in groups:
                    pats = [result_models[mod]['patcor'][j] for mod in groups[grp]]
                    rmss = [result_models[mod]['RMS'][j]/nsqr for mod in groups[grp]]
                    ax.scatter(pats, rmss, color = color_dict[grp], s = 25, marker = group_symbols[grp])

                ax.set_xlim(0.35, 1.0)
                ax.set_ylim(0., 27.0)
                ax.tick_params(labelsize=14)
                plt.gca().invert_xaxis()
                ax.set_xlabel('Pattern correlation', fontsize = 18)
                ax.set_ylabel('RMS (m)', fontsize = 18)
                ax.grid()

            plt.tight_layout()
            # plt.subplots_adjust(top = 0.9)
            # plt.suptitle('Average performance of PRIMAVERA stream1 coupled models', fontsize = 28)
            fig.savefig(cart_out + 'ellipse_plot_{}.pdf'.format(tag))
            all_figures.append(fig)


    filename = cart_out + 'WRtool_{}_allfig.pdf'.format(tag)
    ctl.plot_pdfpages(filename, all_figures)

    return


def plot_WRtool_singlemodel(cart_out, tag, results, model_name = None, patnames = None, patnames_short = None, central_lat_lon = (70, 0), visualization = 'Nstereo', bounding_lat = 30, plot_margins = None, draw_rectangle_area = None, taylor_mark_dim = 100, use_seaborn = True, color_palette = 'hls', show_transitions = False, draw_grid = False):
    """
    Plot the results of WRtool.

    < results > : dict, output of WRtool, either for single or multiple member analysis
    < model_name > : str.
    < central_lat_lon > : tuple. Latitude and longitude of the central point in the maps. Usually (70,0) for EAT, (70,-90) per PNA.
    """
    cart_out = cart_out + tag + '/'
    if not os.path.exists(cart_out): os.mkdir(cart_out)

    n_clus = len(results['cluspattern'])

    if model_name is None:
        model_name = 'model'

    all_figures = []
    color = 'blue'

    patt_ref = results['cluspattern']
    lat = results['lat']
    lon = results['lon']

    if patnames is None:
        patnames = ['clus_{}'.format(i) for i in range(len(patt_ref))]
    if patnames_short is None:
        patnames_short = ['c{}'.format(i) for i in range(len(patt_ref))]

    # PLOTTIN the frequency histogram
    if 'freq_clus' in results.keys():
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.grid(axis = 'y', zorder = 0)
        wi = 0.8
        for j in range(n_clus):
            ax.bar(j, results['freq_clus'][j], width = wi,  color = color, zorder = 5)

        ax.legend(fontsize = 'small', loc = 1)
        ax.set_title('Regimes frequencies')
        ax.set_xticks([j for j in range(n_clus)], minor = False)
        ax.set_xticklabels(patnames_short, size='small')
        ax.set_ylabel('Frequency')
        fig.savefig(cart_out+'Regime_frequency_{}.pdf'.format(tag))
        all_figures.append(fig)

    i1 = int(np.ceil(np.sqrt(n_clus)))
    i2 = n_clus//i1
    if i2*i1 < n_clus:
        i2 = i2 + 1

    # PLOTTIN the persistence histograms
    if 'resid_times' in results.keys():
        axes = []
        fig = plt.figure()
        # binzzz = np.arange(0,36,5)
        for j in range(n_clus):
            ax = fig.add_subplot(i1,i2,j+1)
            ax.set_title(patnames[j])

            max_days = 29
            numarr, frek_obs = ctl.count_occurrences(results['resid_times'][j], num_range = (0, max_days))
            ax.bar(numarr, frek_obs, alpha = 0.5, label = model_name, color = 'indianred')
            coso_obs = ctl.running_mean(frek_obs[:-1], 3)
            ax.plot(numarr[:-1], coso_obs, color = 'indianred')
            ax.legend()
            ax.set_xlim(0, max_days+2)
            tics = np.arange(0,max_days+2,5)
            labs = ['{}'.format(ti) for ti in tics[:-1]]
            labs.append('>{}'.format(max_days))
            ax.set_xticks(tics, minor = False)
            ax.set_xticklabels(labs, size='small')
            ax.set_xlabel('Days')
            ax.set_ylabel('Frequency')
            axes.append(ax)

        plt.suptitle('Residence times - {}'.format(model_name))
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)

        ctl.adjust_ax_scale(axes)
        all_figures.append(fig)


    if 'trans_matrix' in results.keys() and show_transitions:
        cmapparu = cm.get_cmap('RdBu_r')
        mappe = []

        fig = plt.figure(figsize=(16,6))
        ax = fig.add_subplot(121)
        gigi = ax.imshow(results[lab]['trans_matrix'], norm = LogNorm(vmin = 0.01, vmax = 1.0))
        ax.xaxis.tick_top()
        #ax.invert_yaxis()
        # ax.set_xticks(np.arange(n_clus)+0.5, minor=False)
        # ax.set_yticks(np.arange(n_clus)+0.5, minor=False)
        ax.set_xticks(np.arange(n_clus), minor = False)
        ax.set_xticklabels(patnames_short, size='small')
        ax.set_yticks(np.arange(n_clus), minor = False)
        ax.set_yticklabels(patnames_short, size='small')
        cb = plt.colorbar(gigi)
        cb.set_label('Transition probability')

        ax = fig.add_subplot(122)
        numat = results['trans_matrix'][:]
        for i in range(numat.shape[0]):
            numat[i,i] = np.nan
        gigi = ax.imshow(numat, vmin = 0.01, vmax = 0.1, cmap = cmapparu)
        ax.xaxis.tick_top()
        ax.set_xticks(np.arange(n_clus), minor = False)
        ax.set_xticklabels(patnames_short, size='small')
        ax.set_yticks(np.arange(n_clus), minor = False)
        ax.set_yticklabels(patnames_short, size='small')
        cb = plt.colorbar(gigi)
        cb.set_label('Transitions only'.format(lab, obs_name))
        mappe.append(gigi)

        fig.suptitle(model_name)
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        all_figures.append(fig)

    patt = results['cluspattern']
    lat = results['lat']
    lon = results['lon']
    filename = cart_out+'Allclus_{}.pdf'.format(model_name)
    figs = ctl.plot_multimap_contour(patt, lat, lon, filename, visualization = visualization, central_lat_lon = central_lat_lon, cmap = 'RdBu_r', title = 'Weather regimes in {}'.format(model_name), subtitles = patnames, cb_label = 'Geopotential height anomaly (m)', color_percentiles = (0.5,99.5), number_subplots = False, bounding_lat = bounding_lat, plot_margins = plot_margins, add_rectangles = draw_rectangle_area, draw_grid = draw_grid)
    all_figures += figs
    figs[0].savefig(filename)

    filename = cart_out + 'WRtool_{}_allfig.pdf'.format(tag)
    ctl.plot_pdfpages(filename, all_figures)

    return
