#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os

from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as PathEffects

import netCDF4 as nc
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
        var, lat, lon, dates, time_units, var_units, time_cal = ctl.readxDncfield(ifile, extract_level = extract_level_4D)
        print(type(var))
        print(var.shape)

        var_season, dates_season = ctl.sel_season(var, dates, season)
    else:
        print('Concatenating {} input files..\n'.format(len(ifile)))
        var = []
        dates = []
        for fil in ifile:
            var, lat, lon, dates, time_units, var_units, time_cal = ctl.readxDncfield(fil, extract_level = extract_level_4D)

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
    for flun in fluxes_levels.keys():
        fluxes_cross[flun] = np.mean(fluxes_levels[flun], axis = -1)*zonal_factor
        fluxes_maps[flun] = np.zeros(fluxes_levels[flun].shape[1:])

    print('Starting vertical integration\n')
    for seas in press0.keys():
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
    for fu in fluxes_maps.keys():
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
            varuna, level, lat, lon, dates, time_units, var_units, time_cal = ctl.readxDncfield(file_list, select_var = varnames[0])
        else:
            varuna, level, lat, lon, dates, time_units, var_units, time_cal = ctl.readxDncfield(file_list[0])
            if varuna.keys()[0] not in varnames[0]:
                raise ValueError('{} is not v. Please give input files in order v,t,z,q.'.format(varuna.keys()[0]))

        # Changing to standard names
        for varna in varuna.keys():
            for varnamok in varnames:
                if varna in varnamok: vars[varnamok[0]] = varuna.pop(varna)

        print(varuna.keys(), vars.keys())

        # leggo t
        for i, (varnamok, flun) in enumerate(zip(varnames[1:], fluxnames)):
            varname = varnamok[0]
            print('Extracting {}\n'.format(varname))

            if type(file_list) is str:
                varuna, level, lat, lon, dates, time_units, var_units, time_cal = ctl.readxDncfield(file_list, select_var = varnamok)
            else:
                varuna, level, lat, lon, dates, time_units, var_units, time_cal = ctl.readxDncfield(file_list[i+1])
                if varuna.keys()[0] not in varnamok:
                    raise ValueError('{} is not t. Please give input files in order v,t,z,q.'.format(varuna.keys()[0]))

            vars[varname] = varuna.pop(varuna.keys()[0])
            del varuna

            if flun == 'PE' and var_units.values()[0] == 'm**2 s**-2':
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
            varuna, level, lat, lon, dates, time_units, var_units, time_cal = ctl.readxDncfield(file_list[i])
            varuna = varuna[varuna.keys()[0]]

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

    for fu in results.keys():
        filena = cart_out + '{}flux_map_{}_{}seasons.nc'.format(fu, tag, len(seasons))
        vals = [results[fu]['maps'][key] for key in seasons]
        vals.append(results[fu]['maps']['year'])
        ctl.save_N_2Dfields(lat,lon,np.stack(vals), fu, 'W/m', filena)

    # for fu in era_fluxes_zonal.keys():
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
        for flu in results.keys():
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
        cset = ctl.color_set(len(seasons)+1, bright_thres = 1)
        for seas, col in zip(seasons+['year'], cset):
            plt.plot(lat, results[flun]['zonal'][seas], label = seas, color = col, linewidth = 2.)
            #plt.plot(lat, era_fluxes_zonal_itrp[(seas, flun)], label = 'ERA '+seas, color = col, linewidth = 0.7, linestyle = '--')
        plt.legend()
        plt.grid()
        plt.ylim(margins[(flun, 'zonal')])
        plt.xlabel('Latitude')
        plt.ylabel('Integrated Net Heat Flux (W)')
        figures_exp.append(fig)

    pickle.dump(results, open(cart_out+'out_hfc_{}_.p'.format(tag), 'w'))

    print('Saving figures...\n')
    ctl.plot_pdfpages(figure_file_exp, figures_exp)
    ctl.plot_pdfpages(figure_file_exp_maps, figures_exp_maps)

    return results

#############################################################################
#############################################################################

##########              Plots and visualization                    ##########

#############################################################################
#############################################################################

def plot_WRtool_results(cart_out, tag, n_ens, result_models, result_obs, model_name = None, obs_name = None, patnames = None, patnames_short = None):#, groups = None, cluster_names = None):
    """
    Plot the results of WRtool.

    < n_ens > : int, number of ensemble members
    < result_models > : dict, output of WRtool, either for single or multiple member analysis
    < result_obs > : dict, output of WRtool for a single reference observation

    < model_name > : str, only needed for single member. For the multi-member the names are taken from results.keys().
    #< groups > : dict, only needed for multimember. Each entry contains a list of results.keys() belonging to that group. Group names are the group dict keys().

    """
    symbols = ['o', 'd', 'v', '*', 'P', 'h', 'X', 'p', '1']
    cart_out = cart_out + tag + '/'
    if not os.path.exists(cart_out): os.mkdir(cart_out)

    if model_name is None:
        model_name = 'model'
    if obs_name is None:
        obs_name = 'Obs'

    if n_ens == 1:
        resultooo = copy(result_models)
        result_models = dict()
        result_models[model_name] = resultooo

    all_figures = []

    # syms = []
    # labels = []
    # colors = []
    # if groups is not None:
    #     for grounam in groups.keys()
    #     labels = results.keys()
    #     colors = ctl.color_set(len(models)+1)
    colors = ctl.color_set(len(result_models)+1, only_darker_colors = True)
    labels = result_models.keys()

    if 'significance' in result_models.values()[0].keys():
        wi = 0.6
        fig = plt.figure()
        for i, (mod, col) in enumerate(zip(labels, colors)):
            plt.bar(i, result_models[mod]['significance'], width = wi, color = col, label = mod)
        plt.bar(i+1, result_obs['significance'], width = wi,  color = 'black', label = obs_name)
        plt.legend(fontsize = 'small', loc = 4)
        plt.title('Significance of regime structure')
        plt.xticks(range(len(labels+[obs_name])), labels+[obs_name], size='small')
        plt.ylabel('Significance')
        fig.savefig(cart_out+'Significance_{}.pdf'.format(tag))
        all_figures.append(fig)

    patt_ref = result_obs['cluspattern']
    lat = result_obs['lat']
    lon = result_obs['lon']

    if len(patt_ref) == 4:
        if patnames is None:
            patnames = ['NAO +', 'Blocking', 'NAO -', 'Atl. Ridge']
        if patnames_short is None:
            patnames_short = ['NP', 'BL', 'NN', 'AR']
    else:
        patnames = ['clus_{}'.format(i) for i in range(len(patt_ref))]
        patnames_short = ['c{}'.format(i) for i in range(len(patt_ref))]

    for lab in labels:
        patt = result_models[lab]['cluspattern']
        if np.any(np.isnan(patt)):
            print('There are {} NaNs in this patt.. replacing with zeros\n'.format(np.sum(np.isnan(patt))))
            patt[np.isnan(patt)] = 0.0
        cartout_mod = cart_out + 'mod_{}/'.format(lab)
        if not os.path.exists(cartout_mod): os.mkdir(cartout_mod)

        filename = cartout_mod+'Allclus_'+lab+'.pdf'
        figs = ctl.plot_multimap_contour(patt, lat, lon, filename, visualization = 'polar', central_lat_lon = (50.,0.), cmap = 'RdBu_r', title = 'North-Atlantic weather regimes - {}'.format(tag), subtitles = patnames, cb_label = 'Geopotential height anomaly (m)', color_percentiles = (0.5,99.5), fix_subplots_shape = (2,2), number_subplots = False)
        all_figures += figs
        for patuno, patuno_ref, pp, pps in zip(patt, patt_ref, patnames, patnames_short):
            nunam = cartout_mod+'clus_'+pps+'_'+lab+'.pdf'
            print(patuno.max(), patuno.min())
            print(lat.max(), lat.min())
            print(lon.max(), lon.min())
            #fig = ctl.plot_double_sidebyside(patuno, patuno_ref, lat, lon, filename = nunam, visualization = 'polar', central_lat_lon = (50., 0.), title = pp, cb_label = 'Geopotential height anomaly (m)', stitle_1 = tag, stitle_2 = 'ERA', color_percentiles = (0.5,99.5))
            fig = ctl.plot_triple_sidebyside(patuno, patuno_ref, lat, lon, filename = nunam, visualization = 'polar', central_lat_lon = (50., 0.), title = pp, cb_label = 'Geopotential height anomaly (m)', stitle_1 = tag, stitle_2 = 'ERA', color_percentiles = (0.5,99.5))
            all_figures.append(fig)

    # Taylor plots
    for num, patt in enumerate(patnames):
        obs = result_obs['cluspattern_area'][num, ...]
        modpats = [result_models[lab]['cluspattern_area'][num, ...] for lab in labels]

        colors = ctl.color_set(len(modpats), bright_thres = 0.3)

        filename = cart_out + 'TaylorPlot_{}.pdf'.format(patnames_short[num])
        label_ERMS_axis = 'Total RMS error (m)'
        label_bias_axis = 'Pattern mean (m)'

        figs = ctl.Taylor_plot(modpats, obs, filename, title = patt, label_bias_axis = label_bias_axis, label_ERMS_axis = label_ERMS_axis, colors = colors, markers = None, only_first_quarter = False, legend = True, marker_edge = None, labels = labels, obs_label = obs_name, mod_points_size = 50, obs_points_size = 70)
        all_figures += figs

    fig = plt.figure(figsize=(16,12))
    for num, patt in enumerate(patnames):
        ax = plt.subplot(2, 2, num+1, polar = True)

        obs = result_obs['cluspattern_area'][num, ...]
        modpats = [result_models[lab]['cluspattern_area'][num, ...] for lab in labels]

        colors = ctl.color_set(len(modpats), bright_thres = 0.3)

        legok = False
        ctl.Taylor_plot(modpats, obs, ax = ax, title = None, colors = colors, markers = None, only_first_quarter = True, legend = legok, labels = labels, obs_label = obs_name, mod_points_size = 50, obs_points_size = 70)


    #Custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = []
    for col, lab in zip(colors, labels):
        if lab is None:
            break
        legend_elements.append(Line2D([0], [0], marker='o', color=col, label=lab, linestyle = ''))

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

    filename = cart_out + 'WRtool_{}_allfig.pdf'.format(tag)
    ctl.plot_pdfpages(filename, all_figures)

    return
