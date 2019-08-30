#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
from matplotlib import pyplot as plt
from matplotlib import cm

import pickle

import climtools_lib as ctl
import climdiags as cd

from matplotlib.colors import LogNorm
from datetime import datetime

from scipy import stats

#######################################
cart_out = '/home/fabiano/Research/articoli/Papers/primavera_regimes/figures_v3_nodtr/'
if not os.path.exists(cart_out): os.mkdir(cart_out)

# cart = '/home/fabiano/Research/lavori/WeatherRegimes/prima_coup_v3/'
# filogen = cart + 'out_prima_coup_v3_DJF_EAT_4clus_4pcs_1957-2014_refEOF.p'
cart = '/home/fabiano/Research/lavori/WeatherRegimes/prima_coup_v4/'
filogen = cart + 'out_prima_coup_v4_DJF_EAT_4clus_4pcs_1957-2014_refEOF.p'

# model_names = ['CMCC-CM2-HR4', 'CMCC-CM2-VHR4', 'CNRM-CM6-1', 'CNRM-CM6-1-HR', 'EC-Earth-3-LR', 'EC-Earth-3-HR', 'ECMWF-IFS-LR', 'ECMWF-IFS-HR', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-XR', 'HadGEM3-GC31-MM', 'HadGEM3-GC31-HM', 'HadGEM3-GC31-LL-det', 'HadGEM3-GC31-LL-stoc', 'EC-Earth-3P-LR-det', 'EC-Earth-3P-LR-stoc']
# model_coups = ['CMCC-CM2', 'CNRM-CM6-1', 'EC-Earth-3', 'ECMWF-IFS', 'MPI-ESM1-2', 'HadGEM3-GC31', 'HadGEM3-GC31 (det vs stoc)', 'EC-Earth-3P (det vs stoc)']
model_names = ['CMCC-CM2-HR4', 'CMCC-CM2-VHR4', 'CNRM-CM6-1', 'CNRM-CM6-1-HR', 'EC-Earth3P', 'EC-Earth3P-HR', 'ECMWF-IFS-LR', 'ECMWF-IFS-HR', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-XR', 'HadGEM3-GC31-MM', 'HadGEM3-GC31-HM', 'HadGEM3-GC31-LL-det', 'HadGEM3-GC31-LL-stoc', 'EC-Earth3P-det', 'EC-Earth3P-stoc']
model_coups = ['CMCC-CM2', 'CNRM-CM6-1', 'EC-Earth3P', 'ECMWF-IFS', 'MPI-ESM1-2', 'HadGEM3-GC31', 'HadGEM3-GC31 (det vs stoc)', 'EC-Earth3P (det vs stoc)']
model_names_all = model_names + ['ERA']

colors = ctl.color_set(len(model_names), sns_palette = 'Paired')
colors_wERA = colors + ['black']

regnam = ['NAO +', 'Sc. BL', 'AR', 'NAO -']

################################################################################

results, results_ref = pickle.load(open(filogen, 'rb'))
results['ERA'] = results_ref

ref_cen = results['ERA']['centroids']

##
n_choice = 30
n_bootstrap = 100

filo = open(cart_out + 'res_bootstrap_v4.p', 'wb')

for mod in model_names_all:
    print(mod)
    #results[mod]['significance'] = significance[mod]
    results[mod]['varopt'] = ctl.calc_varopt_molt(results[mod]['pcs'], results[mod]['centroids'], results[mod]['labels'])

    results[mod]['autocorr_wlag'] = ctl.calc_autocorr_wlag(results[mod]['pcs'], results[mod]['dates'], out_lag1 = True)

    pcs_seas_set, dates_seas_set = ctl.seasonal_set(results[mod]['pcs'], results[mod]['dates'], 'DJF')
    labels_seas_set, dates_seas_set = ctl.seasonal_set(results[mod]['labels'], results[mod]['dates'], 'DJF')

    n_seas = len(dates_seas_set)
    years_set = np.array([dat[0].year for dat in dates_seas_set])
    bootstraps = dict()

    for nam in ['significance', 'varopt', 'autocorr', 'freq', 'dist_cen', 'resid_times_av', 'resid_times_90', 'trans_matrix', 'centroids']:
        bootstraps[nam] = []

    t0 = datetime.now()
    for i in range(n_bootstrap):
        #if i % 10 == 0:
        print(i)
        ok_yea = np.sort(np.random.choice(list(range(n_seas)), n_choice))
        pcs = np.concatenate(pcs_seas_set[ok_yea])
        labels = np.concatenate(labels_seas_set[ok_yea])
        dates = np.concatenate(dates_seas_set[ok_yea])

        centroids = []
        for iclu in range(4):
            okla = labels == iclu
            centroids.append(np.mean(pcs[okla], axis = 0))
        centroids = np.stack(centroids)

        sig = ctl.clusters_sig(pcs, centroids, labels, dates, nrsamp = 200)
        # if sig < 10:
        #     sig2 = ctl.clusters_sig(pcs, centroids, labels, dates, nrsamp = 1000)
        #     print('RECHECK ', sig, sig2, sig3)

        varopt = ctl.calc_varopt_molt(pcs, centroids, labels)
        autocorr = ctl.calc_autocorr_wlag(pcs, dates, out_lag1 = True)
        bootstraps['significance'].append(sig)
        bootstraps['varopt'].append(varopt)
        bootstraps['autocorr'].append(autocorr)

        bootstraps['freq'].append(ctl.calc_clus_freq(labels))

        centdist = np.array([ctl.distance(centroids[iclu], ref_cen[iclu]) for iclu in range(4)])
        bootstraps['dist_cen'].append(centdist)
        bootstraps['centroids'].append(centroids)

        resid_times = ctl.calc_regime_residtimes(labels, dates = dates)[0]
        av_res = np.array([np.mean(resid_times[reg]) for reg in range(4)])
        av_res_90 = np.array([np.percentile(resid_times[reg], 90) for reg in range(4)])
        bootstraps['resid_times_av'].append(av_res)
        bootstraps['resid_times_90'].append(av_res_90)

        bootstraps['trans_matrix'].append(ctl.calc_regime_transmatrix(1, labels, dates))

    #results[mod]['bootstraps'] = bootstraps
    t1 = datetime.now()
    print('Performed in {:10.1f} min\n'.format((t1-t0).total_seconds()/60.))
    pickle.dump(bootstraps, filo)

filo.close()
