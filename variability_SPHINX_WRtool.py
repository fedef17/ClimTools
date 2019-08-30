#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
from matplotlib import pyplot as plt
import pickle

import climtools_lib as ctl
import climdiags as cd

from matplotlib.colors import LogNorm

#######################################

season = 'DJF'
area = 'EAT'
n_bootstrap = 1000
n_choice = 50

cart_out = '/home/fabiano/Research/lavori/WeatherRegimes/variability_SPHINX/'
if not os.path.exists(cart_out): os.mkdir(cart_out)

# ctl.openlog('.')

file_in = '/data-hobbes/fabiano/OBS/ERA/ERA40+Int_daily_1957-2018_zg500_remap25_meters.nc'

cart_in = '/data-hobbes/fabiano/SPHINX/zg_daily/'
filenam = dict()
filenam['base'] = 'lcb{}-1850-2100-NDJFM_zg500_NH_14473.nc'
filenam['stoc'] = 'lcs{}-1850-2100-NDJFM_zg500_NH_14473.nc'

print(ctl.datestamp())

var, coords, aux_info = ctl.read_iris_nc(file_in, extract_level_hPa = 500)
lat = coords['lat']
var = var[:, lat >= 0, :]
lat = lat[lat >= 0]
lon = coords['lon']
dates = coords['dates']

var_anoms = ctl.anomalies_daily_detrended(var, dates)
var_season, dates_season = ctl.sel_season(var_anoms, dates, season)
all_years = np.arange(dates[0].year, dates[-1].year+1)

kwar = dict()
kwar['numclus'] = 4
kwar['run_significance_calc'] = False
kwar['numpcs'] = 4
kwar['detrended_eof_calculation'] = False
kwar['detrended_anom_for_clustering'] = False
kwar['nrsamp_sig'] = 500

results_ref = cd.WRtool_core(var_season, lat, lon, dates_season, area, heavy_output = True, **kwar)
kwar['ref_solver'] = results_ref['solver']
kwar['ref_patterns_area'] = results_ref['cluspattern_area']
kwar['use_reference_eofs'] = True

var_era_set, dates_era_set = ctl.seasonal_set(var_season, dates_season, season)
n_seas = len(var_era_set)
years_set = np.array([dat[0].year for dat in dates_era_set])

# LEGGO SPHINX E LANCIO IL BOOTSTRAP
var_mod = dict()
dates_mod = dict()

all_res = dict()
#for mod in ['base', 'stoc', 'era']:
#    print(mod)
#    if mod in ['base', 'stoc']:
#        var_mod = []
#        dates_mod = []
#        for i in range(3):
#            file_in = cart_in + filenam[mod].format(i)
#            var, coords, aux_info = ctl.read_iris_nc(file_in, extract_level_hPa = 500)
#            var_time, dates_time = ctl.sel_time_range(var, coords['dates'], (dates_season[0], dates_season[-1]))
#            var_anoms = ctl.anomalies_daily_detrended(var_time, dates_time)
#            var_seas_set, dates_seas_set = ctl.seasonal_set(var_anoms, dates_time, season)
#            var_mod.append(var_seas_set)
#            dates_mod.append(dates_seas_set)
#
#        var_mod = np.concatenate(var_mod)
#        dates_mod = np.concatenate(dates_mod)
#    elif mod == 'era':
#        var_mod = var_era_set
#        dates_mod = dates_era_set
#
#    all_res[mod] = []
#    n_seas = len(var_mod)
#    years_set = np.array([da[0].year for da in dates_mod])
#    for i in range(n_bootstrap):
#        ok_yea = np.sort(np.random.choice(range(n_seas), n_choice))
#        print(i,'\n', years_set[ok_yea])
#        var_ok = np.concatenate(var_mod[ok_yea])
#        dat_ok = np.concatenate(dates_mod[ok_yea])
#
#        results = cd.WRtool_core(var_ok, lat, lon, dat_ok, area, **kwar)
#        results['years_set'] = years_set[ok_yea]
#
#        all_res[mod].append(results)
#
#    pickle.dump(all_res[mod], open(cart_out + 'res_bootstrap_Sphinx_{}yr_{}_{}.p'.format(n_choice, n_bootstrap, mod), 'wb'))

for mod in ['base', 'stoc', 'era']:
    all_res[mod] = pickle.load(open(cart_out + 'res_bootstrap_Sphinx_{}yr_{}_{}.p'.format(n_choice, n_bootstrap, mod)))

ens_mean = dict()
ens_std = dict()

for mod in all_res:
    max_days = 29
    numclus = kwar['numclus']
    for i in range(len(all_res[mod])):
        all_res[mod][i]['histo_resid_times'] = []
        for j in range(numclus):
            numarr, histoco = ctl.count_occurrences(all_res[mod][i]['resid_times'][j], num_range = (0, max_days))
            all_res[mod][i]['histo_resid_times'].append(histoco)

    for key in all_res[mod][0]:
        try:
            mea = np.mean([koso[key] for koso in all_res[mod]], axis = 0)
            st = np.std([koso[key] for koso in all_res[mod]], axis = 0)
            ens_mean[(mod, key)] = mea
            ens_std[(mod, key)] = st
        except Exception as prro:
            print(key, repr(prro))
            pass

for mod in all_res:
    print(mod, all_res[mod][17].keys())

all_figures = []

# ellipse plot
nsqr = np.sqrt(results_ref['cluspattern_area'].size)
pattnames = ['NAO +', 'Sc. Blocking', 'Atl. Ridge', 'NAO -']

fig = plt.figure(figsize = (16,12))
for i in range(4):
    ax = fig.add_subplot(2, 2, i+1)
    ax.set_title(pattnames[i], fontsize = 18, fontweight = 'bold')

    ax.grid()
    colors = ctl.color_set(len(all_res.keys()))
    ctl.ellipse_plot([ens_mean[(mod, 'patcor')][i] for mod in all_res.keys()], [ens_mean[(mod, 'RMS')][i]/nsqr for mod in all_res.keys()], [ens_std[(mod, 'patcor')][i] for mod in all_res.keys()], [ens_std[(mod, 'RMS')][i]/nsqr for mod in all_res.keys()], alpha = 0.7, ax = ax, colors = colors)

    for mod, col in zip(all_res.keys(), colors):
        pats = [coso['patcor'][i] for coso in all_res[mod]]
        rmss = [coso['RMS'][i]/nsqr for coso in all_res[mod]]
        ax.scatter(pats, rmss, color = col, s = 5)

    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0., 30.)
    ax.tick_params(labelsize=14)
    plt.gca().invert_xaxis()
    ax.set_xlabel('Pattern correlation', fontsize = 18)
    ax.set_ylabel('RMS (m)', fontsize = 18)

plt.tight_layout()
plt.subplots_adjust(top = 0.9)
plt.suptitle('Performance of {}-yr subsets'.format(n_choice), fontsize = 28)
fig.savefig(cart_out + 'ellipse_plot_{}yr_{}.pdf'.format(n_choice, n_bootstrap))
all_figures.append(fig)


# plot resid times w std dev
for mod in all_res:
    axes = []
    fig = plt.figure()

    for j in range(kwar['numclus']):
        ax = fig.add_subplot(2, 2, j+1)
        ax.set_title(pattnames[j])

        histomea = np.mean([coso['histo_resid_times'][j] for coso in all_res[mod]], axis = 0)
        histostd = np.std([coso['histo_resid_times'][j] for coso in all_res[mod]], axis = 0)

        ax.bar(numarr, histomea, alpha = 0.5, label = 'mean', color = 'indianred')
        cosomea = ctl.running_mean(histomea[:-1], 3)
        ax.bar(numarr, histostd, alpha = 0.5, label = 'stddev', color = 'steelblue')
        cosostd = ctl.running_mean(histostd[:-1], 3)

        ax.plot(numarr[:-1], cosomea, color = 'indianred')
        ax.plot(numarr[:-1], cosostd, color = 'steelblue')
        ax.legend()
        ax.set_xlim(0, max_days+2)
        tics = np.arange(0,max_days+2,5)
        labs = ['{}'.format(ti) for ti in tics[:-1]]
        labs.append('>{}'.format(max_days))
        ax.set_xticks(tics, minor = False)
        ax.set_xticklabels(labs, size='small')
        ax.set_xlabel('Days')
        ax.set_ylabel('Frequency')
        ax.set_ylim(0, 0.2)
        axes.append(ax)

    plt.suptitle('Residence times - {} - {} yrs - {} runs'.format(mod, n_choice, n_bootstrap))
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    ctl.adjust_ax_scale(axes)
    fig.savefig(cart_out+'Regime_residtimes_{}_{}yr_{}.pdf'.format(mod, n_choice, n_bootstrap))
    all_figures.append(fig)

    # plot regime freq w std dev; anche hist delle regime freqs
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.grid(axis = 'y', zorder = 0)
    ax.set_axisbelow(True)
    wi = 0.8

    for j in range(kwar['numclus']):
        central = j*(kwar['numclus']*1.5)

        sig = np.mean([coso['freq_clus'][j] for coso in all_res[mod]])
        stddev = np.std([coso['freq_clus'][j] for coso in all_res[mod]])

        ax.bar(central, sig, width = wi, label = pattnames[j], zorder = 5)
        ax.errorbar(central, sig, yerr = stddev, color = 'black', capsize = 3, zorder = 6)

    ax.set_title('Regimes frequencies')
    ax.set_xticks([j*(kwar['numclus']*1.5) for j in range(kwar['numclus'])], minor = False)
    ax.set_xticklabels(pattnames, size='small')
    ax.set_ylabel('Frequency')
    ax.set_ylim(0, 35)
    fig.savefig(cart_out+'Regime_frequency_{}_{}yr_{}.pdf'.format(mod, n_choice, n_bootstrap))
    all_figures.append(fig)

    fig = plt.figure()
    axes = []
    for j in range(kwar['numclus']):
        ax = fig.add_subplot(2, 2, j+1)

        all_freqs = np.array([coso['freq_clus'][j] for coso in all_res[mod]])

        cof = ctl.calc_pdf(all_freqs)
        cofvals2 = np.array([cof(i) for i in np.linspace(10, 40, 1000)])
        ax.plot(np.linspace(10, 40, 1000), n_bootstrap*cofvals2)

        binzz = np.arange(10,40,1)
        ax.hist(all_freqs, bins = binzz)

        ax.set_xlim(10, 40)
        ax.set_ylim(0, 250)

        ax.set_xlabel('Frequency')
        ax.set_title(pattnames[j])
        axes.append(ax)

    ctl.adjust_ax_scale(axes)
    fig.tight_layout()
    fig.savefig(cart_out+'Regime_freqhisto_{}_{}yr_{}.pdf'.format(mod, n_choice, n_bootstrap))
    all_figures.append(fig)

    # plot trans_matrix e std dev.
    fig = plt.figure(figsize = (16, 6))
    all_trm = [coso['trans_matrix'] for coso in all_res[mod]]
    mean_trm = np.mean(all_trm, axis = 0)
    std_trm = np.std(all_trm, axis = 0)

    ax = fig.add_subplot(121)
    gigi = ax.imshow(mean_trm, norm = LogNorm(vmin = 0.01, vmax = 1.0))
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(numclus), minor = False)
    ax.set_xticklabels(pattnames, size='small')
    ax.set_yticks(np.arange(numclus), minor = False)
    ax.set_yticklabels(pattnames, size='small')
    cb = plt.colorbar(gigi, orientation = 'horizontal')

    ax = fig.add_subplot(122)
    gigi = ax.imshow(std_trm/mean_trm, vmin = 0., vmax = 0.5)
    #ax.set_title('Ratio std dev/ mean')
    cb = plt.colorbar(gigi, orientation = 'horizontal')

    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(numclus), minor = False)
    ax.set_xticklabels(pattnames, size='small')
    ax.set_yticks(np.arange(numclus), minor = False)
    ax.set_yticklabels(pattnames, size='small')

    fig.savefig(cart_out+'Transmatrix_{}_{}yr_{}.pdf'.format(mod, n_choice, n_bootstrap))
    all_figures.append(fig)

ctl.plot_pdfpages(cart_out+'SPHINX_variability_{}yr_{}.pdf'.format(n_choice, n_bootstrap), all_figures)
