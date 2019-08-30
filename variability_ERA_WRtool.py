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
n_choice = 25

numblo = 50

# idee:
# Progettino. Analizzare WR in ERA (40+Interim), studiando:
# - variabilità dei patterns: bootstrap sugli anni e vedo come cambiano i centroidi nello spazio delle PC e nel Taylor plot rispetto al riferimento.
# - variabilità della significance (luuuungo). Idem con bootstrap.
# Per questi primi due devo ripetere l'analisi ogni volta. Ci mette un po'.
#
# - variabilità dei resid_times: più facile, prendo serie totale dei labels e ne taglio via dei pezzetti per vedere come cambiano i resid times.
# - variabilità delle frequenze e delle prob. di transizione: come sopra, prendo la serie completa e ne taglio via dei pezzetti stagionali per vedere quanto cambiano.
#
# Bootstrap: quante combinazioni prendo? Ho circa 60 anni. Direi di partire con 100 combinazioni (serie di 60 con ripetizione).
#
# Poi. Devo riprendere la cosa dei cluster visualizzati nello spazio delle PCs. E guardare:
# - Distribuzione dei centroidi.
# - Distribuzione dei resid_times nella pdf del cluster. (Quale spazio? Better PCs? Rotazione che massimizza variabilità intra-cluster?)
# - Distribuzione dei punti in transizione rispetto alla pdf del cluster.

cart_out = '/home/fabiano/Research/lavori/WeatherRegimes/variability_ERA_v2/'
if not os.path.exists(cart_out): os.mkdir(cart_out)

file_in = '/data-hobbes/fabiano/OBS/ERA/ERA40+Int_daily_1957-2018_zg500_remap25_meters.nc'

print(ctl.datestamp())

# lat = np.arange(-90, 91, 2.5)
# lon = np.arange(0., 360, 2.5)
var, coords, aux_info = ctl.read_iris_nc(file_in, extract_level_hPa = 500)
lat = coords['lat']
lon = coords['lon']
dates = coords['dates']

var_season, dates_season = ctl.sel_season(var, dates, season)
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
pickle.dump(results_ref, open(cart_out + 'res_bootstrap_v2_ref.p', 'wb'))

#results_ref = pickle.load(open(cart_out + 'res_bootstrap_v2_ref.p'))

var_seas_set, dates_seas_set = ctl.seasonal_set(var_season, dates_season, season)
n_seas = len(var_seas_set)
years_set = np.array([dat[0].year for dat in dates_seas_set])

kwar['run_significance_calc'] = False

#for n_choice in [5, 10, 20, 30, 40, 50, 60]:
for n_choice in [5, 10, 15, 20, 25, 30, 40, 50, 60]:
    print('######################   num years: {}   ##########################\n'.format(n_choice))
    # if n_choice == 60:
    #     kwar['run_significance_calc'] = False
    filpic = open(cart_out + 'res_bootstrap_v2_{}yr_{}.p'.format(n_choice, n_bootstrap), 'wb')
    for iblo in range(numblo):
        all_res = []
        for i in range(n_bootstrap/numblo):
            ok_yea = np.sort(np.random.choice(list(range(n_seas)), n_choice))
            print(i, years_set[ok_yea])
            var_ok = np.concatenate(var_seas_set[ok_yea])
            dat_ok = np.concatenate(dates_seas_set[ok_yea])

            results = cd.WRtool_core(var_ok, lat, lon, dat_ok, area, **kwar)
            results['years_set'] = years_set[ok_yea]

            all_res.append(results)

        print('Saving....\n')
        pickle.dump(all_res, filpic)

        del all_res

    filpic.close()

    # filpic = open(cart_out + 'res_bootstrap_v2_{}yr_{}.p'.format(n_choice, n_bootstrap), 'rb')
    # iblo = 0
    # all_res = []
    # for iblo in numblo:
    #     print(iblo)
    #     try:
    #         coso = pickle.load(filpic)
    #         all_res += coso
    #     except EOFError:
    #         break
    # filpic.close()

    #all_res = pickle.load(open(cart_out + 'res_bootstrap_v2_{}yr_{}.p'.format(n_choice, n_bootstrap)))
    # print('Making figures....\n')
    #
    # max_days = 29
    # numclus = kwar['numclus']
    # for i in range(len(all_res)):
    #     all_res[i]['histo_resid_times'] = []
    #     for j in range(numclus):
    #         numarr, histoco = ctl.count_occurrences(all_res[i]['resid_times'][j], num_range = (0, max_days))
    #         all_res[i]['histo_resid_times'].append(histoco)
    #
    # ens_mean = dict()
    # ens_std = dict()
    # for key in all_res[0]:
    #     try:
    #         mea = np.mean([koso[key] for koso in all_res], axis = 0)
    #         st = np.std([koso[key] for koso in all_res], axis = 0)
    #         ens_mean[key] = mea
    #         ens_std[key] = st
    #     except Exception as prro:
    #         print(key, repr(prro))
    #         pass
    #
    # all_figures = []
    #
    # # significance
    # if 'significance' in all_res[0].keys():
    #     all_sigs = np.array([koso['significance'] for koso in all_res])
    #     fig = plt.figure()
    #     plt.hist(all_sigs, bins = np.arange(60,101,2))
    #
    #     cof = ctl.calc_pdf(all_sigs)
    #     cofvals2 = np.array([cof(i) for i in np.linspace(60, 100, 1000)])
    #     plt.plot(np.linspace(60, 100, 1000), n_bootstrap*cofvals2)
    #
    #     plt.xlabel('Significance')
    #     plt.ylabel('Counts ({} runs)'.format(n_bootstrap))
    #     plt.title('Variability of significance in {}-years sub-ensembles of ERA'.format(n_choice))
    #     fig.savefig(cart_out + 'Significance_histo_{}yr_{}.pdf'.format(n_choice, n_bootstrap))
    #     all_figures.append(fig)
    #
    # # ellipse plot
    # nsqr = np.sqrt(results_ref['cluspattern_area'].size)
    # pattnames = ['NAO +', 'Sc. Blocking', 'Atl. Ridge', 'NAO -']
    #
    # fig = plt.figure(figsize = (16,12))
    # axes = []
    # for i in range(4):
    #     ax = fig.add_subplot(2, 2, i+1)
    #     ax.set_title(pattnames[i], fontsize = 18, fontweight = 'bold')
    #
    #     # x = [patcor_LR[i], patcor_HR[i]]
    #     # y = [rms_LR[i], rms_HR[i]]
    #     # errx = [err_patcor_LR[i], err_patcor_HR[i]]
    #     # erry = [err_rms_LR[i], err_rms_HR[i]]
    #     ax.grid()
    #     ctl.ellipse_plot([ens_mean['patcor'][i]], [ens_mean['RMS'][i]/nsqr], [ens_std['patcor'][i]], [ens_std['RMS'][i]/nsqr], alpha = 0.7, ax = ax)
    #
    #     pats = [coso['patcor'][i] for coso in all_res]
    #     rmss = [coso['RMS'][i]/nsqr for coso in all_res]
    #     ax.scatter(pats, rmss, s = 5)
    #
    #     #ax.set_xlim(min(pats), 1.0)
    #     #ax.set_ylim(0., max(rmss))
    #     ax.set_xlim(0.4, 1.0)
    #     ax.set_ylim(0., 30.)
    #     ax.tick_params(labelsize=14)
    #     plt.gca().invert_xaxis()
    #     ax.set_xlabel('Pattern correlation', fontsize = 18)
    #     ax.set_ylabel('RMS (m)', fontsize = 18)
    #     axes.append(ax)
    #
    # ctl.adjust_ax_scale(axes)
    #
    # plt.tight_layout()
    # plt.subplots_adjust(top = 0.9)
    # plt.suptitle('Performance of {}-yr subset of ERA40+Interim'.format(n_choice), fontsize = 28)
    # fig.savefig(cart_out + 'ellipse_plot_{}yr_{}.pdf'.format(n_choice, n_bootstrap))
    # all_figures.append(fig)
    #
    # # plot correlation of significance vs total RMS
    # pats = [np.mean(coso['patcor']) for coso in all_res]
    # rmss = [np.mean(coso['RMS'])/nsqr for coso in all_res]
    #
    # fig = ctl.plotcorr(rmss, all_sigs, cart_out + 'corr_rms_sig_{}yr_{}.pdf'.format(n_choice, n_bootstrap), xlabel = 'RMS (m)', ylabel = 'Sharpness')
    # #all_figures.append(fig)
    # fig = ctl.plotcorr(pats, all_sigs, cart_out + 'corr_patc_sig_{}yr_{}.pdf'.format(n_choice, n_bootstrap), xlabel = 'Pattern correlation', ylabel = 'Sharpness')
    # #all_figures.append(fig)
    #
    # # plot mean pattern e std dev patterns
    # all_patts = [coso['cluspattern'] for coso in all_res]
    # meanpatt = np.mean(all_patts, axis = 0)
    # stdpatt = np.std(all_patts, axis = 0)
    #
    # for nu in range(kwar['numclus']):
    #     fig = ctl.plot_double_sidebyside(meanpatt[nu], stdpatt[nu], lat, lon, filename = cart_out + 'cluspatt_{}_{}yr_{}.pdf'.format(nu, n_choice, n_bootstrap), visualization = 'Nstereo', central_lat_lon = (90, 0), title = 'Mean and std dev of: {}'.format(pattnames[nu]), cb_label = 'Geopotential height anomaly (m)', color_percentiles = (0.5,99.5), draw_contour_lines = True, stitle_1 = 'Ens. mean', stitle_2 = 'Ens. std dev')
    #     all_figures.append(fig)
    #
    # # plot resid times w std dev
    # axes = []
    # fig = plt.figure()
    #
    # for j in range(kwar['numclus']):
    #     ax = fig.add_subplot(2, 2, j+1)
    #     ax.set_title(pattnames[j])
    #
    #     histomea = np.mean([coso['histo_resid_times'][j] for coso in all_res], axis = 0)
    #     histostd = np.std([coso['histo_resid_times'][j] for coso in all_res], axis = 0)
    #
    #     ax.bar(numarr, histomea, alpha = 0.5, label = 'mean', color = 'indianred')
    #     cosomea = ctl.running_mean(histomea[:-1], 3)
    #     ax.bar(numarr, histostd, alpha = 0.5, label = 'stddev', color = 'steelblue')
    #     cosostd = ctl.running_mean(histostd[:-1], 3)
    #
    #     ax.plot(numarr[:-1], cosomea, color = 'indianred')
    #     ax.plot(numarr[:-1], cosostd, color = 'steelblue')
    #     ax.legend()
    #     ax.set_xlim(0, max_days+2)
    #     tics = np.arange(0,max_days+2,5)
    #     labs = ['{}'.format(ti) for ti in tics[:-1]]
    #     labs.append('>{}'.format(max_days))
    #     ax.set_xticks(tics, minor = False)
    #     ax.set_xticklabels(labs, size='small')
    #     ax.set_xlabel('Days')
    #     ax.set_ylabel('Frequency')
    #     ax.set_ylim(0, 0.2)
    #     axes.append(ax)
    #
    # plt.suptitle('Residence times - {} yrs - {} runs'.format(n_choice, n_bootstrap))
    # fig.tight_layout()
    # fig.subplots_adjust(top=0.88)
    # ctl.adjust_ax_scale(axes)
    # fig.savefig(cart_out+'Regime_residtimes_{}yr_{}.pdf'.format(n_choice, n_bootstrap))
    # all_figures.append(fig)
    #
    # # plot regime freq w std dev; anche hist delle regime freqs
    # fig = plt.figure()
    # ax = plt.subplot(111)
    # ax.grid(axis = 'y', zorder = 0)
    # ax.set_axisbelow(True)
    # wi = 0.8
    #
    # for j in range(kwar['numclus']):
    #     central = j*(kwar['numclus']*1.5)
    #
    #     sig = np.mean([coso['freq_clus'][j] for coso in all_res])
    #     stddev = np.std([coso['freq_clus'][j] for coso in all_res])
    #
    #     ax.bar(central, sig, width = wi, label = pattnames[j], zorder = 5)
    #     ax.errorbar(central, sig, yerr = stddev, color = 'black', capsize = 3, zorder = 6)
    #
    # ax.set_title('Regimes frequencies')
    # ax.set_xticks([j*(kwar['numclus']*1.5) for j in range(kwar['numclus'])], minor = False)
    # ax.set_xticklabels(pattnames, size='small')
    # ax.set_ylabel('Frequency')
    # ax.set_ylim(0, 35)
    # fig.savefig(cart_out+'Regime_frequency_{}yr_{}.pdf'.format(n_choice, n_bootstrap))
    # all_figures.append(fig)
    #
    # fig = plt.figure()
    # axes = []
    # for j in range(kwar['numclus']):
    #     ax = fig.add_subplot(2, 2, j+1)
    #
    #     all_freqs = np.array([coso['freq_clus'][j] for coso in all_res])
    #
    #     cof = ctl.calc_pdf(all_freqs)
    #     cofvals2 = np.array([cof(i) for i in np.linspace(10, 40, 1000)])
    #     ax.plot(np.linspace(10, 40, 1000), n_bootstrap*cofvals2)
    #
    #     binzz = np.arange(10,40,1)
    #     ax.hist(all_freqs, bins = binzz)
    #
    #     ax.set_xlim(10, 40)
    #     ax.set_ylim(0, 250)
    #
    #     ax.set_xlabel('Frequency')
    #     ax.set_title(pattnames[j])
    #     axes.append(ax)
    #
    # ctl.adjust_ax_scale(axes)
    # fig.tight_layout()
    # fig.savefig(cart_out+'Regime_freqhisto_{}yr_{}.pdf'.format(n_choice, n_bootstrap))
    # all_figures.append(fig)
    #
    # # plot trans_matrix e std dev.
    # fig = plt.figure(figsize = (16, 6))
    # all_trm = [coso['trans_matrix'] for coso in all_res]
    # mean_trm = np.mean(all_trm, axis = 0)
    # std_trm = np.std(all_trm, axis = 0)
    #
    # ax = fig.add_subplot(121)
    # gigi = ax.imshow(mean_trm, norm = LogNorm(vmin = 0.01, vmax = 1.0))
    # ax.xaxis.tick_top()
    # ax.set_xticks(np.arange(numclus), minor = False)
    # ax.set_xticklabels(pattnames, size='small')
    # ax.set_yticks(np.arange(numclus), minor = False)
    # ax.set_yticklabels(pattnames, size='small')
    # cb = plt.colorbar(gigi, orientation = 'horizontal')
    #
    # ax = fig.add_subplot(122)
    # gigi = ax.imshow(std_trm/mean_trm, vmin = 0., vmax = 0.5)
    # #ax.set_title('Ratio std dev/ mean')
    # cb = plt.colorbar(gigi, orientation = 'horizontal')
    #
    # ax.xaxis.tick_top()
    # ax.set_xticks(np.arange(numclus), minor = False)
    # ax.set_xticklabels(pattnames, size='small')
    # ax.set_yticks(np.arange(numclus), minor = False)
    # ax.set_yticklabels(pattnames, size='small')
    #
    # fig.savefig(cart_out+'Transmatrix_{}yr_{}.pdf'.format(n_choice, n_bootstrap))
    # all_figures.append(fig)
    #
    # ctl.plot_pdfpages(cart_out+'ERA_variability_{}yr_{}.pdf'.format(n_choice, n_bootstrap), all_figures)
