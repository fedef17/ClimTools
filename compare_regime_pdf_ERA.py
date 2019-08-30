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

from scipy import stats

#######################################

season = 'DJF'
area = 'EAT'
n_bootstrap = 1000
n_choice = 50
USE_RECLUSTERING = True

cart_out = '/home/fabiano/Research/lavori/WeatherRegimes/compare_regime_pdf/'
if not os.path.exists(cart_out): os.mkdir(cart_out)

# ctl.openlog('.')
file_in = '/data-hobbes/fabiano/OBS/ERA/ERA40+Int_daily_1957-2018_zg500_remap25_meters.nc'

print(ctl.datestamp())

kwar = dict()
kwar['numclus'] = 4
kwar['run_significance_calc'] = False
kwar['numpcs'] = 4
kwar['detrended_eof_calculation'] = False # detrendo io all'inizio
kwar['detrended_anom_for_clustering'] = False
kwar['nrsamp_sig'] = 500

# var, coords, aux_info = ctl.read_iris_nc(file_in, extract_level_hPa = 500)
# lat = coords['lat']
# lon = coords['lon']
# dates = coords['dates']
#
# var_anoms = ctl.anomalies_daily_detrended(var, dates)
# var_season, dates_season = ctl.sel_season(var_anoms, dates, season)
# all_years = np.arange(dates[0].year, dates[-1].year+1)
#
# results_ref = cd.WRtool_core(var_season, lat, lon, dates_season, area, heavy_output = True, **kwar)
# kwar['ref_solver'] = results_ref['solver']
# kwar['ref_patterns_area'] = results_ref['cluspattern_area']
# kwar['use_reference_eofs'] = True
#
# # Now selecting 2 periods
# results_all = dict()
# results_all['reference'] = results_ref
#
# varseas1, dates1 = ctl.sel_time_range(var_season, dates_season, ctl.range_years(1957,1987))
# results1 = cd.WRtool_core(varseas1, lat, lon, dates1, area, heavy_output = True, **kwar)
# results_all['period1_reclustering'] = results1
#
# varseas2, dates2 = ctl.sel_time_range(var_season, dates_season, ctl.range_years(1988,2018))
# results2 = cd.WRtool_core(varseas2, lat, lon, dates2, area, heavy_output = True, **kwar)
# results_all['period2_reclustering'] = results2
#
# kwar['use_reference_clusters'] = True
# kwar['ref_clusters_centers'] = results_ref['centroids']
#
# varseas1, dates1 = ctl.sel_time_range(var_season, dates_season, ctl.range_years(1957,1987))
# results1 = cd.WRtool_core(varseas1, lat, lon, dates1, area, heavy_output = True, **kwar)
# results_all['period1_assigntoclosest'] = results1
#
# varseas2, dates2 = ctl.sel_time_range(var_season, dates_season, ctl.range_years(1988,2018))
# results2 = cd.WRtool_core(varseas2, lat, lon, dates2, area, heavy_output = True, **kwar)
# results_all['period2_assigntoclosest'] = results2
#
# #pickle.dump([results_ref, results1, results2], open(cart_out+'out_test.p', 'wb'))
# # [results_ref, results1, results2] = pickle.load(open(cart_out+'out_test.p'))
# pickle.dump(results_all, open(cart_out+'out_test_all.p', 'wb'))
results_all = pickle.load(open(cart_out+'out_test_all.p'))
######################################################

if 'USE_RECLUSTERING':
    results1 = results_all['period1_reclustering']
    results2 = results_all['period2_reclustering']
else:
    results1 = results_all['period1_assigntoclosest']
    results2 = results_all['period2_assigntoclosest']
results_ref = results_all['reference']

sys.exit()

# Proietto sulla nuova base
cfrom = 0
cto = 1
# newbase, new_pcs = ctl.rotate_space_interclus_section_3D(results_ref['centroids'], cfrom, cto, results_ref['pcs'])
# newbase1, new_pcs1 = ctl.rotate_space_interclus_section_3D(results_ref['centroids'], cfrom, cto, results1['pcs'])
# newbase2, new_pcs2 = ctl.rotate_space_interclus_section_3D(results_ref['centroids'], cfrom, cto, results2['pcs'])

FILTER_4DAYS = False

if not FILTER_4DAYS:
    nulabs_ref = results_ref['labels']
    nulabs_1 = results1['labels']
    nulabs_2 = results2['labels']
    filter_tag = ''
else:
    nulabs_ref = ctl.regime_filter_long(results_ref['labels'], results_ref['dates'])
    nulabs_1 = ctl.regime_filter_long(results1['labels'], results1['dates'])
    nulabs_2 = ctl.regime_filter_long(results2['labels'], results2['dates'])
    filter_tag = '_4days'

new_pcs = results_ref['pcs']
new_pcs1 = results1['pcs']
new_pcs2 = results2['pcs']

ngp = 100
(x0, x1) = (np.percentile(new_pcs[:,0], 1), np.percentile(new_pcs[:,0], 99))
(y0, y1) = (np.percentile(new_pcs[:,1], 1), np.percentile(new_pcs[:,1], 99))
xss = np.linspace(x0,x1,ngp)
yss = np.linspace(y0,y1,ngp)
xi2, yi2 = np.meshgrid(xss, yss)

(z0, z1) = (np.percentile(new_pcs[:,2], 1), np.percentile(new_pcs[:,2], 99))
zss = np.linspace(z0,z1,ngp)
xi3, yi3, zi3 = np.meshgrid(xss, yss, zss)

xib, zib = np.meshgrid(xss, zss)
yic, zic = np.meshgrid(yss, zss)

cordss = [xss, yss, zss]
namz = ['x', 'y', 'z']
coups = [(0,1), (0,2), (1,2)]

figs = []
for reg in range(4):
    for nomecoso in ['reclustering', 'assigntoclosest']:
        results1 = results_all['period1_{}'.format(nomecoso)]
        results2 = results_all['period2_{}'.format(nomecoso)]
        new_pcs1 = results1['pcs']
        new_pcs2 = results2['pcs']
        nulabs_1 = results1['labels']
        nulabs_2 = results2['labels']

        fig = plt.figure(figsize=(24, 6), dpi=150)
        for nucou, cou in enumerate(coups):
            regime_pdf_2D = []
            regime_pdf_2D_func = []
            xi, yi = np.meshgrid(cordss[cou[0]], cordss[cou[1]])

            ax = fig.add_subplot(1, 3, nucou+1)

            okclu = nulabs_ref == reg
            okpc = new_pcs[okclu, :]
            kufu = ctl.calc_pdf(okpc[:,cou].T)
            zi = kufu(np.vstack([xi.flatten(), yi.flatten()]))
            ax.contour(xi, yi, zi.reshape(xi.shape), cmap = cm.get_cmap('Blues'), label = 'ref')

            okclu = nulabs_1 == reg
            okpc = new_pcs1[okclu, :]
            kufu = ctl.calc_pdf(okpc[:,cou].T)
            zi = kufu(np.vstack([xi.flatten(), yi.flatten()]))
            ax.contour(xi, yi, zi.reshape(xi.shape), cmap = cm.get_cmap('Greens'), label = 'pre')

            okclu = nulabs_2 == reg
            okpc = new_pcs2[okclu, :]
            kufu = ctl.calc_pdf(okpc[:,cou].T)
            zi = kufu(np.vstack([xi.flatten(), yi.flatten()]))
            ax.contour(xi, yi, zi.reshape(xi.shape), cmap = cm.get_cmap('Reds'), label = 'post')

            ax.set_xlabel('EOF {}'.format(cou[0]))
            ax.set_ylabel('EOF {}'.format(cou[1]))

        ax.legend()
        plt.suptitle('Regime {} - {}'.format(reg, nomecoso))
        figs.append(fig)
        fig.savefig(cart_out + 'regime{}_pdf_proj{}.pdf'.format(reg, filter_tag))

ctl.plot_pdfpages(cart_out + 'ERApre1988_vs_post{}.pdf'.format(filter_tag), figs)

sys.exit()

pcs_ref_1, dates1 = ctl.sel_time_range(results_ref['pcs'], results_ref['dates'], ctl.range_years(1957,1987))
pcs_ref_2, dates2 = ctl.sel_time_range(results_ref['pcs'], results_ref['dates'], ctl.range_years(1988,2018))
lab_ref_1, dates1 = ctl.sel_time_range(nulabs_ref, results_ref['dates'], ctl.range_years(1957,1987))
lab_ref_2, dates2 = ctl.sel_time_range(nulabs_ref, results_ref['dates'], ctl.range_years(1988,2018))

for reg in range(4):
    figs = []
    print(reg)
    extra1 = (lab_ref_1 != reg) & (nulabs_1 == reg)
    extra2 = (lab_ref_2 != reg) & (nulabs_2 == reg)
    print('Extra in series 1: {} of {}\n'.format(np.sum(extra1), np.sum(lab_ref_1 == reg)))
    print('Extra in series 2: {} of {}\n'.format(np.sum(extra2), np.sum(lab_ref_2 == reg)))

    outra1 = (lab_ref_1 == reg) & (nulabs_1 != reg)
    outra2 = (lab_ref_2 == reg) & (nulabs_2 != reg)
    print('Outra in series 1: {} of {}\n'.format(np.sum(outra1), np.sum(lab_ref_1 == reg)))
    print('Outra in series 2: {} of {}\n'.format(np.sum(outra2), np.sum(lab_ref_2 == reg)))

    fig = plt.figure(figsize=(24, 6), dpi=150)
    for nucou, cou in enumerate(coups):
        regime_pdf_2D = []
        regime_pdf_2D_func = []
        xi, yi = np.meshgrid(cordss[cou[0]], cordss[cou[1]])

        ax = fig.add_subplot(1, 3, nucou+1)

        okclu = lab_ref_1 == reg
        okpc = pcs_ref_1[okclu, :]
        kufu = ctl.calc_pdf(okpc[:,cou].T)
        zi = kufu(np.vstack([xi.flatten(), yi.flatten()]))
        ax.contour(xi, yi, zi.reshape(xi.shape), cmap = cm.get_cmap('Blues'), label = 'ref')

        okclu = extra1
        okpc = pcs_ref_1[okclu, :]
        kufu = ctl.calc_pdf(okpc[:,cou].T)
        zi = kufu(np.vstack([xi.flatten(), yi.flatten()]))
        ax.contour(xi, yi, zi.reshape(xi.shape), cmap = cm.get_cmap('Oranges'), label = 'extra')

        okclu = outra1
        okpc = pcs_ref_1[okclu, :]
        kufu = ctl.calc_pdf(okpc[:,cou].T)
        zi = kufu(np.vstack([xi.flatten(), yi.flatten()]))
        ax.contour(xi, yi, zi.reshape(xi.shape), cmap = cm.get_cmap('Greys'), label = 'outra')

        ax.set_xlabel('EOF {}'.format(cou[0]))
        ax.set_ylabel('EOF {}'.format(cou[1]))
        ax.legend()

    plt.suptitle('Period 1 - Regime {}'.format(reg))
    figs.append(fig)

    fig = plt.figure(figsize=(24, 6), dpi=150)
    for nucou, cou in enumerate(coups):
        regime_pdf_2D = []
        regime_pdf_2D_func = []
        xi, yi = np.meshgrid(cordss[cou[0]], cordss[cou[1]])

        ax = fig.add_subplot(1, 3, nucou+1)

        okclu = lab_ref_2 == reg
        okpc = pcs_ref_2[okclu, :]
        kufu = ctl.calc_pdf(okpc[:,cou].T)
        zi = kufu(np.vstack([xi.flatten(), yi.flatten()]))
        ax.contour(xi, yi, zi.reshape(xi.shape), cmap = cm.get_cmap('Blues'), label = 'ref')

        okclu = extra2
        okpc = pcs_ref_2[okclu, :]
        kufu = ctl.calc_pdf(okpc[:,cou].T)
        zi = kufu(np.vstack([xi.flatten(), yi.flatten()]))
        ax.contour(xi, yi, zi.reshape(xi.shape), cmap = cm.get_cmap('Oranges'), label = 'extra')

        okclu = outra2
        okpc = pcs_ref_2[okclu, :]
        kufu = ctl.calc_pdf(okpc[:,cou].T)
        zi = kufu(np.vstack([xi.flatten(), yi.flatten()]))
        ax.contour(xi, yi, zi.reshape(xi.shape), cmap = cm.get_cmap('Greys'), label = 'outra')

        ax.set_xlabel('EOF {}'.format(cou[0]))
        ax.set_ylabel('EOF {}'.format(cou[1]))
        ax.legend()

    plt.suptitle('Period 2 - Regime {}'.format(reg))
    figs.append(fig)

    ctl.plot_pdfpages(cart_out + 'regime{}_pdf_proj_diffclustering{}.pdf'.format(reg, filter_tag), figs)

filos = open(cart_out + 'results_kolmog_ERA.dat', 'w')

for test in ['KolmogSmir', 'MannWhit', 'Anderson']:
    filos.write('\n\n-------------- TEST: {} --------------\n\n'.format(test))
    for nomecoso in ['reclustering', 'assigntoclosest']:
        filos.write(nomecoso)
        results1 = results_all['period1_{}'.format(nomecoso)]
        results2 = results_all['period2_{}'.format(nomecoso)]
        new_pcs1 = results1['pcs']
        new_pcs2 = results2['pcs']
        nulabs_1 = results1['labels']
        nulabs_2 = results2['labels']

        for reg in range(4):
            filos.write('\n Regime {}\n'.format(reg))

            okclu = nulabs_1 == reg
            okpc1 = new_pcs1[okclu, :]
            okclu = nulabs_2 == reg
            okpc2 = new_pcs2[okclu, :]

            for eof in range(4):
                if test == 'KolmogSmir':
                    D, pval = stats.ks_2samp(okpc1[:, eof], okpc2[:, eof])
                    filos.write('eof {:3d}: D -> {:8.4f} , pval -> {:12.3e}\n'.format(eof, D, pval))
                elif test == 'MannWhit':
                    D, pval = stats.mannwhitneyu(okpc1[:, eof], okpc2[:, eof], alternative='two-sided')
                    filos.write('eof {:3d}: D -> {:8.4f} , pval -> {:12.3e}\n'.format(eof, D, pval))
                elif test == 'Anderson':
                    D, critvals, pval = stats.anderson_ksamp([okpc1[:, eof], okpc2[:, eof]])
                    filos.write('eof {:3d}: D -> {:8.4f} , pval -> {:12.3e}\n'.format(eof, D, pval))

filos.close()
sys.exit()
# ORA prendo results_ref e faccio anno per anno i miniclusters e faccio il plottino

var_set, dates_set = ctl.seasonal_set(nulabs_ref, results_ref['dates'], 'DJF')
pcs_set, dates_set = ctl.seasonal_set(results_ref['pcs'], results_ref['dates'], 'DJF')
years = [da[0].year for da in dates_set]

# for numy in [6, 10, 20, 30]:
#     for reg in range(4):
#         figs = []
#         #for va, pcs, ye in zip(var_set, pcs_set, years):
#         #for cyr in np.arange(len(years))[::10][:-1]:
#         for cyr in np.arange(numy/2, len(years)-numy/2-1):
#             va = np.concatenate(var_set[cyr-numy/2:cyr+numy/2], axis = 0)
#             pcs = np.concatenate(pcs_set[cyr-numy/2:cyr+numy/2], axis = 0)
#             ye = years[cyr]-numy/2
#             ye2 = years[cyr]+numy/2
#
#             fig = plt.figure(figsize=(24, 6), dpi=150)
#             for nucou, cou in enumerate(coups):
#                 regime_pdf_2D = []
#                 regime_pdf_2D_func = []
#                 xi, yi = np.meshgrid(cordss[cou[0]], cordss[cou[1]])
#
#                 ax = fig.add_subplot(1, 3, nucou+1)
#
#                 okclu = nulabs_ref == reg
#                 okpc = new_pcs[okclu, :]
#                 kufu = ctl.calc_pdf(okpc[:,cou].T)
#                 zi = kufu(np.vstack([xi.flatten(), yi.flatten()]))
#                 ax.contour(xi, yi, zi.reshape(xi.shape), cmap = cm.get_cmap('Blues'))
#
#                 okclu = va == reg
#                 okpc = pcs[okclu, :]
#                 print(ye, okpc.shape)
#                 kufu = ctl.calc_pdf(okpc[:,cou].T)
#                 zi = kufu(np.vstack([xi.flatten(), yi.flatten()]))
#                 ax.contour(xi, yi, zi.reshape(xi.shape), cmap = cm.get_cmap('Reds'))
#
#                 ax.set_xlabel('EOF {}'.format(cou[0]))
#                 ax.set_ylabel('EOF {}'.format(cou[1]))
#
#             plt.suptitle('Regime {} - year {}-{}'.format(reg, ye, ye2))
#             figs.append(fig)
#
#         ctl.plot_pdfpages(cart_out + 'ERA_var_{}yr_reg{}{}.pdf'.format(numy, reg, filter_tag), figs)
#         plt.close('all')

##########################################################################









################# DALLA HISTORY ###############################

# len(lab_ref_1)
# len(results1['labels'])
# plt.ion()
# plt.plot(lab_ref_1 == 0)
# plt.plot(results1['labels'] == 0)
# plt.show()
# plt.close('all')
# plt.plot(lab_ref_1 == 0)
# np.sum(results_ref['labels'] == 0)
# plt.plot(results1['labels'] == 0)
# np.sum(lab_ref_1 == 0)
# np.sum(lab_ref_1 == 0)
# np.sum(nulabs_1 == 0)
# np.sum(nulabs_1 != lab_ref_1)
# np.sum((nulabs_1 != lab_ref_1) & (lab_ref_1 == 0))
# np.sum((nulabs_1 != lab_ref_1) & (nulabs_1 == 0))
# np.sum((lab_ref_1 != reg) & (nulabs_1 == reg))
# reg
# reg = 0
# np.sum((lab_ref_1 != reg) & (nulabs_1 == reg))
# pl.figure()
# plt.figure()
# plt.scatter(lab_ref_2, nulabs_2)
# plt.scatter(lab_ref_2)
# plt.scatter(range(len(lab_ref_2)), lab_ref_2)
# plt.scatter(range(len(nulabs_2)), nulabs_2)
# plt.figure()
# coso2, dates2 = ctl.sel_time_range(results_ref['labels'], results_ref['dates'], ctl.range_years(1988,2018))
# plt.scatter(range(len(coso2)), coso2)
# gigio = results2['labels']
# plt.scatter(range(len(gigio)), gigio)
# plt.ylim(-1.2, None)
# plt.scatter(range(len(coso2)), coso2-0.05)
# plt.scatter(range(len(gigio)), gigio+0.05)
# plt.figure()
# plt.scatter(range(len(gigio)), gigio+0.05, s = 1)
# plt.scatter(range(len(coso2)), coso2-0.05, s = 1)
# plt.scatter(range(len(gigio)), gigio+0.05, s = 1)
# plt.scatter(range(len(coso2)), coso2-0.05, s = 1, label = 'reference')
# plt.scatter(range(len(gigio)), gigio+0.05, s = 1, label = 'reclustering')
# plt.legend()
# len(lab_ref_2)
# np.sum(lab_ref_2 == 0)
# np.sum(nulabs_2 == 0)
# ctl.calc_varopt_molt(results_ref['pcs'], results_ref['centroids'], results_ref['labels'])
# ctl.calc_varopt_molt(results1['pcs'], results1['centroids'], results1['labels'])
# ctl.calc_varopt_molt(results2['pcs'], results2['centroids'], results2['labels'])
# ctl.calc_varopt_molt(results2['pcs'], results2['centroids'], coso2)
# ctl.calc_varopt_molt(results2['pcs'], results_ref['centroids'], coso2)
# ctl.calc_varopt_molt(results2['pcs'], results_ref['centroids'], results2['labels'])
