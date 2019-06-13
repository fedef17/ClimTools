#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
from matplotlib import pyplot as plt
from matplotlib import cm

import pickle
from scipy import stats
from datetime import datetime

import climtools_lib as ctl
import climdiags as cd

from matplotlib.colors import LogNorm

from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
base = importr('base')
utils = importr('utils')

importr('Peacock.test')
importr('cramer')

peacock2 = robjects.r('peacock2')
peacock3 = robjects.r('peacock3')
cramertest = robjects.r('cramer.test')

#######################################
np.random.seed(27071988)

cart_out = '/home/fabiano/Research/lavori/WeatherRegimes/phasespace_ERA/'

cart_ERAvar = '/home/fabiano/Research/lavori/WeatherRegimes/variability_ERA/'
n_bootstrap = 1000

# all_KS_stat = dict()
# all_cent_dist = dict()
# all_cent_std = dict()
# all_refcent_KS_dist = dict()
#
# for n_choice in [5, 10, 15, 20, 25, 30, 40, 50, 60]:
#     print('numcho:', n_choice)
#     results_ref, all_res = pickle.load(open(cart_ERAvar + 'res_bootstrap_{}yr_{}.p'.format(n_choice, n_bootstrap)))
#
#     for reg in range(4):
#         all_KS_stat[(n_choice, reg)] = []
#         all_cent_dist[(n_choice, reg)] = []
#         all_cent_std[(n_choice, reg)] = []
#         all_refcent_KS_dist[(n_choice, reg)] = []
#
#     t0 = datetime.now()
#     for res in all_res:
#         new_pcs1 = res['pcs']
#         new_pcs2 = results_ref['pcs']
#         nulabs_1 = res['labels']
#         nulabs_2 = results_ref['labels']
#
#         for reg in range(4):
#             okclu = nulabs_1 == reg
#             okpc1 = new_pcs1[okclu, :]
#             okclu = nulabs_2 == reg
#             okpc2 = new_pcs2[okclu, :]
#
#             kss = np.mean([stats.ks_2samp(okpc1[:, eof], okpc2[:, eof]).statistic for eof in range(4)])
#             all_KS_stat[(n_choice, reg)].append(kss)
#
#             # voglio std_dev della distanza dal centroide suo
#             cen = res['centroids'][reg]
#             dist = np.mean([ctl.distance(po, cen) for po in okpc1])
#             stddist = np.std([ctl.distance(po, cen) for po in okpc1])
#             all_cent_dist[(n_choice, reg)].append(dist)
#             all_cent_std[(n_choice, reg)].append(stddist)
#
#             # voglio ks su distanza dal centroide ref
#             # non ha senso
#             cen = results_ref['centroids'][reg]
#             dist1 = [ctl.distance(po, cen) for po in okpc1]
#             dist2 = [ctl.distance(po, cen) for po in okpc2]
#             kss = stats.ks_2samp(dist1, dist2).statistic
#             all_refcent_KS_dist[(n_choice, reg)].append(kss)
#
#     t1 = datetime.now()
#     print('{} yr - {:6.2f}'.format(n_choice, (t1-t0).total_seconds()))
#
# pickle.dump([all_KS_stat, all_cent_dist, all_cent_std, all_refcent_KS_dist], open(cart_out + 'KS_stat_1000.p', 'w'))

# all_centtocent_dist = dict()
#
# for n_choice in [5, 10, 15, 20, 25, 30, 40, 50, 60]:
#     print('numcho:', n_choice)
#     results_ref, all_res = pickle.load(open(cart_ERAvar + 'res_bootstrap_{}yr_{}.p'.format(n_choice, n_bootstrap)))
#
#     for reg in range(4):
#         all_centtocent_dist[(n_choice, reg)] = []
#
#     t0 = datetime.now()
#     for res in all_res:
#         for reg in range(4):
#             dist = ctl.distance(res['centroids'][reg], results_ref['centroids'][reg])
#             all_centtocent_dist[(n_choice, reg)].append(dist)
#
#     t1 = datetime.now()
#     print('{} yr - {:6.2f}'.format(n_choice, (t1-t0).total_seconds()))
#
# pickle.dump(all_centtocent_dist, open(cart_out + 'centocen_stat_1000.p', 'w'))
all_centtocent_dist = pickle.load(open(cart_out + 'centocen_stat_1000.p'))

results_ref, _ = pickle.load(open(cart_ERAvar + 'res_bootstrap_5yr_1000.p'))
del _

ref_cent_dist = []
ref_cent_std = []

for reg in range(4):
    okclu = results_ref['labels'] == reg
    okpc1 = results_ref['pcs'][okclu, :]

    # voglio std_dev della distanza dal centroide suo
    cen = results_ref['centroids'][reg]
    ref_cent_dist.append(np.mean([ctl.distance(po, cen) for po in okpc1]))
    ref_cent_std.append(np.std([ctl.distance(po, cen) for po in okpc1]))


all_stats = pickle.load(open(cart_out + 'KS_stat_1000.p'))
all_stats += [all_centtocent_dist]

stat_nams = ['KS 1D', 'Clus radius', 'Clus std', 'KS dist refcentroid', 'dist centocen']

titles = ['1D Kolmogorov-Smirnov statistics on regime pdfs', 'Cluster radius', 'Cluster internal std dev', '1D KS on distance to reference centroid', 'Distance of centroid to reference centroid']

ERA_ref_thresholds = dict()

for stat, nam, tit in zip(all_stats, stat_nams, titles):
    fig = plt.figure(figsize = (16, 12))
    #yrcho = [5, 10, 15, 20, 25, 30, 40, 50, 60]
    yrcho = [5, 10, 20, 30, 40, 50, 60]
    colors = ctl.color_set(len(yrcho))
    ind = 0
    axes = []
    for reg in range(4):
        ind += 1
        ax = plt.subplot(2, 2, ind)
        for n_choice, col in zip(yrcho, colors):
            allvals = np.array(stat[(n_choice, reg)])
            for perc in [1, 5, 10, 100/3., 50, 200/3., 90, 95, 99]:
                ERA_ref_thresholds[(nam, n_choice, reg, perc)] = np.percentile(allvals, perc)

            kpdf = ctl.calc_pdf(allvals)

            #xvec = np.linspace(0., 0.4, 500)

            if 'KS' in nam:
                xvec = np.linspace(0., 0.5, 500)
                ax.axvline(0.06, color = 'grey', alpha = 0.6)
            elif nam == 'Clus radius':
                xvec = np.linspace(1000, 2200, 500)
                ax.axvline(ref_cent_dist[reg], color = 'grey', alpha = 0.6)
            elif nam == 'Clus std':
                xvec = np.linspace(300, 800, 500)
                ax.axvline(ref_cent_std[reg], color = 'grey', alpha = 0.6)
            elif nam == 'dist centocen':
                xvec = np.linspace(0, 2300, 500)
            else:
                xvec = np.linspace(0., 1.2*allvals.max(), 500)

            pdfxvec = kpdf(xvec)

            ax.plot(xvec, pdfxvec, color = col)

        ax.set_title('Reg {}'.format(reg))
        axes.append(ax)
    ctl.adjust_ax_scale(axes)

    fig.suptitle(tit)
    plt.subplots_adjust(top = 0.9)
    fig = ctl.custom_legend(fig, colors, ['{} yr'.format(yr) for yr in yrcho])
    fig.savefig(cart_out + '_'.join(nam.split())+'_ERA_allyears.pdf')

pickle.dump(ERA_ref_thresholds, open(cart_out + 'ERA_ref_thresholds_allstats.p', 'w'))
