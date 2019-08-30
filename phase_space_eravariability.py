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

all_stats_choices = dict()

# for n_choice in [5, 10, 15, 20, 25, 30, 40, 50, 60]:
#     print('numcho:', n_choice)
#     results_ref, all_res = pickle.load(open(cart_ERAvar + 'res_bootstrap_{}yr_{}.p'.format(n_choice, n_bootstrap)))
#
#     eof_axis_lim = (-3000., 3000.)
#     numtoplot = np.random.choice(range(1000), 100)
#
#     all_res_dict = dict()
#     names = []
#     for nu in numtoplot:
#         all_res_dict['sim_{}'.format(nu)] = all_res[nu]
#         names.append('sim_{}'.format(nu))
#
#     names.append('ref')
#     all_res_dict['ref'] = results_ref
#
#     #ctl.plot_multimodel_regime_pdfs(all_res_dict, model_names = model_names, filename = cart_out + 'all_regimes_ERAvar1000_y{}_refEOF.pdf'.format(n_choice), reference = refff, eof_axis_lim = eof_axis_lim, nolegend = True)
#
#     ctl.plot_multimodel_regime_pdfs(all_res_dict, model_names = names, filename = cart_out + 'all_regimes_ERAvar1000_y{}_refEOF_4eofs.pdf'.format(n_choice), eof_proj = [(0,1), (1,2), (2,3), (3,0)], figsize = (20,12), reference = 'ref', eof_axis_lim = eof_axis_lim, nolegend = True, check_for_eofs = False)
#
#     for reg in range(4):
#         for statna in ['KolmogSmir', 'MannWhit', 'Anderson', 'multi_peacock', 'multi_cramer']:
#             all_stats_choices[(n_choice, statna, reg)] = []
#
#     t0 = datetime.now()
#     for nam in names[:-1]:
#         new_pcs1 = all_res_dict[nam]['pcs']
#         new_pcs2 = results_ref['pcs']
#         nulabs_1 = all_res_dict[nam]['labels']
#         nulabs_2 = results_ref['labels']
#
#         for reg in range(4):
#             okclu = nulabs_1 == reg
#             okpc1 = new_pcs1[okclu, :]
#             okclu = nulabs_2 == reg
#             okpc2 = new_pcs2[okclu, :]
#
#             #for eof in range(4):
#             #D, pval = stats.ks_2samp(okpc1[:, eof], okpc2[:, eof])
#             res = [stats.ks_2samp(okpc1[:, eof], okpc2[:, eof]) for eof in range(4)]
#             all_stats_choices[(n_choice, 'KolmogSmir', reg)].append(res)
#
#             #D, pval = stats.mannwhitneyu(okpc1[:, eof], okpc2[:, eof], alternative='two-sided')
#             res = [stats.mannwhitneyu(okpc1[:, eof], okpc2[:, eof], alternative='two-sided') for eof in range(4)]
#             all_stats_choices[(n_choice, 'MannWhit', reg)].append(res)
#
#             res = [stats.anderson_ksamp([okpc1[:, eof], okpc2[:, eof]]) for eof in range(4)]
#             all_stats_choices[(n_choice, 'Anderson', reg)].append(res)
#
#
#     t3 = datetime.now()
#     print('single for 100 cases took: {:8.2f} s\n'.format((t3-t0).total_seconds()))
#
#     # prendo 5 casi rappresentativi e calcolo le due multi
#     allks = [np.mean([pu[0] for pu in cos]) for cos in all_stats_choices[(n_choice, 'KolmogSmir', reg)]]
#     ordks = np.argsort(allks)
#     numoks = ordks[::19]
#
#     for nu in numoks:
#         nam = names[nu]
#         new_pcs1 = all_res_dict[nam]['pcs']
#         new_pcs2 = results_ref['pcs']
#         nulabs_1 = all_res_dict[nam]['labels']
#         nulabs_2 = results_ref['labels']
#
#         for reg in range(4):
#             t1 = datetime.now()
#             okclu = nulabs_1 == reg
#             okpc1 = new_pcs1[okclu, :]
#             okclu = nulabs_2 == reg
#             okpc2 = new_pcs2[okclu, :]
#             # v = robjects.FloatVector(okpc1[:, :3].flatten())
#             # m = robjects.r['matrix'](v, nrow = len(okpc1), ncol = 3, byrow = True)
#             # v2 = robjects.FloatVector(okpc2[:, :3].flatten())
#             # m2 = robjects.r['matrix'](v2, nrow = len(okpc2), ncol = 3, byrow = True)
#             #
#             # D = peacock3(m, m2) ### più di 5 minuti, mai finito
#             v = robjects.FloatVector(okpc1[:, :2].flatten())
#             m = robjects.r['matrix'](v, nrow = len(okpc1), ncol = 2, byrow = True)
#             v2 = robjects.FloatVector(okpc2[:, :2].flatten())
#             m2 = robjects.r['matrix'](v2, nrow = len(okpc2), ncol = 2, byrow = True)
#
#             D = peacock2(m, m2) # circa 20 sec (con 2 dim)
#             t2 = datetime.now()
#             print((t2-t1).total_seconds())
#             # all_stats['multi_peacock'].append(D)
#             all_stats_choices[(n_choice, 'multi_peacock', reg)].append(D)
#
#             coso = cramertest(m, m2) # circa 100 sec (con 2 dim)
#             t2 = datetime.now()
#             print((t2-t1).total_seconds())
#             D, critval, pval = coso[4], coso[6], coso[7]
#             # all_stats['multi_cramer'].append((D, pval))
#             all_stats_choices[(n_choice, 'multi_cramer', reg)].append((D, pval))
#
#     t4 = datetime.now()
#     print('multi for {} cases took: {:8.2f} s\n'.format(len(numoks), (t4-t3).total_seconds()))
#
#
# pickle.dump(all_stats_choices, open(cart_out + 'all_stats_ERAvariab.p', 'wb'))

all_stats_choices = pickle.load(open(cart_out + 'all_stats_ERAvariab.p'))

all_ks_compare = []
all_peacock2_compare = []
all_cramer_compare = []
all_cramer_pvals = []

for n_choice in [5, 10, 15, 20, 25, 30, 40, 50, 60]:
    fig = plt.figure(figsize = (16, 12))

    colors = ctl.color_set(5)

    ind = 0
    for tai, (statnam, lims) in enumerate(zip(['KolmogSmir', 'MannWhit', 'Anderson'], [(0.,0.4), (200000, 900000), (0., 150)])):
        axes = []
        for reg in range(4):
            ind += 1
            ax = plt.subplot(5, 4, ind)
            allvals = np.array([np.mean([cos[0] for cos in all_stats_choices[(n_choice, statnam, reg)][i]]) for i in range(100)])
            kpdf = ctl.calc_pdf(allvals)

            if statnam == 'KolmogSmir':
                xvec = np.linspace(lims[0], lims[1], 500)
                ax.axvline(0.06, color = 'indianred', alpha = 0.5)
            else:
                xvec = np.linspace(0.8*allvals.min(), 1.2*allvals.max(), 100)
            ax.plot(xvec, kpdf(xvec), color = colors[tai])

            axes.append(ax)
        ctl.adjust_ax_scale(axes)

    axes = []
    for reg in range(4):
        ind += 1

        allks = np.array([np.mean([pu[0] for pu in cos]) for cos in all_stats_choices[(n_choice, 'KolmogSmir', reg)]])
        allks_2D = np.array([np.mean([cos[0][0], cos[1][0]]) for cos in all_stats_choices[(n_choice, 'KolmogSmir', reg)]])
        allks_max = np.array([np.max([pu[0] for pu in cos]) for cos in all_stats_choices[(n_choice, 'KolmogSmir', reg)]])
        ordks = np.argsort(allks)
        numoks = ordks[::19]
        allks_ok = allks[numoks]
        allks_ok_max = allks_max[numoks]

        ax = plt.subplot(5, 4, ind)
        allvals = np.array([all_stats_choices[(n_choice, 'multi_peacock',reg)][i][0] for i in range(6)])
        ax.scatter(allks_2D[numoks], allvals, marker = 'o', color = colors[3])

        all_ks_compare.append(allks_2D[numoks])
        all_peacock2_compare.append(allvals)

        ax.set_xlabel('1D KolmogSmir')
        ax.set_ylabel('2D peacock')
        #ax.scatter(allks_ok_max, allvals, marker = 'o', color = colors[4])
        axes.append(ax)
    ctl.adjust_ax_scale(axes)

    axes = []
    for reg in range(4):
        ind += 1

        allks = np.array([np.mean([pu[0] for pu in cos]) for cos in all_stats_choices[(n_choice, 'KolmogSmir', reg)]])
        allks_max = np.array([np.max([pu[0] for pu in cos]) for cos in all_stats_choices[(n_choice, 'KolmogSmir', reg)]])
        ordks = np.argsort(allks)
        numoks = ordks[::19]
        allks_ok = allks[numoks]
        allks_ok_max = allks_max[numoks]

        ax = plt.subplot(5, 4, ind)
        allvals = np.array([all_stats_choices[(n_choice, 'multi_cramer',reg)][i][0][0] for i in range(6)])
        ax.scatter(allks_ok, allvals, marker = 'D', color = colors[4])
        all_cramer_compare.append(allvals)
        allvalspval = np.array([all_stats_choices[(n_choice, 'multi_cramer',reg)][i][1][0] for i in range(6)])
        all_cramer_pvals.append(allvalspval)
        #ax.scatter(allks_ok_max, allvals, marker = 'D', color = colors[4])
        ax.set_xlabel('1D KolmogSmir')
        ax.set_ylabel('2D cramer')
        axes.append(ax)
    ctl.adjust_ax_scale(axes)

    fig.tight_layout()

    fig = ctl.custom_legend(fig, colors, labels = ['KolmogSmir', 'MannWhit', 'Anderson', 'peacock2', 'cramer'])
    fig.savefig(cart_out + 'all_stats_ERA_pdfs_{}yr.pdf'.format(n_choice))


# Only KolmogSmir 1D
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
        allvals = np.array([np.mean([cos[0] for cos in all_stats_choices[(n_choice, 'KolmogSmir', reg)][i]]) for i in range(100)])
        kpdf = ctl.calc_pdf(allvals)

        xvec = np.linspace(0., 0.4, 500)
        ax.plot(xvec, kpdf(xvec), color = col)

    ax.axvline(0.06, color = 'grey', alpha = 0.6)
    ax.set_title('Reg {}'.format(reg))
    axes.append(ax)
ctl.adjust_ax_scale(axes)


fig.suptitle('1D Kolmogorov-Smirnov statistics on regime pdfs')
plt.subplots_adjust(top = 0.9)
fig = ctl.custom_legend(fig, colors, ['{} yr'.format(yr) for yr in yrcho])
fig.savefig(cart_out + 'KS_stat1D_ERA_allyears.pdf')

# All KolmogSmir vs peacock2
all_ks_compare = np.concatenate(all_ks_compare)
all_peacock2_compare = np.concatenate(all_peacock2_compare)
all_cramer_compare = np.concatenate(all_cramer_compare)
all_cramer_pvals = np.concatenate(all_cramer_pvals)

ctl.plotcorr(all_ks_compare, all_peacock2_compare, cart_out + 'KS_vs_peacock2.pdf', xlabel = 'KolmogSmir 1D', ylabel = 'peacock2')

plt.close('all')
plt.ion()

ctl.plotcorr(all_ks_compare, all_cramer_compare, cart_out + 'KS_vs_cramer.pdf', xlabel = 'KolmogSmir 1D', ylabel = 'cramer')

fig = plt.figure()
plt.scatter(all_cramer_compare, all_cramer_pvals)
plt.xlabel('cramer statistic')
plt.ylabel('p value')
fig.savefig(cart_out + 'cramer_statistics.pdf')
