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
#cart_out = '/home/fabiano/Research/lavori/WeatherRegimes/phasespace_ERA/'
cart_out = '/home/fabiano/Research/articoli/Papers/primavera_regimes/figures/figures_phasespace_v4/'
if not os.path.exists(cart_out): os.mkdir(cart_out)

cart_out_ERA = '/home/fabiano/Research/lavori/WeatherRegimes/phasespace_ERA/'

# cart = '/home/fabiano/Research/lavori/WeatherRegimes/primavera_coupled_1957-2014/'
# filo = cart + 'out_primavera_coupled_1957-2014_DJF_EAT_4clus_4pcs_1957-2014_dtr.p'

tag = 'v4'
cart = '/home/fabiano/Research/lavori/WeatherRegimes/prima_coup_v4/'
filo = cart + 'out_prima_coup_v4_DJF_EAT_4clus_4pcs_1957-2014_refEOF.p'

model_names = ['CMCC-CM2-HR4', 'CMCC-CM2-VHR4', 'CNRM-CM6-1', 'CNRM-CM6-1-HR', 'EC-Earth3P', 'EC-Earth3P-HR', 'ECMWF-IFS-LR', 'ECMWF-IFS-HR', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-XR', 'HadGEM3-GC31-MM', 'HadGEM3-GC31-HM', 'HadGEM3-GC31-LL-det', 'HadGEM3-GC31-LL-stoc', 'EC-Earth3P-det', 'EC-Earth3P-stoc']
model_coups = ['CMCC-CM2', 'CNRM-CM6-1', 'EC-Earth3P', 'ECMWF-IFS', 'MPI-ESM1-2', 'HadGEM3-GC31', 'HadGEM3-GC31 (det vs stoc)', 'EC-Earth3P (det vs stoc)']
model_names_all = model_names + ['ERA']

model_coups = ['CMCC-CM2', 'CNRM-CM6-1', 'EC-Earth3P', 'ECMWF-IFS', 'MPI-ESM1-2', 'HadGEM3-GC31', 'HadGEM3-GC31 (det vs stoc)', 'EC-Earth3P (det vs stoc)']

colors = ctl.color_set(len(model_names), sns_palette = 'Paired')
colors_wERA = colors + ['black']

# # Ora tutte le statistiche
resolution = np.array([100, 25, 250, 50, 100, 50, 50, 25, 100, 50, 100, 50, 250, 250, 250, 100, 100])
sym_all = ['o', 'P','o', 'P','o', 'P','o', 'P','o', 'P','o', 'P', 's', '*', 's', '*']
wi = 0.6

# # Ora tutte le statistiche
stat_nams = ['KS 1D', 'Clus radius', 'Clus std', 'KS dist refcentroid', 'dist centocen']
titles = ['1D Kolmogorov-Smirnov statistics on regime pdfs', 'Cluster radius', 'Cluster internal std dev', '1D KS on distance to reference centroid', 'Distance of centroid to reference centroid']

allfigs = []

eof_axis_lim = (-3000., 3000.)

check_for_eofs = False
results, results_ref = pickle.load(open(filo, 'r'))
results['ERA'] = results_ref

fig = ctl.plot_multimodel_regime_pdfs(results, model_names = model_names_all, filename = cart_out + 'all_regimes_primacoup_{}_2eofs.pdf'.format(tag), eof_proj = [(0,1)], reference = 'ERA', eof_axis_lim = eof_axis_lim, check_for_eofs = check_for_eofs, colors = colors_wERA, fix_subplots_shape = (2, 2))
allfigs.append(fig)

fig = ctl.plot_multimodel_regime_pdfs(results, model_names = model_names_all, filename = cart_out + 'all_regimes_primacoup_{}_3eofs.pdf'.format(tag), reference = 'ERA', eof_axis_lim = eof_axis_lim, check_for_eofs = check_for_eofs, colors = colors_wERA)
allfigs.append(fig)

fig = ctl.plot_multimodel_regime_pdfs(results, model_names = model_names_all, filename = cart_out + 'all_regimes_primacoup_{}_4eofs.pdf'.format(tag), eof_proj = [(0,1), (1,2), (2,3), (3,0)], figsize = (20,12), reference = 'ERA', eof_axis_lim = eof_axis_lim, check_for_eofs = check_for_eofs, colors = colors_wERA)
allfigs.append(fig)

# filo = filogen.format('_refEOF')
# results, results_ref = pickle.load(open(filo, 'r'))

all_mod_stats = dict()

t0 = datetime.now()
for mod in model_names:
    for reg in range(4):
        okclu = results[mod]['labels'] == reg
        okpc1 = results[mod]['pcs'][okclu, :]
        okclu = results_ref['labels'] == reg
        okpc2 = results_ref['pcs'][okclu, :]

        # 1D KS on the pcs
        kss = np.mean([stats.ks_2samp(okpc1[:, eof], okpc2[:, eof]).statistic for eof in range(4)])
        all_mod_stats[('KS 1D', mod, reg)] = kss

        # voglio std_dev della distanza dal centroide suo
        cen = results[mod]['centroids'][reg]
        dist = np.mean([ctl.distance(po, cen) for po in okpc1])
        stddist = np.std([ctl.distance(po, cen) for po in okpc1])
        all_mod_stats[('Clus radius', mod, reg)] = dist
        all_mod_stats[('Clus std', mod, reg)] = stddist

        # voglio ks su distanza dal centroide ref
        cen = results_ref['centroids'][reg]
        dist1 = [ctl.distance(po, cen) for po in okpc1]
        dist2 = [ctl.distance(po, cen) for po in okpc2]
        kss = stats.ks_2samp(dist1, dist2).statistic
        all_mod_stats[('KS dist refcentroid', mod, reg)] = kss

        # cen to cen distance
        dist = ctl.distance(results[mod]['centroids'][reg], results_ref['centroids'][reg])
        all_mod_stats[('dist centocen', mod, reg)] = dist

pickle.dump(all_mod_stats, open(cart_out + 'all_stats_models_primacoup_{}.p'.format(tag), 'w'))

all_mod_stats = pickle.load(open(cart_out + 'all_stats_models_primacoup_{}.p'.format(tag)))
ERA_ref_thresholds = pickle.load(open(cart_out_ERA + 'ERA_ref_thresholds_allstats.p'))
print('TO BE REPLACED WITH NEW ERA v3 PHASESPACE STATISTICS')

for nam, tit in zip(stat_nams, titles):
    fig = plt.figure(figsize = (16, 12))
    ind = 0
    axes = []
    for reg in range(4):
        ind += 1
        ax = plt.subplot(2, 2, ind)

        allvals = np.array([all_mod_stats[(nam, mod, reg)] for mod in model_names])
        #ax.bar(np.arange(len(model_names)), allvals, color = colors)
        #ax.scatter(np.arange(len(model_names)), allvals, color = 'grey', marker = 'x', s = 200)
        ax.scatter(np.arange(len(model_names)), allvals, color = colors, marker = 'x', s = 100, linewidth=4)

        ax.axhline(ERA_ref_thresholds[(nam, 60, reg, 50)], color = 'grey', alpha = 0.6)
        ax.axhline(ERA_ref_thresholds[(nam, 60, reg, 10)], color = 'grey', alpha = 0.6, linestyle = '--')
        ax.axhline(ERA_ref_thresholds[(nam, 60, reg, 90)], color = 'grey', alpha = 0.6, linestyle = '--')
        ax.axhline(ERA_ref_thresholds[(nam, 60, reg, 1)], color = 'grey', alpha = 0.6, linestyle = ':')
        ax.axhline(ERA_ref_thresholds[(nam, 60, reg, 99)], color = 'grey', alpha = 0.6, linestyle = ':')

        #ax.set_xticklabels(model_names, size='small', rotation = 60)
        ax.set_xticklabels([])

        ax.set_title('Reg {}'.format(reg))
        axes.append(ax)
    ctl.adjust_ax_scale(axes)

    fig.suptitle(tit)
    plt.subplots_adjust(top = 0.9)
    fig = ctl.custom_legend(fig, colors, model_names)
    fig.savefig(cart_out + '_'.join(nam.split())+'_models_primacoup_{}.pdf'.format(tag))
    allfigs.append(fig)


    fig = plt.figure(figsize = (16, 12))
    ind = 0
    axes = []
    for reg in range(4):
        ind += 1
        ax = plt.subplot(2, 2, ind)

        eraval = ERA_ref_thresholds[(nam, 60, reg, 50)]
        allvals = np.array([all_mod_stats[(nam, mod, reg)]-eraval for mod in model_names])
        allvalsdiff = np.array([abs(cu)-abs(ci) for cu, ci in zip(allvals[::2], allvals[1::2])])

        i = 0
        for val, col, modcou in zip(allvalsdiff, colors[1::2], model_coups):
            ax.bar(i, val, color = col, width = wi)
            i+=0.7
            if modcou == 'HadGEM3-GC31': i+=0.5

        ax.set_xticks([])
        ax.set_title('Reg {}'.format(reg))
        ax.axhline(0., color = 'grey', alpha = 0.6)
        axes.append(ax)
    ctl.adjust_ax_scale(axes)

    fig.suptitle(tit)
    plt.subplots_adjust(top = 0.9)
    fig = ctl.custom_legend(fig, colors[1::2], model_coups)
    fig.savefig(cart_out + '_'.join(nam.split())+'_models_primacoup_{}_1vs1.pdf'.format(tag))
    allfigs.append(fig)


# 250 -> 100:     HadGEM3 GC3.1    AWI-CM 1.0
# 250 -> 50:     HadGEM3 GC3.1    CNRM-CM6
# 100 -> 50:     HadGEM3 GC3.1    EC-Earth3P    MPIESM-1-2
# 100 -> 25:    CMCC-CM2
# 50 -> 25:    IFS

# model_names = ['CMCC-CM2-HR4', 'CMCC-CM2-VHR4', 'CNRM-CM6-1', 'CNRM-CM6-1-HR', 'EC-Earth-3-LR', 'EC-Earth-3-HR', 'ECMWF-IFS-LR', 'ECMWF-IFS-HR', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-XR', 'HadGEM3-GC31-MM', 'HadGEM3-GC31-HM', 'HadGEM3-GC31-LL']
# resolution = np.array([100, 25, 250, 50, 100, 50, 50, 25, 100, 50, 100, 50, 250])
# sym_all = ['o', 'o', 'v','v', 's','s', 'P','P', 'X','X', 'd', 'd', 'd']

for nam, tit in zip(stat_nams, titles):
    fig = plt.figure(figsize = (16, 12))
    ind = 0
    axes = []
    for reg in range(4):
        ind += 1
        ax = plt.subplot(2, 2, ind)

        allvals = np.array([all_mod_stats[(nam, mod, reg)] for mod in model_names])
        #ax.bar(np.arange(len(model_names)), allvals, color = colors)
        #ax.scatter(np.arange(len(model_names)), allvals, color = 'grey', marker = 'x', s = 200)
        for res, val, col, mark in zip(resolution, allvals, colors, sym_all):
            ax.scatter(res, val, color = col, marker = mark, s = 100, linewidth = 3)

        ax.axhline(ERA_ref_thresholds[(nam, 60, reg, 50)], color = 'grey', alpha = 0.6)
        ax.axhline(ERA_ref_thresholds[(nam, 60, reg, 10)], color = 'grey', alpha = 0.6, linestyle = '--')
        ax.axhline(ERA_ref_thresholds[(nam, 60, reg, 90)], color = 'grey', alpha = 0.6, linestyle = '--')
        ax.axhline(ERA_ref_thresholds[(nam, 60, reg, 1)], color = 'grey', alpha = 0.6, linestyle = ':')
        ax.axhline(ERA_ref_thresholds[(nam, 60, reg, 99)], color = 'grey', alpha = 0.6, linestyle = ':')

        ax.set_title('Reg {}'.format(reg))
        axes.append(ax)
    ctl.adjust_ax_scale(axes)

    fig.suptitle(tit)
    plt.subplots_adjust(top = 0.9)
    fig = ctl.custom_legend(fig, colors, model_names)
    fig.savefig(cart_out + '_'.join(nam.split())+'_models_primacoup_{}_wresolution.pdf'.format(tag))
    allfigs.append(fig)

ctl.plot_pdfpages(cart_out + 'allstats_models_primacoup_{}.pdf'.format(tag), allfigs, save_single_figs = False)
