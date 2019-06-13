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
cart_out = '/home/fabiano/Research/articoli/Papers/primavera_regimes/figures/'

cart = '/home/fabiano/Research/lavori/WeatherRegimes/prima_coup_v2/'
filogen = cart + 'out_prima_coup_v2_DJF_EAT_4clus_4pcs_1957-2014{}_dtr.p'

model_names = ['CMCC-CM2-HR4', 'CMCC-CM2-VHR4', 'CNRM-CM6-1', 'CNRM-CM6-1-HR', 'EC-Earth-3-LR', 'EC-Earth-3-HR', 'ECMWF-IFS-LR', 'ECMWF-IFS-HR', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-XR', 'HadGEM3-GC31-MM', 'HadGEM3-GC31-HM', 'HadGEM3-GC31-LL']
model_names_all = model_names + ['ERA']

colors = ctl.color_set(len(model_names), sns_palette = 'Paired')
colors_wERA = colors + ['black']

# # Ora tutte le statistiche
resolution = np.array([100, 25, 250, 50, 100, 50, 50, 25, 100, 50, 100, 50, 250])
sym_all = ['o', 'o', 'v','v', 's','s', 'P','P', 'X','X', 'd', 'd', 'd']
# sym_all = len(model_names)*['X']
lw = 2

regnam = ['NAO +', 'Sc. BL', 'NAO -', 'AR']

for tip in ['_refEOF', '_refCLUS', '']:
    print(tip)
    filo = filogen.format(tip)
    results, results_ref = pickle.load(open(filo, 'r'))
    results['ERA'] = results_ref

    # significance = dict()
    # print('Running clus sig\n')
    # for mod in results.keys():
    #     print(mod)
    #     significance[mod] = ctl.clusters_sig(results[mod]['pcs'], results[mod]['centroids'], results[mod]['labels'], results[mod]['dates'], nrsamp = 5000)
    #     print(significance[mod])
    #
    # pickle.dump(significance, open(cart + 'significance'+tip+'.p', 'w'))
    significance = pickle.load(open(cart + 'significance'+tip+'.p'))

    for mod in results:
        results[mod]['significance'] = significance[mod]

        results[mod]['varopt'] = ctl.calc_varopt_molt(results[mod]['pcs'], results[mod]['centroids'], results[mod]['labels'])

    # Qui ci metto: RMS, patcor, significance, optimal_ratio, clus_frequency, clus_persistence
    allnams = ['significance', 'varopt', 'RMS', 'patcor', 'freq_clus', 'resid_times']
    longnams = ['Sharpness', 'Optimal ratio', 'RMS', '1 - Pattern correlation', 'Regime frequency bias', 'Average regime persistence']

    figures = []
    for cos, tit in zip(allnams[:2], longnams[:2]):
        fig = plt.figure(figsize = (16, 12))

        ax = plt.subplot(111)
        allvals = np.array([results[mod][cos] for mod in model_names])

        for res, val, col, mark in zip(resolution, allvals, colors, sym_all):
            ax.scatter(res, val, color = col, marker = mark, s = 100, linewidth = lw)

        ax.axhline(results['ERA'][cos], color = 'grey', alpha = 0.6)

        ax.set_ylabel(tit)
        ax.set_xlabel('Eff. atm. resolution (km)')

        fig = ctl.custom_legend(fig, colors, model_names)
        fig.savefig(cart_out + cos+'_primacoup_v2{}_wresolution.pdf'.format(tip))
        figures.append(fig)

    for cos, tit in zip(allnams[2:5], longnams[2:5]):
        fig = plt.figure(figsize = (16, 12))
        ind = 0
        axes = []
        for reg in range(4):
            ind += 1
            ax = plt.subplot(2, 2, ind)

            if cos == 'patcor':
                allvals = np.array([(1.-results[mod][cos][reg]) for mod in model_names])
            elif cos == 'RMS':
                allvals = np.array([results[mod][cos][reg] for mod in model_names])
            elif cos == 'freq_clus':
                allvals = np.array([results[mod][cos][reg]-results['ERA'][cos][reg] for mod in model_names])
            #ax.bar(np.arange(len(model_names)), allvals, color = colors)
            #ax.scatter(np.arange(len(model_names)), allvals, color = 'grey', marker = 'x', s = 200)
            for res, val, col, mark in zip(resolution, allvals, colors, sym_all):
                ax.scatter(res, val, color = col, marker = mark, s = 100, linewidth = lw)

            ax.set_ylabel(tit)
            ax.set_xlabel('Eff. atm. resolution (km)')

            ax.axhline(0., color = 'grey', alpha = 0.6)

            # ax.set_title('Reg {}'.format(reg))
            ax.set_title(regnam[reg])
            axes.append(ax)
        ctl.adjust_ax_scale(axes)

        fig.suptitle(tit)
        plt.subplots_adjust(top = 0.9)
        fig = ctl.custom_legend(fig, colors, model_names)
        fig.savefig(cart_out + cos+'_primacoup_v2{}_wresolution.pdf'.format(tip))
        figures.append(fig)

    cos = allnams[5]
    tit = longnams[5]
    fig = plt.figure(figsize = (16, 12))
    ind = 0
    axes = []
    for reg in range(4):
        ind += 1
        ax = plt.subplot(2, 2, ind)

        #allvals = np.array([stats.ks_2samp(results[mod][cos][reg], results['ERA'][cos][reg]).statistic for mod in model_names])

        allvals = np.array([np.mean(results[mod][cos][reg]) for mod in model_names])

        #ax.bar(np.arange(len(model_names)), allvals, color = colors)
        #ax.scatter(np.arange(len(model_names)), allvals, color = 'grey', marker = 'x', s = 200)
        for res, val, col, mark in zip(resolution, allvals, colors, sym_all):
            ax.scatter(res, val, color = col, marker = mark, s = 100, linewidth = lw)

        ax.axhline(np.mean(results['ERA'][cos][reg]), color = 'grey', alpha = 0.6)

        ax.set_ylabel(tit)
        ax.set_xlabel('Eff. atm. resolution (km)')

        ax.axhline(0., color = 'grey', alpha = 0.6)

        #ax.set_title('Reg {}'.format(reg))
        ax.set_title(regnam[reg])
        axes.append(ax)
    ctl.adjust_ax_scale(axes)

    fig.suptitle(tit)
    plt.subplots_adjust(top = 0.9)
    fig = ctl.custom_legend(fig, colors, model_names)
    fig.savefig(cart_out + cos+'_primacoup_v2{}_wresolution.pdf'.format(tip))
    figures.append(fig)

    ctl.plot_pdfpages(cart_out + 'allfigs_primacoup_v2{}_wresolution.pdf'.format(tip), figures)
