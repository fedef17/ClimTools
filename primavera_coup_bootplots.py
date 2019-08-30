#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
from matplotlib import pyplot as plt
from matplotlib import cm
import pickle

import climtools_lib as ctl
#import climdiags as cd

from matplotlib.colors import LogNorm

#######################################

cart_out = '/home/fabiano/Research/articoli/Papers/primavera_regimes/figures/figures_bootstraps_v4/'
#'/home/fedefab/Scrivania/Research/Post-doc/abstractsepapers/primavera_regimes/figures/figures_bootstraps_v4/'

filon = open(cart_out + 'res_bootstrap_v4.p', 'r')

model_names = ['CMCC-CM2-HR4', 'CMCC-CM2-VHR4', 'CNRM-CM6-1', 'CNRM-CM6-1-HR', 'EC-Earth3P', 'EC-Earth3P-HR', 'ECMWF-IFS-LR', 'ECMWF-IFS-HR', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-XR', 'HadGEM3-GC31-MM', 'HadGEM3-GC31-HM', 'HadGEM3-GC31-LL-det', 'HadGEM3-GC31-LL-stoc', 'EC-Earth3P-det', 'EC-Earth3P-stoc']
model_coups = ['CMCC-CM2', 'CNRM-CM6-1', 'EC-Earth3P', 'ECMWF-IFS', 'MPI-ESM1-2', 'HadGEM3-GC31', 'HadGEM3-GC31 (det vs stoc)', 'EC-Earth3P (det vs stoc)']
model_names_all = model_names + ['ERA']

nmods = len(model_names_all)

allnams = ['significance', 'varopt', 'autocorr', 'freq', 'dist_cen', 'resid_times_av', 'resid_times_90', 'trans_matrix', 'centroids']
alltits = ['Sharpness of regime structure', 'Optimal variance ratio', 'lag-1 autocorrelation', 'Regime frequency', 'RMS in reduced state space', 'Average residence time', '90th percentile of residence time', 'Transition matrix', 'Regime centroid']
alltits = dict(zip(allnams, alltits))

bootstri = dict()
for mod in model_names_all:
    bootstri[mod] = pickle.load(filon)

colors = ctl.color_set(len(model_names), sns_palette = 'Paired') + [cm.colors.ColorConverter.to_rgb('black')]

nam = 'significance'
regnam = ['NAO +', 'Sc. BL', 'AR', 'NAO -']

#plt.ion()
allfigs = []
for nam in ['significance', 'varopt', 'autocorr']:
    fig = plt.figure(figsize=(16,12))
    allsigs = [np.mean(bootstri[mod][nam]) for mod in model_names_all]
    allerrs = [np.std(bootstri[mod][nam]) for mod in model_names_all]
    allp90 = [(np.percentile(bootstri[mod][nam], 10), np.percentile(bootstri[mod][nam], 90)) for mod in model_names_all]
    plt.scatter(list(range(nmods)), allsigs, c = colors, s = 20)
    for imod, sig, err, errp9, col in zip(range(nmods), allsigs, allerrs, allp90, colors):
        #plt.errorbar(imod, sig, yerr = err, ecolor = col, linestyle = 'None', elinewidth = 2, capsize = 10)
        plt.errorbar([imod], [sig], yerr = [[sig - errp9[0]], [errp9[1]-sig]], ecolor = col, linestyle = 'None', elinewidth = 2, capsize = 10)
    plt.title(alltits[nam])
    plt.axhline(allsigs[-1], color = 'grey', alpha = 0.6)
    plt.axhline(allp90[-1][0], color = 'grey', alpha = 0.6, ls = '--')
    plt.axhline(allp90[-1][1], color = 'grey', alpha = 0.6, ls = '--')

    ctl.custom_legend(fig, colors, model_names_all)
    fig.savefig(cart_out + '{}_bootstraps_v4.pdf'.format(nam))
    allfigs.append(fig)


for nam in ['freq', 'dist_cen', 'resid_times_av', 'resid_times_90']:
    fig = plt.figure(figsize=(16,12))
    axes = []
    for ii, reg in enumerate(regnam):
        ax = plt.subplot(2, 2, ii+1)
        allsigs = [np.mean([cos[ii] for cos in bootstri[mod][nam]]) for mod in model_names_all]
        allerrs = [np.std([cos[ii] for cos in bootstri[mod][nam]]) for mod in model_names_all]
        allp90 = [(np.percentile([cos[ii] for cos in bootstri[mod][nam]], 10), np.percentile([cos[ii] for cos in bootstri[mod][nam]], 90)) for mod in model_names_all]
        plt.scatter(list(range(nmods)), allsigs, c = colors, s = 20)
        for imod, sig, err, errp9, col in zip(range(nmods), allsigs, allerrs, allp90, colors):
            #plt.errorbar(imod, sig, yerr = err, ecolor = col, linestyle = 'None', elinewidth = 2, capsize = 10)
            plt.errorbar([imod], [sig], yerr = [[sig - errp9[0]], [errp9[1]-sig]], ecolor = col, linestyle = 'None', elinewidth = 2, capsize = 5)
        ax.set_title(reg)
        ax.axhline(allsigs[-1], color = 'grey', alpha = 0.6)
        ax.axhline(allp90[-1][0], color = 'grey', alpha = 0.6, ls = '--')
        ax.axhline(allp90[-1][1], color = 'grey', alpha = 0.6, ls = '--')
        axes.append(ax)

    ctl.adjust_ax_scale(axes)

    fig.suptitle(alltits[nam])
    plt.subplots_adjust(top = 0.9)

    ctl.custom_legend(fig, colors, model_names_all)
    fig.savefig(cart_out + '{}_bootstraps_v4.pdf'.format(nam))
    allfigs.append(fig)


nam = 'trans_matrix'
figs = []
axes_diff = []
axes_pers = []
for ireg, reg in enumerate(regnam):
    fig = plt.figure(figsize=(16,12))
    for ii in range(4):
        ax = plt.subplot(2, 2, ii+1)
        allsigs = [np.mean([cos[ireg, ii] for cos in bootstri[mod][nam]]) for mod in model_names_all]
        allerrs = [np.std([cos[ireg, ii] for cos in bootstri[mod][nam]]) for mod in model_names_all]
        allp90 = [(np.percentile([cos[ireg, ii] for cos in bootstri[mod][nam]], 10), np.percentile([cos[ireg, ii] for cos in bootstri[mod][nam]], 90)) for mod in model_names_all]
        plt.scatter(list(range(nmods)), allsigs, c = colors, s = 20)
        for imod, sig, err, errp9, col in zip(range(nmods), allsigs, allerrs, allp90, colors):
            #plt.errorbar(imod, sig, yerr = err, ecolor = col, linestyle = 'None', elinewidth = 2, capsize = 10)
            plt.errorbar([imod], [sig], yerr = [[sig - errp9[0]], [errp9[1]-sig]], ecolor = col, linestyle = 'None', elinewidth = 2, capsize = 5)
        ax.set_title('trans {} -> {}'.format(regnam[ireg], regnam[ii]))
        if ii != ireg:
            axes_diff.append(ax)
        else:
            axes_pers.append(ax)

        ax.axhline(allsigs[-1], color = 'grey', alpha = 0.6)
        ax.axhline(allp90[-1][0], color = 'grey', alpha = 0.6, ls = '--')
        ax.axhline(allp90[-1][1], color = 'grey', alpha = 0.6, ls = '--')

    fig.suptitle(alltits[nam])
    plt.subplots_adjust(top = 0.9)

    ctl.custom_legend(fig, colors, model_names_all)
    figs.append(fig)
    allfigs.append(fig)

ctl.adjust_ax_scale(axes_diff)
ctl.adjust_ax_scale(axes_pers)

for fig, ireg in zip(figs, range(4)):
    fig.savefig(cart_out + '{}_reg{}_bootstraps_v4.pdf'.format(nam, ireg))


nam = 'trans_matrix'
figs = []
axes_diff = []
axes_pers = []
for ireg, reg in enumerate(regnam):
    fig = plt.figure(figsize=(24,8))
    ind = 0
    normpers = [np.array([(1-cos[ireg, ireg]) for cos in bootstri[mod][nam]]) for mod in model_names_all]
    normpers = dict(zip(model_names_all, normpers))
    for ii in range(4):
        if ii == ireg: continue
        ind += 1
        ax = plt.subplot(1, 3, ind)
        allsigs = [np.mean(np.array([cos[ireg, ii] for cos in bootstri[mod][nam]])/normpers[mod]) for mod in model_names_all]
        allerrs = [np.std(np.array([cos[ireg, ii] for cos in bootstri[mod][nam]])/normpers[mod]) for mod in model_names_all]
        allp90 = [(np.percentile(np.array([cos[ireg, ii] for cos in bootstri[mod][nam]])/normpers[mod], 10), np.percentile(np.array([cos[ireg, ii] for cos in bootstri[mod][nam]])/normpers[mod], 90)) for mod in model_names_all]
        plt.scatter(list(range(nmods)), allsigs, c = colors, s = 20)
        for imod, sig, err, errp9, col in zip(range(nmods), allsigs, allerrs, allp90, colors):
            #plt.errorbar(imod, sig, yerr = err, ecolor = col, linestyle = 'None', elinewidth = 2, capsize = 10)
            plt.errorbar([imod], [sig], yerr = [[sig - errp9[0]], [errp9[1]-sig]], ecolor = col, linestyle = 'None', elinewidth = 2, capsize = 5)
        ax.set_title('trans {} -> {}'.format(regnam[ireg], regnam[ii]))
        axes_diff.append(ax)

        ax.axhline(allsigs[-1], color = 'grey', alpha = 0.6)
        ax.axhline(allp90[-1][0], color = 'grey', alpha = 0.6, ls = '--')
        ax.axhline(allp90[-1][1], color = 'grey', alpha = 0.6, ls = '--')

    fig.suptitle(alltits[nam])
    plt.subplots_adjust(top = 0.9)

    ctl.custom_legend(fig, colors, model_names_all)
    figs.append(fig)
    allfigs.append(fig)

ctl.adjust_ax_scale(axes_diff)

for fig, ireg in zip(figs, range(4)):
    fig.savefig(cart_out + 'transonly_reg{}_bootstraps_v4.pdf'.format(ireg))


ctl.plot_pdfpages(cart_out + 'all_bootplots.pdf', allfigs, save_single_figs = False)
