#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os

from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as PathEffects

import netCDF4 as nc
import cartopy.crs as ccrs
import pandas as pd

from numpy import linalg as LA
from eofs.standard import Eof
from scipy import stats
import itertools as itt

from sklearn.cluster import KMeans
import ctool
import ctp

from datetime import datetime
import pickle

import climtools_lib as ctl

from copy import deepcopy as cp

##############################################
############ INPUTS #########

cart = '/home/fabiano/Research/lavori/SPHINX_for_lisboa/WRtool/'
dates = pickle.load(open(cart+'dates.p'))

ensmem = ['lcb0', 'lcb1', 'lcb2', 'lcs0', 'lcs1', 'lcs2']
base_ens = ensmem[:3]
stoc_ens = ensmem[3:]

# results = pickle.load(open(cart+'results_SPHINX_10yr.p','r'))
# years1 = np.arange(1850,2071,10)
# years2 = np.arange(1880,2101,10)
#
#
# yr_ranges = []
# for y1, y2 in zip(years1, years2):
#     yr_ranges.append((y1,y2))
#
# cyea = [np.mean(ran) for ran in yr_ranges]
#
# histran = (1850,2005)
# futran = (2006,2100)
#
#
# sigs_EAT = []
# sigs_PNA = []
# for ens in ensmem:
#     sig_a = np.array([results[(ens,'EAT',ran)]['significance'] for ran in yr_ranges])
#     sig_p = np.array([results[(ens,'PNA',ran)]['significance'] for ran in yr_ranges])
#     sigs_EAT.append(sig_a)
#     sigs_PNA.append(sig_p)
#
# colors = ctl.color_set(6, bright_thres = 0.2)
#
# for area in ['EAT', 'PNA']:
#     fig = plt.figure()
#     for ens in base_ens:
#         sig = np.array([results[(ens,area,ran)]['significance'] for ran in yr_ranges])
#         plt.plot(cyea, sig, label = ens)
#         # plt.scatter(cyea, sig, color = col, marker = sym, label = ens)
#     plt.legend()
#     plt.ylabel('Significance')
#     plt.xlabel('central year of 30yr period')
#     plt.title(area+' - base runs')
#     fig.savefig(cart+'Sig_{}_base.pdf'.format(area))
#
#     fig = plt.figure()
#     for ens in stoc_ens:
#         sig = np.array([results[(ens,area,ran)]['significance'] for ran in yr_ranges])
#         plt.plot(cyea, sig, label = ens)
#         # plt.scatter(cyea, sig, color = col, marker = sym, label = ens)
#     plt.legend()
#     plt.ylabel('Significance')
#     plt.xlabel('central year of 30yr period')
#     plt.title(area+' - stoc runs')
#     fig.savefig(cart+'Sig_{}_stoc.pdf'.format(area))
#
#
# hist_runs = dict()
# ran = (1850,2005)
# for area in ['EAT', 'PNA']:
#     reference = results[('lcb1',area,ran)]
#     for ens in ensmem:
#         res_ens = results[(ens,area,ran)]
#         perm, centroids, labels, et, patcor = ctl.clus_compare_patternsonly(res_ens['centroids'], res_ens['labels'], res_ens['cluspattern_area'], reference['cluspattern_area'])
#         hist_runs[(ens, area)] = cp(res_ens)
#         hist_runs[(ens, area)]['centroids'] = cp(centroids)
#         hist_runs[(ens, area)]['labels'] = cp(labels)
#         hist_runs[(ens, area)]['cluspattern'] = cp(res_ens['cluspattern'][perm])
#         hist_runs[(ens, area)]['cluspattern_area'] = cp(res_ens['cluspattern_area'][perm])
#
# ran = (1850,2005)
# for area in ['EAT', 'PNA']:
#     for num in range(4):
#         if area == 'PNA' and num == 3:
#             break
#         lat = results[(ens,area,ran)]['lat']
#         lon = results[(ens,area,ran)]['lon']
#
#         #patts = [results[(ens,area,ran)]['cluspattern'][num] for ens in ensmem]
#         patts = [hist_runs[(ens,area)]['cluspattern'][num] for ens in ensmem]
#
#         filename = cart + 'Clus_{}{}_all_1850-2005.pdf'.format(area,num)
#         ctl.plot_multimap_contour(patts, lat, lon, filename, visualization = 'polar', central_lat_lon = (90.,0.), cmap = 'RdBu_r', title = 'Regime {} on {}'.format(num, area), subtitles = ensmem, cb_label = 'Geopotential height anomaly (m)', color_percentiles = (0.5,99.5), fix_subplots_shape = (2,3), number_subplots = False, figsize = (15,15))



results = pickle.load(open(cart+'results_SPHINX_new_oksig.p','r'))
years1 = np.arange(1850,2071,5)
years2 = np.arange(1880,2101,5)
yr_ranges = []
for y1, y2 in zip(years1, years2):
    yr_ranges.append((y1,y2))

cyea = [np.mean(ran) for ran in yr_ranges]


sigs_EAT = []
sigs_PNA = []
for ens in ensmem:
    sig_a = np.array([results[(ens,'EAT',ran)]['significance'] for ran in yr_ranges])
    sig_p = np.array([results[(ens,'PNA',ran)]['significance'] for ran in yr_ranges])
    sigs_EAT.append(sig_a)
    sigs_PNA.append(sig_p)

colors = ctl.color_set(6, bright_thres = 0.2)

for area in ['EAT', 'PNA']:
    fig = plt.figure()
    for ens in base_ens:
        sig = np.array([results[(ens,area,ran)]['significance'] for ran in yr_ranges])
        plt.plot(cyea, sig, label = ens)
        # plt.scatter(cyea, sig, color = col, marker = sym, label = ens)
    plt.legend()
    plt.ylabel('Significance')
    plt.xlabel('central year of 30yr period')
    plt.title(area+' - base runs')
    fig.savefig(cart+'Sig_{}_base_5yr.pdf'.format(area))

    fig = plt.figure()
    for ens in stoc_ens:
        sig = np.array([results[(ens,area,ran)]['significance'] for ran in yr_ranges])
        plt.plot(cyea, sig, label = ens)
        # plt.scatter(cyea, sig, color = col, marker = sym, label = ens)
    plt.legend()
    plt.ylabel('Significance')
    plt.xlabel('central year of 30yr period')
    plt.title(area+' - stoc runs')
    fig.savefig(cart+'Sig_{}_stoc_5yr.pdf'.format(area))

    fig = plt.figure()
    for ens in base_ens:
        sig = np.array([results[(ens,area,ran)]['significance'] for ran in yr_ranges])
        sig = ctl.running_mean(sig, 20)
        plt.plot(cyea, sig, label = ens)
        # plt.scatter(cyea, sig, color = col, marker = sym, label = ens)
    plt.legend()
    plt.ylabel('Significance')
    plt.xlabel('central year of 30yr period')
    plt.title(area+' - base runs')
    fig.savefig(cart+'Sig_{}_base_20yr_smooth.pdf'.format(area))

    fig = plt.figure()
    for ens in stoc_ens:
        sig = np.array([results[(ens,area,ran)]['significance'] for ran in yr_ranges])
        sig = ctl.running_mean(sig, 20)
        plt.plot(cyea, sig, label = ens)
        # plt.scatter(cyea, sig, color = col, marker = sym, label = ens)
    plt.legend()
    plt.ylabel('Significance')
    plt.xlabel('central year of 30yr period')
    plt.title(area+' - stoc runs')
    fig.savefig(cart+'Sig_{}_stoc_20yr_smooth.pdf'.format(area))



lat = results[(ens,area,ran)]['lat']
lon = results[(ens,area,ran)]['lon']
newlons = lon % 180

ran = (1850,2005)
for area in ['EAT', 'PNA']:
    for num in range(4):
        if area == 'PNA' and num == 3:
            break

        patts = [results[(ens,area,ran)]['cluspattern'][num] for ens in ensmem]
        #patts = [hist_runs[(ens,area)]['cluspattern'][num] for ens in ensmem]

        filename = cart + 'Clus_{}{}_all_1850-2005_new.pdf'.format(area,num)
        print(filename)
        for pa in patts:
            print(ens, pa.min())
        ctl.plot_multimap_contour(patts, lat, newlons, filename, visualization = 'polar', central_lat_lon = (90.,0.), cmap = 'RdBu_r', title = 'Regime {} on {}'.format(num, area), subtitles = ensmem, cb_label = 'Geopotential height anomaly (m)', color_percentiles = (0.5,99.5), fix_subplots_shape = (2,3), number_subplots = False, figsize = (15,15))

# RESIDENCE TIMES
base_labels = np.concatenate([results[('lcb{}'.format(i), 'EAT', ran)]['labels'] for i in range(3)])
stoc_labels = np.concatenate([results[('lcs{}'.format(i), 'EAT', ran)]['labels'] for i in range(3)])
dates_long = np.concatenate([dates,dates,dates])

freq_base = ctl.calc_clus_freq(base_labels)
freq_stoc = ctl.calc_clus_freq(stoc_labels)
print('stoc', freq_stoc)
print('base', freq_base)

rs_base, dates_init_b = ctl.calc_residence_times(base_labels, dates = dates_long)
rs_stoc, dates_init_s = ctl.calc_residence_times(stoc_labels, dates = dates_long)

patnames = ['NAO +', 'Blocking', 'NAO -', 'Atl. Ridge']
patnames_short = ['NP', 'BL', 'NN', 'AR']
binzzz = np.arange(0,37,2)
for clu, clunam in zip(range(4), patnames):
    pts = patnames_short[clu]
    fig = plt.figure()
    plt.title(clunam)
    n, bins, patches = plt.hist(rs_base[clu], bins = binzzz, alpha = 0.5, density = True, label = 'base')
    n2, bins2, patches2 = plt.hist(rs_stoc[clu], bins = binzzz, alpha = 0.5, density = True, label = 'stoc')
    plt.legend()
    plt.xlabel('days')
    plt.ylabel('freq')
    fig.savefig(cart+'persistence_{}_base_vs_stoc_1850-2005.pdf'.format(pts))


clus_freq_ens = dict()

dists = results[('lcb0', 'EAT', (1850,2005))]['dist_centroid']
thres = np.percentile(dists, 80)

for ens in ensmem:
    for area in ['EAT', 'PNA']:
        clusfreq = []
        for ran in yr_ranges:
            freqs = ctl.calc_clus_freq(results[(ens, area, ran)]['labels'])
            clusfreq.append(freqs)
        clus_freq_ens[(ens, area)] = np.stack(clusfreq)

        clusfreq = []
        for ran in yr_ranges:
            labs = results[(ens, area, ran)]['labels']
            dists = results[(ens, area, ran)]['dist_centroid']
            greylabs = dists > thres
            labs[greylabs] = np.max(labs)+1
            freqs = ctl.calc_clus_freq(labs)
            clusfreq.append(freqs)
        clus_freq_ens[(ens, area, 'filt80')] = np.stack(clusfreq)

for area in ['EAT', 'PNA']:
    clus_freq_ens[('base', area, 'filt80')] = np.mean(np.stack([clus_freq_ens[(ens, area, 'filt80')] for ens in base_ens]), axis = 0)
    clus_freq_ens[('base', area)] = np.mean(np.stack([clus_freq_ens[(ens, area)] for ens in base_ens]), axis = 0)
    clus_freq_ens[('stoc', area, 'filt80')] = np.mean(np.stack([clus_freq_ens[(ens, area, 'filt80')] for ens in stoc_ens]), axis = 0)
    clus_freq_ens[('stoc', area)] = np.mean(np.stack([clus_freq_ens[(ens, area)] for ens in stoc_ens]), axis = 0)

patnames = dict()
patnames['EAT'] = ['NAO +', 'Blocking', 'NAO -', 'Atl. Ridge']
patnames['PNA'] = ['Ala. Ridge', 'Pac. Trough', 'Arctic High']

for ens in ensmem:
    for area in ['EAT', 'PNA']:
        fig = plt.figure()
        for nu, nam in enumerate(patnames[area]):
            frq = clus_freq_ens[(ens, area)][:,nu]
            plt.plot(cyea, frq, label = nam)
        plt.legend()
        plt.grid()
        plt.ylabel('Frequency')
        plt.xlabel('central year of 30yr period')
        plt.title('{} - {}'.format(area, ens))
        fig.savefig(cart+'Freq_{}_{}.pdf'.format(area,ens))

        fig = plt.figure()
        for nu, nam in enumerate(patnames[area]):
            frq = clus_freq_ens[(ens, area, 'filt80')][:,nu]
            plt.plot(cyea, frq, label = nam)
        frq = clus_freq_ens[(ens, area, 'filt80')][:,-1]
        plt.plot(cyea, frq, label = 'wandering')
        plt.legend()
        plt.grid()
        plt.ylabel('Frequency')
        plt.xlabel('central year of 30yr period')
        plt.title('{} - {}'.format(area, ens))
        fig.savefig(cart+'Freq_{}_{}_filt80.pdf'.format(area,ens))

for area in ['EAT', 'PNA']:
    for sim in ['base', 'stoc']:
        fig = plt.figure()
        for nu, nam in enumerate(patnames[area]):
            frq = clus_freq_ens[(sim, area)][:,nu]
            plt.plot(cyea, frq, label = nam)
        plt.legend()
        plt.grid()
        plt.ylabel('Frequency')
        plt.xlabel('central year of 30yr period')
        plt.title('{} - {}'.format(area, sim))
        fig.savefig(cart+'Freq_{}_{}.pdf'.format(area,sim))

        fig = plt.figure()
        for nu, nam in enumerate(patnames[area]):
            frq = clus_freq_ens[(sim, area, 'filt80')][:,nu]
            plt.plot(cyea, frq, label = nam)
        frq = clus_freq_ens[(sim, area, 'filt80')][:,-1]
        plt.plot(cyea, frq, label = 'wandering')
        plt.legend()
        plt.grid()
        plt.ylabel('Frequency')
        plt.xlabel('central year of 30yr period')
        plt.title('{} - {}'.format(area, sim))
        fig.savefig(cart+'Freq_{}_{}_filt80.pdf'.format(area, sim))



#################################################################################################

corrpat_ens = []
for ens in ensmem:
    corrpat = []
    for ran in yr_ranges:
        pat_EAT = results[(ens, 'EAT', ran)]['cluspattern'][0]
        pat_PNA = results[(ens, 'PNA', ran)]['cluspattern'][0]
        corrpat.append(ctl.Rcorr(pat_EAT, pat_PNA))
    corrpat_ens.append(np.array(corrpat))

fig = plt.figure()
for ens, cpa in zip(base_ens, corrpat_ens[:3]):
    plt.plot(cyea, cpa, label = ens)
    # plt.scatter(cyea, sig, color = col, marker = sym, label = ens)
plt.legend()
plt.grid()
plt.ylabel('Corr')
plt.xlabel('central year of 30yr period')
plt.title('base runs')
fig.savefig(cart+'Corr_EATPNA_base.pdf')

fig = plt.figure()
for ens, cpa in zip(stoc_ens, corrpat_ens[3:]):
    plt.plot(cyea, cpa, label = ens)
    # plt.scatter(cyea, sig, color = col, marker = sym, label = ens)
plt.legend()
plt.grid()
plt.ylabel('Corr')
plt.xlabel('central year of 30yr period')
plt.title('base runs')
fig.savefig(cart+'Corr_EATPNA_stoc.pdf')


fig = plt.figure()
for ens, cpa in zip(stoc_ens, corrpat_ens[3:]):
    plt.plot(cyea, cpa, label = ens)
    # plt.scatter(cyea, sig, color = col, marker = sym, label = ens)
plt.legend()
plt.grid()
plt.ylabel('Corr')
plt.xlabel('central year of 30yr period')
plt.title('base runs')
fig.savefig(cart+'Corr_EATPNA_stoc.pdf')

# videino dei cluster patterns
#for clus
#ctl.plot_double_sidebyside(patuno, patuno_ref, lat, lon, filename = nunam, visualization = 'polar', central_lat_lon = (50., 0.), title = pp, cb_label = 'Geopotential height anomaly (m)', stitle_1 = tag, stitle_2 = 'ERA', color_percentiles = (0.5,99.5))
