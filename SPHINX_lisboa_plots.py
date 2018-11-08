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

cart2 = '/home/fabiano/Research/lavori/WP2_deliverable_Oct2018/Results_WP2/'
results_prima2 = pickle.load(open(cart2+'res_primavera.p','r'))

pat_era = results_prima2['cluspattern_area']['ERA']
lab_era = results_prima2['labels']['ERA']

ensmem = ['lcb0', 'lcb1', 'lcb2', 'lcs0', 'lcs1', 'lcs2']
base_ens = ensmem[:3]
stoc_ens = ensmem[3:]

patnames = dict()
patnames['EAT'] = ['NAO +', 'Blocking', 'NAO -', 'Atl. Ridge']
patnames['PNA'] = ['Ala. Ridge', 'Pac. Trough', 'Arctic High']
patnames_short = dict()
patnames_short['EAT'] = ['NP', 'BL', 'NN', 'AR']
patnames_short['PNA'] = ['AR', 'PT', 'AH']

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

results_fullp = pickle.load(open(cart+'results_SPHINX_fullperiod_detrended.p','r'))

results = pickle.load(open(cart+'results_SPHINX_new_oksig.p','r'))
years1 = np.arange(1850,2071,5)
years2 = np.arange(1880,2101,5)
yr_ranges = []
for y1, y2 in zip(years1, years2):
    yr_ranges.append((y1,y2))

cyea = [np.mean(ran) for ran in yr_ranges]

histran = (1850, 2005)
futran = (2006, 2100)

cartsig = cart + 'sig/'

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
    fig.savefig(cartsig+'Sig_{}_base_5yr.pdf'.format(area))

    fig = plt.figure()
    for ens in stoc_ens:
        sig = np.array([results[(ens,area,ran)]['significance'] for ran in yr_ranges])
        plt.plot(cyea, sig, label = ens)
        # plt.scatter(cyea, sig, color = col, marker = sym, label = ens)
    plt.legend()
    plt.ylabel('Significance')
    plt.xlabel('central year of 30yr period')
    plt.title(area+' - stoc runs')
    fig.savefig(cartsig+'Sig_{}_stoc_5yr.pdf'.format(area))

    fig = plt.figure()
    for ens in base_ens:
        sig = np.array([results[(ens,area,ran)]['significance'] for ran in yr_ranges])
        sig = ctl.running_mean(sig, 5)
        plt.plot(cyea, sig, label = ens)
        # plt.scatter(cyea, sig, color = col, marker = sym, label = ens)
    plt.legend()
    plt.ylabel('Significance')
    plt.xlabel('central year of 30yr period')
    plt.title(area+' - base runs (25 yr smooth)')
    fig.savefig(cartsig+'Sig_{}_base_20yr_smooth.pdf'.format(area))

    fig = plt.figure()
    for ens in stoc_ens:
        sig = np.array([results[(ens,area,ran)]['significance'] for ran in yr_ranges])
        sig = ctl.running_mean(sig, 5)
        plt.plot(cyea, sig, label = ens)
        # plt.scatter(cyea, sig, color = col, marker = sym, label = ens)
    plt.legend()
    plt.ylabel('Significance')
    plt.xlabel('central year of 30yr period')
    plt.title(area+' - stoc runs (25 yr smooth)')
    fig.savefig(cartsig+'Sig_{}_stoc_20yr_smooth.pdf'.format(area))



lat = np.array(results[(ens,area,ran)]['lat'])
lon = np.array(results[(ens,area,ran)]['lon'])
#
# cartmap = cart + 'Maps/'
# for ran in [histran, futran]:
#     for area in ['EAT', 'PNA']:
#         for num, nam in enumerate(patnames[area]):
#             patts = [results[(ens,area,ran)]['cluspattern'][num] for ens in ensmem]
#             #patts = [hist_runs[(ens,area)]['cluspattern'][num] for ens in ensmem]
#
#             filename = cartmap + 'Clus_{}{}_all_new_{}-{}.pdf'.format(area,num,ran[0],ran[1])
#             print(filename)
#             for pa in patts:
#                 print(ens, pa.min())
#             if area == 'EAT':
#                 clatlo = (70.,0.)
#             else:
#                 clatlo = (70.,-90.)
#             ctl.plot_multimap_contour(patts, lat, lon, filename, visualization = 'polar', central_lat_lon = clatlo, cmap = 'RdBu_r', title = nam, subtitles = ensmem, cb_label = 'Geopotential height anomaly (m)', color_percentiles = (0.5,99.5), fix_subplots_shape = (2,3), number_subplots = False, figsize = (15,15), draw_contour_lines = False)

# RESIDENCE TIMES
thres = dict()
dists = results[('lcb0', 'EAT', (1850,2005))]['dist_centroid']
thres['EAT'] = np.percentile(dists, 80)
dists = results[('lcb0', 'PNA', (1850,2005))]['dist_centroid']
thres['PNA'] = np.percentile(dists, 80)

dates_long = np.concatenate([dates,dates,dates])
freq = dict()
resid_times = dict()
for area in ['EAT', 'PNA']:
    for ran in [(1850,2005), (2006,2100)]:
        base_labels = np.concatenate([results[('lcb{}'.format(i), 'EAT', ran)]['labels'] for i in range(3)])
        base_dists =  np.concatenate([results[('lcb{}'.format(i), 'EAT', ran)]['dist_centroid'] for i in range(3)])
        stoc_labels = np.concatenate([results[('lcs{}'.format(i), 'EAT', ran)]['labels'] for i in range(3)])
        stoc_dists =  np.concatenate([results[('lcs{}'.format(i), 'EAT', ran)]['dist_centroid'] for i in range(3)])

        freq[('base', area, ran)] = ctl.calc_clus_freq(base_labels)
        freq[('stoc', area, ran)] = ctl.calc_clus_freq(stoc_labels)

        greylabs = base_dists > thres[area]
        gigi = base_labels
        gigi[greylabs] = np.max(base_labels)+1
        freq[('base', area, ran, 'filt80')] = ctl.calc_clus_freq(gigi)

        greylabs = stoc_dists > thres[area]
        gigi = stoc_labels
        gigi[greylabs] = np.max(stoc_labels)+1
        freq[('stoc', area, ran, 'filt80')] = ctl.calc_clus_freq(gigi)

        rs_base, dates_init_b = ctl.calc_regime_residtimes(base_labels, dates = dates_long)
        resid_times[('base', area, ran)] = rs_base
        rs_stoc, dates_init_s = ctl.calc_regime_residtimes(stoc_labels, dates = dates_long)
        resid_times[('stoc', area, ran)] = rs_stoc

cartper = cart + 'resid_times/'
for area in ['EAT', 'PNA']:
    binzzz = np.arange(0,36,5)
    for clu, clunam in enumerate(patnames[area]):
        ran = histran
        pts = patnames_short[area][clu]
        fig = plt.figure()
        plt.title(clunam)
        n, bins, patches = plt.hist(resid_times[('base', area, ran)][clu], bins = binzzz, alpha = 0.5, density = True, label = 'base')
        n2, bins2, patches2 = plt.hist(resid_times[('stoc', area, ran)][clu], bins = binzzz, alpha = 0.5, density = True, label = 'stoc')
        plt.legend()
        plt.xlabel('days')
        plt.ylabel('freq')
        fig.savefig(cartper+'persistence_{}_{}_base_vs_stoc_{}-{}.pdf'.format(area, pts, ran[0], ran[1]))

        pts = patnames_short[area][clu]
        fig = plt.figure()
        plt.title(clunam)
        n, bins, patches = plt.hist(resid_times[('stoc', area, histran)][clu], bins = binzzz, alpha = 0.5, density = True, label = 'hist')
        n2, bins2, patches2 = plt.hist(resid_times[('stoc', area, futran)][clu], bins = binzzz, alpha = 0.5, density = True, label = 'fut')
        plt.legend()
        plt.xlabel('days')
        plt.ylabel('freq')
        fig.savefig(cartper+'persistence_{}_{}_fut_vs_hist__stoc.pdf'.format(area, pts))

        pts = patnames_short[area][clu]
        fig = plt.figure()
        plt.title(clunam)
        n, bins, patches = plt.hist(resid_times[('base', area, histran)][clu], bins = binzzz, alpha = 0.5, density = True, label = 'hist')
        n2, bins2, patches2 = plt.hist(resid_times[('base', area, futran)][clu], bins = binzzz, alpha = 0.5, density = True, label = 'fut')
        plt.legend()
        plt.xlabel('days')
        plt.ylabel('freq')
        fig.savefig(cartper+'persistence_{}_{}_fut_vs_hist__base.pdf'.format(area, pts, ran[0], ran[1]))


#####################################################################################
#####################################################################################
### CLUSTER Frequency

era_freq = ctl.calc_clus_freq(lab_era)

timed = pd.Timedelta('90 days')
dates_pdh = pd.to_datetime(dates)
pi = (dates_pdh.month == 12) | (dates_pdh.month == 1) | (dates_pdh.month == 2)
pi2 = (pi) & (dates_pdh > dates_pdh[0]+timed) & (dates_pdh < dates_pdh[-1]-timed)
datespi2 = dates[pi2]

labs_fullp = []
dist_fullp = []
freqs_yr_all = dict()
for ens in ensmem:
    labs_fullp.append(results_fullp[(ens, 'EAT', (1850,2100))]['labels'])
    dist_fullp.append(results_fullp[(ens, 'EAT', (1850,2100))]['dist_centroid'])
    if ens == 'lcb1':
        dates_ok = datespi2[90:]
    else:
        dates_ok = datespi2
    freqs_yr_all[ens] = ctl.calc_seasonal_clus_freq(labs_fullp[-1], dates_ok)

cartfr = cart+'freq_fullp/'
if not os.path.exists(cartfr):
    os.mkdir(cartfr)

plt.close('all')

yearsall = np.arange(1851, 2101)

for ens in ensmem:
    fig = plt.figure()
    if ens == 'lcb1':
        years = yearsall[1:]
    else:
        years = yearsall
    #plt.bar(years, freqs_yr_all[ens])
    for clu, clunam in enumerate(patnames['EAT']):
        smut = ctl.running_mean(freqs_yr_all[ens][:,clu], wnd = 10)
        plt.plot(years, smut, label = clunam)

    plt.legend()
    fig.savefig(cartfr+'freq_fullp_{}.pdf'.format(ens))

freqs_yr_all['lcb1'] = np.vstack([freqs_yr_all['lcb1'][0,:],freqs_yr_all['lcb1']])

for cos in ['base', 'stoc']:
    fig = plt.figure()
    if cos == 'base':
        okfrq = np.mean([freqs_yr_all[ens][1:] for ens in ensmem[:3]], axis = 0)
    else:
        okfrq = np.mean([freqs_yr_all[ens][1:] for ens in ensmem[3:]], axis = 0)
    #plt.bar(years, freqs_yr_all[ens])
    for clu, clunam in enumerate(patnames['EAT']):
        smut = ctl.running_mean(okfrq[:,clu], wnd = 30)
        plt.plot(years[1:], smut, label = clunam)

    plt.legend()
    fig.savefig(cartfr+'freq_fullp_{}.pdf'.format(cos))

cartmap = cart + 'Maps/'
area = 'EAT'
for num, nam in enumerate(patnames[area]):
    patts = [results_fullp[(ens, area, (1850,2100))]['cluspattern'][num] for ens in ensmem]
    #patts = [hist_runs[(ens,area)]['cluspattern'][num] for ens in ensmem]

    filename = cartmap + 'Clusfullp_{}{}_all_new.pdf'.format('EAT',num)
    print(filename)
    for pa in patts:
        print(ens, pa.min())
    if area == 'EAT':
        clatlo = (70.,0.)
    else:
        clatlo = (70.,-90.)
    ctl.plot_multimap_contour(patts, lat, lon, filename, visualization = 'polar', central_lat_lon = clatlo, cmap = 'RdBu_r', title = nam, subtitles = ensmem, cb_label = 'Geopotential height anomaly (m)', color_percentiles = (0.5,99.5), fix_subplots_shape = (2,3), number_subplots = False, figsize = (15,15), draw_contour_lines = False)

sys.exit()

cartfr = cart+'freq_histo/'

for area in ['EAT', 'PNA']:
    for ran in [histran, futran]:
        fig = plt.figure()
        lab1 = 'base'
        lab2 = 'stoc'
        wi = 0.3
        for clu, clunam in enumerate(patnames[area]):
            if clu > 0:
                lab1 = None
                lab2 = None
            plt.bar(clu-wi/2, freq[('base', area, ran)][clu], width = wi, color = 'green', label = lab1)
            plt.bar(clu+wi/2, freq[('stoc', area, ran)][clu], width = wi, color = 'orange', label = lab2)

        pts = patnames_short[area]
        plt.xticks(range(len(pts)), pts, size='small')
        plt.title('Regime frequency - base vs stoc')
        plt.legend()
        plt.xlabel('Regime')
        plt.ylabel('Freq')
        fig.savefig(cartfr+'frequency_{}_base_vs_stoc_{}-{}.pdf'.format(area, ran[0], ran[1]))

        fig = plt.figure()
        lab1 = 'base'
        lab2 = 'stoc'
        lab3 = 'ERA'
        wi = 0.2
        for clu, clunam in enumerate(patnames[area]):
            if clu > 0:
                lab1 = None
                lab2 = None
                lab3 = None
            plt.bar(clu-wi, era_freq[clu], width = wi, color = 'lightsteelblue', label = lab3)
            plt.bar(clu, freq[('base', area, ran)][clu], width = wi, color = 'green', label = lab1)
            plt.bar(clu+wi, freq[('stoc', area, ran)][clu], width = wi, color = 'orange', label = lab2)

        pts = patnames_short[area]
        plt.xticks(range(len(pts)), pts, size='small')
        plt.title('Regime frequency - base vs stoc')
        plt.legend()
        plt.xlabel('Regime')
        plt.ylabel('Freq')
        fig.savefig(cartfr+'frequency_{}_base_vs_stoc_{}-{}_compera.pdf'.format(area, ran[0], ran[1]))

    for exp in ['base', 'stoc']:
        fig = plt.figure()
        lab1 = 'hist'
        lab2 = 'future'
        wi = 0.3
        for clu, clunam in enumerate(patnames[area]):
            if clu > 0:
                lab1 = None
                lab2 = None
            plt.bar(clu-wi/2, freq[(exp, area, histran)][clu], width = wi, color = 'green', label = lab1)
            plt.bar(clu+wi/2, freq[(exp, area, futran)][clu], width = wi, color = 'orange', label = lab2)

        pts = patnames_short[area]
        plt.xticks(range(len(pts)), pts, size='small')
        plt.title('Regime frequency - hist vs future')
        plt.legend()
        plt.xlabel('Regime')
        plt.ylabel('Freq')
        fig.savefig(cartfr+'frequency_{}_{}_hist_vs_fut.pdf'.format(area, exp))

    for ran in [histran, futran]:
        fig = plt.figure()
        lab1 = 'base'
        lab2 = 'stoc'
        wi = 0.3
        for clu, clunam in enumerate(patnames[area]):
            if clu > 0:
                lab1 = None
                lab2 = None
            plt.bar(clu-wi/2, freq[('base', area, ran, 'filt80')][clu], width = wi, color = 'green', label = lab1)
            plt.bar(clu+wi/2, freq[('stoc', area, ran, 'filt80')][clu], width = wi, color = 'orange', label = lab2)

        plt.bar(clu+1-wi/2, freq[('base', area, ran, 'filt80')][-1], width = wi, color = 'green', label = lab1)
        plt.bar(clu+1+wi/2, freq[('stoc', area, ran, 'filt80')][-1], width = wi, color = 'orange', label = lab2)
        pts = patnames_short[area]
        plt.xticks(range(len(pts)+1), pts+['W'], size='small')
        plt.title('Regime frequency - base vs stoc')
        plt.legend()
        plt.xlabel('Regime')
        plt.ylabel('Freq')
        fig.savefig(cartfr+'frequency_{}_base_vs_stoc_{}-{}_wandering.pdf'.format(area, ran[0], ran[1]))

    for exp in ['base', 'stoc']:
        fig = plt.figure()
        lab1 = 'hist'
        lab2 = 'future'
        wi = 0.3
        for clu, clunam in enumerate(patnames[area]):
            if clu > 0:
                lab1 = None
                lab2 = None
            plt.bar(clu-wi/2, freq[(exp, area, histran, 'filt80')][clu], width = wi, color = 'green', label = lab1)
            plt.bar(clu+wi/2, freq[(exp, area, futran, 'filt80')][clu], width = wi, color = 'orange', label = lab2)

        plt.bar(clu+1-wi/2, freq[('base', area, histran, 'filt80')][-1], width = wi, color = 'green', label = lab1)
        plt.bar(clu+1+wi/2, freq[('stoc', area, futran, 'filt80')][-1], width = wi, color = 'orange', label = lab2)
        pts = patnames_short[area]
        plt.xticks(range(len(pts)+1), pts+['W'], size='small')
        pts = patnames_short[area]
        plt.xticks(range(len(pts)), pts, size='small')
        plt.title('Regime frequency - hist vs future')
        plt.legend()
        plt.xlabel('Regime')
        plt.ylabel('Freq')
        fig.savefig(cartfr+'frequency_{}_{}_hist_vs_fut_wandering.pdf'.format(area, exp))


clus_freq_ens = dict()
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
            greylabs = dists > thres[area]
            labs[greylabs] = np.max(labs)+1
            freqs = ctl.calc_clus_freq(labs)
            clusfreq.append(freqs)
        clus_freq_ens[(ens, area, 'filt80')] = np.stack(clusfreq)

for area in ['EAT', 'PNA']:
    clus_freq_ens[('base', area, 'filt80')] = np.mean(np.stack([clus_freq_ens[(ens, area, 'filt80')] for ens in base_ens]), axis = 0)
    clus_freq_ens[('base', area)] = np.mean(np.stack([clus_freq_ens[(ens, area)] for ens in base_ens]), axis = 0)
    clus_freq_ens[('stoc', area, 'filt80')] = np.mean(np.stack([clus_freq_ens[(ens, area, 'filt80')] for ens in stoc_ens]), axis = 0)
    clus_freq_ens[('stoc', area)] = np.mean(np.stack([clus_freq_ens[(ens, area)] for ens in stoc_ens]), axis = 0)


for smooth in [False, True]:
    if smooth:
        cartfr = cart + 'Freq_20smooth/'
        sm = '(20 yr smooth)'
    else:
        cartfr = cart + 'Freq/'
        sm = ''

    for ens in ensmem:
        for area in ['EAT', 'PNA']:
            fig = plt.figure()
            for nu, nam in enumerate(patnames[area]):
                frq = clus_freq_ens[(ens, area)][:,nu]
                if smooth: frq = ctl.running_mean(frq, 5)
                plt.plot(cyea, frq, label = nam)
            plt.legend()
            #plt.grid()
            plt.ylabel('Frequency')
            plt.xlabel('central year of 30yr period')
            plt.title('{} - {} {}'.format(area, ens, sm))
            fig.savefig(cartfr+'Freq_{}_{}.pdf'.format(area,ens))

            fig = plt.figure()
            for nu, nam in enumerate(patnames[area]):
                frq = clus_freq_ens[(ens, area, 'filt80')][:,nu]
                if smooth: frq = ctl.running_mean(frq, 5)
                plt.plot(cyea, frq, label = nam)
            frq = clus_freq_ens[(ens, area, 'filt80')][:,-1]
            if smooth: frq = ctl.running_mean(frq, 5)
            plt.plot(cyea, frq, label = 'wandering', color = 'grey')
            plt.legend()
            #plt.grid()
            plt.ylabel('Frequency')
            plt.xlabel('central year of 30yr period')
            plt.title('{} - {} {}'.format(area, ens, sm))
            fig.savefig(cartfr+'Freq_{}_{}_filt80.pdf'.format(area,ens))

    for area in ['EAT', 'PNA']:
        for sim in ['base', 'stoc']:
            fig = plt.figure()
            for nu, nam in enumerate(patnames[area]):
                frq = clus_freq_ens[(sim, area)][:,nu]
                if smooth: frq = ctl.running_mean(frq, 5)
                plt.plot(cyea, frq, label = nam)
            plt.legend()
            #plt.grid()
            plt.ylabel('Frequency')
            plt.xlabel('central year of 30yr period')
            plt.title('{} - {} {}'.format(area, sim, sm))
            fig.savefig(cartfr+'Freq_{}_{}.pdf'.format(area,sim))

            fig = plt.figure()
            for nu, nam in enumerate(patnames[area]):
                frq = clus_freq_ens[(sim, area, 'filt80')][:,nu]
                if smooth: frq = ctl.running_mean(frq, 5)
                plt.plot(cyea, frq, label = nam)
            frq = clus_freq_ens[(sim, area, 'filt80')][:,-1]
            if smooth: frq = ctl.running_mean(frq, 5)
            plt.plot(cyea, frq, label = 'wandering', color = 'grey')
            plt.legend()
            #plt.grid()
            plt.ylabel('Frequency')
            plt.xlabel('central year of 30yr period')
            plt.title('{} - {} {}'.format(area, sim, sm))
            fig.savefig(cartfr+'Freq_{}_{}_filt80.pdf'.format(area, sim))



#################################################################################################
cartcp = cart + 'corrpat/'

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
fig.savefig(cartcp+'Corr_EATPNA_base.pdf')

fig = plt.figure()
for ens, cpa in zip(stoc_ens, corrpat_ens[3:]):
    plt.plot(cyea, cpa, label = ens)
    # plt.scatter(cyea, sig, color = col, marker = sym, label = ens)
plt.legend()
plt.grid()
plt.ylabel('Corr')
plt.xlabel('central year of 30yr period')
plt.title('base runs')
fig.savefig(cartcp+'Corr_EATPNA_stoc.pdf')


fig = plt.figure()
for ens, cpa in zip(stoc_ens, corrpat_ens[3:]):
    plt.plot(cyea, cpa, label = ens)
    # plt.scatter(cyea, sig, color = col, marker = sym, label = ens)
plt.legend()
plt.grid()
plt.ylabel('Corr')
plt.xlabel('central year of 30yr period')
plt.title('base runs')
fig.savefig(cartcp+'Corr_EATPNA_stoc.pdf')



# Taylor plots
fig = plt.figure(figsize=(16,12))

for num, patt in enumerate(patnames['EAT']):
    ax = plt.subplot(2, 2, num+1, polar = True)

    modpats = [results[(ens,'EAT',histran)]['cluspattern_area'][num] for ens in ensmem]
    modpats += [results[(ens,'EAT',futran)]['cluspattern_area'][num] for ens in ensmem]

    obs = pat_era[num]

    colors = 6*['green'] + 6*['orange']
    syms = 3*['$B$'] + ['$S$']*3 + 3*['$B$'] + ['$S$']*3
    labels = ['base hist', None, None, 'stoc hist', None, None] + ['base future', None, None, 'stoc future', None, None]

    ctl.Taylor_plot(modpats, obs, ax = ax, title = None, colors = colors, markers = syms, only_first_quarter = True, legend = False, labels = labels, obs_label = 'ERA', mod_points_size = 50, obs_points_size = 70)

#Custom legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = []
for col, lab in zip(colors, labels):
    if lab is None:
        continue
    legend_elements.append(Line2D([0], [0], marker='o', color=col, label=lab, linestyle = ''))

legend_elements.append(Line2D([0], [0], marker='D', color='black', label='ERA', linestyle = ''))
fig.legend(handles=legend_elements, loc=1, fontsize = 'large')

fig.tight_layout()
n1 = plt.text(0.15,0.6,'NAO +',transform=fig.transFigure, fontsize = 20)
n2 = plt.text(0.15,0.1,'NAO -',transform=fig.transFigure, fontsize = 20)
n3 = plt.text(0.6,0.6,'Blocking',transform=fig.transFigure, fontsize = 20)
n4 = plt.text(0.6,0.1,'Atl.Ridge',transform=fig.transFigure, fontsize = 20)
bbox=dict(facecolor = 'lightsteelblue', alpha = 0.7, edgecolor='black', boxstyle='round,pad=0.2')
n1.set_bbox(bbox)
n2.set_bbox(bbox)
n3.set_bbox(bbox)
n4.set_bbox(bbox)

fig.savefig(cart+'TaylorPlot_ALL.pdf')


# Taylor plots
for gi, pio in zip(['lcb', 'lcs'], ['base', 'stoc']):
    fig = plt.figure(figsize=(16,12))

    for num, patt in enumerate(patnames['EAT']):
        ax = plt.subplot(2, 2, num+1, polar = True)

        modpats = []
        for ran in yr_ranges:
            mdpt = np.stack([results[(ens,'EAT',ran)]['cluspattern_area'][num] for ens in ensmem if gi in ens])
            modpats.append(np.mean(mdpt, axis = 0))

        obs = pat_era[num]

        colors = ctl.color_set(len(yr_ranges), bright_thres = 0.2)

        ctl.Taylor_plot(modpats, obs, ax = ax, title = None, colors = colors, only_first_quarter = True, legend = False, obs_label = 'ERA', mod_points_size = 50, obs_points_size = 70)

    fig.tight_layout()
    n1 = plt.text(0.15,0.6,'NAO +',transform=fig.transFigure, fontsize = 20)
    n2 = plt.text(0.15,0.1,'NAO -',transform=fig.transFigure, fontsize = 20)
    n3 = plt.text(0.6,0.6,'Blocking',transform=fig.transFigure, fontsize = 20)
    n4 = plt.text(0.6,0.1,'Atl.Ridge',transform=fig.transFigure, fontsize = 20)
    bbox=dict(facecolor = 'lightsteelblue', alpha = 0.7, edgecolor='black', boxstyle='round,pad=0.2')
    n1.set_bbox(bbox)
    n2.set_bbox(bbox)
    n3.set_bbox(bbox)
    n4.set_bbox(bbox)

    fig.savefig(cart+'TaylorPlot_30yrwnd_{}.pdf'.format(pio))
# videino dei cluster patterns
#for clus
#ctl.plot_double_sidebyside(patuno, patuno_ref, lat, lon, filename = nunam, visualization = 'polar', central_lat_lon = (50., 0.), title = pp, cb_label = 'Geopotential height anomaly (m)', stitle_1 = tag, stitle_2 = 'ERA', color_percentiles = (0.5,99.5))
