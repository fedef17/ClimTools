#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os

from matplotlib import pyplot as plt
import netCDF4 as nc
import pandas as pd
from numpy import linalg as LA
from scipy import stats
from scipy import interpolate as itrp
import itertools as itt

from datetime import datetime
import pickle

import climtools_lib as ctl
import climdiags as cd

from copy import deepcopy as cp
###############################################

# ###############################################################################
# ### constants
cp = 1005.0 # specific enthalpy dry air - J kg-1 K-1
cpw = 1840.0 # specific enthalpy water vapor - J kg-1 K-1
# per la moist air sarebbe cpm = cp + q*cpw, ma conta al massimo per un 3 %
L = 2501000.0 # J kg-1
Lsub = 2835000.0 # J kg-1
g = 9.81 # m s-2
Rearth = 6371.0e3 # mean radius
KtoC = 273.15
# ###############################################################################

namefi = '{}_mon_{}_{}.nc'
# Calculates radiation balances
rad_file = '/home/fabiano/Research/lavori/SPHINX_for_lisboa/radiation_balance/radclim_yearly.p'
cloud_file = '/home/fabiano/Research/lavori/SPHINX_for_lisboa/cloud_cover/cloudcover_yearly.p'
tas_file = '/data-hobbes/fabiano/SPHINX/tas_mon/global_tasmean_yearly.p'

radclim = pickle.load(open(rad_file, 'rb'))
cloudclim = pickle.load(open(cloud_file, 'rb'))
globalme, zonalme = pickle.load(open(tas_file, 'rb'))

cart_out = '/home/fabiano/Research/lavori/SPHINX_for_lisboa/diffusion_plots/'
if not os.path.exists(cart_out): os.mkdir(cart_out)

cart_in = '/data-hobbes/fabiano/SPHINX/radiation/'
varniuu, lat, lon, dates, time_units, var_units = ctl.read3Dncfield(cart_in+namefi.format('lcb0',1988,'rsut'))
del varniuu

serie = dict() # serie contiene tutte le serie puramente temporali che vado a plottare
zonals = dict() # zonals contiene varie medie zonali, per ognuna c'è un valore pi (1850-1900), uno pd (1985-2005) e uno fut (2080-2100). Più uno tra 14.0 e 14.5 e uno tra 16.5 e 17.0
locseascyc = dict() # seasonal cycle locale in diverse zone che sono 'TROP', 'ML' e 'ART'
years = np.arange(1850,2101)

ensmems = ['lcb0', 'lcb1', 'lcb2', 'lcs0', 'lcs1', 'lcs2']

radvars = ['rlut', 'rsut', 'toa_balance', 'heat_flux']
cloudvars = ['hcc', 'mcc', 'lcc', 'tcw']

allvars = radvars+cloudvars+['tas']
allvarsbl = ['rlut', 'rsut', 'toa_balance', 'toa_balance*cos(lat)', 'heat_flux']+cloudvars+['tas']
allvarunits = 'W/m**2 W/m**2 W/m**2 W Cover Cover Cover Kg/m**2 C'.split()
allvarunitsbl = 'W/m**2 W/m**2 W/m**2 W/m**2 W Cover Cover Cover Kg/m**2 C'.split()

#cominciamo da zonals
time_horiz = dict()
time_horiz['pi'] = (1850,1900)
time_horiz['pd'] = (1985,2005)
time_horiz['fut'] = (2080,2100)

temp_horiz = [(13.5, 14.5), (14.5, 15.5), (15.5, 16.5), (16.5, 17.5)]

#for coso in ['base', 'stoc']:
for coso in ensmems:
    for th in time_horiz:
        yok = (years >= time_horiz[th][0]) & (years <= time_horiz[th][1])
        for varna in radvars:
            zonals[(varna, th, coso)] = np.mean(radclim[('zonal', coso, varna)][yok, ...], axis = 0)
        for varna in cloudvars:
            zonals[(varna, th, coso)] = np.mean(cloudclim[('zonal', coso, varna)][yok, ...], axis = 0)
        zonals[('tas', th, coso)] = np.mean(zonalme[coso][yok, ...], axis = 0)
    for th in temp_horiz:
        yok = (globalme[(coso, 'tas')] >= th[0]) & (globalme[(coso, 'tas')] <= th[1])
        for varna in radvars:
            zonals[(varna, th, coso)] = np.mean(radclim[('zonal', coso, varna)][yok, ...], axis = 0)
        for varna in cloudvars:
            zonals[(varna, th, coso)] = np.mean(cloudclim[('zonal', coso, varna)][yok, ...], axis = 0)
        zonals[('tas', th, coso)] = np.mean(zonalme[coso][yok, ...], axis = 0)


for th in temp_horiz + list(time_horiz.keys()):
    for varna in allvars:
        zonals[(varna, th, 'base')] = np.mean([zonals[(varna, th, ens)] for ens in ensmems[:3]], axis = 0)
        zonals[(varna, th, 'stoc')] = np.mean([zonals[(varna, th, ens)] for ens in ensmems[3:]], axis = 0)

for ky in zonals:
    if 'toa_balance' in ky:
        ky2 = ('toa_balance*cos(lat)', ky[1], ky[2])
        zonals[ky2] = zonals[ky]*np.cos(np.deg2rad(lat))

# ora le serie
lat_bands = dict()
lat_bands['trop'] = (-20, 20)
lat_bands['NP'] = (70, 90)
lat_bands['NML'] = (40, 60)
# lat_bands['SP'] = (-90, -70)
# lat_bands['SML'] = (-60, -40)

for coso in ['base', 'stoc']+ensmems:
    for laok, lb in zip([20, 40, 70], ['trop', 'NML', 'NP']):
        ind = np.argmin(abs(lat-laok))
        print('ueue', lat[ind], laok)
        serie[('heat_flux', lb, coso)] = radclim[('zonal', coso, 'heat_flux')][:, ind]
        print(np.mean(serie[('heat_flux', lb, coso)]))
    for lb in lat_bands:
        for varna in radvars[:-1]:
            serie[(varna, lb, coso)] = ctl.band_mean_from_zonal(radclim[('zonal', coso, varna)], lat, lat_bands[lb][0], lat_bands[lb][1])
        for varna in cloudvars:
            serie[(varna, lb, coso)] = ctl.band_mean_from_zonal(cloudclim[('zonal', coso, varna)], lat, lat_bands[lb][0], lat_bands[lb][1])
        serie[('tas', lb, coso)] = ctl.band_mean_from_zonal(zonalme[coso], lat, lat_bands[lb][0], lat_bands[lb][1])

# faccio i plots
#for th in time_horiz:
figure_file = cart_out+'all_zonals_timeave.pdf'
figures = []

for var in allvarsbl:
    fig = plt.figure()
    plt.title('{} mean - stoc-base diff'.format(var))
    mea = np.max(abs(zonals[(var, 'pi', 'base')]))
    for th in time_horiz:
        plt.plot(lat, 100.*(zonals[(var, th, 'stoc')]-zonals[(var, th, 'base')])/mea, label = th, linewidth = 1.5)
    plt.xlabel('Latitude')
    plt.ylabel('Diff (% of var abs max)')
    plt.legend()
    plt.grid()
    figures.append(fig)


ctl.plot_pdfpages(figure_file, figures)
plt.close('all')

figure_file = cart_out+'all_zonals_tempave.pdf'
figures = []

for var in allvarsbl:
    fig = plt.figure()
    plt.title('{} mean - stoc-base diff'.format(var))
    mea = np.max(abs(zonals[(var, temp_horiz[0], 'base')]))
    for th in temp_horiz:
        plt.plot(lat, 100.*(zonals[(var, th, 'stoc')]-zonals[(var, th, 'base')])/mea, label = '{}-{} C'.format(th[0], th[1]), linewidth = 1.5)
    plt.xlabel('Latitude')
    plt.ylabel('Diff (% of var abs max)')
    plt.legend()
    plt.grid()
    figures.append(fig)

ctl.plot_pdfpages(figure_file, figures)
plt.close('all')


figure_file = cart_out+'all_zonals_timeave_abs.pdf'
figures = []

for var, varunits in zip(allvarsbl, allvarunitsbl):
    fig = plt.figure()
    plt.title('{} mean - stoc-base diff'.format(var))
    for th in time_horiz:
        plt.plot(lat, (zonals[(var, th, 'stoc')]-zonals[(var, th, 'base')]), label = th, linewidth = 1.5)
    plt.xlabel('Latitude')
    plt.ylabel(varunits)
    plt.legend()
    plt.grid()
    figures.append(fig)

ctl.plot_pdfpages(figure_file, figures)
plt.close('all')

figure_file = cart_out+'all_zonals_tempave_abs.pdf'
figures = []

for var, varunits in zip(allvarsbl, allvarunitsbl):
    fig = plt.figure()
    plt.title('{} mean - stoc-base diff'.format(var))
    for th in temp_horiz:
        plt.plot(lat, (zonals[(var, th, 'stoc')]-zonals[(var, th, 'base')]), label = '{}-{} C'.format(th[0], th[1]), linewidth = 1.5)
    plt.xlabel('Latitude')
    plt.ylabel(varunits)
    plt.legend()
    plt.grid()
    figures.append(fig)

ctl.plot_pdfpages(figure_file, figures)
plt.close('all')


figure_file = cart_out+'all_zonals_futvspi.pdf'
figures = []

for var, varunits in zip(allvarsbl, allvarunitsbl):
    fig = plt.figure()
    plt.title('{} fut-pi change'.format(var))
    difsto = zonals[(var, 'fut', 'stoc')]-zonals[(var, 'pi', 'stoc')]
    difbas = zonals[(var, 'fut', 'base')]-zonals[(var, 'pi', 'base')]
    plt.plot(lat, difsto, label = 'stoc', linewidth = 1.5)
    plt.plot(lat, difbas, label = 'base', linewidth = 1.5)
    plt.plot(lat, difsto-difbas, label = 'diff (s-b)', linewidth = 1.0, linestyle = '--')
    plt.xlabel('Latitude')
    plt.ylabel(varunits)
    plt.legend()
    plt.grid()
    figures.append(fig)

ctl.plot_pdfpages(figure_file, figures)
plt.close('all')

figure_file = cart_out+'all_zonals_futvspi_temp.pdf'
figures = []

for var, varunits in zip(allvarsbl, allvarunitsbl):
    fig = plt.figure()
    plt.title('{} change btw 14 and 17 C'.format(var))
    difsto = zonals[(var, temp_horiz[3], 'stoc')]-zonals[(var, temp_horiz[0], 'stoc')]
    difbas = zonals[(var, temp_horiz[3], 'base')]-zonals[(var, temp_horiz[0], 'base')]
    plt.plot(lat, difsto, label = 'stoc', linewidth = 1.5)
    plt.plot(lat, difbas, label = 'base', linewidth = 1.5)
    plt.plot(lat, difsto-difbas, label = 'diff (s-b)', linewidth = 1.0, linestyle = '--')
    plt.xlabel('Latitude')
    plt.ylabel(varunits)
    plt.legend()
    plt.grid()
    figures.append(fig)

ctl.plot_pdfpages(figure_file, figures)
plt.close('all')

for lb in lat_bands:
    figure_file = cart_out+'all_serie_{}.pdf'.format(lb)
    figures = []

    var1 = 'rlut'
    all_stoc_1 = np.concatenate([serie[(var1, lb, ens)] for ens in ensmems[3:]])
    all_base_1 = np.concatenate([serie[(var1, lb, ens)] for ens in ensmems[:3]])
    for var2 in ['hcc', 'mcc', 'lcc', 'tcw', 'tas', 'heat_flux']:
        fig = plt.figure()
        ax = plt.subplot(1,1,1)
        plt.title('{} vs {} - {}'.format(var1, var2, lb))
        all_stoc_2 = np.concatenate([serie[(var2, lb, ens)] for ens in ensmems[3:]])
        all_base_2 = np.concatenate([serie[(var2, lb, ens)] for ens in ensmems[:3]])

        #print(var1, var2, all_base_1.shape, all_base_2.shape)
        sc1 = ax.scatter(all_base_1, all_base_2, label = 'base', s = 3)
        sc2 = ax.scatter(all_stoc_1, all_stoc_2, label = 'stoc', s = 3)
        rb = ctl.Rcorr(all_base_1, all_base_2)
        rs = ctl.Rcorr(all_stoc_1, all_stoc_2)
        plt.text(0.1, 0.95, 'R = {:5.2f}'.format(rb), transform = ax.transAxes, color = sc1.get_facecolor()[0])
        plt.text(0.1, 0.9, 'R = {:5.2f}'.format(rs), transform = ax.transAxes, color = sc2.get_facecolor()[0])

        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.legend(loc = 1)
        plt.grid()
        figures.append(fig)

    var1 = 'rsut'
    all_stoc_1 = np.concatenate([serie[(var1, lb, ens)] for ens in ensmems[3:]])
    all_base_1 = np.concatenate([serie[(var1, lb, ens)] for ens in ensmems[:3]])
    for var2 in ['hcc', 'mcc', 'lcc', 'tcw', 'tas', 'heat_flux']:
        fig = plt.figure()
        ax = plt.subplot(1,1,1)
        plt.title('{} vs {} - {}'.format(var1, var2, lb))
        all_stoc_2 = np.concatenate([serie[(var2, lb, ens)] for ens in ensmems[3:]])
        all_base_2 = np.concatenate([serie[(var2, lb, ens)] for ens in ensmems[:3]])

        sc1 = ax.scatter(all_base_1, all_base_2, label = 'base', s = 3)
        sc2 = ax.scatter(all_stoc_1, all_stoc_2, label = 'stoc', s = 3)
        rb = ctl.Rcorr(all_base_1, all_base_2)
        rs = ctl.Rcorr(all_stoc_1, all_stoc_2)
        plt.text(0.1, 0.95, 'R = {:5.2f}'.format(rb), transform = ax.transAxes, color = sc1.get_facecolor()[0])
        plt.text(0.1, 0.9, 'R = {:5.2f}'.format(rs), transform = ax.transAxes, color = sc2.get_facecolor()[0])

        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.legend(loc = 1)
        plt.grid()
        figures.append(fig)

    var1 = 'toa_balance'
    all_stoc_1 = np.concatenate([serie[(var1, lb, ens)] for ens in ensmems[3:]])
    all_base_1 = np.concatenate([serie[(var1, lb, ens)] for ens in ensmems[:3]])
    for var2 in ['tas', 'heat_flux', 'hcc', 'tcw', 'mcc', 'lcc']:
        fig = plt.figure()
        ax = plt.subplot(1,1,1)
        plt.title('{} vs {} - {}'.format(var1, var2, lb))
        all_stoc_2 = np.concatenate([serie[(var2, lb, ens)] for ens in ensmems[3:]])
        all_base_2 = np.concatenate([serie[(var2, lb, ens)] for ens in ensmems[:3]])

        sc1 = ax.scatter(all_base_1, all_base_2, label = 'base', s = 3)
        sc2 = ax.scatter(all_stoc_1, all_stoc_2, label = 'stoc', s = 3)
        rb = ctl.Rcorr(all_base_1, all_base_2)
        rs = ctl.Rcorr(all_stoc_1, all_stoc_2)
        plt.text(0.1, 0.95, 'R = {:5.2f}'.format(rb), transform = ax.transAxes, color = sc1.get_facecolor()[0])
        plt.text(0.1, 0.9, 'R = {:5.2f}'.format(rs), transform = ax.transAxes, color = sc2.get_facecolor()[0])

        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.legend(loc = 1)
        plt.grid()
        figures.append(fig)

    var1 = 'tas'
    all_stoc_1 = np.concatenate([serie[(var1, lb, ens)] for ens in ensmems[3:]])
    all_base_1 = np.concatenate([serie[(var1, lb, ens)] for ens in ensmems[:3]])
    for var2 in ['heat_flux', 'tcw', 'hcc', 'mcc', 'lcc']:
        fig = plt.figure()
        ax = plt.subplot(1,1,1)
        plt.title('{} vs {} - {}'.format(var1, var2, lb))
        all_stoc_2 = np.concatenate([serie[(var2, lb, ens)] for ens in ensmems[3:]])
        all_base_2 = np.concatenate([serie[(var2, lb, ens)] for ens in ensmems[:3]])

        sc1 = ax.scatter(all_base_1, all_base_2, label = 'base', s = 3)
        sc2 = ax.scatter(all_stoc_1, all_stoc_2, label = 'stoc', s = 3)
        rb = ctl.Rcorr(all_base_1, all_base_2)
        rs = ctl.Rcorr(all_stoc_1, all_stoc_2)
        plt.text(0.1, 0.95, 'R = {:5.2f}'.format(rb), transform = ax.transAxes, color = sc1.get_facecolor()[0])
        plt.text(0.1, 0.9, 'R = {:5.2f}'.format(rs), transform = ax.transAxes, color = sc2.get_facecolor()[0])

        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.legend(loc = 1)
        plt.grid()
        figures.append(fig)

    ctl.plot_pdfpages(figure_file, figures)
    plt.close('all')
