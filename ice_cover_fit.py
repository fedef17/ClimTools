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
import iris
###############################################

cart_in = '/data/fabiano/SPHINX/ice_cover/'
cart_out = '/home/fabiano/Research/lavori/SPHINX_for_lisboa/ice_cover_virna/'
if not os.path.exists(cart_out): os.mkdir(cart_out)

filista = os.listdir(cart_in)

icec = dict()
for seas in ['mam', 'son']:
    icefils = [fi for fi in filista if seas in fi]
    for fi in icefils:
        coso = iris.load(cart_in+fi)
        for cos in coso:
            varna = cos.var_name
            if varna[-1] == 'm': continue
            icec[(seas, varna)] = cos.data.squeeze()

tasfil = [fi for fi in filista if 'tas' in fi][0]
tas = dict()
coso = iris.load(cart_in + tasfil)
for cos in coso:
    varna = cos.var_name
    tas[varna] = cos.data.squeeze()

# prima il caso son: fitto una retta per ognuno e vedo come viene il plot e se ci sono differenze significative tra uno e l'altro.
expmems = ['lcb0', 'lcb1', 'lcb2', 'lcs0', 'lcs1', 'lcs2']

print('\n\n---------- DOUBLE CUT FIT ----------------------\n')

figures = []
axes = []
#plt.ion()

p0s = dict()
p0s[('mam', 'lcs')] = (-5e11, -2.5e12, 1.e13, 15)
p0s[('mam', 'lcb')] = (-5e11, -2.5e12, -5e12, 1.e13, 15.5, 17.5)
p0s[('son', 'lcs')] = (-5e11, -2.5e12, 1.e13, 15)
p0s[('son', 'lcb')] = (-5e11, -2.5e12, -1.e9, 1.e13, 15, 17)

results = dict()

for seas in ['mam', 'son']:
    fig = plt.figure(figsize=(24, 12), dpi=150)
    for i, exp in enumerate(expmems):
        n_cut = 1
        if exp[:-1] == 'lcb': n_cut = 2

        x = tas[exp]
        y = icec[(seas, exp)]

        xcuts, lines = ctl.cutline2_fit(x, y, n_cut = n_cut, approx_par = p0s[(seas, exp[:-1])])
        results[(seas, exp)] = (np.array(xcuts), np.array([line[0] for line in lines]), lines[0][1])

        #print(seas, exp, xcuts)
        xlin = np.linspace(min(x)-0.05*(max(x)-min(x)),max(x)+0.05*(max(x)-min(x)),11)

        xlins = []
        if n_cut == 1:
            xlins.append(np.linspace(xlin[0], xcuts[0], 5))
            xlins.append(np.linspace(xcuts[0], xlin[-1], 5))
        elif n_cut == 2:
            xlins.append(np.linspace(xlin[0], xcuts[0], 5))
            xlins.append(np.linspace(xcuts[0], xcuts[1], 5))
            xlins.append(np.linspace(xcuts[1], xlin[-1], 5))

        ax = fig.add_subplot(2,3,i+1)
        plt.grid()
        ax.scatter(x, y, label='Data', color = 'lightsteelblue', s=6)
        for i, (xli, line) in enumerate(zip(xlins, lines)):
            ax.plot(xli, xli*line[0]+line[1], label='y{} = {:9.2e} x + {:9.2e}'.format(i, line[0], line[1]), linewidth = 1.5)
        plt.legend(fontsize = 'small', loc = 3)
        if n_cut == 1:
            plt.title("{} - {} - T_cut = {:6.2f} C".format(exp, seas, xcuts[0]))
        elif n_cut == 2:
            plt.title("{} - {} - T_cuts = {:6.2f}, {:6.2f} C".format(exp, seas, xcuts[0], xcuts[1]))
        axes.append(ax)
    plt.tight_layout()
    figures.append(fig)

ctl.adjust_ax_scale(axes, sel_axis = 'both')
for fig, seas in zip(figures, ['mam', 'son']):
    fig.savefig(cart_out+'icec_{}_fit_doublecut.pdf'.format(seas))

for seas in ['mam', 'son']:
    for cos in ['lcb', 'lcs']:
        agag = [results[(seas, exp)][0] for exp in expmems if cos in exp]
        xcuts = np.mean(agag, axis = 0)
        delta_xcuts = np.std(agag, axis = 0)
        agag = [results[(seas, exp)][1] for exp in expmems if cos in exp]
        agag = np.array(agag)/1.e6
        ms = np.mean(agag, axis = 0)
        delta_ms = np.std(agag, axis = 0)
        agag = [results[(seas, exp)][2] for exp in expmems if cos in exp]
        agag = np.array(agag)/1.e6
        c1 = np.mean(agag, axis = 0)
        delta_c1 = np.std(agag, axis = 0)
        if cos == 'lcb':
            print('------- {} - {} ------------\n'.format(seas, cos))
            print('Cuts: {:8.2f} pm {:8.2f} C, {:8.2f} pm {:8.2f} C \n'.format(xcuts[0], delta_xcuts[0], xcuts[1], delta_xcuts[1]))
            print('Slopes: {:9.4e} pm {:9.4e} km2/C, {:9.4e} pm {:9.4e} km2/C, {:9.4e} pm {:9.4e} km2/C \n'.format(ms[0], delta_ms[0], ms[1], delta_ms[1], ms[2], delta_ms[2]))
            print('Intercept: {:9.4e} pm {:9.4e} km2 \n'.format(c1, delta_c1))
        elif cos == 'lcs':
            print('------- {} - {} ------------\n'.format(seas, cos))
            print('Cut: {:8.2f} pm {:8.2f} C \n'.format(xcuts[0], delta_xcuts[0]))
            print('Slopes: {:9.4e} pm {:9.4e} km2/C, {:9.4e} pm {:9.4e} km2/C \n'.format(ms[0], delta_ms[0], ms[1], delta_ms[1]))
            print('Intercept: {:9.4e} pm {:9.4e} km2 \n'.format(c1, delta_c1))


print('\n\n---------- SINGLE CUT FIT ----------------------\n')

figures = []
axes = []
#plt.ion()

p0s = dict()
p0s[('mam', 'lcs')] = (-2.5e12, 1.e13)
p0s[('mam', 'lcb')] = (-2.5e12, -5e12, 1.e13, 17.5)
p0s[('son', 'lcs')] = (-2.5e12, 1.e13)
p0s[('son', 'lcb')] = (-2.5e12, -1.e9, 1.e13, 17)

results = dict()

for seas in ['mam', 'son']:
    fig = plt.figure(figsize=(24, 12), dpi=150)
    for i, exp in enumerate(expmems):
        n_cut = 0
        if exp[:-1] == 'lcb': n_cut = 1

        x = tas[exp]
        y = icec[(seas, exp)]

        xcuts, lines = ctl.cutline2_fit(x, y, n_cut = n_cut, approx_par = p0s[(seas, exp[:-1])])
        results[(seas, exp)] = (np.array(xcuts), np.array([line[0] for line in lines]), lines[0][1])

        #print(seas, exp, xcuts)
        xlin = np.linspace(min(x)-0.05*(max(x)-min(x)),max(x)+0.05*(max(x)-min(x)),11)

        xlins = []
        if n_cut == 0:
            xlins.append(xlin)
        elif n_cut == 1:
            xlins.append(np.linspace(xlin[0], xcuts[0], 5))
            xlins.append(np.linspace(xcuts[0], xlin[-1], 5))
        elif n_cut == 2:
            xlins.append(np.linspace(xlin[0], xcuts[0], 5))
            xlins.append(np.linspace(xcuts[0], xcuts[1], 5))
            xlins.append(np.linspace(xcuts[1], xlin[-1], 5))

        ax = fig.add_subplot(2,3,i+1)
        plt.grid()
        ax.scatter(x, y, label='Data', color = 'lightsteelblue', s=6)
        for i, (xli, line) in enumerate(zip(xlins, lines)):
            ax.plot(xli, xli*line[0]+line[1], label='y{} = {:9.2e} x + {:9.2e}'.format(i, line[0], line[1]), linewidth = 1.5)
        plt.legend(fontsize = 'small', loc = 3)
        if n_cut == 1:
            plt.title("{} - {} - T_cut = {:6.2f} C".format(exp, seas, xcuts[0]))
        elif n_cut == 2:
            plt.title("{} - {} - T_cuts = {:6.2f}, {:6.2f} C".format(exp, seas, xcuts[0], xcuts[1]))
        axes.append(ax)
    plt.tight_layout()
    figures.append(fig)

ctl.adjust_ax_scale(axes, sel_axis = 'both')
for fig, seas in zip(figures, ['mam', 'son']):
    fig.savefig(cart_out+'icec_{}_fit_singlecut.pdf'.format(seas))

for seas in ['mam', 'son']:
    for cos in ['lcb', 'lcs']:
        agag = [results[(seas, exp)][0] for exp in expmems if cos in exp]
        xcuts = np.mean(agag, axis = 0)
        delta_xcuts = np.std(agag, axis = 0)
        agag = [results[(seas, exp)][1] for exp in expmems if cos in exp]
        agag = np.array(agag)/1.e6
        ms = np.mean(agag, axis = 0)
        delta_ms = np.std(agag, axis = 0)
        agag = [results[(seas, exp)][2] for exp in expmems if cos in exp]
        agag = np.array(agag)/1.e6
        c1 = np.mean(agag, axis = 0)
        delta_c1 = np.std(agag, axis = 0)
        if cos == 'lcb':
            print('------- {} - {} ------------\n'.format(seas, cos))
#            print('Cuts: {:8.2f} pm {:8.2f} C, {:8.2f} pm {:8.2f} C \n'.format(xcuts[0], delta_xcuts[0], xcuts[1], delta_xcuts[1]))
#            print('Slopes: {:9.4e} pm {:9.4e} km2/C, {:9.4e} pm {:9.4e} km2/C, {:9.4e} pm {:9.4e} km2/C \n'.format(ms[0], delta_ms[0], ms[1], delta_ms[1], ms[2], delta_ms[2]))
            print('Cuts: {:8.2f} pm {:8.2f} C \n'.format(xcuts[0], delta_xcuts[0]))
            print('Slopes: {:9.4e} pm {:9.4e} km2/C, {:9.4e} pm {:9.4e} km2/C \n'.format(ms[0], delta_ms[0], ms[1], delta_ms[1]))
            print('Intercept: {:9.4e} pm {:9.4e} km2 \n'.format(c1, delta_c1))
        elif cos == 'lcs':
            print('------- {} - {} ------------\n'.format(seas, cos))
#            print('Cut: {:8.2f} pm {:8.2f} C \n'.format(xcuts[0], delta_xcuts[0]))
#            print('Slopes: {:9.4e} pm {:9.4e} km2/C, {:9.4e} pm {:9.4e} km2/C \n'.format(ms[0], delta_ms[0], ms[1], delta_ms[1]))
            print('Slopes: {:9.4e} pm {:9.4e} km2/C \n'.format(ms[0], delta_ms[0]))
            print('Intercept: {:9.4e} pm {:9.4e} km2 \n'.format(c1, delta_c1))

# CLIMATE SENSITIVITY
print('\n\n-----------------  CLIMATE SENSITIVITY --------------------\n')

for delta in [10,20,30]:
    print('-----------------  {} yr window --------------------\n'.format(delta))
    clim_sens = dict()
    for exp in expmems:
        t_pi = np.mean(tas[exp][:delta], axis = 0)
        t_fu = np.mean(tas[exp][-delta:], axis = 0)
        clim_sens[exp] = t_fu - t_pi

    for cos in ['lcb', 'lcs']:
        agag = [clim_sens[exp] for exp in expmems if cos in exp]
        cs = np.mean(agag)
        delta_cs = np.std(agag)
        print('--------------- {} ------------\n'.format(cos))
        print('T diff {}-2100 to 1850-{}: {:8.2f} pm {:6.2f}\n'.format(2100-delta, 1850+delta, cs, delta_cs))

