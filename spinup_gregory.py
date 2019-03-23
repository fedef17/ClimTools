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

cart_in = '/data-hobbes/fabiano/SPHINX/SPINUP/'
cart_out = '/home/fabiano/Research/lavori/SPHINX_for_lisboa/spinup_gregory/'
if not os.path.exists(cart_out): os.mkdir(cart_out)

ctl.openlog(cart_out, redirect_stderr = False)

file_stoc = '/data-hobbes/fabiano/SPHINX/SPINUP/sss0/atmosphere/sss0_1850_2169_time-series_atmo.nc'
file_base = '/data-hobbes/fabiano/SPHINX/SPINUP/ssb0/atmosphere/ssb0_1850_2169_time-series_atmo.nc'

data = dict()
data['stoc'] = iris.load(file_stoc)
data['base'] = iris.load(file_base)

#plt.ion()

series = dict()

col = dict()
col['base'] = ['lightgreen', 'forestgreen']
col['stoc'] = ['lightcoral', 'indianred']

for var2 in ['NetTOA', 'rlut', 'rsut']:
    print('\n\n\n ----------------- {} ------------------ \n\n'.format(var2))
    axes = []
    axes2 = []
    axes1 = []
    axes3 = []
    fig6 = plt.figure(figsize=(8, 6), dpi=150)
    axtot3 = fig6.add_subplot(111)
    fig5 = plt.figure(figsize=(8, 6), dpi=150)
    axtot2 = fig5.add_subplot(111)
    fig4 = plt.figure(figsize=(8, 6), dpi=150)
    axtot = fig4.add_subplot(111)
    fig3 = plt.figure(figsize=(16, 6), dpi=150)
    fig2 = plt.figure(figsize=(16, 6), dpi=150)
    fig = plt.figure(figsize=(16, 6), dpi=150)

    linesall = dict()
    linesall_y = dict()

    for i, exp in enumerate(['stoc', 'base']):
        gigi_names = [gi.var_name for gi in data[exp]]
        gigidict = dict(zip(gigi_names, data[exp]))
        tas = gigidict['tas']
        netTOA = gigidict[var2]

        time_mon = tas.coord('time').points
        # dates = time.units.num2date(time.points)
        #
        # tas_yr = ctl.yearly_average(tas, dates)
        # netTOA_yr = ctl.yearly_average(netTOA, dates)

        x_mon = tas.data
        y_mon = netTOA.data
        x = []
        y = []
        time = []
        for gg in range(len(time_mon)/12):
            time.append(np.mean(time_mon[12*gg:12*gg+12]))
            x.append(np.mean(x_mon[12*gg:12*gg+12]))
            y.append(np.mean(y_mon[12*gg:12*gg+12]))

        x = np.array(x)
        y = np.array(y)
        time = np.array(time)

        line = ctl.linear_regre_witherr(x, y)
        linefirst = ctl.linear_regre_witherr(x[:50], y[:50])
        line_y = ctl.linear_regre_witherr(y, x)
        linefirst_y = ctl.linear_regre_witherr(y[:50], x[:50])

        xlin = np.linspace(min(x)-0.05*(max(x)-min(x)),max(x)+0.05*(max(x)-min(x)),11)
        #ylin = np.linspace(min(y)-0.05*(max(y)-min(y)),max(y)+0.05*(max(y)-min(y)),11)

        coso10x = np.array([np.mean(x[10*j:10*(j+1)]) for j in range(len(x)/10)])
        coso10y = np.array([np.mean(y[10*j:10*(j+1)]) for j in range(len(y)/10)])

        line3 = ctl.linear_regre_witherr(coso10x, coso10y)
        line3first = ctl.linear_regre_witherr(coso10x[:5], coso10y[:5])
        line3_y = ctl.linear_regre_witherr(coso10y, coso10x)
        line3first_y = ctl.linear_regre_witherr(coso10y[:5], coso10x[:5])

        coso25x = np.array([np.mean(x[25*j:25*(j+1)]) for j in range(len(x)/25)])
        coso25y = np.array([np.mean(y[25*j:25*(j+1)]) for j in range(len(y)/25)])
        line4 = ctl.linear_regre_witherr(coso25x, coso25y)
        line4_y = ctl.linear_regre_witherr(coso25y, coso25x)

        ax = fig.add_subplot(1,2,i+1)
        plt.grid()
        ax.scatter(x, y, label='Data', color = 'lightsteelblue', s=4)
        # ax.scatter(coso5x, coso5y, label='5 yr mean', color = 'lightcoral', s=6)
        ax.scatter(coso10x, coso10y, label='10 yr mean', color = 'lightcoral', s=10, marker = 'D')
        ax.plot(xlin, xlin*line[0]+line[1], label='slope = {:5.2f} +/- {:4.2f} (sl. yx = {:5.2f} +/- {:4.2f} )'.format(line[0], line[2], 1/line[0], (line[2]/line[0])*1/line[0]), linewidth = 1.5, color = 'steelblue')
        ax.plot(xlin, xlin*line3[0]+line3[1], label='slope = {:5.2f} +/- {:4.2f} (sl. yx = {:5.2f} +/- {:4.2f} )'.format(line3[0], line3[2], 1/line3[0], (line3[2]/line3[0])*1/line3[0]), linewidth = 1.5, color = 'indianred')
        ylin = (xlin-line_y[1])/line_y[0]
        ax.plot(ylin*line_y[0]+line_y[1], ylin, label='slope = {:5.2f} +/- {:4.2f} (sl. yx = {:5.2f} +/- {:4.2f} )'.format(1/line_y[0], (line_y[2]/line_y[0])*1/line_y[0], line_y[0], line_y[2]), linewidth = 1.5, color = 'steelblue', linestyle = '--')
        # ax.plot(xlin, xlin*line2[0]+line2[1], label='slope = {:8.3f} +/- {:5.2f}'.format(line2[0], line2[2]), linewidth = 1.5, color = 'indianred')
        ylin = (xlin-line3_y[1])/line3_y[0]
        ax.plot(ylin*line3_y[0]+line3_y[1], ylin, label='slope = {:5.2f} +/- {:4.2f} (sl. yx = {:5.2f} +/- {:4.2f} )'.format(1/line3_y[0], (line3_y[2]/line3_y[0])*1/line3_y[0], line3_y[0], line3_y[2]), linewidth = 1.5, color = 'indianred', linestyle = '--')
        plt.legend(fontsize = 'small', loc = 3)
        axes.append(ax)
        ax.set_title(exp)

        ax3 = fig3.add_subplot(1,2,i+1)
        ax3.scatter(coso10x, coso10y, label='10 yr mean', color = 'lightcoral', s=4)
        ax3.scatter(coso25x, coso25y, label='25 yr mean', color = 'lightgreen', s=10, marker = 'D')
        ax3.plot(xlin, xlin*line3[0]+line3[1], label='slope = {:5.2f} +/- {:4.2f} (sl. yx = {:5.2f} +/- {:4.2f} )'.format(line3[0], line3[2], 1/line3[0], (line3[2]/line3[0])*1/line3[0]), linewidth = 1.5, color = 'indianred')
        ax3.plot(xlin, xlin*line4[0]+line4[1], label='slope = {:5.2f} +/- {:4.2f}  (sl. yx = {:5.2f} +/- {:4.2f} )'.format(line4[0], line4[2],  1/line4[0], (line4[2]/line4[0])*1/line4[0]), linewidth = 1.5, color = 'forestgreen')
        ylin = (xlin-line3_y[1])/line3_y[0]
        ax3.plot(ylin*line3_y[0]+line3_y[1], ylin, label='slope = {:5.2f} +/- {:4.2f} (sl. yx = {:5.2f} +/- {:4.2f} )'.format(1/line3_y[0], (line3_y[2]/line3_y[0])*1/line3_y[0], line3_y[0], line3_y[2]), linewidth = 1.5, color = 'indianred', linestyle = '--')
        ylin = (xlin-line4_y[1])/line4_y[0]
        ax3.plot(ylin*line4_y[0]+line4_y[1], ylin, label='slope = {:5.2f} +/- {:4.2f} (sl. yx = {:5.2f} +/- {:4.2f} )'.format(1/line4_y[0], (line4_y[2]/line4_y[0])*1/line4_y[0], line4_y[0], line4_y[2]), linewidth = 1.5, color = 'forestgreen', linestyle = '--')
        ax3.legend(fontsize = 'small', loc = 3)
        axes3.append(ax3)
        ax3.set_title(exp)

        axtot.scatter(coso10x, coso10y, label=exp+' 10 yr', color = col[exp][0], s=8, marker = 'D')
        axtot.plot(xlin, xlin*line3[0]+line3[1], label='slope = {:8.3f} +/- {:5.2f}'.format(line3[0], line3[2]), linewidth = 1.5, color = col[exp][1])

        axtot3.scatter(coso10x[5:], coso10y[5:], label=exp+' 10 yr', color = col[exp][0], s=6, marker = 'D')
        axtot3.scatter(coso10x[:5], coso10y[:5], color = col[exp][0], s=10, marker = 'D')
        axtot3.plot(xlin, xlin*line3first[0]+line3first[1], label='slope = {:8.3f} +/- {:5.2f}'.format(line3first[0], line3first[2]), linewidth = 1.5, color = col[exp][1])

        axtot2.scatter(x[50:], y[50:], label=exp, color = col[exp][0], s=2)
        axtot2.scatter(x[:50], y[:50], color = col[exp][0], s=8, marker = '*')
        axtot2.plot(xlin, xlin*linefirst[0]+linefirst[1], label='slope = {:8.3f} +/- {:5.2f}'.format(linefirst[0], linefirst[2]), linewidth = 1.5, color = col[exp][1])

        ax1 = fig2.add_subplot(1,2,i+1)
        print(i)
        ax1.scatter(time, x, label='tas', color = 'lightsteelblue', s=3)
        coso = ctl.running_mean(x, 10)
        ax1.plot(time, coso, color = 'steelblue')
        axes1.append(ax1)
        ax1.set_title(exp)

        ax2 = ax1.twinx()
        ax2.scatter(time, y, label=var2, color = 'lightcoral', s=3)
        coso = ctl.running_mean(y, 10)
        ax2.plot(time, coso, color = 'indianred')
        axes2.append(ax2)
        #for i, (xli, line) in enumerate(zip(xlins, lines)):
        #    ax.plot(xli, xli*line[0]+line[1], label='y{} = {:9.2e} x + {:9.2e}'.format(i, line[0], line[1]), linewidth = 1.5)
        ax1.legend(fontsize = 'small', loc = 3)
        ax2.legend(fontsize = 'small', loc = 4)
        #axes.append(ax)

        alllines = [line, line3, line4, linefirst, line3first]
        alllines_y = [line_y, line3_y, line4_y, linefirst_y, line3first_y]

        linesall[exp] = alllines
        linesall_y[exp] = alllines_y

    alltitles = ['Yearly', '10 years', '25 years', 'Yearly [:50]', '10 years [:50]']
    print('\n\n-------------- {} -----------------\n'.format(var2))

    titleform1 = '{:20s}||  {:31s}   ||   {:31s}   ||   {:31s}   ||   {:31s}\n'.format('Type', 'Sl. xy (W m-2 K-1)', 'Sl. yx (K W-1 m2)', 'ECS_xy (K)', 'ECS_yx (K)')
    titleform2 = '{:20s}||  {:14s}   {:14s}   ||   {:14s}   {:14s}   ||   {:14s}   {:14s}   ||   {:14s}   {:14s}\n'.format('', 'base', 'stoc', 'base', 'stoc', 'base', 'stoc', 'base', 'stoc')
    stringform = '{:20s}||  {:5.2f} +/- {:4.2f}   {:5.2f} +/- {:4.2f}   ||   {:5.2f} +/- {:4.2f}   {:5.2f} +/- {:4.2f}   ||   {:5.2f} +/- {:4.2f}   {:5.2f} +/- {:4.2f}   ||   {:5.2f} +/- {:4.2f}   {:5.2f} +/- {:4.2f}\n'
    stringform2 = '{:20s}||  {:5.2f} +/- {:4.2f}   {:5.2f} +/- {:4.2f}   ||   {:5.2f} +/- {:4.2f}   {:5.2f} +/- {:4.2f}\n'
    print(titleform1)
    print(titleform2)
    for num, tit in enumerate(alltitles):
        lix_base = linesall['base'][num]
        liy_base = linesall_y['base'][num]
        lix_stoc = linesall['stoc'][num]
        liy_stoc = linesall_y['stoc'][num]
        if var2=='NetTOA':
            ecsx_base = abs(3.7/lix_base[0])
            ecsx_baseerr = abs(lix_base[2]/lix_base[0])*ecsx_base
            ecsy_base = abs(3.7*liy_base[0])
            ecsy_baseerr = 3.7*abs(liy_base[2])
            ecsx_stoc = abs(3.7/lix_stoc[0])
            ecsx_stocerr = abs(lix_stoc[2]/lix_stoc[0])*ecsx_stoc
            ecsy_stoc = abs(3.7*liy_stoc[0])
            ecsy_stocerr = 3.7*abs(liy_stoc[2])

            print(stringform.format(tit, lix_base[0], lix_base[2], lix_stoc[0], lix_stoc[2], liy_base[0], liy_base[2], liy_stoc[0], liy_stoc[2], ecsx_base, ecsx_baseerr, ecsx_stoc, ecsx_stocerr, ecsy_base, ecsy_baseerr, ecsy_stoc, ecsy_stocerr))
        else:
            print(stringform2.format(tit, lix_base[0], lix_base[2], lix_stoc[0], lix_stoc[2], liy_base[0], liy_base[2], liy_stoc[0], liy_stoc[2]))

    axtot.legend()
    axtot.grid()
    axtot.set_title('Regression on 10 yr mean')
    axtot.set_xlabel('TAS (C)')
    axtot.set_ylabel(var2+' (W/m2)')

    axtot2.legend()
    axtot2.grid()
    axtot2.set_title('Regression on first 50 years only')
    axtot2.set_xlabel('TAS (C)')
    axtot2.set_ylabel(var2+' (W/m2)')

    axtot3.legend()
    axtot3.grid()
    axtot3.set_title('Regression on first 5 points of 10 yr mean')
    axtot3.set_xlabel('TAS (C)')
    axtot3.set_ylabel(var2+' (W/m2)')

    plt.tight_layout()
    ctl.adjust_ax_scale(axes, sel_axis = 'both')
    ctl.adjust_ax_scale(axes1, sel_axis = 'both')
    ctl.adjust_ax_scale(axes2, sel_axis = 'both')
    ctl.adjust_ax_scale(axes3, sel_axis = 'both')

    fig.savefig(cart_out+'{}_greg_base_vs_stoc_invslope.pdf'.format(var2))
    fig3.savefig(cart_out+'{}_greg2_base_vs_stoc_invslope.pdf'.format(var2))
    # fig2.savefig(cart_out+'{}_timeseries_base_vs_stoc.pdf'.format(var2))
    # fig4.savefig(cart_out+'{}_gregall10_base_vs_stoc.pdf'.format(var2))
    # fig5.savefig(cart_out+'{}_gregfirst_base_vs_stoc.pdf'.format(var2))
    # fig6.savefig(cart_out+'{}_gregfirst10_base_vs_stoc.pdf'.format(var2))

    plt.close('all')
