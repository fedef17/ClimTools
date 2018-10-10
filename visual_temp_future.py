#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import netCDF4 as nc
import climtools_lib as ctl
import pandas as pd
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib import cm
import pickle

import matplotlib.animation as animation
from matplotlib.animation import ImageMagickFileWriter

import cartopy.crs as ccrs

#cart = '/home/fabiano/Research/lavori/SPHINX_for_lisboa/'
cart = '/home/fedefab/Scrivania/Research/Post-doc/SPHINX/'

ref_period = ctl.range_years(1850,1900)

filena = 'lcb0-1850-2100-tas_mon.nc'

var, lat, lon, dates, time_units, var_units = ctl.read3Dncfield(cart+filena)
dates_pdh = pd.to_datetime(dates)

# Global stuff
global_mean = ctl.global_mean(var, lat)
zonal_mean = ctl.zonal_mean(var)

climat_mon, dates_mon, climat_std = ctl.monthly_climatology(var, dates, dates_range = ref_period)
climat_year = np.mean(climat_mon, axis = 0)

yearly_anom, years = ctl.yearly_average(var, dates)
yearly_anom = yearly_anom - climat_year

zonal_anom = ctl.zonal_mean(yearly_anom)
global_anom = ctl.global_mean(yearly_anom, lat)

del var

# GIF animation
plt.ion()
fig = plt.figure(figsize=(8,6), dpi=150)

years_pdh = pd.to_datetime(years)
anni = years_pdh.year

lonstep = 10.
#proj = ccrs.Orthographic(central_longitude=0., central_latitude=30.)
proj = ccrs.PlateCarree()
ax = plt.subplot(projection = proj)
cmappa = cm.get_cmap('RdBu_r')
cbar_range = ctl.get_cbar_range(yearly_anom, symmetrical = True)
clevels = np.linspace(cbar_range[0], cbar_range[1], 21)
cset = ctl.color_set(len(anni), bright_thres = 0., full_cb_range = True)

def animate(i):
    proj = ccrs.PlateCarree()
    #proj = ccrs.Orthographic(central_longitude=lonstep*i, central_latitude=30.)
    ax = plt.subplot(projection = proj)
    ax.clear()
    map_plot = ctl.plot_mapc_on_ax(ax, yearly_anom[i], lat, lon, proj, cmappa, cbar_range)
    year = anni[i]
    color = cset[i]
    tit.set_text('{}'.format(year))
    #showdate.set_text('{}'.format(year))#, color = color)
    #showdate.update(color = color)
    ax.relim()
    ax.autoscale_view()
    return

# Plotting figure
map_plot = ctl.plot_mapc_on_ax(ax, yearly_anom[0], lat, lon, proj, cmappa, cbar_range)

cax = plt.axes([0.1, 0.11, 0.8, 0.05]) #horizontal
cb = plt.colorbar(map_plot, cax=cax, orientation='horizontal')#, labelsize=18)
cb.ax.tick_params(labelsize=14)
cb.set_label('Temp anomaly (K)', fontsize=16)

top    = 0.88  # the top of the subplots
bottom = 0.20    # the bottom of the subplots
left   = 0.02    # the left side
right  = 0.98  # the right side
hspace = 0.20   # height reserved for white space
wspace = 0.05    # width reserved for blank space
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

tit = plt.title('1850')
#showdate = ax.text(0.5, 0.95, '1850', fontweight = 'bold', color = cset[0], bbox=dict(facecolor='lightsteelblue', edgecolor='black', boxstyle='round,pad=1'))

save = True
if save:
    metadata = dict(title='Temperature anomaly (SPHINX lcb0 experiment)', artist='Federico Fabiano')
    writer = ImageMagickFileWriter(fps = 10, metadata = metadata)#, frame_size = (1200, 900))
    with writer.saving(fig, cart + "Temp_anomaly_animation_flat.gif", 150):
        for i, (year, col) in enumerate(zip(anni, cset)):
            print(year)
            animate(i)
            writer.grab_frame()
else:
    line_ani = animation.FuncAnimation(fig, animate, len(anni), interval=100, blit=False)
