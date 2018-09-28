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

cart = '/home/fabiano/Research/lavori/SPHINX_for_lisboa/'

cart_data = '/home/fabiano/DATA/SPHINX/tas_mon/'

ensmem = ['lcb0','lcb1','lcb2','lcs0','lcs1','lcs2']

zonal_anom_ens = []
global_anom_ens = []
med_anom_ens = []
med_anom_sum_ens = []
yearly_anom_ens = []
month_climat_ens = []

ref_period = ctl.range_years(1850,1900)

# for ens in ensmem:
#     filena = ens+'-1850-2100-tas_mon.nc'
#
#     var, lat, lon, dates, time_units, var_units = ctl.read3Dncfield(cart_data+filena)
#     dates_pdh = pd.to_datetime(dates)
#
#     # Global stuff
#     global_mean = ctl.global_mean(var, lat)
#     zonal_mean = ctl.zonal_mean(var)
#
#     climat_mon, dates_mon, climat_std = ctl.monthly_climatology(var, dates, dates_range = ref_period)
#     climat_year = np.mean(climat_mon, axis = 0)
#
#     climat_mons_future = []
#     climat_mons_future.append(climat_mon)
#     ann = [1990,2020,2050,2080,2100]
#     for a,b in zip(ann[:-1], ann[1:]):
#         cli, _, _ = ctl.monthly_climatology(var, dates, dates_range = ctl.range_years(a,b))
#         climat_mons_future.append(cli)
#     climat_mons_future = np.stack(climat_mons_future)
#
#     yearly_anom, years = ctl.yearly_average(var, dates)
#     yearly_anom = yearly_anom - climat_year
#
#     zonal_anom = ctl.zonal_mean(yearly_anom)
#     global_anom = ctl.global_mean(yearly_anom, lat)
#
#     month_anom = ctl.anomalies_monthly(var, dates, climat_mean = climat_mon, dates_climate_mean = dates_mon)
#
#     # On the Mediterranean area
#     var_med, lat_med, lon_med = ctl.sel_area(lat,lon,var,'Med')
#
#     climat_monmed, dates_monmed, climat_stdmed = ctl.monthly_climatology(var_med, dates, dates_range = ref_period)
#     climat_year_med = np.mean(climat_monmed, axis = 0)
#
#     yearly_anom_med, years = ctl.yearly_average(var_med, dates)
#     yearly_anom_med = yearly_anom_med - climat_year_med
#     med_anom = ctl.global_mean(yearly_anom_med, lat_med)
#
#     month_anom_med = ctl.anomalies_monthly(var_med, dates, climat_mean = climat_monmed, dates_climate_mean = dates_monmed)
#
#     # On Mediterranean, summer
#     var_med_sum, dates_sum = ctl.sel_season(var_med, dates, 'JJA')
#     med_anom_sumbl, years = ctl.yearly_average(var_med_sum, dates_sum)
#     sum_mean_med, _ = ctl.sel_season(climat_monmed, dates_monmed, 'JJA')
#     med_anom_sumbl = med_anom_sumbl - np.mean(sum_mean_med, axis = 0)
#     med_anom_sum = ctl.global_mean(med_anom_sumbl, lat_med)
#
#     zonal_anom_ens.append(zonal_anom)
#     global_anom_ens.append(global_anom)
#     med_anom_ens.append(med_anom)
#     med_anom_sum_ens.append(med_anom_sum)
#     yearly_anom_ens.append(yearly_anom)
#     #month_climat_ens.append(climat_mons_future)
#
#     del var, var_med, var_med_sum, month_anom, med_anom
#
#
# zonal_anom_ens = np.stack(zonal_anom_ens)
# global_anom_ens = np.stack(global_anom_ens)
# med_anom_ens = np.stack(med_anom_ens)
# med_anom_sum_ens = np.stack(med_anom_sum_ens)
# yearly_anom_ens = np.stack(yearly_anom_ens)
# #month_climat_ens = np.stack(month_climat_ens)
#

# LEGGO DAL PICKLE 
# pickle.dump([lat, lon, years, zonal_anom_ens, global_anom_ens, yearly_anom_ens], open(cart+'temp_proj.p','w'))
lat, lon, years, zonal_anom_ens, global_anom_ens, yearly_anom_ens = pickle.load(open(cart+'temp_proj.p'))




zonal_anom_b = np.mean(zonal_anom_ens[:3, ...], axis = 0)
zonal_anom_s = np.mean(zonal_anom_ens[3:, ...], axis = 0)
global_anom_b = np.mean(global_anom_ens[:3, ...], axis = 0)
global_anom_s = np.mean(global_anom_ens[3:, ...], axis = 0)
# med_anom_b = np.mean(med_anom_ens[:3, ...], axis = 0)
# med_anom_s = np.mean(med_anom_ens[3:, ...], axis = 0)
# med_anom_sum_b = np.mean(med_anom_sum_ens[:3, ...], axis = 0)
# med_anom_sum_s = np.mean(med_anom_sum_ens[3:, ...], axis = 0)
yearly_anom_b = np.mean(yearly_anom_ens[:3, ...], axis = 0)
yearly_anom_s = np.mean(yearly_anom_ens[3:, ...], axis = 0)
#monthly_climat_b = np.mean(month_climat_ens[:3, ...], axis = 0)
#monthly_climat_s = np.mean(month_climat_ens[3:, ...], axis = 0)

#plt.ion()
fig = plt.figure()
years_pdh = pd.to_datetime(years)

cset = ctl.color_set(len(years), bright_thres = 0., full_cb_range = True)
for year, col in zip(years_pdh.year, cset):
    plt.plot(lat, zonal_anom_b[years_pdh.year == year].squeeze(), color = col)

plt.xlabel('Latitude')
plt.ylabel('Temperature anomaly [K]')

fig.subplots_adjust(right=0.8)
plt.grid()
plt.xlim(-90,90)
plt.ylim(-3.,18.)
plt.title('Zonal temperature anomaly (base)')

cbar_ax = fig.add_axes([0.83, 0.12, 0.05, 0.7])

norm = mpl_colors.Normalize(vmin=years_pdh.year[0],vmax=years_pdh.year[-1])
sm = plt.cm.ScalarMappable(cmap=cm.get_cmap('nipy_spectral'), norm=norm)
sm.set_array([0.05, 0.95])
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label('Year')

fig.savefig(cart+'Zonal_temp_anomaly_lcb.pdf')


diff = zonal_anom_b-zonal_anom_s
coslat = abs(np.cos(np.deg2rad(lat)))

years_pdh = pd.to_datetime(years)
anni = years_pdh.year

decades = anni[10::20]
diffdec = []
anomdecbase = []
for dec in decades:
    mask = abs(years_pdh.year - dec) <= 10
    diffdec.append(np.mean(diff[mask,:], axis=0))
    anomdecbase.append(np.mean(zonal_anom_b[mask,:], axis = 0))

fig = plt.figure()

cset_short = ctl.color_set(len(decades), bright_thres = 0., full_cb_range = True)
for dec, col, diff_ok in zip(decades, cset_short, diffdec):
    plt.plot(lat, coslat*diff_ok, color = col)

plt.xlabel('Latitude')
plt.ylabel('Difference base-stoc [K]')

fig.subplots_adjust(right=0.8)
plt.grid()
plt.xlim(-90,90)
plt.title('Zonal temperature anomaly difference (weighted)')

cbar_ax = fig.add_axes([0.83, 0.12, 0.05, 0.7])

norm = mpl_colors.Normalize(vmin=years_pdh.year[0],vmax=years_pdh.year[-1])
sm = plt.cm.ScalarMappable(cmap=cm.get_cmap('nipy_spectral'), norm=norm)
sm.set_array([0.05, 0.95])
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label('Year')

fig.savefig(cart+'Zonal_temp_anomaly_base_vs_stoc_weighted.pdf')


fig = plt.figure()
years_pdh = pd.to_datetime(years)
anni = years_pdh.year

cset_short = ctl.color_set(len(decades), bright_thres = 0., full_cb_range = True)
for dec, col, diff_ok in zip(decades, cset_short, diffdec):
    plt.plot(lat, diff_ok, color = col)

plt.xlabel('Latitude')
plt.ylabel('Difference base-stoc [K]')

fig.subplots_adjust(right=0.8)
plt.grid()
plt.xlim(-90,90)
plt.title('Zonal temperature anomaly difference')

cbar_ax = fig.add_axes([0.83, 0.12, 0.05, 0.7])

norm = mpl_colors.Normalize(vmin=years_pdh.year[0],vmax=years_pdh.year[-1])
sm = plt.cm.ScalarMappable(cmap=cm.get_cmap('nipy_spectral'), norm=norm)
sm.set_array([0.05, 0.95])
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label('Year')

fig.savefig(cart+'Zonal_temp_anomaly_base_vs_stoc.pdf')


fig = plt.figure()
years_pdh = pd.to_datetime(years)
anni = years_pdh.year

cset_short = ctl.color_set(len(decades[5:]), bright_thres = 0., full_cb_range = True)
for dec, col, diff_ok, anomb in zip(decades[5:], cset_short, diffdec[5:], anomdecbase[5:]):
    plt.plot(lat, diff_ok/anomb, color = col)

plt.xlabel('Latitude')
plt.ylabel('Difference base-stoc wrt tot anomaly (base)')

fig.subplots_adjust(right=0.8)
plt.grid()
plt.xlim(-90,90)
plt.title('Zonal temperature anomaly difference wrt tot anomaly')

cbar_ax = fig.add_axes([0.83, 0.12, 0.05, 0.7])

norm = mpl_colors.Normalize(vmin=1960,vmax=2100)
sm = plt.cm.ScalarMappable(cmap=cm.get_cmap('nipy_spectral'), norm=norm)
sm.set_array([0.05, 0.95])
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label('Year')

fig.savefig(cart+'Zonal_temp_anomaly_base_vs_stoc_relative.pdf')


fig = plt.figure()
years_pdh = pd.to_datetime(years)

cset = ctl.color_set(len(years), bright_thres = 0., full_cb_range = True)
for year, col in zip(years_pdh.year, cset):
    plt.plot(lat, zonal_anom_s[years_pdh.year == year].squeeze(), color = col)

plt.xlabel('Latitude')
plt.ylabel('Temperature anomaly [K]')

fig.subplots_adjust(right=0.8)
plt.grid()
plt.xlim(-90,90)
plt.ylim(-3.,18.)
plt.title('Zonal temperature anomaly (stoc)')

cbar_ax = fig.add_axes([0.83, 0.12, 0.05, 0.7])

norm = mpl_colors.Normalize(vmin=years_pdh.year[0],vmax=years_pdh.year[-1])
sm = plt.cm.ScalarMappable(cmap=cm.get_cmap('nipy_spectral'), norm=norm)
sm.set_array([0.05, 0.95])
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label('Year')
fig.savefig(cart+'Zonal_temp_anomaly_lcs.pdf')


anni = years_pdh.year
pino_b = pd.Series(global_anom_b)
rollpi_b = pino_b.rolling(5, center = True).mean()

pino_s = pd.Series(global_anom_s)
rollpi_s = pino_s.rolling(5, center = True).mean()

fig2 = plt.figure()

linest = 3*['-']+3*['--']
for temp, lst in zip(global_anom_ens, linest):
    tempser = pd.Series(temp)
    rollpi_temp = tempser.rolling(5, center = True).mean()
    plt.plot(anni, rollpi_temp, linewidth = 0.7, color = 'gray', linestyle = lst)

plt.plot(anni, rollpi_b, linewidth = 2.5, label = 'base')
plt.plot(anni, rollpi_s, linewidth = 2.5, label = 'stoc')
plt.grid()
plt.title('Global temperature anomaly')

plt.xlabel('Year')
plt.ylabel('Temperature anomaly (K)')
plt.legend()
fig2.savefig(cart+'Global_anomaly_base_vs_stoc.pdf')


rollpi_b_20 = pino_b.rolling(20, center = True).mean()
rollpi_s_20 = pino_s.rolling(20, center = True).mean()

fig2 = plt.figure()
plt.plot(anni, rollpi_b - rollpi_s, linewidth = 1.5)
plt.plot(anni, rollpi_b_20 - rollpi_s_20, linewidth = 2.5)
plt.title('Global temperature anomaly difference (base-stoc)')
plt.xlabel('Year')
plt.ylabel('Temperature anomaly (K)')
plt.grid()
fig2.savefig(cart+'Global_anomaly_base_vs_stoc_difference.pdf')

fig2 = plt.figure()
linest = 3*['-']+3*['--']
for temp, lst in zip(global_anom_ens, linest):
    tempser = pd.Series(temp)
    rollpi_temp = tempser.rolling(5, center = True).mean()
    rollpi_temp_20 = tempser.rolling(20, center = True).mean()
    plt.plot(anni, rollpi_temp-rollpi_temp_20, linewidth = 0.7, color = 'gray', linestyle = lst)
plt.plot(anni, rollpi_b - rollpi_b_20, linewidth = 2.5, label = 'base')
plt.plot(anni, rollpi_s - rollpi_s_20, linewidth = 2.5, label = 'stoc')
plt.title('Global temperature anomaly 5-year oscillations')
plt.xlabel('Year')
plt.ylabel('Temperature anomaly diff. (K)')
plt.grid()
fig2.savefig(cart+'Global_anomaly_base_vs_stoc_oscillations.pdf')

fig2 = plt.figure()
plt.plot(anni, rollpi_b_20, linewidth = 2.5, label = 'base')
plt.plot(anni, rollpi_s_20, linewidth = 2.5, label = 'stoc')
plt.grid()
plt.title('Global temperature anomaly (20 yr smooth)')

plt.xlabel('Year')
plt.ylabel('Temperature anomaly (K)')
plt.legend()
fig2.savefig(cart+'Global_anomaly_base_vs_stoc.pdf')


fig2 = plt.figure()
plt.plot(anni, rollpi_b, linewidth = 2.5)
plt.grid()

year_thres_base = []
thress = [1.5, 2., 3., 4.]
cols = ['yellow', 'orange', 'red', 'violet']
for th,c  in zip(thress, cols):
    mask = rollpi_b > th
    an = anni[mask][0]
    year_thres_base.append(an)
    plt.scatter(an, th, s=30, c = c, edgecolors='black', zorder = 10)
    plt.text(an, th-0.35, '{}'.format(an), fontweight = 'bold', color = 'black', bbox=dict(facecolor=c, edgecolor='black', alpha = 0.7, boxstyle='round'))

plt.title('Global temperature anomaly (base)')
plt.xlabel('Year')
plt.ylabel('Temperature anomaly (K)')
fig2.savefig(cart+'Global_anomaly_base_thresholds.pdf')


fig2 = plt.figure()

plt.plot(anni, rollpi_s, linewidth = 2.5)
plt.grid()

year_thres_stoc = []
thress = [1.5, 2., 3., 4.]
cols = ['yellow', 'orange', 'red', 'violet']
for th,c  in zip(thress, cols):
    mask = rollpi_s > th
    an = anni[mask][0]
    year_thres_stoc.append(an)
    plt.scatter(an, th, s=30, c = c, edgecolors='black', zorder = 10)
    plt.text(an, th-0.35, '{}'.format(an), fontweight = 'bold', color = 'black', bbox=dict(facecolor=c, edgecolor='black', alpha = 0.7, boxstyle='round'))

plt.title('Global temperature anomaly (stoc)')
plt.xlabel('Year')
plt.ylabel('Temperature anomaly (K)')
fig2.savefig(cart+'Global_anomaly_stoc_thresholds.pdf')


yas_b = []
yas_s = []
for an, th in zip(year_thres_base, thress):
    mask = abs(years_pdh.year - an) <= 10
    yas_b.append(np.mean(yearly_anom_b[mask, ...], axis=0))
for an, th in zip(year_thres_stoc, thress):
    mask = abs(years_pdh.year - an) <= 10
    yas_s.append(np.mean(yearly_anom_s[mask, ...], axis=0))

yas_b = np.stack(yas_b)
yas_s = np.stack(yas_s)
(cmin, cmax) = ctl.get_cbar_range(yas_b, symmetrical = True)

for okanom, an, th in zip(yas_b, year_thres_base, thress):
    nomfi = cart+'map_anom_{}_th{}_base.pdf'.format(an, int(10*th))
    tit = 'Year {} - 20yr ave anomaly - +{:3.1f} K globally'.format(an, th)
    ctl.plot_map_contour(okanom, lat, lon, filename = nomfi, visualization = 'polar', central_lat_lon = (30,0), cbar_range = (cmin,cmax), title = tit, cb_label = 'Temp. anomaly (K)')

for okanom, an, th in zip(yas_s, year_thres_stoc, thress):
    tit = 'Year {} - 20yr averaged anomaly - +{:3.1f} K globally'.format(an, th)
    nomfi = cart+'map_anom_{}_th{}_stoc.pdf'.format(an, int(10*th))
    ctl.plot_map_contour(okanom, lat, lon, filename = nomfi, visualization = 'polar', central_lat_lon = (30,0), cbar_range = (cmin,cmax), title = tit, cb_label = 'Temp. anomaly (K)')


# pinomed = pd.Series(med_anom_b)
# rollpimed = pinomed.rolling(5, center = True).mean()
#
# pinomedsum = pd.Series(med_anom_sum_b)
# rollpimedsum = pinomedsum.rolling(5, center = True).mean()
#
#
# fig3 = plt.figure()
# plt.plot(anni, rollpi_b, linewidth = 2.0, label = 'Global')
# plt.grid()
# #plt.plot(anni, pinomed)
# plt.plot(anni, rollpimed, linewidth = 2.0, label = 'Med')
# #plt.plot(anni, pinomedsum)
# plt.plot(anni, rollpimedsum, linewidth = 2.0, label = 'Med summer')
# plt.xlabel('Year')
# plt.ylabel('Temperature anomaly (K)')
# plt.legend()
# plt.title('Global vs Mediterranean (base)')
# fig3.savefig(cart+'Global_vs_Med_anomaly_base.pdf')
#
#
#
# pinomed = pd.Series(med_anom_s)
# rollpimed = pinomed.rolling(5, center = True).mean()
#
# pinomedsum = pd.Series(med_anom_sum_s)
# rollpimedsum = pinomedsum.rolling(5, center = True).mean()
#
#
# fig5 = plt.figure()
# plt.plot(anni, rollpi_s, linewidth = 2.0, label = 'Global')
# plt.grid()
# #plt.plot(anni, pinomed)
# plt.plot(anni, rollpimed, linewidth = 2.0, label = 'Med')
# #plt.plot(anni, pinomedsum)
# plt.plot(anni, rollpimedsum, linewidth = 2.0, label = 'Med summer')
# plt.xlabel('Year')
# plt.ylabel('Temperature anomaly (K)')
# plt.legend()
# plt.title('Global vs Mediterranean (stoc)')
# fig5.savefig(cart+'Global_vs_Med_anomaly_stoc.pdf')


filt_artico = lat > 80
diff_filt = np.mean(diff[:, filt_artico], axis = 1)

artico_b = np.mean(zonal_anom_b[:, filt_artico], axis = 1)
artico_s = np.mean(zonal_anom_s[:, filt_artico], axis = 1)

artico_b_smooth = pd.Series(artico_b).rolling(5, center = True).mean()
artico_s_smooth = pd.Series(artico_s).rolling(5, center = True).mean()
diff_filt_smooth = pd.Series(diff_filt).rolling(5, center = True).mean()

fig32 = plt.figure()
plt.plot(anni, artico_b_smooth, label = 'base', linewidth = 2)
plt.plot(anni, artico_s_smooth, label = 'stoc', linewidth = 2)
plt.plot(anni, diff_filt_smooth, label = 'diff', linewidth = 2)
plt.grid()
plt.legend()


artico_ens = []
for ens, lst in zip(zonal_anom_ens, linest):
    coso = np.mean(ens[:, filt_artico], axis=1)
    coso2 = pd.Series(coso).rolling(5, center = True).mean()
    artico_ens.append(coso2)
    plt.plot(anni, coso2, linewidth = 0.7, color = 'gray', linestyle = lst, label = None)

plt.xlabel('Year')
plt.ylabel('Temperature anomaly (K)')
plt.title('Temp anomaly in the Arctic (> 80N)')
fig32.savefig(cart+'Temp_difference_arctic.pdf')


# GIF animation
fig = plt.figure(figsize=(8,6), dpi=150)
ax = fig.add_subplot(111)

cset = ctl.color_set(len(anni), bright_thres = 0., full_cb_range = True)

plt.xlabel('Latitude')
plt.ylabel('Temperature anomaly [K]')
plt.grid()
plt.xlim(-90,90)
plt.title('Zonal temperature anomaly')

def update_lines(num):
    year = anni[num]
    line = zonal_anom_b[anni == year].squeeze()
    color = cset[num]

    ax.plot(lat, line, color = color)
    showdate.set_text('{}'.format(year))#, color = color)
    showdate.update(color = color)
    return

line = zonal_anom_b[anni == anni[0]].squeeze()
ax.plot(lat, line, color = cset[0])
plt.ylim(-3.,18.)
showdate = ax.text(0., 24., '1850', fontweight = 'bold', color = cset[0], bbox=dict(facecolor='lightsteelblue', edgecolor='black', boxstyle='round,pad=1'))

save = True

if save:
    metadata = dict(title='Zonal temperature anomaly (SPHINX lcb experiments)', artist='Federico Fabiano')
    writer = ImageMagickFileWriter(fps = 10, metadata = metadata)#, frame_size = (1200, 900))
    with writer.saving(fig, cart + "Zonal_temp_anomaly_animation.gif", 100):
        for i, (year, col) in enumerate(zip(anni, cset)):
            print(year)
            update_lines(i)
            writer.grab_frame()
else:
    line_ani = animation.FuncAnimation(fig, update_lines, len(anni), interval=100, blit=False)
