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

#######################################
#ùèòìàúéíóá
plt.ion()

cart_out = '/home/fabiano/Research/lavori/WeatherRegimes/phasespace_ERA/'
if not os.path.exists(cart_out): os.mkdir(cart_out)

[results_ref, all_res] = pickle.load(open('/home/fabiano/Research/lavori/WeatherRegimes/variability_ERA/res_bootstrap_1000.p'))

#trans_all = ctl.find_transition_pcs(6, lab_all, dates_all, pcs_all, filter_longer_than = 3, skip_persistence = True)
trans_all = results_ref['regime_transition_pcs']

cfrom = 0
cto = 1
# Guardo il regime 0
rsd_tim, rsd_dat, rsd_num = ctl.calc_regime_residtimes(results_ref['labels'], dates = results_ref['dates'])
ok_times = rsd_tim[0]
ok_nums = rsd_num[0]

days_event, length_event = ctl.calc_days_event(results_ref['labels'], rsd_tim, rsd_num)
okclu = results_ref['labels'] == 0

# Proietto sulla nuova base
newbase, new_pcs, new_trans = ctl.rotate_space_interclus_section_3D(results_ref['centroids'], cfrom, cto, results_ref['pcs'], transitions = trans_all[cfrom, cto])

ngp = 100
(x0, x1) = (np.percentile(new_pcs[:,0], 1), np.percentile(new_pcs[:,0], 99))
(y0, y1) = (np.percentile(new_pcs[:,1], 1), np.percentile(new_pcs[:,1], 99))
xss = np.linspace(x0,x1,ngp)
yss = np.linspace(y0,y1,ngp)
xi2, yi2 = np.meshgrid(xss, yss)

(z0, z1) = (np.percentile(new_pcs[:,2], 1), np.percentile(new_pcs[:,2], 99))
zss = np.linspace(z0,z1,ngp)
xi3, yi3, zi3 = np.meshgrid(xss, yss, zss)

xib, zib = np.meshgrid(xss, zss)
yic, zic = np.meshgrid(yss, zss)

cordss = [xss, yss, zss]
namz = ['x', 'y', 'z']
coups = [(0,1), (0,2), (1,2)]

for cou in coups:
    regime_pdf_2D = []
    regime_pdf_2D_func = []
    xi, yi = np.meshgrid(cordss[cou[0]], cordss[cou[1]])

    for clus in range(2):
        print(clus)
        okclus = results_ref['labels'] == clus
        okpc = new_pcs[okclus, :]
        kufu = ctl.calc_pdf(okpc[:,cou].T)
        print('fatto kufu\n')
        zi = kufu(np.vstack([xi.flatten(), yi.flatten()]))
        regime_pdf_2D.append(zi)
        regime_pdf_2D_func.append(kufu)

    fig = plt.figure(figsize=(8, 6), dpi=150)
    for li in new_trans:
        li3 = li[:,cou]
        p0 = li3[0]
        dp = li3[1]-li3[0]
        plt.arrow(p0[0], p0[1], dp[0], dp[1], color = 'red', linestyle = ':', linewidth = 0.5)
        #plt.plot(li3[0], li3[1], color = 'grey')

    for li in new_trans:
        li3 = li[:,cou]
        p0 = li3[0]
        dp = li3[1]-li3[0]
        plt.arrow(p0[0], p0[1], dp[0], dp[1], color = 'blue', linestyle = '--', linewidth = 0.4)

    for clus, namcm in enumerate(['Purples']):#,'Blues']):#,'Greens','Oranges']):
        plt.contour(xi, yi, regime_pdf_2D[clus].reshape(xi.shape), cmap = cm.get_cmap(namcm))

    okclu = results_ref['labels'] == 0
    okpc = new_pcs[okclu, :]
    #plt.scatter(okpc[:, cou][:,0], okpc[:, cou][:,1], c = days_event[okclu], s = 1)
    #plt.scatter(okpc[:, cou][:,0], okpc[:, cou][:,1], c = length_event[okclu], s = 1)
    shortev = length_event[okclu] <= 5
    kufu = ctl.calc_pdf(okpc[shortev,:][:,cou].T)
    zi = kufu(np.vstack([xi.flatten(), yi.flatten()]))
    plt.contour(xi, yi, zi.reshape(xi.shape), cmap = cm.get_cmap('Blues'))

    shortev = (length_event[okclu] > 5) & (length_event[okclu] <= 10)
    kufu = ctl.calc_pdf(okpc[shortev,:][:,cou].T)
    zi = kufu(np.vstack([xi.flatten(), yi.flatten()]))
    plt.contour(xi, yi, zi.reshape(xi.shape), cmap = cm.get_cmap('Greens'))

    shortev = length_event[okclu] > 10
    kufu = ctl.calc_pdf(okpc[shortev,:][:,cou].T)
    zi = kufu(np.vstack([xi.flatten(), yi.flatten()]))
    plt.contour(xi, yi, zi.reshape(xi.shape), cmap = cm.get_cmap('Reds'))

    plt.title('Projection on {} and {} rotated eofs'.format(cou[0], cou[1]))
    plt.xlabel('EOF {}'.format(cou[0]))
    plt.ylabel('EOF {}'.format(cou[1]))

    #plt.contourf()


# transvecs01 = np.stack([(li[1,:3]-li[0,:3]).T for li in new_trans])
# transvecs01_mean = np.mean(transvecs01, axis = 0)
#
# transvecs10 = np.stack([(li[1,:3]-li[0,:3]).T for li in trans_all[1,0]])
# transvecs10_mean = np.mean(transvecs10, axis = 0)
