#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
from matplotlib import pyplot as plt
import netCDF4 as nc
import cartopy.crs as ccrs

# Begin simple program to deal with .nc files

# Open nc file

cart = '/data/fabiano/ECEARTHdata/SPHINX/AMIP/Z500T1279regrid/Z500base/zg500uab/'
filename = 'zg500_Aday_EC-EARTH31_T1279base_regrid25_0_1979-2008.nc'

data = nc.Dataset(cart+filename)

#print(data.variables)
field = data.variables['zg']
print(field.shape)

# Extract a single map
time_num = 0
gigi = field[time_num,0,:,:]

# Creating the projection from cartopy
clon = 0.
clat = 0.
proj = ccrs.PlateCarree(central_longitude=0.)
#proj = ccrs.Orthographic(central_longitude=clon, central_latitude=clat)

plt.ion()
#pl.imshow(gigi, interpolation='gaussian')
fig = plt.figure()
ax = plt.axes(projection = proj)
ax.set_global()
ax.coastlines(resolution='110m')
ax.gridlines()


nlons = 144
nlats = 73
lons = np.linspace(0.,360.,nlons+1)
lats = np.linspace(-90.,90.,nlats)
print(gigi.shape)
gigi2 = np.concatenate((gigi,gigi[:,-1:]), axis = 1)
print(gigi2.shape)

nlev = 20
fill = ax.contourf(lons,lats,gigi2,nlev,cmap=plt.cm.RdBu_r, transform = proj)

"""
clat=lat_area.min()+abs(lat_area.max()-lat_area.min())/2
clon=lon_area.min()+abs(lon_area.max()-lon_area.min())/2
var_allens=np.concatenate(var_ensList[:])

delta=30
if area=='PNA':
    rangecolorbar=np.arange(-270, 300, delta)
if area=='EAT':
    rangecolorbar=np.arange(-240, 270, delta)
    #clus_ordered=['NAO+','Blocking','Altantic Ridge','NAO-']  #Order of observed clusters
    clus_ordered=['clus1','clus2','clus3','clus4']
ax = plt.subplots(2, 2, figsize=(18,14), sharex='col', sharey='row')
txt='Clusters for {0} PCs'.format(numpcs)
print(txt)

cluspatt=[]
for nclus in range(numberclusters):
    cl=list(np.where(indcl==nclus)[0])
    freq_perc=len(cl)*100./len(indcl)
    tclus='{0} {1}%\n'.format(clus_ordered[nclus],"%.2f" %freq_perc)
    print(tclus)
    cluspattern=np.mean(var_allens[cl,:,:],axis=0)
    lonzero=cluspattern[:,0:1]
    cluspattern_ext=np.concatenate([cluspattern,lonzero],axis=1)
    lon_ext=np.append(lon,lon[-1]+(lon[1]-lon[0]))
    cluspatt.append(cluspattern)
    ax = plt.subplot(2, 2, nclus+1, projection=proj)
    ax.set_global()
    ax.coastlines()
    #ax.gridlines()
    fill = ax.contourf(lon_ext,lat,cluspattern_ext,rangecolorbar,cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    plt.title(tclus, fontsize=30, fontweight='bold')
#             ([x,y,thickness, height])
cax = plt.axes([0.85, 0.1, 0.02, 0.8])
cb=plt.colorbar(fill,cax=cax)#, labelsize=18)
cb.set_label('(m)', rotation=0, fontsize=22)
cb.ax.tick_params(labelsize=22)
plt.suptitle(tit, fontsize=40, fontweight='bold')
#ax.annotate(txt, xy=(.26, .875),xycoords='figure fraction',fontsize=24)
plt.subplots_adjust(top=0.85)

left   = 0    # the left side of the subplots of the figure
right  = 0.8  # the right side of the subplots of the figure
bottom = 0    # the bottom of the subplots of the figure
top    = 0.82  # the top of the subplots of the figure
wspace = 0    # the amount of width reserved for blank space between subplots
hspace = 0.32   # the amount of height reserved for white space between subplots

plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

"""
