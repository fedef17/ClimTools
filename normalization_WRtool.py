#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
from matplotlib import pyplot as plt
import netCDF4 as nc
import cartopy.crs as ccrs
from numpy import linalg as LA
import pickle
from eofs.standard import Eof

# try:
#     from EOFtool import eof_computation
# except ModuleNotFoundError:
#     sys.path.insert(0, '/home/fabiano/Research/git/WRtool/CLUS_tool/WRtool/')
#     from EOFtool import eof_computation

# Calculate normalization of 2D fields

def cosine(x,y):
    """
    Calculates the cosine of the angle between x and y. If x and y are 2D, the scalar product is taken using the np.vdot() function.
    """
    if x.ndim != y.ndim:
        raise ValueError('x and y have different dimension')
    elif x.shape != y.shape:
        raise ValueError('x and y have different shapes')

    if x.ndim == 1:
        return np.dot(x,y)/(LA.norm(x)*LA.norm(y))
    elif x.ndim == 2:
        return np.vdot(x,y)/(LA.norm(x)*LA.norm(y))
    else:
        raise ValueError('Too many dimensions')

# Open nc file

fold_0 = '/home/fabiano/Research/lavori/WeatherRegimes/'

out_fold = fold_0 + 'PP_normalization/'

cart = fold_0 + 'OUT_WRTOOL/OUTPUT/OUTnc/'
cartpy = fold_0 + 'OUT_WRTOOL/OUTPUT/OUTpy/'
filename = 'EOFscal2_zg500_day_ERAInterim_obs_144x73_1ens_DJF_EAT_1979-2008_4pcs.nc'

data = nc.Dataset(cart+filename)

#print(data.variables)
field = data.variables['EOFscal2']
print(field.shape)

print('\n Normalizzazione delle EOFscal2\n')

for i in range(4):
    norma = LA.norm(field[i,:,:])
    print(norma)

filename = 'EOFunscal_zg500_day_ERAInterim_obs_144x73_1ens_DJF_EAT_1979-2008_4pcs.nc'

data = nc.Dataset(cart+filename)

#print(data.variables)
eofs_u = data.variables['EOFunscal']
print(eofs_u.shape)

print('\n Normalizzazione delle EOFunscal \n')

for i in range(4):
    norma = LA.norm(eofs_u[i,:,:])
    print(norma)

filename = 'EOFunscal_zg500_day_ECEARTH31_obs_144x73_1ens_DJF_EAT_1979-2008_4pcs.nc'

data = nc.Dataset(cart+filename)

#print(data.variables)
eofs_u_ec = data.variables['EOFunscal']
print(eofs_u_ec.shape)

print('\n Proiezione delle EOFunscal di ECEARTH su quelle di ERA\n')

riga = 'EOF {}: '+ 4*' {:6.2f} '

for i in range(4):
    coef1 = []
    for j in range(4):
        coeff = np.vdot(eofs_u_ec[i,:,:],eofs_u[j,:,:])
        coef1.append(coeff)

    print(riga.format(i+1,*coef1))

riga = 'WR {}: '+ 4*' {:6.2f} '

filesolver = cartpy + 'solver_zg500_day_ERAInterim_obs_144x73_1ens_DJF_EAT_1979-2008_4pcs.p'
solver_era = pickle.load(open(filesolver, 'rb'))

filesolver = cartpy + 'solver_zg500_day_ECEARTH31_obs_144x73_1ens_DJF_EAT_1979-2008_4pcs.p'
solver_ec = pickle.load(open(filesolver, 'rb'))

filesolver = cartpy + 'solver_zg500_day_NCEPNCAR_obs_144x73_1ens_DJF_EAT_1979-2008_4pcs.p'
solver_nc = pickle.load(open(filesolver, 'rb'))


file1 = 'cluspatternORDasREF_4clus_zg500_day_ECEARTH31_obs_144x73_1ens_DJF_EAT_1979-2008_4pcs.nc'
file2 = 'cluspatternORDasREF_4clus_zg500_day_ERAInterim_obs_144x73_1ens_DJF_EAT_1979-2008_4pcs.nc'
file3 = 'cluspatternORDasREF_4clus_zg500_day_NCEPNCAR_obs_144x73_1ens_DJF_EAT_1979-2008_4pcs.nc'

data1 = nc.Dataset(cart+file1)
data2 = nc.Dataset(cart+file2)
data3 = nc.Dataset(cart+file3)

#print(data.variables)
field1 = data1.variables['cluspattern']
field2 = data2.variables['cluspattern']
field3 = data3.variables['cluspattern']

rigaera = 'WR {} era-int: '+ 4*' {:6.2f} '
rigaec = 'WR {} ec-earth: '+ 4*' {:6.2f} '

print('\n Proiezione dei WR di ECEARTH e di ERA sulle EOFunscal di ERA\n')

for i in range(4):
    coef1 = []
    coef2 = []
    for j in range(4):
        coeff = np.vdot(field1[i,:,:],eofs_u[j,:,:])
        coef1.append(coeff)
        coeff = np.vdot(field2[i,:,:],eofs_u[j,:,:])
        coef2.append(coeff)

    print(rigaec.format(i+1,*coef1))
    print(rigaera.format(i+1,*coef2))

print('\n Distanza (norm della diff) dei WR di ECEARTH da quelli di ERA\n')

for i in range(4):
    coef1 = []
    for j in range(4):
        coeff = LA.norm(field1[i,:,:]-field2[j,:,:])
        coef1.append(coeff)
    print(riga.format(i+1,*coef1))


print('\n Coseno dell angolo relativo tra i WR di ECEARTH e quelli di ERA\n')

for i in range(4):
    coef1 = []
    for j in range(4):
        coeff = cosine(field1[i,:,:],field2[j,:,:])
        coef1.append(coeff)
    print(riga.format(i+1,*coef1))


print('\n Distanza (norm della diff) dei WR di NCEP da quelli di ERA\n')

for i in range(4):
    coef1 = []
    for j in range(4):
        coeff = LA.norm(field3[i,:,:]-field2[j,:,:])
        coef1.append(coeff)
    print(riga.format(i+1,*coef1))


print('\n Coseno dell angolo relativo tra i WR di NCEP e quelli di ERA\n')

for i in range(4):
    coef1 = []
    for j in range(4):
        coeff = cosine(field3[i,:,:],field2[j,:,:])
        coef1.append(coeff)
    print(riga.format(i+1,*coef1))


print('\n Rifaccio i conti sopra ma proietto prima sulla base di ERA e faccio i conti sulle PC \n')

WR_NCEP_proera = solver_era.projectField(field3[:,:,:], neofs=4, eofscaling=0, weighted=True)
WR_ECEARTH_proera = solver_era.projectField(field1[:,:,:], neofs=4, eofscaling=0, weighted=True)
WR_ERA = solver_era.projectField(field2[:,:,:], neofs=4, eofscaling=0, weighted=True)

print('\n Distanza dei WR di NCEP da quelli di ERA - nella base di ERA\n')

for i in range(4):
    coef1 = []
    for j in range(4):
        coeff = LA.norm(WR_NCEP_proera[i]-WR_ERA[j])
        coef1.append(coeff)
    print(riga.format(i+1,*coef1))

print('\n Distanza dei WR di ECEARTH da quelli di ERA - nella base di ERA\n')

for i in range(4):
    coef1 = []
    for j in range(4):
        coeff = LA.norm(WR_ECEARTH_proera[i]-WR_ERA[j])
        coef1.append(coeff)
    print(riga.format(i+1,*coef1))

print('-------------\n')

n_eofs = 25
ind_pcs = [0,25]

eigenval_era = solver_era.eigenvalues()
print(len(eigenval_era))
pcs_era = solver_era.pcs()
print(np.shape(pcs_era))

plt.ion()

fig1 = plt.figure()
nor = np.sum(eigenval_era)
# plt.plot(eigenval/nor)
# plt.scatter(range(0,len(eigenval)), eigenval/nor)
plt.grid()
plt.bar(range(0,len(eigenval_era)), eigenval_era/nor)
plt.xlim(-1,n_eofs)
plt.xlabel('EOF')
plt.ylabel(r'$\lambda_i \div \sum_j \lambda_j$')
plt.title('Normalized eigenvalues of EOFs - ERA')
plt.tight_layout()
plt.savefig(out_fold+'ERA_eigenvalues.pdf', format = 'pdf')

tito = 'Showing only the pcs from {} to {}, with {} eofs..'

fig2 = plt.figure()
plt.xlim(-0.5,n_eofs+0.5)
plt.ylim(ind_pcs[0]-0.5,ind_pcs[1]+0.5)
print(tito.format(ind_pcs[0],ind_pcs[1],n_eofs))
plt.imshow(pcs_era)
gigi = plt.colorbar()
gigi.set_label('PC value (m)')
plt.xlabel('EOF')
plt.ylabel('Time (seq. number)')
plt.title('PCs - ERA')
plt.savefig(out_fold+'ERA_PCs.pdf', format = 'pdf')


eigenval_ec = solver_ec.eigenvalues()
print(len(eigenval_ec))
pcs_ec = solver_ec.pcs()
print(np.shape(pcs_ec))

plt.ion()

fig3 = plt.figure()
nor = np.sum(eigenval_ec)
# plt.plot(eigenval/nor)
# plt.scatter(range(0,len(eigenval)), eigenval/nor)
plt.bar(range(0,len(eigenval_ec)), eigenval_ec/nor)
plt.grid()
plt.xlim(-1,n_eofs)
plt.xlabel('EOF')
plt.ylabel(r'$\lambda_i \div \sum_j \lambda_j$')
plt.title('Normalized eigenvalues of EOFs - ECEARTH')
plt.tight_layout()
plt.savefig(out_fold+'ECE_eigenvalues.pdf', format = 'pdf')

fig4 = plt.figure()
plt.xlim(-0.5,n_eofs+0.5)
plt.ylim(ind_pcs[0]-0.5,ind_pcs[1]+0.5)
print(tito.format(ind_pcs[0],ind_pcs[1],n_eofs))
plt.imshow(pcs_ec)
gigi = plt.colorbar()
gigi.set_label('PC value (m)')
plt.xlabel('EOF')
plt.ylabel('Time (seq. number)')
plt.title('PCs - ECEARTH')
plt.savefig(out_fold+'ECE_PCs.pdf', format = 'pdf')


# Studying the difference between the two sets of EOFs
eofs_ec = solver_ec.eofs()
eofs_era = solver_era.eofs()

cosines = []
for i in range(10):
    cosines.append(cosine(eofs_ec[i], eofs_era[i]))

cosines = np.array(cosines)
# fig5 = plt.figure()
#
# plt.bar(range(0,len(cosines)), cosines)
# plt.xlabel('EOF')


cosines_2D = []
for i in range(10):
    cosi = []
    for j in range(10):
        cosi.append(cosine(eofs_ec[i], eofs_era[j]))
    cosines_2D.append(cosi)

riga = 'EOF ECEARTH {:2d}: '+ 10*' {:6.2f} '
print('\n Cosine of relative angle between ECEARTH and ERA EOFs:\n')
for i, cosi in zip(range(10), cosines_2D):
    print(riga.format(i+1,*cosi))
print('-------------\n')

cosines_2D = np.array(cosines_2D)
fig6 = plt.figure()

plt.imshow(cosines_2D)
plt.xlabel('EOFs ecearth')
plt.ylabel('EOFs era')
plt.title('Cosine of relative angle')
plt.colorbar()
plt.savefig(out_fold+'ERA_ECE_cosines_EOFs.pdf', format = 'pdf')


eigenval_nc = solver_nc.eigenvalues()
print(len(eigenval_nc))
pcs_nc = solver_nc.pcs()
print(np.shape(pcs_nc))

eofs_nc = solver_nc.eofs()

cosines_2D = []
for i in range(10):
    cosi = []
    for j in range(10):
        cosi.append(cosine(eofs_nc[i], eofs_era[j]))
    cosines_2D.append(cosi)

riga = 'EOF NCEP {:2d}: '+ 10*' {:6.2f} '
print('\n Cosine of relative angle between NCEP and ERA EOFs:\n')
for i, cosi in zip(range(10), cosines_2D):
    print(riga.format(i+1,*cosi))

cosines_2D = np.array(cosines_2D)
fig6 = plt.figure()

plt.imshow(cosines_2D)
plt.xlabel('EOFs ncep')
plt.ylabel('EOFs era')
plt.title('Cosine of relative angle')
plt.colorbar()
plt.savefig(out_fold+'ERA_NCEP_cosines_EOFs.pdf', format = 'pdf')
