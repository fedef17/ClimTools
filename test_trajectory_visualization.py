#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import pickle
import os
import datetime
from shutil import copy2

from matplotlib import pyplot as plt
from numpy import linalg as LA
import netCDF4 as nc
from netCDF4 import Dataset, num2date
import cartopy.crs as ccrs

import Tkinter
import tkMessageBox
from copy import deepcopy as cp

sys.path.insert(0,inputs['dir_WRtool']+'WRtool/')
import lib_WRtool as lwr
import clus_manipulate as clum

cart = ''
nameout = ''
numens = 1
numpcs = 4
numclus = 4

out_precompute = lwr.read_out_precompute(cart, nameout, numens)
var_ensList_glob, lat, lon, var_ensList_area, lat_area, lon_area, dates, time_units, var_units = out_precompute
pickle.dump(out_precompute, open(cart+'out_precompute.p','wb'), protocol = 2)

solver = read_out_compute(cart, nameout, numpcs)


centrORD, indclORD, cluspattORD, varopt = lwr.read_out_clustering(cart, nameout, numpcs, numclus)
pickle.dump([centrORD, indclORD, cluspattORD, varopt], open(cart+'out_clustering_{}pcs_{}clus.p','wb'), protocol = 2)

# anomalies = var_ensList_area
# pvec_anom = clum.compute_pvectors(numpcs, solver_ERA, anomalies)
