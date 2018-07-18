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


def clus_visualize_2D():
    """
    Makes a 2D plot using the coordinates in the first 2 EOFs of single realizations and clusters.
    """

    return


def clus_visualize_3D():
    """
    Makes a 3D plot using the coordinates in the first 3 EOFs of single realizations and clusters.
    """

    return
