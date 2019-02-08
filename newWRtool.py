#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
from matplotlib import pyplot as plt
import pickle

import climtools_lib as ctl
import climdiags as cd

#######################################

# # open our log file
# logname = 'log_WRtool.log'
# logfile = open(logname,'w') #self.name, 'w', 0)
#
# # re-open stdout without buffering
# sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
#
# # redirect stdout and stderr to the log file opened above
# os.dup2(logfile.fileno(), sys.stdout.fileno())
# os.dup2(logfile.fileno(), sys.stderr.fileno())

print('*******************************************************')
print('Running {0}'.format(sys.argv[0]))
print('********************************************************')

if len(sys.argv) > 1:
    file_input = sys.argv[1] # Name of input file (relative path)
else:
    file_input = 'input_WRtool.in'

keys = 'exp_name cart_in cart_out filenames model_names level season area numclus numpcs perc ERA_ref_orig ERA_ref_out run_sig_calc run_compare patnames patnames_short heavy_output'
keys = keys.split()
itype = [str, str, str, list, list, float, str, str, int, int, float, str, str, bool, bool, list, list, bool]

if len(itype) != len(keys):
    raise RuntimeError('Ill defined input keys in {}'.format(__file__))
itype = dict(zip(keys, itype))

defaults = dict()
defaults['numclus'] = 4 # 4 clusters
defaults['perc'] = None
defaults['exp_name'] = 'WRtool'
defaults['run_sig_calc'] = False
defaults['run_compare'] = True
defaults['heavy_output'] = False

inputs = ctl.read_inputs(file_input, keys, n_lines = None, itype = itype, defaults = defaults)

if not os.path.exists(inputs['cart_out']): os.mkdir(inputs['cart_out'])

if inputs['area'] == 'EAT':
    if inputs['patnames'] is None:
        patnames = ['NAO +', 'Sc. Blocking', 'Atl. Ridge', 'NAO -']
    if inputs['patnames_short'] is None:
        patnames_short = ['NP', 'BL', 'NN', 'AR']
elif inputs['area'] == 'PNA':
    if inputs['patnames'] is None:
        patnames = ['Ala. Ridge', 'Pac. Trough', 'Arctic Low', 'Arctic High']
    if inputs['patnames_short'] is None:
        patnames_short = ['AR', 'PT', 'AL', 'AH']

if inputs['ERA_ref_out'] is None:
    ERA_ref = cd.WRtool_from_file(inputs['ERA_ref_orig'], inputs['season'], inputs['area'], extract_level_4D = inputs['level'], numclus = inputs['numclus'], heavy_output = True, run_significance_calc = inputs['run_sig_calc'])
else:
    ERA_ref = pickle.load(open(inputs['ERA_ref_out'], 'r'))

model_outs = dict()
for modfile, modname in zip(inputs['filenames'], inputs['model_names']):
    model_outs[modname] = cd.WRtool_from_file(inputs['cart_in']+modfile, inputs['season'], inputs['area'], extract_level_4D = inputs['level'], numclus = inputs['numclus'], heavy_output = inputs['heavy_output'], run_significance_calc = inputs['run_sig_calc'], ref_solver = ERA_ref['solver'], ref_patterns_area = ERA_ref['cluspattern_area'])

pickle.dump([model_outs, ERA_ref], open(inputs['cart_out']+'out_{}_{}.p'.format(inputs['exp_name'], inputs['area']), 'w'))

clatlo = dict()
clatlo['EAT'] = (70, 0)
clatlo['PNA'] = (70, -90)

n_models = len(model_outs.keys())

cd.plot_WRtool_results(inputs['cart_out'], '{}_{}'.format(inputs['exp_name'], inputs['area']), n_models, model_outs, ERA_ref, obs_name = 'ERA', patnames = inputs['patnames'], patnames_short = inputs['patnames_short'], central_lat_lon = clatlo[inputs['area']])#, custom_model_colors = ['indianred', 'forestgreen', 'black'], compare_models = [('stoc', 'base')])

print('define out netcdf')
