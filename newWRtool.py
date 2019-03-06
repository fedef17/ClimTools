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
def std_outname(tag, inputs):
    name_outputs = '{}_{}_{}_{}clus'.format(tag, inputs['season'], inputs['area'], inputs['numclus'])

    if inputs['flag_perc']:
        name_outputs += '_{}perc'.format(inputs['perc'])
    else:
        name_outputs += '_{}pcs'.format(inputs['numpcs'])

    if inputs['year_range'] is not None:
        name_outputs += '_{}-{}'.format(inputs['year_range'][0], inputs['year_range'][1])
    else:
        name_outputs += '_allyrs'

    if inputs['use_reference_eofs']:
        name_outputs += '_refEOF'

    if inputs['detrended_anom_for_clustering']:
        name_outputs += '_dtr'

    return name_outputs

if np.any(['log_WRtool_' in cos for cos in os.listdir('.')]):
    os.system('rm log_WRtool_*log')

# open our log file
logname = 'log_WRtool_{}.log'.format(ctl.datestamp())
logfile = open(logname,'w') #self.name, 'w', 0)

# re-open stdout without buffering
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

# redirect stdout and stderr to the log file opened above
os.dup2(logfile.fileno(), sys.stdout.fileno())
os.dup2(logfile.fileno(), sys.stderr.fileno())

print('*******************************************************')
print('Running {0}\n'.format(sys.argv[0]))
print(ctl.datestamp()+'\n')
print('********************************************************')

if len(sys.argv) > 1:
    file_input = sys.argv[1] # Name of input file (relative path)
else:
    file_input = 'input_WRtool.in'

keys = 'exp_name cart_in cart_out_general filenames model_names level season area numclus numpcs flag_perc perc ERA_ref_orig ERA_ref_folder run_sig_calc run_compare patnames patnames_short heavy_output model_tags year_range groups group_symbols group_compare_style reference_group detrended_eof_calculation detrended_anom_for_clustering use_reference_eofs obs_name filelist visualization bounding_lat plot_margins'
keys = keys.split()
itype = [str, str, str, list, list, float, str, str, int, int, bool, float, str, str, bool, bool, list, list, bool, list, list, dict, dict, str, str, bool, bool, bool, str, str, str, float, list]

if len(itype) != len(keys):
    raise RuntimeError('Ill defined input keys in {}'.format(__file__))
itype = dict(zip(keys, itype))

defaults = dict()
defaults['numclus'] = 4 # 4 clusters
defaults['perc'] = None
defaults['flag_perc'] = False
defaults['exp_name'] = 'WRtool'
defaults['run_sig_calc'] = False
defaults['run_compare'] = True
defaults['heavy_output'] = False
defaults['detrended_anom_for_clustering'] = True
defaults['detrended_eof_calculation'] = True
defaults['use_reference_eofs'] = False
defaults['ERA_ref_folder'] = 'ERA_ref'
defaults['obs_name'] = 'ERA'
defaults['visualization'] = 'Nstereo'
defaults['bounding_lat'] = 30.
defaults['plot_margins'] = None

inputs = ctl.read_inputs(file_input, keys, n_lines = None, itype = itype, defaults = defaults)

if inputs['cart_in'][-1] != '/': inputs['cart_in'] += '/'
if inputs['cart_out_general'][-1] != '/': inputs['cart_out_general'] += '/'
if inputs['ERA_ref_folder'][-1] != '/': inputs['ERA_ref_folder'] += '/'

if inputs['filenames'] is None:
    if inputs['filelist'] is None:
        raise ValueError('Set either [filenames] or [filelist]')
    else:
        filo = open(inputs['filelist'], 'r')
        allfi = [fi.strip() for fi in filo.readlines()]
        filo.close()
        inputs['filenames'] = allfi

if len(inputs['filenames']) == 0:
    raise ValueError('No filenames specified. Set either [filenames] or [filelist].')

if inputs['model_names'] is None:
    n_mod = len(inputs['filenames'])
    inputs['model_names'] = ['ens_{}'.format(i) for i in range(n_mod)]

print('filenames: ', inputs['filenames'])
print('model names: ', inputs['model_names'])

print(inputs['groups'])
if inputs['group_symbols'] is not None:
    for k in inputs['group_symbols'].keys():
        inputs['group_symbols'][k] = inputs['group_symbols'][k][0]
print(inputs['group_symbols'])

if not inputs['flag_perc']:
    print('Considering fixed numpcs = {}\n'.format(inputs['numpcs']))
    inputs['perc'] = None
else:
    print('Considering pcs to explain {}% of variance\n'.format(inputs['perc']))

if inputs['groups'] is not None and inputs['reference_group'] is None:
    print('Setting default reference group to: {}\n'.format(inputs['groups'].keys()[0]))
    inputs['reference_group'] = inputs['groups'].keys()[0]

if not os.path.exists(inputs['cart_out_general']): os.mkdir(inputs['cart_out_general'])
inputs['cart_out'] = inputs['cart_out_general'] + inputs['exp_name'] + '/'
if not os.path.exists(inputs['cart_out']): os.mkdir(inputs['cart_out'])

if inputs['year_range'] is not None:
    inputs['year_range'] = map(int, inputs['year_range'])
# dictgrp = dict()
# dictgrp['all'] = inputs['dictgroup']
# inputs['dictgroup'] = dictgrp

if inputs['area'] == 'EAT' and inputs['numclus'] == 4:
    if inputs['patnames'] is None:
        input['patnames'] = ['NAO +', 'Sc. Blocking', 'Atl. Ridge', 'NAO -']
    if inputs['patnames_short'] is None:
        input['patnames_short'] = ['NP', 'BL', 'AR', 'NN']
elif inputs['area'] == 'PNA' and inputs['numclus'] == 4:
    if inputs['patnames'] is None:
        input['patnames'] = ['Ala. Ridge', 'Pac. Trough', 'Arctic Low', 'Arctic High']
    if inputs['patnames_short'] is None:
        input['patnames_short'] = ['AR', 'PT', 'AL', 'AH']

if inputs['patnames'] is None:
    inputs['patnames'] = ['clus_{}'.format(i) for i in range(inputs['numclus'])]
if inputs['patnames_short'] is None:
    inputs['patnames_short'] = ['c{}'.format(i) for i in range(inputs['numclus'])]

if inputs['plot_margins'] is not None:
    inputs['plot_margins'] = map(float, inputs['plot_margins'])

outname = 'out_' + std_outname(inputs['exp_name'], inputs) + '.p'
nomeout = inputs['cart_out'] + outname

ERA_ref_out = inputs['cart_out_general'] + inputs['ERA_ref_folder'] + 'out_' + std_outname(inputs['obs_name'], inputs) + '.p'
if not os.path.exists(inputs['cart_out_general'] + inputs['ERA_ref_folder']):
    os.mkdir(inputs['cart_out_general'] + inputs['ERA_ref_folder'])

if not os.path.exists(nomeout):
    print('{} not found, this is the first run. Setting up the computation..\n'.format(nomeout))
    if not os.path.exists(ERA_ref_out) is None:
        ERA_ref = cd.WRtool_from_file(inputs['ERA_ref_orig'], inputs['season'], inputs['area'], extract_level_hPa = inputs['level'], numclus = inputs['numclus'], heavy_output = True, run_significance_calc = inputs['run_sig_calc'], sel_yr_range = inputs['year_range'], numpcs = inputs['numpcs'], perc = inputs['perc'], detrended_eof_calculation = inputs['detrended_eof_calculation'], detrended_anom_for_clustering = inputs['detrended_anom_for_clustering'])
        pickle.dump(ERA_ref, open(ERA_ref_out, 'w'))
    else:
        ERA_ref = pickle.load(open(ERA_ref_out, 'r'))

    model_outs = dict()
    for modfile, modname in zip(inputs['filenames'], inputs['model_names']):
        model_outs[modname] = cd.WRtool_from_file(inputs['cart_in']+modfile, inputs['season'], inputs['area'], extract_level_hPa = inputs['level'], numclus = inputs['numclus'], heavy_output = inputs['heavy_output'], run_significance_calc = inputs['run_sig_calc'], ref_solver = ERA_ref['solver'], ref_patterns_area = ERA_ref['cluspattern_area'], sel_yr_range = inputs['year_range'], numpcs = inputs['numpcs'], perc = inputs['perc'], detrended_eof_calculation = inputs['detrended_eof_calculation'], detrended_anom_for_clustering = inputs['detrended_anom_for_clustering'], use_reference_eofs = inputs['use_reference_eofs'])

    pickle.dump([model_outs, ERA_ref], open(nomeout, 'w'))
else:
    print('Computation already performed. Reading output from {}\n'.format(nomeout))
    [model_outs, ERA_ref] = pickle.load(open(nomeout, 'r'))

latc = np.mean(ERA_ref['lat_area'])
lonc = np.mean(ERA_ref['lon_area'])
clatlo = (latc, lonc)

n_models = len(model_outs.keys())

file_res = inputs['cart_out'] + 'results_' + std_outname(inputs['exp_name'], inputs) + '.dat'
cd.out_WRtool_mainres(file_res, model_outs, ERA_ref, inputs)

cd.plot_WRtool_results(inputs['cart_out'], std_outname(inputs['exp_name'], inputs), n_models, model_outs, ERA_ref, obs_name = inputs['obs_name'], patnames = inputs['patnames'], patnames_short = inputs['patnames_short'], central_lat_lon = clatlo, groups = inputs['groups'], group_compare_style = inputs['group_compare_style'], group_symbols = inputs['group_symbols'], reference_group = inputs['reference_group'], visualization = inputs['visualization'], bounding_lat = inputs['bounding_lat'], plot_margins = inputs['plot_margins'])#, custom_model_colors = ['indianred', 'forestgreen', 'black'], compare_models = [('stoc', 'base')])

cart_out_nc = inputs['cart_out'] + 'outnc_' + std_outname(inputs['exp_name'], inputs) + '/'
if not os.path.exists(cart_out_nc): os.mkdir(cart_out_nc)
cd.out_WRtool_netcdf(cart_out_nc, model_outs, ERA_ref, inputs)

print('Check results in directory: {}\n'.format(inputs['cart_out']))
print(ctl.datestamp()+'\n')
print('Ended successfully!\n')

os.system('mv {} {}'.format(logname, inputs['cart_out']))
os.system('cp {} {}'.format(file_input, inputs['cart_out']))
