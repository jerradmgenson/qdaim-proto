"""
Copyright 2020, 2021 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import json
import subprocess
from pathlib import Path

import ingest_raw_uci_data
import ingest_cleveland_data
import gen_model

GIT_ROOT = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
GIT_ROOT = Path(GIT_ROOT.decode('utf-8').strip())
BUILD_DIR = GIT_ROOT / 'build'
INGEST_DIR = BUILD_DIR / 'ingest'
CFG_DIR = GIT_ROOT / 'cfg'
GENERAL_CONFIG = CFG_DIR / 'general.json'
PARAMETER_GRID = CFG_DIR / 'grid_search.json'

with GENERAL_CONFIG.open() as model_gen_config_fp:
    model_gen_config = json.load(model_gen_config_fp)

training_dataset = str(BUILD_DIR / model_gen_config['training_dataset'])
test_dataset = str(BUILD_DIR / model_gen_config['test_dataset'])
validation_dataset = str(BUILD_DIR / model_gen_config['validation_dataset'])

with PARAMETER_GRID.open() as parameter_grid_fp:
    parameter_grid = parameter_grid_fp.read().strip()


def build_ingest_raw_uci_data(target, source, env):
    return ingest_raw_uci_data.main([str(INGEST_DIR), str(source[0])])

ingest_raw_uci_data_builder = Builder(action=build_ingest_raw_uci_data,
                                      suffix='.csv',
                                      src_suffix='.data')

def build_ingest_cleveland_data(target, source, env):
    return ingest_cleveland_data.main([str(INGEST_DIR), str(source[0])])

ingest_cleveland_data_builder = Builder(action=build_ingest_cleveland_data,
                                        suffix='.csv',
                                        src_suffix='.csv')

def build_preprocess(target, source, env):
    print(f'source: {source}')
    print(f'target: {target}')
    args = ['src/preprocess.R',
            str(target[0]),
            str(target[1]),
            str(target[2]),
            str(INGEST_DIR),
	    model_gen_config['test_pool'],
	    '--validation-fraction', '0',
            '--random-state', str(model_gen_config['random_state']),
            '--features']

    args.extend(model_gen_config['features'])
    if 'impute_methods' in model_gen_config:
        # Ensure that empty strings will show up when passed on the command line.
        methods = [x or '""' for x in model_gen_config['impute_methods']]
        args.append('--impute-methods')
        args.extend(methods)

    if model_gen_config.get('impute_missing'):
        args.append('--impute-missing')

    if model_gen_config.get('impute_multiple'):
        args.append('--impute-multiple')

    return subprocess.call(args)

preprocess_builder = Builder(action=build_preprocess)

def build_gen_model(target, source, env):
    return gen_model.main([str(target[0]), str(source[0]), str(source[1]),
                           '--model', model_gen_config['model'],
                           '--random-state', str(model_gen_config['random_state']),
                           '--scoring', model_gen_config['scoring'],
                           '--parameter-grid', parameter_grid,
                           '--outlier-scores',
                           '--preprocessing'] + model_gen_config['preprocessing'])

gen_model_builder = Builder(action=build_gen_model,
                            suffix='.dat',
                            src_suffix='.json')

env = Environment(BUILDERS=dict(
    Ingest_raw_uci_data=ingest_raw_uci_data_builder,
    Ingest_cleveland_data=ingest_cleveland_data_builder,
    Preprocess=preprocess_builder,
    Gen_model=gen_model_builder,
))

Export('GIT_ROOT')
Export('INGEST_DIR')
Export('env')
Export('training_dataset')
Export('validation_dataset')
Export('test_dataset')
SConscript('src/SConscript', variant_dir=BUILD_DIR.name)
