"""
Copyright 2020 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import subprocess
from pathlib import Path

import ingest_raw_uci_data
import ingest_cleveland_data
import gen_model

GIT_ROOT = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
GIT_ROOT = Path(GIT_ROOT.decode('utf-8').strip())
BUILD_DIR = GIT_ROOT / 'build'
INGEST_DIR = BUILD_DIR / 'ingest'

# Number of folds (or "splits") to use in cross-validation.
N_SPLITS = 20


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
    return subprocess.call(['src/preprocess.R',
                            BUILD_DIR.name,
                            str(INGEST_DIR),
                            '--random-seed', '1467756838'])

preprocess_builder = Builder(action=build_preprocess,
                             src_suffix='.csv')

def build_gen_model(target, source, env):
    return gen_model.main([str(target[0]), str(source[0]),
                           '--cross-validate', str(N_SPLITS),
                           '--outlier-scores'])

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
SConscript('src/SConscript', variant_dir=BUILD_DIR.name)
