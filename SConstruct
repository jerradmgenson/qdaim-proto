"""
Copyright 2020 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import subprocess
from pathlib import Path

from src import preprocess_stage1
from src import preprocess_stage2
from src import gen_model

GIT_ROOT = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
GIT_ROOT = Path(GIT_ROOT.decode('utf-8').strip())
BUILD_DIR = GIT_ROOT / Path('build')


def build_preprocess_stage1(target, source, env):
    return preprocess_stage1.main([str(target[0])] + [str(s) for s in source])

preprocess_stage1_builder = Builder(action=build_preprocess_stage1,
                                    suffix='.csv',
                                    src_suffix='.data')

def build_preprocess_stage2(target, source, env):
   return preprocess_stage2.main([str(BUILD_DIR), str(source[0])])

preprocess_stage2_builder = Builder(action=build_preprocess_stage2,
                                    src_suffix='.csv')

def build_gen_model(target, source, env):
    return gen_model.main([str(target[0]), str(source[0]), '--roc-curve'])

gen_model_builder = Builder(action=build_gen_model,
                            suffix='.dat',
                            src_suffix='.json')

env = Environment(BUILDERS=dict(
    Preprocess_stage1=preprocess_stage1_builder,
    Preprocess_stage2=preprocess_stage2_builder,
    Gen_model=gen_model_builder,
))

Export('GIT_ROOT')
Export('env')
SConscript('src/SConscript', variant_dir=BUILD_DIR.name)
