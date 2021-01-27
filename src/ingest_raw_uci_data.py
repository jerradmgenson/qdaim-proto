#!/usr/bin/python3
"""
Ingest raw (unprocessed) UCI Heart Disease datasets.
This script is designed to be run before the Feature Selection
notebook - the output of this script will be read by that notebook,
and by preprocess.R.

Inputs to this script are three individual datasets containing heart
disease data - switzerland.data, hungarian.data, and long_beach.data.

Steps performed by this script include:
- Convert each dataset from a one-dimensional list to a dataframe.
- Create a subset of only the columns we are interested in
  (see SUBSET_COLUMNS).
- Rename num to target.

Copyright 2020, 2021 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import re
import sys
import subprocess
from pathlib import Path

import pandas as pd

from ingester_clparser import parse_command_line


# Column names for raw UCI heart disease datasets in the order that they
# appear in the datasets.
COLUMN_NAMES = ('id', 'ccf', 'age', 'sex', 'painloc', 'painexer', 'relrest', 'pncaden', 'cp', 'trestbps', 'htn', 'chol', 'smoke', 'cigs', 'years', 'fbs', 'dm', 'famhist', 'restecg', 'ekgmo', 'ekgday', 'ekgyr', 'dig', 'prop', 'nitr', 'pro', 'diuretic', 'proto', 'thaldur', 'thaltime', 'met', 'thalach', 'thalrest', 'tpeakbps', 'tpeakbpd', 'dummy', 'trestbpd', 'exang', 'xhypo', 'oldpeak', 'slope', 'rldv5', 'rldv5e', 'ca', 'restckm', 'exerckm', 'restef', 'restwm', 'exeref', 'exerwm', 'thal', 'thalsev', 'thalpul', 'earlobe', 'cmo', 'cday', 'cyr', 'num', 'lmt', 'ladprox', 'laddist', 'diag', 'cxmain', 'ramus', 'om1', 'om2', 'rcaprox', 'rcadist', 'lvx1', 'lvx2', 'lvx3', 'lvx4', 'lvf', 'cathef', 'junk', 'name')
GIT_ROOT = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
GIT_ROOT = Path(GIT_ROOT.decode('utf-8').strip())
DATA = GIT_ROOT / 'data'


def main(argv):
    """
    Program's 'main' function. Main execution starts here.

    """

    command_line_arguments = parse_command_line(argv)
    dataset = load_dataset(command_line_arguments.source)

    # Rename num to target.
    dataset.rename(mapper=dict(num='target'), axis=1, inplace=True)
    output_path = (command_line_arguments.target / command_line_arguments.source.name).with_suffix('.csv')
    dataset.to_csv(output_path, index=False)

    return 0


def load_dataset(path):
    """
    Load a dataset from a file containing whitespace-separated values
    and return a pandas dataframe.

    """

    with path.open() as dataset_fp:
        raw_data = dataset_fp.read()

    samples = []
    current_sample = []
    for count, data_point in enumerate(re.split(r'\s+', raw_data)):
        if count != 0 and count % len(COLUMN_NAMES) == 0:
            samples.append(current_sample)
            current_sample = []

        if data_point == '-9':
            data_point = None

        current_sample.append(data_point)

    dataset = pd.DataFrame(data=samples, columns=COLUMN_NAMES)
    return dataset


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
