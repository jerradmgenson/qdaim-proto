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

Copyright 2020 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import re
import sys
import argparse
import subprocess
from pathlib import Path

import pandas as pd


GIT_ROOT = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
GIT_ROOT = Path(GIT_ROOT.decode('utf-8').strip())
BUILD = GIT_ROOT / Path('build')
DATA = GIT_ROOT / Path('data')

# Path to the file containing column names for the above three datasets.
COLUMNS_FILE = DATA / Path('column_names')

# Names of columns we are interested in studying.
# Discard all other columns from the dataset.
SUBSET_COLUMNS = ['age', 'sex', 'cp', 'thalrest', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'smoke', 'cigs', 'years', 'famhist', 'num']


def main(argv):
    """
    Program's 'main' function. Main execution starts here.

    """

    command_line_arguments = parse_command_line(argv)
    dataset = load_dataset(command_line_arguments.source)
    dataset = dataset[SUBSET_COLUMNS]

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

    with COLUMNS_FILE.open() as attributes_fp:
        attributes = attributes_fp.read().split(', ')

    with path.open() as dataset_fp:
        raw_data = dataset_fp.read()

    samples = []
    current_sample = []
    for count, data_point in enumerate(re.split(r'\s+', raw_data)):
        if count != 0 and count % len(attributes) == 0:
            samples.append(current_sample)
            current_sample = []

        if data_point == '-9':
            data_point = None

        current_sample.append(data_point)

    dataset = pd.DataFrame(data=samples, columns=attributes)
    return dataset


def parse_command_line(argv):
    """
    Parse the command line using argparse.

    Args
      argv: A list of command line arguments, excluding the program name.

    Returns
      The output of parse_args().

    """

    parser = argparse.ArgumentParser(description='Stage 1 preprocessor')
    parser.add_argument('target',
                        type=Path,
                        help='Path to output the result of preprocess_stage1.py.')

    parser.add_argument('source',
                        type=Path,
                        help='Raw dataset to preprocess.')

    return parser.parse_args(argv)


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
