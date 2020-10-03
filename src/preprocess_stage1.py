"""
Perform preprocessing activities that occur before feature selection.
Ergo, this script is designed to be run before the Feature Selection
notebook - the output of this script will be read by that notebook,
and by preprocess_stage2.py.

Inputs to this script are three individual datasets containing heart
disease data - switzerland.data, hungarian.data, and long_beach.data.

Preprocessing steps performed by this script include:
- Convert each dataset from a one-dimensional list to a dataframe.
- Create a subset of only the columns we are interested in
  (see SUBSET_COLUMNS).
- Combine the three individual datasets into a single dataset with all
  rows from the individual sets.
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
    combined_dataset = None
    for dataset_path in command_line_arguments.source:
        dataset = load_dataset(dataset_path)
        dataset_subset = dataset[SUBSET_COLUMNS]
        if combined_dataset is not None:
            combined_dataset = combined_dataset.append(dataset_subset)

        else:
            combined_dataset = dataset_subset

    # Rename num to target.
    combined_dataset.rename(mapper=dict(num='target'), axis=1, inplace=True)
    combined_dataset.to_csv(command_line_arguments.target, index=False)

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
                        nargs='+',
                        help='Raw data files to preprocess.')

    return parser.parse_args(argv)


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
