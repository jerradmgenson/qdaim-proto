"""
A common command line parser for ingest scripts.

Copyright 2020 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import argparse
from pathlib import Path

# Names of columns we are interested in studying.
# Discard all other columns from the dataset.
SUBSET_COLUMNS = ['age', 'sex', 'cp', 'trestbps', 'fbs', 'chol', 'restecg', 'thalach',
                  'exang', 'oldpeak', 'slope', 'target']


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
