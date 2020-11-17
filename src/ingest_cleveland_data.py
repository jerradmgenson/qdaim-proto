#!/usr/bin/python3
"""
Ingest data from the preprocessed UCI Heart Disease Cleveland dataset.

Steps performed by this script include:
- Create a subset of only the columns we are interested in
  (see SUBSET_COLUMNS).
- Replace occurences of '?' with 'na'.


Copyright 2020 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import sys

import pandas as pd

from ingester_clparser import parse_command_line


# Names of columns we are interested in studying.
# Discard all other columns from the dataset.
SUBSET_COLUMNS = ['age', 'sex', 'cp', 'trestbps', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'target']


def main(argv):
    """
    Program's 'main' function. Main execution starts here.

    """

    command_line_arguments = parse_command_line(argv)
    dataset = pd.read_csv(command_line_arguments.source)
    dataset = dataset[SUBSET_COLUMNS]
    dataset.replace(to_replace='?', inplace=True)
    output_path = (command_line_arguments.target / command_line_arguments.source.name).with_suffix('.csv')
    dataset.to_csv(output_path, index=False)

    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
