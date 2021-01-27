"""
A common command line parser for ingest scripts.

Copyright 2020, 2021 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import argparse
from pathlib import Path


def parse_command_line(argv):
    """
    Parse the command line using argparse.

    Args
      argv: A list of command line arguments, excluding the program name.

    Returns
      The output of parse_args().

    """

    parser = argparse.ArgumentParser(description='Heart disease data ingester.')
    parser.add_argument('target',
                        type=Path,
                        help='Path to write the ingested CSV data file.')

    parser.add_argument('source',
                        type=Path,
                        help='Raw input dataset to ingest.')

    return parser.parse_args(argv)
