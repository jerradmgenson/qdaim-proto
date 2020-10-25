#!/usr/bin/python3
"""
Run linters on code in src (excluding tests) and report the results.

Copyright 2020 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import re
import sys
import time
import argparse
import subprocess
import multiprocessing
from pathlib import Path
from collections import namedtuple

from pylint import epylint as lint

import run_tests

# Path to the root of the git repository.
GIT_ROOT = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
GIT_ROOT = Path(GIT_ROOT.decode('utf-8').strip())
SRC_PATH = GIT_ROOT / Path('src')
TESTS_PATH = SRC_PATH / Path('tests')
UNIT_PATH = TESTS_PATH / Path('unit')
INTEGRATION_PATH = TESTS_PATH / Path('integration')

# The minimum pylint score (/10) required for the pylint test to pass.
MIN_PYLINT_SCORE = 9


def main(argv):
    start_time = time.time()
    command_line_arguments = parse_command_line(argv)
    pylint_files = run_tests.get_python_files(SRC_PATH,
                                              exclude=['unit', 'integration', '__init__.py'])

    with multiprocessing.Pool(command_line_arguments.cpu) as pool:
        pylint_results = pool.map(run_pylint, pylint_files)

    print('Pylint scores')
    failure = False
    for pylint_result in pylint_results:
        if pylint_result.errors:
            print(f'{pylint_result.path}.... error')
            failure = True

        else:
            print(f'{pylint_result.path}.... {pylint_result.score}/10')

        if pylint_result.score < MIN_PYLINT_SCORE:
            failure = True

        if pylint_result.errors or pylint_result.score < MIN_PYLINT_SCORE:
            print(pylint_result.report)

    if failure:
        print('\nFinal status: FAIL')

    else:
        print('\nFinal status: SUCCESS')

    print(f'Total runtime: {time.time() - start_time:.2f}')

    return failure


PylintResult = namedtuple('PylintResult', 'path score errors report')


def run_pylint(path):
    """
    Run pylint on the target file at `path` and return a PylintResult object.

    """

    try:
        pylint_stdout, pylint_stderr = lint.py_run(path, return_std=True)
        pylint_report = pylint_stdout.read()
        pylint_error = pylint_stderr.read()

    finally:
        pylint_stdout.close()
        pylint_stderr.close()

    if pylint_error:
        return PylintResult(path=path, score=0.0, errors=True)

    errors = bool(re.search(r':\d+: error', pylint_report))
    try:
        score = float(re.search(r'Your code has been rated at (\d+\.\d+)',
                                pylint_report).group(1))

    except AttributeError:
        score = 0
        errors = True

    pylint_result = PylintResult(path=path,
                                 score=score,
                                 errors=errors,
                                 report=pylint_report)

    return pylint_result


def parse_command_line(argv):
    """
    Parse the command line arguments given by argv.

    """

    description = 'Test source code for QDAIM.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cpu',
                        type=int,
                        default=multiprocessing.cpu_count(),
                        help='Number of processes to use for running tests.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
