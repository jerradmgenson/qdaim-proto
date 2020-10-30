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
import json
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
MIN_LINTER_SCORE = 9


def main(argv):
    start_time = time.time()
    command_line_arguments = parse_command_line(argv)
    python_files = run_tests.get_files_with_extension(SRC_PATH,
                                                      '.py',
                                                      exclude=['unit', 'integration', '__init__.py'])

    r_files = run_tests.get_files_with_extension(SRC_PATH,
                                                 '.R',
                                                 exclude=['unit', 'integration'])

    linter_results = []
    with multiprocessing.Pool(command_line_arguments.cpu) as pool:
        linter_results.extend(pool.map(run_pylint, python_files))
        linter_results.extend(pool.map(run_lintr, r_files))

    print('Linter scores')
    failure = False
    for linter_result in linter_results:
        if linter_result.errors:
            print(f'{linter_result.path}.... error')
            failure = True

        else:
            print(f'{linter_result.path}.... {linter_result.score}/10')

        if linter_result.score < MIN_LINTER_SCORE:
            failure = True

        if linter_result.errors or linter_result.score < MIN_LINTER_SCORE:
            print(linter_result.report)

    if failure:
        print('\nFinal status: FAIL')

    else:
        print('\nFinal status: SUCCESS')

    print(f'Total runtime: {time.time() - start_time:.2f}')

    return failure


LinterResult = namedtuple('LinterResult', 'path score errors report')


def run_pylint(path):
    """
    Run pylint on the target file at `path` and return a LinterResult object.

    """

    try:
        pylint_stdout, pylint_stderr = lint.py_run(path, return_std=True)
        pylint_report = pylint_stdout.read()
        pylint_error = pylint_stderr.read()

    finally:
        pylint_stdout.close()
        pylint_stderr.close()

    if pylint_error:
        return LinterResult(path=path, score=0.0, errors=True)

    errors = bool(re.search(r':\d+: error', pylint_report))
    try:
        score = float(re.search(r'Your code has been rated at (\d+\.\d+)',
                                pylint_report).group(1))

    except AttributeError:
        score = 0
        errors = True

    pylint_result = LinterResult(path=path,
                                 score=score,
                                 errors=errors,
                                 report=pylint_report)

    return pylint_result


def run_lintr(path):
    """
    Run lintr on the target file at `path` and return a LinterResult object.

    """

    try:
        lintr_stdout = subprocess.check_output(['R', '-e', f'library(lintr); lint("{str(path)}")']).decode('utf-8')

    except subprocess.CalledProcessError:
        return LinterResult(path=path,
                            score=0,
                            errors=True,
                            report='')

    errors = re.search(r'^Error:', lintr_stdout) is not None
    issues = len(re.findall(r'^\w+:', lintr_stdout))
    cloc_stdout = subprocess.check_output(['cloc', str(path), '--json']).decode('utf-8')
    cloc_json = json.loads(cloc_stdout)
    lines_of_code = cloc_json['SUM']['code']
    score = round((1 - issues / lines_of_code) * 10, 2)

    return LinterResult(path=path,
                        score=score,
                        errors=errors,
                        report=re.split(r'> library\(lintr\); lint\(.+\)', lintr_stdout)[1])


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
