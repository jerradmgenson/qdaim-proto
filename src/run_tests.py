#!/usr/bin/python3
"""
Run all unit and integration tests and report the results.

Copyright 2020 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import io
import re
import os
import sys
import enum
import time
import argparse
import unittest
import subprocess
import multiprocessing
from pathlib import Path
from collections import namedtuple

from coverage import Coverage

# Path to the root of the git repository.
GIT_ROOT = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
GIT_ROOT = Path(GIT_ROOT.decode('utf-8').strip())
SRC_PATH = GIT_ROOT / Path('src')
TESTS_PATH = SRC_PATH / Path('tests')
UNIT_PATH = TESTS_PATH / Path('unit')
INTEGRATION_PATH = TESTS_PATH / Path('integration')

# The minimum coverage percentage required for the coverage test to pass.
MIN_COVERAGE_PERCENT = 100


class Verdict(enum.Enum):
    """
    Enumerates possible testcase verdicts.

    """

    SUCCESS = enum.auto()
    FAILURE = enum.auto()
    ERROR = enum.auto()
    SKIPPED = enum.auto()
    EXPECTED_FAILURE = enum.auto()
    UNEXPECTED_SUCCESS = enum.auto()


def main(argv):
    start_time = time.time()
    command_line_arguments = parse_command_line(argv)
    coverage_files = get_python_files(SRC_PATH,
                                      exclude=['tests',
                                               'run_tests.py',
                                               'run_linters.py'])

    jobs = [(run_unittest, coverage_files)]
    with multiprocessing.Pool(command_line_arguments.cpu) as pool:
        test_results = pool.map(run_job, jobs)

    unittest_results = test_results[0]
    verdicts = unittest_results.verdicts
    coverage_percentage = unittest_results.coverage_percentage
    total_tests = len(verdicts)
    successes = verdicts.count(Verdict.SUCCESS)
    failures = verdicts.count(Verdict.FAILURE)
    errors = verdicts.count(Verdict.ERROR)
    skipped = verdicts.count(Verdict.SKIPPED)
    expected_failures = verdicts.count(Verdict.EXPECTED_FAILURE)
    unexpected_successes = verdicts.count(Verdict.UNEXPECTED_SUCCESS)

    report = f'\nTotal tests:             {total_tests}\n'
    report += f'Successes:               {successes}\n'
    report += f'Failures:                {failures}\n'
    report += f'Errors:                  {errors}\n'
    report += f'Skipped:                 {skipped}\n'
    report += f'Expected failures:       {expected_failures}\n'
    report += f'Unexpected successes:    {unexpected_successes}\n'
    print(report)
    print(unittest_results.coverage_report)
    failed = (failures
              or errors
              or unexpected_successes
              or coverage_percentage < MIN_COVERAGE_PERCENT)

    if failed:
        print('\nFinal status: FAIL')

    else:
        print('\nFinal status: SUCCESS')

    print(f'Total runtime: {time.time() - start_time:.2f}')

    return failed


def run_test(test_case):
    """
    Run a single test case and print errors/failures to stderr.

    Args
      test_case: An instance of unittest.TestCase.

    Returns
      An attribute of Verdict.

    """

    print(f'{test_case.id()}.... ', end='')
    with open(os.devnull, 'w') as null_stream:
        prev_stdout = sys.stdout
        prev_stderr = sys.stderr
        try:
            sys.stdout = null_stream
            sys.stderr = null_stream
            test_result = test_case.run()

        finally:
            sys.stdout = prev_stdout
            sys.stderr = prev_stderr

    if test_result is None:
        print('skipped')
        return Verdict.SKIPPED

    assert test_result.testsRun == 1
    if test_result.failures:
        print('failure\n')
        print(test_result.failures[0][1], file=sys.stderr)
        return Verdict.FAILURE

    elif test_result.errors:
        print('error\n')
        print(test_result.errors[0][1], file=sys.stderr)
        return Verdict.ERROR

    elif test_result.expectedFailures:
        print('expected failure')
        return Verdict.EXPECTED_FAILURE

    elif test_result.unexpectedSuccesses:
        print('unexpected success')
        return Verdict.UNEXPECTED_SUCCESS

    else:
        print('success')
        return Verdict.SUCCESS


def extract_tests(testsuite):
    """
    Extract individual TestCases from a TestSuite and return them in a list.

    """

    testsuite_components = list(testsuite)
    testcases = []
    for component in testsuite_components:
        if isinstance(component, unittest.TestCase):
            testcases.append(component)

        elif isinstance(component, unittest.TestSuite):
            testcases.extend(extract_tests(component))

        else:
            assert False

    return testcases


UnittestResult = namedtuple('UnittestResult',
                            'verdicts coverage_percentage coverage_report')


def run_unittest(coverage_files):
    """
    Run all unittest testcases.

    Args
      coverage_files: List of files to include in test coverage report.

    Returns
      An instance of UnittestResult.

    """

    coverage = Coverage()
    coverage.start()
    unit_testsuite = unittest.defaultTestLoader.discover(UNIT_PATH,
                                                         top_level_dir=SRC_PATH)

    integration_testsuite = unittest.defaultTestLoader.discover(INTEGRATION_PATH,
                                                                top_level_dir=SRC_PATH)

    testcases = (extract_tests(unit_testsuite)
                 + extract_tests(integration_testsuite))

    verdicts = list(map(run_test, testcases))
    coverage.stop()
    coverage.save()
    if coverage_files:
        coverage_stream = io.StringIO()
        coverage_percentage = coverage.report(file=coverage_stream,
                                              include=coverage_files,
                                              show_missing=True)

        coverage_report = coverage_stream.getvalue()
        coverage_stream.close()

    else:
        # coverage not run on any file changes.
        # Set coverage percentage to the highest possible value (100).
        coverage_percentage = 100
        coverage_report = ''

    unittest_result = UnittestResult(verdicts=verdicts,
                                     coverage_percentage=coverage_percentage,
                                     coverage_report=coverage_report)

    return unittest_result


def run_job(job):
    """
    Run a job and return its results.

    Args
      job: A tuple of `(function, function_argument)`.

    Returns
      The value returned by `function`.

    """

    return job[0](job[1])


def get_changed_files():
    """
    Get files that changed since the last commit. If there have been no
    changes since the last commit, return the files that changed between
    the last commit and master.

    """

    git_diff = subprocess.check_output(['git', 'diff']).decode('utf-8')
    if not git_diff.strip():
        git_diff = subprocess.check_output(['git', 'diff', 'origin/master']).decode('utf-8')

    changed_files = set(str(GIT_ROOT / Path(x)) for x in re.findall(r'\+\+\+ b/(.+)', git_diff))

    return changed_files


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


def get_python_files(path, exclude=None):
    """
    Get all Python files that live in `path` or any of its children.

    Args
      path: The path to the directory to search for Python files as either a
            string or a Path object.
      exclude: (Optional) A list of directory names not to descend into.

    Returns
      A set of path strings to all Python files within the directory at `path`.

    """

    if exclude is None:
        return set(str(x) for x in Path(path).glob('**/*.py'))

    python_files = set()
    for child in Path(path).iterdir():
        if child.is_dir() and child.name not in exclude:
            python_files.update(get_python_files(child, exclude=exclude))

        elif child.suffix == '.py' and child.name not in exclude:
            python_files.add(str(child))

    return python_files


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))