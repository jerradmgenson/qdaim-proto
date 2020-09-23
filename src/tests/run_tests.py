"""
Run all unit and integration tests and report the results.

"""

import re
import os
import sys
import enum
import unittest
import subprocess
from pathlib import Path
from collections import namedtuple

from coverage import Coverage
from pylint import epylint as lint

# Path to the root of the git repository.
GIT_ROOT = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
GIT_ROOT = Path(GIT_ROOT.decode('utf-8').strip())
SRC_PATH = GIT_ROOT / Path('src')
TESTS_PATH = SRC_PATH / Path('tests')
UNIT_PATH = TESTS_PATH / Path('unit')
INTEGRATION_PATH = TESTS_PATH / Path('integration')


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


def main():
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

    src_files = (p for p in SRC_PATH.iterdir() if p.is_file())
    py_files = [str(p) for p in src_files if p.suffix == '.py']
    coverage_percentage = coverage.report(file=sys.stdout, include=py_files)
    pylint_results = map(run_pylint, py_files)
    pylint_failure = False
    print('\nPylint scores')
    for pylint_result in pylint_results:
        if pylint_result.errors:
            print(f'{pylint_result.path}.... error')
            pylint_failure = True

        else:
            print(f'{pylint_result.path}.... {pylint_result.score}/10')

        if pylint_result.score < 9:
            pylint_failure = True

        if pylint_result.errors or pylint_result.score < 9:
            print(pylint_result.report)

    return (failures
            or errors
            or unexpected_successes
            or coverage_percentage < 100
            or pylint_failure)


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

    assert test_result.testsRun == 1
    if test_result.failures:
        print('failure\n')
        print(test_result.failures[0][1], file=sys.stderr)
        return Verdict.FAILURE

    elif test_result.errors:
        print('error\n')
        print(test_result.errors[0][1], file=sys.stderr)
        return Verdict.ERROR

    elif test_result.skipped:
        print('skipped')
        return Verdict.SKIPPED

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
    score = float(re.search(r'Your code has been rated at (\d+\.\d+)',
                            pylint_report).group(1))

    return PylintResult(path=path,
                        score=score,
                        errors=errors,
                        report=pylint_report)


if __name__ == '__main__':
    sys.exit(main())
