"""
Run all unit and integration tests and report the results.

"""

import io
import re
import os
import sys
import enum
import time
import unittest
import subprocess
import multiprocessing
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


class Testrunner(enum.Enum):
    """
    Enumerates possible testrunner names.

    """

    UNITTEST = enum.auto()
    PYLINT = enum.auto()


def main():
    start_time = time.time()
    src_files = (p for p in SRC_PATH.iterdir() if p.is_file())
    py_files = [str(p) for p in src_files if p.suffix == '.py']
    pylint_jobs = [(run_pylint, p) for p in py_files]
    unittest_jobs = [(run_unittest, py_files)]
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        test_results = pool.map(run_job, unittest_jobs + pylint_jobs)

    pylint_results = [x[1] for x in test_results if x[0] == Testrunner.PYLINT]
    unittest_results = [x for x in test_results if x[0] == Testrunner.UNITTEST][0][1]
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

    failed = (failures
              or errors
              or unexpected_successes
              or coverage_percentage < 100
              or pylint_failure)

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

    pylint_result = PylintResult(path=path,
                                 score=score,
                                 errors=errors,
                                 report=pylint_report)

    return Testrunner.PYLINT, pylint_result


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
    coverage_stream = io.StringIO()
    coverage_percentage = coverage.report(file=coverage_stream,
                                          include=coverage_files)

    coverage_report = coverage_stream.getvalue()
    coverage_stream.close()
    unittest_result = UnittestResult(verdicts=verdicts,
                                     coverage_percentage=coverage_percentage,
                                     coverage_report=coverage_report)

    return Testrunner.UNITTEST, unittest_result


def run_job(job):
    """
    Run a job and return its results.

    Args
      job: A tuple of `(function, function_argument)`.

    Returns
      The value returned by `function`.

    """

    return job[0](job[1])


if __name__ == '__main__':
    sys.exit(main())
