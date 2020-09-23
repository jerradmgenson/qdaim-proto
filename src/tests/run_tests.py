"""
Run all unit and integration tests and report the results.

"""

import os
import sys
import enum
import unittest
import subprocess
from pathlib import Path

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
    unit_testsuite = unittest.defaultTestLoader.discover(UNIT_PATH,
                                                         top_level_dir=SRC_PATH)

    integration_testsuite = unittest.defaultTestLoader.discover(INTEGRATION_PATH,
                                                                top_level_dir=SRC_PATH)

    testcases = (extract_tests(unit_testsuite)
                 + extract_tests(integration_testsuite))

    verdicts = list(map(run_test, testcases))
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

    return not (failures or errors or unexpected_successes)


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


if __name__ == '__main__':
    sys.exit(main())
