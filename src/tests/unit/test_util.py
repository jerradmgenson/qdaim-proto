"""
Unit tests for util.py

Copyright 2020 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import util


class GetCommitHashTest(unittest.TestCase):
    """
    Tests for util.get_commit_hash()

    """

    def test_git_call_success(self):
        """
        Test get_commit_hash() when calls to git are successful.

        """

        def create_run_command_mock():
            def run_command_mock(command):
                if command == 'git diff':
                    return ''

                elif command == 'git rev-parse --verify HEAD':
                    return '26223577219e04975a8ea93b95d0ab047a0ea536'

                assert False

            return run_command_mock

        run_command_patch = patch.object(util,
                                         'run_command',
                                         new_callable=create_run_command_mock)

        with run_command_patch:
            commit_hash = util.get_commit_hash()

        self.assertEqual(commit_hash, '26223577219e04975a8ea93b95d0ab047a0ea536')

    def test_nonempty_git_diff(self):
        """
        Test get_commit_hash() when git diff returns a nonempty string.

        """

        run_command_patch = patch.object(util,
                                         'run_command',
                                         return_value='sdfigh')

        with run_command_patch:
            commit_hash = util.get_commit_hash()

        self.assertEqual(commit_hash, '')

    def test_file_not_found_error(self):
        """
        Test get_commit_hash() when git can not be found

        """

        def create_run_command_mock():
            def run_command_mock(command):
                raise FileNotFoundError()

            return run_command_mock

        run_command_patch = patch.object(util,
                                         'run_command',
                                         new_callable=create_run_command_mock)

        with run_command_patch:
            commit_hash = util.get_commit_hash()

        self.assertEqual(commit_hash, '')


class RunCommandTest(unittest.TestCase):
    """
    Tests for util.run_command()

    """

    def test_run_command(self):
        """
        Test run_command() on a typical input.

        """

        check_output_patch = patch.object(util.subprocess,
                                          'check_output',
                                          return_value=b'bbd155263aeaae63c12ad7498a0594fb2ff8d615\n')

        with check_output_patch as check_output_mock:
            command_output = util.run_command('git rev-parse --verify HEAD')

        self.assertEqual(check_output_mock.call_count, 1)
        self.assertEqual(check_output_mock.call_args[0][0],
                         ['git', 'rev-parse', '--verify', 'HEAD'])

        self.assertEqual(command_output,
                         'bbd155263aeaae63c12ad7498a0594fb2ff8d615')


class SplitInputsTests(unittest.TestCase):
    """
    Tests for util.split_inputs

    """

    def test_split_inputs(self):
        """
        Test split_inputs() on a typical input.

        """

        data = pd.DataFrame([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
        inputs = util.split_inputs(data)
        self.assertTrue((inputs == np.array([[0, 0], [0, 1], [1, 0], [1, 1]])).all())


class SplitTargetTests(unittest.TestCase):
    """
    Tests for util.split_inputs

    """

    def test_split_target(self):
        """
        Test split_target() on a typical input.

        """

        data = pd.DataFrame([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
        inputs = util.split_target(data)
        self.assertTrue((inputs == np.array([0, 1, 1, 0])).all())


if __name__ == '__main__':
    unittest.main()
