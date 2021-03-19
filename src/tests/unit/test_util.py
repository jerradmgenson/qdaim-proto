"""
Unit tests for util.py

Copyright 2020 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

import util


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


class GetCommitHashTests(unittest.TestCase):
    """
    Tests for util.get_commit_hash

    """

    def test_get_commit_hash_with_valid_commit_hash_file(self):
        """
        Test get_commit_hash() with a valid commit_hash file.

        """

        expected_commit_hash = '367138724bb7f51500b9e64cf5536d6836c4a619'
        commit_hash_file_old = util.COMMIT_HASH_FILE
        try:
            temp_fp, commit_hash_file_mock = tempfile.mkstemp()
            os.close(temp_fp)
            util.COMMIT_HASH_FILE = Path(commit_hash_file_mock)
            with open(commit_hash_file_mock, 'w') as commit_hash_fp:
                commit_hash_fp.write(expected_commit_hash + '\n')

            commit_hash = util.get_commit_hash()
            self.assertEqual(commit_hash, expected_commit_hash)

        finally:
            util.COMMIT_HASH_FILE = commit_hash_file_old
            os.remove(commit_hash_file_mock)


if __name__ == '__main__':
    unittest.main()
