"""
Integration testcases for preprocess_stage1.py.

Copyright 2020 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import os
import unittest
import tempfile
import subprocess
from pathlib import Path

import pandas as pd

import preprocess_stage1

GIT_ROOT = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
GIT_ROOT = Path(GIT_ROOT.decode('utf-8').strip())
TEST_DATA = GIT_ROOT / Path('src/tests/data')
TEST_DATASET1 = TEST_DATA / Path('dataset1.data')
TEST_DATASET2 = TEST_DATA / Path('dataset2.data')
TEST_DATASET3 = TEST_DATA / Path('dataset3.data')
TEST_COLUMNS = TEST_DATA / Path('column_names')
EXPECTED_OUTPUT_DEFAULT_PARAMETERS = TEST_DATA / Path('preprocess_stage1_default_parameters.csv')


class PreprocessStage1Test(unittest.TestCase):
    """
    Test cases for preprocess_stage1.py

    """

    def setUp(self):
        setUp(self)

    def tearDown(self):
        tearDown(self)

    def test_default_parameters(self):
        """
        Test preprocess_stage1.py with the default parameters.

        """

        preprocess_stage1.main()
        actual_dataset = pd.read_csv(self.output_path)
        expected_dataset = pd.read_csv(EXPECTED_OUTPUT_DEFAULT_PARAMETERS)
        self.assertTrue(expected_dataset.equals(actual_dataset))


# Define setUp and tearDown functions outside of the class so that they are
# callable from other TestCase classes.
def setUp(self):
    tempfile_descriptor = tempfile.mkstemp()
    os.close(tempfile_descriptor[0])
    self.output_path = Path(tempfile_descriptor[1])
    self.prev_output_path = preprocess_stage1.OUTPUT_PATH
    self.prev_columns_file = preprocess_stage1.COLUMNS_FILE
    self.prev_input_datasets = preprocess_stage1.INPUT_DATASETS
    preprocess_stage1.OUTPUT_PATH = self.output_path
    preprocess_stage1.COLUMNS_FILE = TEST_COLUMNS
    preprocess_stage1.INPUT_DATASETS = (TEST_DATASET1,
                                        TEST_DATASET2,
                                        TEST_DATASET3)


def tearDown(self):
    preprocess_stage1.INPUT_DATASETS = self.prev_input_datasets
    preprocess_stage1.OUTPUT_PATH = self.prev_output_path
    preprocess_stage1.COLUMNS_FILE = self.prev_columns_file
    self.output_path.unlink()


if __name__ == '__main__':
    unittest.main()
