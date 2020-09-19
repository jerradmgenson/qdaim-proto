import os
import unittest
import tempfile
import subprocess
from pathlib import Path

import pandas as pd

import preprocess_stage1


class PreprocessStage1Test(unittest.TestCase):
    """
    Test cases for preprocess_stage1.py

    """

    GIT_ROOT = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
    GIT_ROOT = Path(GIT_ROOT.decode('utf-8').strip())
    TEST_DATA = GIT_ROOT / Path('src/test_data')
    TEST_DATASET1 = TEST_DATA / Path('dataset1.data')
    TEST_DATASET2 = TEST_DATA / Path('dataset2.data')
    TEST_DATASET3 = TEST_DATA / Path('dataset3.data')
    TEST_COLUMNS = TEST_DATA / Path('column_names')
    EXPECTED_OUTPUT_DEFAULT_PARAMETERS = TEST_DATA / Path('preprocess_stage1_default_parameters.csv')

    def setUp(self):
        tempfile_descriptor = tempfile.mkstemp()
        os.close(tempfile_descriptor[0])
        self.output_path = Path(tempfile_descriptor[1])
        self.prev_output_path = preprocess_stage1.OUTPUT_PATH
        self.prev_columns_file = preprocess_stage1.COLUMNS_FILE
        self.prev_input_datasets = preprocess_stage1.INPUT_DATASETS
        preprocess_stage1.OUTPUT_PATH = self.output_path
        preprocess_stage1.COLUMNS_FILE = self.TEST_COLUMNS
        preprocess_stage1.INPUT_DATASETS = (self.TEST_DATASET1,
                                            self.TEST_DATASET2,
                                            self.TEST_DATASET3)

    def tearDown(self):
        preprocess_stage1.INPUT_DATASETS = self.prev_input_datasets
        preprocess_stage1.OUTPUT_PATH = self.prev_output_path
        preprocess_stage1.COLUMNS_FILE = self.prev_columns_file
        self.output_path.unlink()

    def test_default_parameters(self):
        """
        Test preprocess_stage1.py with the default parameters.

        """

        preprocess_stage1.main()
        actual_dataset = pd.read_csv(self.output_path)
        expected_dataset = pd.read_csv(self.EXPECTED_OUTPUT_DEFAULT_PARAMETERS)
        self.assertTrue(expected_dataset.equals(actual_dataset))


if __name__ == '__main__':
    unittest.main()
