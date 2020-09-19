import os
import unittest
import tempfile
import subprocess
from pathlib import Path

import pandas as pd

import preprocess_stage2
from integration_tests import preprocess_stage1_tests


class PreprocessStage2Test(unittest.TestCase):
    """
    Test cases for preprocess_stage2.py

    """

    RANDOM_SEED = 667252912
    GIT_ROOT = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
    GIT_ROOT = Path(GIT_ROOT.decode('utf-8').strip())
    TEST_DATA = GIT_ROOT / Path('src/test_data')
    TESTING_DATASET1_PATH = TEST_DATA / Path('testing_dataset1.csv')
    TRAINING_DATASET1_PATH = TEST_DATA / Path('training_dataset1.csv')
    VALIDATION_DATASET1_PATH = TEST_DATA / Path('validation_dataset1.csv')

    def setUp(self):
        tempfile_descriptor1 = tempfile.mkstemp()
        os.close(tempfile_descriptor1[0])
        self.testing_dataset_path = Path(tempfile_descriptor1[1])
        tempfile_descriptor2 = tempfile.mkstemp()
        os.close(tempfile_descriptor2[0])
        self.training_dataset_path = Path(tempfile_descriptor2[1])
        tempfile_descriptor3 = tempfile.mkstemp()
        os.close(tempfile_descriptor3[0])
        self.validation_dataset_path = Path(tempfile_descriptor3[1])

        self.prev_random_seed = preprocess_stage2.RANDOM_SEED
        self.prev_input_dataset_path = preprocess_stage2.INPUT_DATASET_PATH
        self.prev_testing_dataset_path = preprocess_stage2.TESTING_DATASET_PATH
        self.prev_training_dataset_path = preprocess_stage2.TRAINING_DATASET_PATH
        self.prev_validation_dataset_path = preprocess_stage2.VALIDATION_DATASET_PATH

        preprocess_stage2.RANDOM_SEED = self.RANDOM_SEED
        preprocess_stage2.INPUT_DATASET_PATH = \
            preprocess_stage1_tests.PreprocessStage1Test.EXPECTED_OUTPUT_DEFAULT_PARAMETERS

        preprocess_stage2.TESTING_DATASET_PATH = self.testing_dataset_path
        preprocess_stage2.TRAINING_DATASET_PATH = self.training_dataset_path
        preprocess_stage2.VALIDATION_DATASET_PATH = self.validation_dataset_path

    def tearDown(self):
        preprocess_stage2.RANDOM_SEED = self.prev_random_seed
        preprocess_stage2.INPUT_DATASET_PATH = self.prev_input_dataset_path
        preprocess_stage2.TESTING_DATASET_PATH = self.prev_testing_dataset_path
        preprocess_stage2.TRAINING_DATASET_PATH = self.prev_training_dataset_path
        preprocess_stage2.VALIDATION_DATASET_PATH = self.prev_validation_dataset_path

        self.testing_dataset_path.unlink()
        self.training_dataset_path.unlink()
        self.validation_dataset_path.unlink()

    def test_training_testing_validation(self):
        """
        Test creation of training, testing, and validation datasets.

        """

        preprocess_stage2.main()

        actual_testing_dataset = pd.read_csv(self.testing_dataset_path)
        expected_testing_dataset = pd.read_csv(self.TESTING_DATASET1_PATH)
        self.assertTrue(expected_testing_dataset.equals(actual_testing_dataset))

        actual_training_dataset = pd.read_csv(self.training_dataset_path)
        expected_training_dataset = pd.read_csv(self.TRAINING_DATASET1_PATH)
        self.assertTrue(expected_training_dataset.equals(actual_training_dataset))

        actual_validation_dataset = pd.read_csv(self.validation_dataset_path)
        expected_validation_dataset = pd.read_csv(self.VALIDATION_DATASET1_PATH)
        self.assertTrue(expected_validation_dataset.equals(actual_validation_dataset))


if __name__ == '__main__':
    unittest.main()
