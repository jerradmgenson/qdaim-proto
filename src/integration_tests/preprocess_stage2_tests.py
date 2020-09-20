import os
import unittest
import tempfile
import subprocess
from pathlib import Path

import pandas as pd

import preprocess_stage1
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
    BINARY_TESTING_DATASET1 = TEST_DATA / Path('binary_testing_dataset1.csv')
    BINARY_TRAINING_DATASET1 = TEST_DATA / Path('binary_training_dataset1.csv')
    BINARY_VALIDATION_DATASET1 = TEST_DATA / Path('binary_validation_dataset1.csv')
    BINARY_TESTING_DATASET2 = TEST_DATA / Path('binary_testing_dataset2.csv')
    BINARY_TRAINING_DATASET2 = TEST_DATA / Path('binary_training_dataset2.csv')
    TERNARY_TESTING_DATASET = TEST_DATA / Path('ternary_testing_dataset.csv')
    TERNARY_TRAINING_DATASET = TEST_DATA / Path('ternary_training_dataset.csv')
    TERNARY_VALIDATION_DATASET = TEST_DATA / Path('ternary_validation_dataset.csv')
    MULTICLASS_TESTING_DATASET = TEST_DATA / Path('multiclass_testing_dataset.csv')
    MULTICLASS_TRAINING_DATASET = TEST_DATA / Path('multiclass_training_dataset.csv')
    MULTICLASS_VALIDATION_DATASET = TEST_DATA / Path('multiclass_validation_dataset.csv')


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

        self.prev_classification_type = preprocess_stage2.CLASSIFICATION_TYPE
        self.prev_random_seed = preprocess_stage2.RANDOM_SEED
        self.prev_input_dataset_path = preprocess_stage2.INPUT_DATASET_PATH
        self.prev_testing_dataset_path = preprocess_stage2.TESTING_DATASET_PATH
        self.prev_training_dataset_path = preprocess_stage2.TRAINING_DATASET_PATH
        self.prev_validation_dataset_path = preprocess_stage2.VALIDATION_DATASET_PATH

        preprocess_stage2.CLASSIFICATION_TYPE = preprocess_stage2.ClassificationType.BINARY
        preprocess_stage2.RANDOM_SEED = self.RANDOM_SEED
        preprocess_stage2.INPUT_DATASET_PATH = \
            preprocess_stage1_tests.EXPECTED_OUTPUT_DEFAULT_PARAMETERS

        preprocess_stage2.TESTING_DATASET_PATH = self.testing_dataset_path
        preprocess_stage2.TRAINING_DATASET_PATH = self.training_dataset_path
        preprocess_stage2.VALIDATION_DATASET_PATH = self.validation_dataset_path

    def tearDown(self):
        preprocess_stage2.CLASSIFICATION_TYPE = self.prev_classification_type
        preprocess_stage2.RANDOM_SEED = self.prev_random_seed
        preprocess_stage2.INPUT_DATASET_PATH = self.prev_input_dataset_path
        preprocess_stage2.TESTING_DATASET_PATH = self.prev_testing_dataset_path
        preprocess_stage2.TRAINING_DATASET_PATH = self.prev_training_dataset_path
        preprocess_stage2.VALIDATION_DATASET_PATH = self.prev_validation_dataset_path

        self.testing_dataset_path.unlink()
        self.training_dataset_path.unlink()
        self.validation_dataset_path.unlink()

    def test_binary_classification_datasets(self):
        """
        Test creation of training, testing, and validation datasets for
        binary classification

        """

        preprocess_stage2.main()

        actual_testing_dataset = pd.read_csv(self.testing_dataset_path)
        expected_testing_dataset = pd.read_csv(self.BINARY_TESTING_DATASET1)
        self.assertTrue(expected_testing_dataset.equals(actual_testing_dataset))

        actual_training_dataset = pd.read_csv(self.training_dataset_path)
        expected_training_dataset = pd.read_csv(self.BINARY_TRAINING_DATASET1)
        self.assertTrue(expected_training_dataset.equals(actual_training_dataset))

        actual_validation_dataset = pd.read_csv(self.validation_dataset_path)
        expected_validation_dataset = pd.read_csv(self.BINARY_VALIDATION_DATASET1)
        self.assertTrue(expected_validation_dataset.equals(actual_validation_dataset))

    def test_ternary_classification_datasets(self):
        """
        Test creation of training, testing, and validation datasets for
        ternary classification

        """

        preprocess_stage2.CLASSIFICATION_TYPE = preprocess_stage2.ClassificationType.TERNARY
        preprocess_stage2.main()

        actual_testing_dataset = pd.read_csv(self.testing_dataset_path)
        expected_testing_dataset = pd.read_csv(self.TERNARY_TESTING_DATASET)
        self.assertTrue(expected_testing_dataset.equals(actual_testing_dataset))

        actual_training_dataset = pd.read_csv(self.training_dataset_path)
        expected_training_dataset = pd.read_csv(self.TERNARY_TRAINING_DATASET)
        self.assertTrue(expected_training_dataset.equals(actual_training_dataset))

        actual_validation_dataset = pd.read_csv(self.validation_dataset_path)
        expected_validation_dataset = pd.read_csv(self.TERNARY_VALIDATION_DATASET)
        self.assertTrue(expected_validation_dataset.equals(actual_validation_dataset))

    def test_multiclass_classification_datasets(self):
        """
        Test creation of training, testing, and validation datasets for
        multiclass classification

        """

        preprocess_stage2.CLASSIFICATION_TYPE = preprocess_stage2.ClassificationType.MULTICLASS
        preprocess_stage2.main()

        actual_testing_dataset = pd.read_csv(self.testing_dataset_path)
        expected_testing_dataset = pd.read_csv(self.MULTICLASS_TESTING_DATASET)
        self.assertTrue(expected_testing_dataset.equals(actual_testing_dataset))

        actual_training_dataset = pd.read_csv(self.training_dataset_path)
        expected_training_dataset = pd.read_csv(self.MULTICLASS_TRAINING_DATASET)
        self.assertTrue(expected_training_dataset.equals(actual_training_dataset))

        actual_validation_dataset = pd.read_csv(self.validation_dataset_path)
        expected_validation_dataset = pd.read_csv(self.MULTICLASS_VALIDATION_DATASET)
        self.assertTrue(expected_validation_dataset.equals(actual_validation_dataset))

    def test_training_testing_datasets(self):
        """
        Test creation of only training and testing datasets (no validation).

        """

        prev_validation_fraction = preprocess_stage2.VALIDATION_FRACTION
        preprocess_stage2.VALIDATION_FRACTION = 0
        preprocess_stage2.main()
        preprocess_stage2.VALIDATION_FRACTION = prev_validation_fraction

        actual_testing_dataset = pd.read_csv(self.testing_dataset_path)
        expected_testing_dataset = pd.read_csv(self.BINARY_TESTING_DATASET2)
        self.assertTrue(expected_testing_dataset.equals(actual_testing_dataset))

        actual_training_dataset = pd.read_csv(self.training_dataset_path)
        expected_training_dataset = pd.read_csv(self.BINARY_TRAINING_DATASET2)
        self.assertTrue(expected_training_dataset.equals(actual_training_dataset))

    def test_with_preprocess_stage1(self):
        """
        Test running preprocess_stage2.py on the output of preprocess_stage1.py.

        """

        preprocess_stage1_tests.setUp(self)
        preprocess_stage2.INPUT_DATASET_PATH = self.output_path
        preprocess_stage1.main()
        preprocess_stage2.main()

        actual_testing_dataset = pd.read_csv(self.testing_dataset_path)
        expected_testing_dataset = pd.read_csv(self.BINARY_TESTING_DATASET1)
        self.assertTrue(expected_testing_dataset.equals(actual_testing_dataset))

        actual_training_dataset = pd.read_csv(self.training_dataset_path)
        expected_training_dataset = pd.read_csv(self.BINARY_TRAINING_DATASET1)
        self.assertTrue(expected_training_dataset.equals(actual_training_dataset))

        actual_validation_dataset = pd.read_csv(self.validation_dataset_path)
        expected_validation_dataset = pd.read_csv(self.BINARY_VALIDATION_DATASET1)
        self.assertTrue(expected_validation_dataset.equals(actual_validation_dataset))

        preprocess_stage1_tests.tearDown(self)
