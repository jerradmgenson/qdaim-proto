"""
Integration testcases for preprocess_stage2.R

Copyright 2020 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import unittest
import tempfile
import subprocess
from pathlib import Path

import pandas as pd

import preprocess_stage1
from tests.integration import test_preprocess_stage1

RANDOM_SEED = '667252912'


class PreprocessStage2Test(unittest.TestCase):
    """
    Test cases for preprocess_stage2.R

    """

    GIT_ROOT = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
    GIT_ROOT = Path(GIT_ROOT.decode('utf-8').strip())
    TEST_DATA = GIT_ROOT / 'src/tests/data'
    BINARY_TESTING_DATASET1 = TEST_DATA / 'binary_testing_dataset1.csv'
    BINARY_TRAINING_DATASET1 = TEST_DATA / 'binary_training_dataset1.csv'
    BINARY_VALIDATION_DATASET1 = TEST_DATA / 'binary_validation_dataset1.csv'
    BINARY_TESTING_DATASET2 = TEST_DATA / 'binary_testing_dataset2.csv'
    BINARY_TRAINING_DATASET2 = TEST_DATA / 'binary_training_dataset2.csv'
    TERNARY_TESTING_DATASET = TEST_DATA / 'ternary_testing_dataset.csv'
    TERNARY_TRAINING_DATASET = TEST_DATA / 'ternary_training_dataset.csv'
    TERNARY_VALIDATION_DATASET = TEST_DATA / 'ternary_validation_dataset.csv'
    MULTICLASS_TESTING_DATASET = TEST_DATA / 'multiclass_testing_dataset.csv'
    MULTICLASS_TRAINING_DATASET = TEST_DATA / 'multiclass_training_dataset.csv'
    MULTICLASS_VALIDATION_DATASET = TEST_DATA / 'multiclass_validation_dataset.csv'
    PREPROCESS_STAGE2 = GIT_ROOT / 'src/preprocess_stage2.R'

    def setUp(self):
        setUp(self)

    def tearDown(self):
        tearDown(self)

    def test_binary_classification_datasets(self):
        """
        Test creation of training, testing, and validation datasets for
        binary classification

        """

        subprocess.check_output([str(self.PREPROCESS_STAGE2),
                                 str(self.output_directory),
                                 str(test_preprocess_stage1.EXPECTED_OUTPUT_DEFAULT_PARAMETERS),
                                 '--random-seed', RANDOM_SEED])

        actual_testing_dataset = pd.read_csv(self.testing_path)
        expected_testing_dataset = pd.read_csv(self.BINARY_TESTING_DATASET1)
        self.assertTrue(expected_testing_dataset.equals(actual_testing_dataset))

        actual_training_dataset = pd.read_csv(self.training_path)
        expected_training_dataset = pd.read_csv(self.BINARY_TRAINING_DATASET1)
        self.assertTrue(expected_training_dataset.equals(actual_training_dataset))

        actual_validation_dataset = pd.read_csv(self.validation_path)
        expected_validation_dataset = pd.read_csv(self.BINARY_VALIDATION_DATASET1)
        self.assertTrue(expected_validation_dataset.equals(actual_validation_dataset))

    def test_ternary_classification_datasets(self):
        """
        Test creation of training, testing, and validation datasets for
        ternary classification

        """

        subprocess.check_output([str(self.PREPROCESS_STAGE2),
                                 str(self.output_directory),
                                 str(test_preprocess_stage1.EXPECTED_OUTPUT_DEFAULT_PARAMETERS),
                                 '--random-seed', RANDOM_SEED,
                                 '--classification-type', 'ternary'])

        actual_testing_dataset = pd.read_csv(self.testing_path)
        expected_testing_dataset = pd.read_csv(self.TERNARY_TESTING_DATASET)
        self.assertTrue(expected_testing_dataset.equals(actual_testing_dataset))

        actual_training_dataset = pd.read_csv(self.training_path)
        expected_training_dataset = pd.read_csv(self.TERNARY_TRAINING_DATASET)
        self.assertTrue(expected_training_dataset.equals(actual_training_dataset))

        actual_validation_dataset = pd.read_csv(self.validation_path)
        expected_validation_dataset = pd.read_csv(self.TERNARY_VALIDATION_DATASET)
        self.assertTrue(expected_validation_dataset.equals(actual_validation_dataset))

    def test_multiclass_classification_datasets(self):
        """
        Test creation of training, testing, and validation datasets for
        multiclass classification

        """

        subprocess.check_output([str(self.PREPROCESS_STAGE2),
                                 str(self.output_directory),
                                 str(test_preprocess_stage1.EXPECTED_OUTPUT_DEFAULT_PARAMETERS),
                                 '--random-seed', RANDOM_SEED,
                                 '--classification-type', 'multiclass'])

        actual_testing_dataset = pd.read_csv(self.testing_path)
        expected_testing_dataset = pd.read_csv(self.MULTICLASS_TESTING_DATASET)
        self.assertTrue(expected_testing_dataset.equals(actual_testing_dataset))

        actual_training_dataset = pd.read_csv(self.training_path)
        expected_training_dataset = pd.read_csv(self.MULTICLASS_TRAINING_DATASET)
        self.assertTrue(expected_training_dataset.equals(actual_training_dataset))

        actual_validation_dataset = pd.read_csv(self.validation_path)
        expected_validation_dataset = pd.read_csv(self.MULTICLASS_VALIDATION_DATASET)
        self.assertTrue(expected_validation_dataset.equals(actual_validation_dataset))

    def test_invalid_classification_type(self):
        """
        Test preprocess_stage2.R with an invalid classification type

        """

        stdout = subprocess.check_output([str(self.PREPROCESS_STAGE2),
                                          str(self.output_directory),
                                          str(test_preprocess_stage1.EXPECTED_OUTPUT_DEFAULT_PARAMETERS),
                                          '--classification-type', 'invalid'],
                                         stderr=subprocess.STDOUT)

        self.assertIn('Error: Unknown classification type `invalid`',
                      stdout.decode('utf-8'))

    def test_training_testing_datasets(self):
        """
        Test creation of only training and testing datasets (no validation).

        """

        subprocess.check_output([str(self.PREPROCESS_STAGE2),
                                 str(self.output_directory),
                                 str(test_preprocess_stage1.EXPECTED_OUTPUT_DEFAULT_PARAMETERS),
                                 '--random-seed', RANDOM_SEED,
                                 '--validation-fraction', '0'])

        actual_testing_dataset = pd.read_csv(self.testing_path)
        expected_testing_dataset = pd.read_csv(self.BINARY_TESTING_DATASET2)
        self.assertTrue(expected_testing_dataset.equals(actual_testing_dataset))

        actual_training_dataset = pd.read_csv(self.training_path)
        expected_training_dataset = pd.read_csv(self.BINARY_TRAINING_DATASET2)
        self.assertTrue(expected_training_dataset.equals(actual_training_dataset))

    def test_with_preprocess_stage1(self):
        """
        Test running preprocess_stage2.py on the output of preprocess_stage1.py.

        """

        test_preprocess_stage1.setUp(self)
        preprocess_stage1.main([str(self.output_path)] + test_preprocess_stage1.SOURCE_DATASETS)
        subprocess.check_output([str(self.PREPROCESS_STAGE2),
                                 str(self.output_directory),
                                 str(self.output_path),
                                 '--random-seed', RANDOM_SEED])

        actual_testing_dataset = pd.read_csv(self.testing_path)
        expected_testing_dataset = pd.read_csv(self.BINARY_TESTING_DATASET1)
        self.assertTrue(expected_testing_dataset.equals(actual_testing_dataset))

        actual_training_dataset = pd.read_csv(self.training_path)
        expected_training_dataset = pd.read_csv(self.BINARY_TRAINING_DATASET1)
        self.assertTrue(expected_training_dataset.equals(actual_training_dataset))

        actual_validation_dataset = pd.read_csv(self.validation_path)
        expected_validation_dataset = pd.read_csv(self.BINARY_VALIDATION_DATASET1)
        self.assertTrue(expected_validation_dataset.equals(actual_validation_dataset))

        test_preprocess_stage1.tearDown(self)


# Define setUp and tearDown functions outside of the class so that they are
# callable from other TestCase classes.
def setUp(self):
    self.output_directory = Path(tempfile.mkdtemp())
    self.testing_path = self.output_directory / 'testing.csv'
    self.training_path = self.output_directory / 'training.csv'
    self.validation_path = self.output_directory / 'validation.csv'


def tearDown(self):
    if self.training_path.exists():
        self.training_path.unlink()

    if self.testing_path.exists():
        self.testing_path.unlink()

    if self.validation_path.exists():
        self.validation_path.unlink()

    self.output_directory.rmdir()
