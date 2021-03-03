"""
Integration testcases for preprocess_stage2.R

Copyright 2020, 2021 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import unittest
import tempfile
import subprocess
from pathlib import Path

import pandas as pd

import ingest_raw_uci_data
from tests.integration import test_ingest_raw_uci_data

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
    PREPROCESS = GIT_ROOT / 'src/preprocess.R'
    EXPECTED_TOTAL_ROWS = 16
    EXPECTED_TOTAL_ROWS_RAW_UCI = 6
    SUBSET_COLUMNS = ['age', 'sex', 'cp', 'trestbps', 'fbs', 'restecg', 'thalach',
                      'exang', 'oldpeak', 'chol', 'target']

    MISSING_VALUES_INGEST_DIR = TEST_DATA / 'imputation_ingest'
    EXPECTED_TOTAL_ROWS_SINGLE_IMPUTATION = 10
    TEST_SET_INGEST_DIR = TEST_DATA / 'test_set_ingest'
    EXPECTED_TESTING_ROWS_TEST_SET = 3
    EXPECTED_TOTAL_ROWS_TEST_SET = 16

    def setUp(self):
        setUp(self)

    def tearDown(self):
        tearDown(self)

    def test_binary_classification_datasets(self):
        """
        Test creation of training, testing, and validation datasets for
        binary classification

        """

        subprocess.check_call([str(self.PREPROCESS),
                               str(self.training_path),
                               str(self.testing_path),
                               str(self.validation_path),
                               str(test_ingest_raw_uci_data.INGESTED_DIR),
                               'cleveland1',
                               '--test-fraction', '0.15',
                               '--random-state', RANDOM_SEED,
                               '--features'] + self.SUBSET_COLUMNS)

        actual_testing_dataset = pd.read_csv(self.testing_path)
        actual_training_dataset = pd.read_csv(self.training_path)
        actual_validation_dataset = pd.read_csv(self.validation_path)

        total_rows = (len(actual_testing_dataset)
                      + len(actual_training_dataset)
                      + len(actual_validation_dataset))

        self.assertEqual(total_rows, self.EXPECTED_TOTAL_ROWS)

        testing_set = frozenset(actual_testing_dataset.apply(tuple, axis=1))
        training_set = frozenset(actual_training_dataset.apply(tuple, axis=1))
        validation_set = frozenset(actual_validation_dataset.apply(tuple, axis=1))
        self.assertFalse(testing_set & training_set)
        self.assertFalse(testing_set & validation_set)
        self.assertFalse(training_set & validation_set)
        self.assertEqual(len(testing_set | training_set | validation_set), total_rows)

    def test_ternary_classification_datasets(self):
        """
        Test creation of training, testing, and validation datasets for
        ternary classification

        """

        subprocess.check_call([str(self.PREPROCESS),
                               str(self.training_path),
                               str(self.testing_path),
                               str(self.validation_path),
                               str(test_ingest_raw_uci_data.INGESTED_DIR),
                               'cleveland1',
                               '--test-fraction', '0.15',
                               '--random-state', RANDOM_SEED,
                               '--classification-type', 'ternary',
                               '--features'] + self.SUBSET_COLUMNS)

        actual_testing_dataset = pd.read_csv(self.testing_path)
        actual_training_dataset = pd.read_csv(self.training_path)
        actual_validation_dataset = pd.read_csv(self.validation_path)

        total_rows = (len(actual_testing_dataset)
                      + len(actual_training_dataset)
                      + len(actual_validation_dataset))

        self.assertEqual(total_rows, self.EXPECTED_TOTAL_ROWS)

        testing_set = frozenset(actual_testing_dataset.apply(tuple, axis=1))
        training_set = frozenset(actual_training_dataset.apply(tuple, axis=1))
        validation_set = frozenset(actual_validation_dataset.apply(tuple, axis=1))
        self.assertFalse(testing_set & training_set)
        self.assertFalse(testing_set & validation_set)
        self.assertFalse(training_set & validation_set)
        self.assertEqual(len(testing_set | training_set | validation_set), total_rows)

    def test_multiclass_classification_datasets(self):
        """
        Test creation of training, testing, and validation datasets for
        multiclass classification

        """

        subprocess.check_call([str(self.PREPROCESS),
                               str(self.training_path),
                               str(self.testing_path),
                               str(self.validation_path),
                               str(test_ingest_raw_uci_data.INGESTED_DIR),
                               'cleveland1',
                               '--test-fraction', '0.15',
                               '--random-state', RANDOM_SEED,
                               '--classification-type', 'multiclass',
                               '--features'] + self.SUBSET_COLUMNS)

        actual_testing_dataset = pd.read_csv(self.testing_path)
        actual_training_dataset = pd.read_csv(self.training_path)
        actual_validation_dataset = pd.read_csv(self.validation_path)

        total_rows = (len(actual_testing_dataset)
                      + len(actual_training_dataset)
                      + len(actual_validation_dataset))

        self.assertEqual(total_rows, self.EXPECTED_TOTAL_ROWS)

        testing_set = frozenset(actual_testing_dataset.apply(tuple, axis=1))
        training_set = frozenset(actual_training_dataset.apply(tuple, axis=1))
        validation_set = frozenset(actual_validation_dataset.apply(tuple, axis=1))
        self.assertFalse(testing_set & training_set)
        self.assertFalse(testing_set & validation_set)
        self.assertFalse(training_set & validation_set)
        self.assertEqual(len(testing_set | training_set | validation_set), total_rows)

    def test_invalid_classification_type(self):
        """
        Test preprocess_stage2.R with an invalid classification type

        """

        with self.assertRaises(subprocess.CalledProcessError):
            subprocess.check_call([str(self.PREPROCESS),
                                   str(self.training_path),
                                   str(self.testing_path),
                                   str(self.validation_path),
                                   str(test_ingest_raw_uci_data.INGESTED_DIR),
                                   'cleveland1',
                                   '--classification-type', 'invalid'])

    def test_training_testing_datasets(self):
        """
        Test creation of only training and testing datasets (no validation).

        """

        subprocess.check_call([str(self.PREPROCESS),
                               str(self.training_path),
                               str(self.testing_path),
                               str(self.validation_path),
                               str(test_ingest_raw_uci_data.INGESTED_DIR),
                               'cleveland1',
                               '--test-fraction', '0.15',
                               '--random-state', RANDOM_SEED,
                               '--validation-fraction', '0',
                               '--features'] + self.SUBSET_COLUMNS)

        actual_testing_dataset = pd.read_csv(self.testing_path)
        actual_training_dataset = pd.read_csv(self.training_path)

        total_rows = len(actual_testing_dataset) + len(actual_training_dataset)
        self.assertEqual(total_rows, self.EXPECTED_TOTAL_ROWS)

        testing_set = frozenset(actual_testing_dataset.apply(tuple, axis=1))
        training_set = frozenset(actual_training_dataset.apply(tuple, axis=1))
        self.assertFalse(testing_set & training_set)
        self.assertEqual(len(testing_set | training_set), total_rows)

    def test_with_ingest_raw_uci_data(self):
        """
        Test running preprocess_stage2.py on the output of ingest_raw_uci_data.py.

        """

        test_ingest_raw_uci_data.setUp(self)
        for test_dataset in test_ingest_raw_uci_data.SOURCE_DATASETS:
            ingest_raw_uci_data.main([self.output_path, test_dataset])

        subprocess.check_call([str(self.PREPROCESS),
                               str(self.training_path),
                               str(self.testing_path),
                               str(self.validation_path),
                               str(self.output_path),
                               'cleveland1',
                               '--test-fraction', '0.15',
                               '--random-state', RANDOM_SEED,
                               '--features'] + self.SUBSET_COLUMNS)

        actual_testing_dataset = pd.read_csv(self.testing_path)
        actual_training_dataset = pd.read_csv(self.training_path)
        actual_validation_dataset = pd.read_csv(self.validation_path)

        total_rows = (len(actual_testing_dataset)
                      + len(actual_training_dataset)
                      + len(actual_validation_dataset))

        self.assertEqual(total_rows, self.EXPECTED_TOTAL_ROWS_RAW_UCI)

        testing_set = frozenset(actual_testing_dataset.apply(tuple, axis=1))
        training_set = frozenset(actual_training_dataset.apply(tuple, axis=1))
        validation_set = frozenset(actual_validation_dataset.apply(tuple, axis=1))
        self.assertFalse(testing_set & training_set)
        self.assertFalse(testing_set & validation_set)
        self.assertFalse(training_set & validation_set)
        self.assertEqual(len(testing_set | training_set | validation_set), total_rows)

        test_ingest_raw_uci_data.tearDown(self)

    def test_single_imputation(self):
        """
        Test that single imputation works as expected.

        """

        subprocess.check_call([str(self.PREPROCESS),
                               str(self.training_path),
                               str(self.testing_path),
                               str(self.validation_path),
                               str(self.MISSING_VALUES_INGEST_DIR),
                               'cleveland1',
                               '--test-fraction', '0.15',
                               '--random-state', RANDOM_SEED,
                               '--features'] + self.SUBSET_COLUMNS)

        testing_dataset = pd.read_csv(self.testing_path)
        testing_nans = testing_dataset.isnull().sum().sum()
        self.assertEqual(testing_nans, 0)

        training_dataset = pd.read_csv(self.training_path)
        training_nans = training_dataset.isnull().sum().sum()
        self.assertEqual(training_nans, 2)

        validation_dataset = pd.read_csv(self.validation_path)
        validation_nans = validation_dataset.isnull().sum().sum()
        self.assertEqual(validation_nans, 0)

        total_rows = (len(testing_dataset)
                      + len(training_dataset)
                      + len(validation_dataset))

        self.assertEqual(total_rows, self.EXPECTED_TOTAL_ROWS_SINGLE_IMPUTATION)

    def test_test_set_with_first_dataset(self):
        """
        Test preprocess.R with the test-pool argument as a dataset whose
        name is first in alphabetical order.

        """

        subprocess.check_call([str(self.PREPROCESS),
                               str(self.training_path),
                               str(self.testing_path),
                               str(self.validation_path),
                               self.TEST_SET_INGEST_DIR,
                               'cleveland',
                               '--test-fraction', '0.15',
                               '--random-state', RANDOM_SEED,
                               '--features'] + self.SUBSET_COLUMNS)

        testing_dataset = pd.read_csv(self.testing_path)
        training_dataset = pd.read_csv(self.training_path)
        validation_dataset = pd.read_csv(self.validation_path)
        self.assertEqual(len(testing_dataset),
                         self.EXPECTED_TESTING_ROWS_TEST_SET)

        total_rows = (len(testing_dataset)
                      + len(training_dataset)
                      + len(validation_dataset))

        self.assertEqual(total_rows, self.EXPECTED_TOTAL_ROWS_TEST_SET)

        cleveland_dataset = pd.read_csv(self.TEST_SET_INGEST_DIR / 'cleveland.csv')
        cleveland_subset = frozenset(cleveland_dataset[['age', 'trestbps', 'thalach', 'oldpeak']].apply(tuple, axis=1))
        testing_subset = frozenset(testing_dataset[['age', 'trestbps', 'thalach', 'oldpeak']].apply(tuple, axis=1))
        self.assertTrue(testing_subset <= cleveland_subset)

        training_subset = frozenset(training_dataset[['age', 'trestbps', 'thalach', 'oldpeak']].apply(tuple, axis=1))
        validation_subset = frozenset(validation_dataset[['age', 'trestbps', 'thalach', 'oldpeak']].apply(tuple, axis=1))
        self.assertEqual(len(testing_subset | training_subset | validation_subset), total_rows)

    def test_test_set_with_second_dataset(self):
        """
        Test preprocess.R with the --test-samples-from option on a dataset whose
        name is second in alphabetical order.

        """

        subprocess.check_call([str(self.PREPROCESS),
                               str(self.training_path),
                               str(self.testing_path),
                               str(self.validation_path),
                               self.TEST_SET_INGEST_DIR,
                               'ingest_raw_uci_data1',
                               '--test-fraction', '0.15',
                               '--random-state', RANDOM_SEED,
                               '--features'] + self.SUBSET_COLUMNS)

        testing_dataset = pd.read_csv(self.testing_path)
        training_dataset = pd.read_csv(self.training_path)
        validation_dataset = pd.read_csv(self.validation_path)
        self.assertEqual(len(testing_dataset),
                         self.EXPECTED_TESTING_ROWS_TEST_SET)

        total_rows = (len(testing_dataset)
                      + len(training_dataset)
                      + len(validation_dataset))

        self.assertEqual(total_rows, self.EXPECTED_TOTAL_ROWS_TEST_SET)

        ingest_raw_uci_data1_dataset = pd.read_csv(self.TEST_SET_INGEST_DIR / 'ingest_raw_uci_data1.csv')
        ingest_raw_uci_data1_subset = \
            frozenset(ingest_raw_uci_data1_dataset[['age', 'trestbps', 'thalach', 'oldpeak']].apply(tuple, axis=1))

        testing_subset = frozenset(testing_dataset[['age', 'trestbps', 'thalach', 'oldpeak']].apply(tuple, axis=1))
        self.assertTrue(testing_subset <= ingest_raw_uci_data1_subset)

        training_subset = frozenset(training_dataset[['age', 'trestbps', 'thalach', 'oldpeak']].apply(tuple, axis=1))
        validation_subset = frozenset(validation_dataset[['age', 'trestbps', 'thalach', 'oldpeak']].apply(tuple, axis=1))
        self.assertEqual(len(testing_subset | training_subset | validation_subset), total_rows)

    def test_test_set_with_insufficient_dataset(self):
        """
        Test preprocess.R with the --test-samples-from option on a dataset that
        does now contain enough rows to fill the testing set.

        """

        with self.assertRaises(subprocess.CalledProcessError):
            subprocess.check_call([str(self.PREPROCESS),
                                   str(self.training_path),
                                   str(self.testing_path),
                                   str(self.validation_path),
                                   self.TEST_SET_INGEST_DIR,
                                   'cleveland1',
                                   '--random-state', RANDOM_SEED,
                                   '--features'] + self.SUBSET_COLUMNS)


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
