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
    EXPECTED_TOTAL_ROWS_IMPUTE_MULTIPLE = 13
    TEST_SET_INGEST_DIR = TEST_DATA / 'test_set_ingest'

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

        self.assertEqual(actual_testing_dataset.isna().sum().sum(), 0)
        self.assertEqual(actual_training_dataset.isna().sum().sum(), 0)
        self.assertEqual(actual_validation_dataset.isna().sum().sum(), 0)

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

        self.assertEqual(actual_testing_dataset.isna().sum().sum(), 0)
        self.assertEqual(actual_training_dataset.isna().sum().sum(), 0)
        self.assertEqual(actual_validation_dataset.isna().sum().sum(), 0)

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

        self.assertEqual(actual_testing_dataset.isna().sum().sum(), 0)
        self.assertEqual(actual_training_dataset.isna().sum().sum(), 0)
        self.assertEqual(actual_validation_dataset.isna().sum().sum(), 0)

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

        self.assertEqual(actual_testing_dataset.isna().sum().sum(), 0)
        self.assertEqual(actual_training_dataset.isna().sum().sum(), 0)

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
                               'dataset2',
                               '--test-fraction', '0.15',
                               '--random-state', RANDOM_SEED,
                               '--features'] + self.SUBSET_COLUMNS)

        actual_testing_dataset = pd.read_csv(self.testing_path)
        actual_training_dataset = pd.read_csv(self.training_path)
        actual_validation_dataset = pd.read_csv(self.validation_path)

        self.assertEqual(actual_testing_dataset.isna().sum().sum(), 0)
        self.assertEqual(actual_training_dataset.isna().sum().sum(), 0)
        self.assertEqual(actual_validation_dataset.isna().sum().sum(), 0)

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

    def test_impute_missing(self):
        """
        Test that the --impute-missing flag works as expected.

        """

        subprocess.check_call([str(self.PREPROCESS),
                               str(self.training_path),
                               str(self.testing_path),
                               str(self.validation_path),
                               str(self.MISSING_VALUES_INGEST_DIR),
                               'imputation',
                               '--impute-missing',
                               '--test-fraction', '0.15',
                               '--random-state', RANDOM_SEED,
                               '--features'] + self.SUBSET_COLUMNS)

        testing_dataset = pd.read_csv(self.testing_path)
        training_dataset = pd.read_csv(self.training_path)
        validation_dataset = pd.read_csv(self.validation_path)

        self.assertEqual(testing_dataset.isna().sum().sum(), 0)
        self.assertEqual(training_dataset.isna().sum().sum(), 0)
        self.assertEqual(validation_dataset.isna().sum().sum(), 0)

        total_rows = (len(testing_dataset)
                      + len(training_dataset)
                      + len(validation_dataset))

        self.assertEqual(total_rows, 8)

    def test_impute_multiple(self):
        """
        Test that the --impute-multiple flag works as expected.

        """

        subprocess.check_call([str(self.PREPROCESS),
                               str(self.training_path),
                               str(self.testing_path),
                               str(self.validation_path),
                               str(self.MISSING_VALUES_INGEST_DIR),
                               'imputation',
                               '--impute-multiple',
                               '--test-fraction', '0.15',
                               '--random-state', "2",
                               '--features'] + self.SUBSET_COLUMNS)

        testing_dataset = pd.read_csv(self.testing_path)
        training_dataset = pd.read_csv(self.training_path)
        validation_dataset = pd.read_csv(self.validation_path)

        self.assertEqual(testing_dataset.isna().sum().sum(), 0)
        self.assertEqual(training_dataset.isna().sum().sum(), 0)
        self.assertEqual(validation_dataset.isna().sum().sum(), 0)

        total_rows = (len(testing_dataset)
                      + len(training_dataset)
                      + len(validation_dataset))

        self.assertEqual(total_rows, self.EXPECTED_TOTAL_ROWS_IMPUTE_MULTIPLE)

    def test_impute_multiple_and_impute_missing(self):
        """
        Test that combining --impute-multiple with --impute-missing
        works as expected.

        """

        subprocess.check_call([str(self.PREPROCESS),
                               str(self.training_path),
                               str(self.testing_path),
                               str(self.validation_path),
                               str(self.MISSING_VALUES_INGEST_DIR),
                               'imputation',
                               '--impute-multiple',
                               '--impute-missing',
                               '--test-fraction', '0.15',
                               '--random-state', "2",
                               '--features'] + self.SUBSET_COLUMNS)

        testing_dataset = pd.read_csv(self.testing_path)
        training_dataset = pd.read_csv(self.training_path)
        validation_dataset = pd.read_csv(self.validation_path)

        self.assertEqual(testing_dataset.isna().sum().sum(), 0)
        self.assertEqual(training_dataset.isna().sum().sum(), 0)
        self.assertEqual(validation_dataset.isna().sum().sum(), 0)

        total_rows = (len(testing_dataset)
                      + len(training_dataset)
                      + len(validation_dataset))

        self.assertEqual(total_rows, self.EXPECTED_TOTAL_ROWS_IMPUTE_MULTIPLE)

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

        self.assertEqual(testing_dataset.isna().sum().sum(), 0)
        self.assertEqual(training_dataset.isna().sum().sum(), 0)
        self.assertEqual(validation_dataset.isna().sum().sum(), 0)

        self.assertEqual(len(testing_dataset), 3)
        total_rows = (len(testing_dataset)
                      + len(training_dataset)
                      + len(validation_dataset))

        self.assertEqual(total_rows, 16)

        cleveland_dataset = pd.read_csv(self.TEST_SET_INGEST_DIR / 'cleveland.csv')
        cleveland_subset = frozenset(cleveland_dataset[['age', 'trestbps', 'thalach', 'oldpeak']].apply(tuple, axis=1))
        testing_subset = frozenset(testing_dataset[['age', 'trestbps', 'thalach', 'oldpeak']].apply(tuple, axis=1))
        self.assertTrue(testing_subset <= cleveland_subset)

        training_subset = frozenset(training_dataset[['age', 'trestbps', 'thalach', 'oldpeak']].apply(tuple, axis=1))
        validation_subset = frozenset(validation_dataset[['age', 'trestbps', 'thalach', 'oldpeak']].apply(tuple, axis=1))
        self.assertEqual(len(testing_subset | training_subset | validation_subset), total_rows)

    def test_test_set_with_second_dataset(self):
        """
        Test preprocess.R with a test-pool dataset whose name is second
        in alphabetical order.

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

        self.assertEqual(testing_dataset.isna().sum().sum(), 0)
        self.assertEqual(training_dataset.isna().sum().sum(), 0)
        self.assertEqual(validation_dataset.isna().sum().sum(), 0)

        self.assertEqual(len(testing_dataset), 3)
        total_rows = (len(testing_dataset)
                      + len(training_dataset)
                      + len(validation_dataset))

        self.assertEqual(total_rows, 16)

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

    def test_test_pool_is_constructed_from_correct_dataset1(self):
        """
        Test that the preprocessor constructs the test pool from the correct dataset.

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
        testing_dataset = testing_dataset[['age', 'trestbps', 'thalach']]
        testing_dataset = set(testing_dataset.itertuples(index=False))

        cleveland_dataset = pd.read_csv(self.TEST_SET_INGEST_DIR / 'cleveland.csv')
        cleveland_dataset = cleveland_dataset[['age', 'trestbps', 'thalach']]
        cleveland_dataset = set(cleveland_dataset.itertuples(index=False))

        self.assertTrue(testing_dataset <= cleveland_dataset)
        self.assertTrue(testing_dataset)

    def test_test_pool_is_constructed_from_correct_dataset2(self):
        """
        Test that the preprocessor constructs the test pool from the correct dataset.

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

        testing_dataset = pd.read_csv(self.testing_path)
        testing_dataset = testing_dataset[['age', 'trestbps', 'thalach']]
        testing_dataset = set(testing_dataset.itertuples(index=False))

        cleveland1_dataset = pd.read_csv(test_ingest_raw_uci_data.INGESTED_DIR / 'cleveland1.csv')
        cleveland1_dataset = cleveland1_dataset[['age', 'trestbps', 'thalach']]
        cleveland1_dataset = set(cleveland1_dataset.itertuples(index=False))

        self.assertTrue(testing_dataset <= cleveland1_dataset)
        self.assertTrue(testing_dataset)

    def test_calling_preprocessor_without_chol(self):
        """
        Test calling the preprocessor without the 'chol' feature.

        """

        args = [str(self.PREPROCESS),
                str(self.training_path),
                str(self.testing_path),
                str(self.validation_path),
                str(test_ingest_raw_uci_data.INGESTED_DIR),
                'cleveland1',
                '--test-fraction', '0.15',
                '--random-state', RANDOM_SEED,
                '--features']

        args.extend(['age', 'sex', 'cp', 'trestbps', 'fbs', 'restecg',
                     'thalach', 'exang', 'oldpeak', 'target'])

        subprocess.check_call(args)

        actual_testing_dataset = pd.read_csv(self.testing_path)
        actual_training_dataset = pd.read_csv(self.training_path)
        actual_validation_dataset = pd.read_csv(self.validation_path)

        self.assertNotIn('chol', actual_testing_dataset.columns.values)
        self.assertNotIn('chol', actual_training_dataset.columns.values)
        self.assertNotIn('chol', actual_validation_dataset.columns.values)

        self.assertEqual(actual_testing_dataset.isna().sum().sum(), 0)
        self.assertEqual(actual_training_dataset.isna().sum().sum(), 0)
        self.assertEqual(actual_validation_dataset.isna().sum().sum(), 0)

        total_rows = (len(actual_testing_dataset)
                      + len(actual_training_dataset)
                      + len(actual_validation_dataset))

        self.assertEqual(total_rows, 17)

        testing_set = frozenset(actual_testing_dataset.apply(tuple, axis=1))
        training_set = frozenset(actual_training_dataset.apply(tuple, axis=1))
        validation_set = frozenset(actual_validation_dataset.apply(tuple, axis=1))
        self.assertFalse(testing_set & training_set)
        self.assertFalse(testing_set & validation_set)
        self.assertFalse(training_set & validation_set)
        self.assertEqual(len(testing_set | training_set | validation_set), total_rows)

    def test_calling_preprocessor_with_impute_methods(self):
        """
        Test calling the preprocessor with the --impute-methods argument.

        """

        args = [str(self.PREPROCESS),
                str(self.training_path),
                str(self.testing_path),
                str(self.validation_path),
                str(test_ingest_raw_uci_data.INGESTED_DIR),
                'cleveland1',
                '--impute-missing',
                '--test-fraction', '0.15',
                '--random-state', RANDOM_SEED,
                '--features']

        args.extend(self.SUBSET_COLUMNS)
        args.extend(['--impute-methods', "", "", "", "", "logreg", "polyreg",
                     "", "", "pmm", "pmm", "''"])

        subprocess.check_call(args)

        actual_testing_dataset = pd.read_csv(self.testing_path)
        actual_training_dataset = pd.read_csv(self.training_path)
        actual_validation_dataset = pd.read_csv(self.validation_path)

        self.assertEqual(actual_testing_dataset.isna().sum().sum(), 0)
        self.assertEqual(actual_training_dataset.isna().sum().sum(), 0)
        self.assertEqual(actual_validation_dataset.isna().sum().sum(), 0)

        total_rows = (len(actual_testing_dataset)
                      + len(actual_training_dataset)
                      + len(actual_validation_dataset))

        self.assertEqual(total_rows, 17)

        testing_set = frozenset(actual_testing_dataset.apply(tuple, axis=1))
        training_set = frozenset(actual_training_dataset.apply(tuple, axis=1))
        validation_set = frozenset(actual_validation_dataset.apply(tuple, axis=1))
        self.assertFalse(testing_set & training_set)
        self.assertFalse(testing_set & validation_set)
        self.assertFalse(training_set & validation_set)
        self.assertEqual(len(testing_set | training_set | validation_set), total_rows)

    def test_calling_preprocessor_with_mismatched_impute_methods(self):
        """
        Test calling the preprocessor with impute-methods that don't match the
        number of features.

        """

        args = [str(self.PREPROCESS),
                str(self.training_path),
                str(self.testing_path),
                str(self.validation_path),
                str(test_ingest_raw_uci_data.INGESTED_DIR),
                'cleveland1',
                '--impute-missing',
                '--test-fraction', '0.15',
                '--random-state', RANDOM_SEED,
                '--features']

        args.extend(self.SUBSET_COLUMNS)
        args.extend(['--impute-methods', "", "", "", "", "logreg", "polyreg",
                     "", "", "pmm", "pmm", "", "''"])

        with self.assertRaises(subprocess.CalledProcessError):
            subprocess.check_call(args)


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
