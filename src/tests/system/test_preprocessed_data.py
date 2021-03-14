"""
System tests for training, validation, and testing datasets created by
preprocess.R.

"""

import unittest
import subprocess
from pathlib import Path

import pandas as pd

GIT_ROOT = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
GIT_ROOT = Path(GIT_ROOT.decode('utf-8').strip())
BUILD_DIR = GIT_ROOT / 'build'


class PreprocessedDataTest(unittest.TestCase):
    def setUp(self):
        self.training_data = pd.read_csv(BUILD_DIR / 'training.csv')
        self.validation_data = pd.read_csv(BUILD_DIR / 'validation.csv')
        self.test_data = pd.read_csv(BUILD_DIR / 'test.csv')

    def test_duplicate_samples(self):
        """
        Check for duplicate samples in the testing, training, and
        validation datasets.

        """

        datasets = [self.training_data, self.validation_data, self.test_data]
        datasets = pd.concat(datasets)
        self.assertFalse(datasets.duplicated().any())

    def test_testing_training_validation_ratios(self):
        """
        Check that the testing, training, and validation ratios are correct.

        """

        total_samples = (len(self.training_data)
                         + len(self.validation_data)
                         + len(self.test_data))

        self.assertGreaterEqual(len(self.test_data) / total_samples, 0.2)
        self.assertGreaterEqual(len(self.validation_data) / total_samples, 0.2)

    def test_nas_not_present_in_datasets(self):
        """
        Check that no NAs are present in datasets.

        """

        self.assertEqual(self.test_data.isna().sum().sum(), 0)
        self.assertEqual(self.validation_data.isna().sum().sum(), 0)
        self.assertEqual(self.training_data.isna().sum().sum(), 0)

    def test_numeric_values_are_plausible(self):
        """
        Check that the numeric values in the datasets are all plausible.

        """

        plausible_values = dict(
            age=(28, 77),
            trestbps=(80, 200),
            thalach=(60, 202),
            chol=(40, 603),
            oldpeak=(-2.6, 6.2),
        )

        datasets = [self.training_data, self.validation_data, self.test_data]
        datasets = pd.concat(datasets)
        for feature, values in plausible_values.items():
            self.assertTrue((datasets[feature] >= values[0]).all())
            self.assertTrue((datasets[feature] <= values[1]).all())

    def test_categorical_values_are_plausible(self):
        """
        Check that the categorical values in the datasets are all plausible.

        """

        plausible_values = dict(
            sex=(-1, 1),
            cp=(-1, 1),
            fbs=(-1, 1),
            restecg=(-1, 1),
            exang=(-1, 1),
            target=(-1, 1),
        )

        datasets = [self.training_data, self.validation_data, self.test_data]
        datasets = pd.concat(datasets)
        for feature, values in plausible_values.items():
            self.assertTrue(datasets[feature].isin(values).all())
