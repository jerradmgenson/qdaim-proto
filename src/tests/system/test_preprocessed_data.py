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
