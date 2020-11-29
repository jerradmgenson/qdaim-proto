"""
Integration testcases for ingest_raw_uci_data.py.

Copyright 2020 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import shutil
import unittest
import tempfile
import subprocess
from pathlib import Path

import pandas as pd

import ingest_cleveland_data

GIT_ROOT = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
GIT_ROOT = Path(GIT_ROOT.decode('utf-8').strip())
TEST_DATA = GIT_ROOT / Path('src/tests/data')
TEST_DATASET1 = TEST_DATA / 'cleveland1.csv'
TEST_DATASET2 = TEST_DATA / 'cleveland2.csv'
TEST_DATASET3 = TEST_DATA / 'cleveland3.csv'
SOURCE_DATASETS = [str(TEST_DATASET1), str(TEST_DATASET2), str(TEST_DATASET3)]
TEST_COLUMNS = TEST_DATA / 'column_names'
INGESTED_DIR = TEST_DATA / 'ingested'
EXPECTED_OUTPUT1 = INGESTED_DIR / 'cleveland1.csv'
EXPECTED_OUTPUT2 = INGESTED_DIR / 'cleveland2.csv'
EXPECTED_OUTPUT3 = INGESTED_DIR / 'cleveland3.csv'
SUBSET_COLUMNS = ['age', 'sex', 'cp', 'trestbps', 'fbs', 'restecg', 'thalach',
                  'exang', 'oldpeak', 'target']


class IngestClevelandDataTest(unittest.TestCase):
    """
    Test cases for ingest_cleveland_data.py

    """

    def setUp(self):
        setUp(self)

    def tearDown(self):
        tearDown(self)

    def test_dataset1(self):
        """
        Test ingest_cleveland_data.py with the default parameters.

        """

        ingest_cleveland_data.main([self.output_path, str(TEST_DATASET1),
                                    '--columns'] + SUBSET_COLUMNS)

        actual_dataset = pd.read_csv((Path(self.output_path) / TEST_DATASET1.name).with_suffix('.csv'))
        expected_dataset = pd.read_csv(EXPECTED_OUTPUT1)
        self.assertTrue(expected_dataset.equals(actual_dataset))

    def test_dataset2(self):
        """
        Test ingest_cleveland_data.py with the default parameters.

        """

        ingest_cleveland_data.main([self.output_path, str(TEST_DATASET2),
                                    '--columns'] + SUBSET_COLUMNS)

        actual_dataset = pd.read_csv((Path(self.output_path) / TEST_DATASET2.name).with_suffix('.csv'))
        expected_dataset = pd.read_csv(EXPECTED_OUTPUT2)
        self.assertTrue(expected_dataset.equals(actual_dataset))

    def test_dataset3(self):
        """
        Test ingest_cleveland_data.py with the default parameters.

        """

        ingest_cleveland_data.main([self.output_path, str(TEST_DATASET3),
                                    '--columns'] + SUBSET_COLUMNS)

        actual_dataset = pd.read_csv((Path(self.output_path) / TEST_DATASET3.name).with_suffix('.csv'))
        expected_dataset = pd.read_csv(EXPECTED_OUTPUT3)
        self.assertTrue(expected_dataset.equals(actual_dataset))


# Define setUp and tearDown functions outside of the class so that they are
# callable from other TestCase classes.
def setUp(self):
    self.output_path = tempfile.mkdtemp()


def tearDown(self):
    shutil.rmtree(self.output_path, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
