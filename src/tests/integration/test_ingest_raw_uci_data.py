"""
Integration testcases for ingest_raw_uci_data.py.

Copyright 2020, 2021 Jerrad M. Genson

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

import ingest_raw_uci_data

GIT_ROOT = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
GIT_ROOT = Path(GIT_ROOT.decode('utf-8').strip())
TEST_DATA = GIT_ROOT / Path('src/tests/data')
TEST_DATASET1 = TEST_DATA / 'dataset1.data'
TEST_DATASET2 = TEST_DATA / 'dataset2.data'
TEST_DATASET3 = TEST_DATA / 'dataset3.data'
SOURCE_DATASETS = [str(TEST_DATASET1), str(TEST_DATASET2), str(TEST_DATASET3)]
TEST_COLUMNS = 'id', 'ccf', 'age', 'sex', 'painloc', 'painexer', 'relrest', 'pncaden', 'cp', 'trestbps', 'htn', 'chol', 'smoke', 'cigs', 'years', 'fbs', 'dm', 'famhist', 'restecg', 'ekgmo', 'ekgday', 'ekgyr', 'dig', 'prop', 'nitr', 'pro', 'diuretic', 'proto', 'thaldur', 'thaltime', 'met', 'thalach', 'thalrest', 'tpeakbps', 'tpeakbpd', 'dummy', 'trestbpd', 'exang', 'xhypo', 'oldpeak', 'slope', 'rldv5', 'rldv5e', 'ca', 'restckm', 'exerckm', 'restef', 'restwm', 'exeref', 'exerwm', 'thal', 'thalsev', 'thalpul', 'earlobe', 'cmo', 'cday', 'cyr', 'num', 'lmt', 'ladprox', 'laddist', 'diag', 'cxmain', 'ramus', 'om1', 'om2', 'rcaprox', 'rcadist', 'lvx1', 'lvx2', 'lvx3', 'lvx4', 'lvf', 'cathef', 'junk', 'name'
INGESTED_DIR = TEST_DATA / 'ingested'
EXPECTED_OUTPUT1 = INGESTED_DIR / 'ingest_raw_uci_data1.csv'
EXPECTED_OUTPUT2 = INGESTED_DIR / 'ingest_raw_uci_data2.csv'
EXPECTED_OUTPUT3 = INGESTED_DIR / 'ingest_raw_uci_data3.csv'


class IngestRawUCIDataTest(unittest.TestCase):
    """
    Test cases for ingest_raw_uci_data.py

    """

    def setUp(self):
        setUp(self)

    def tearDown(self):
        tearDown(self)

    def test_dataset1(self):
        """
        Test ingest_raw_uci_data.py with the default parameters.

        """

        ingest_raw_uci_data.main([self.output_path, str(TEST_DATASET1)])
        actual_dataset = pd.read_csv((Path(self.output_path) / TEST_DATASET1.name).with_suffix('.csv'))
        expected_dataset = pd.read_csv(EXPECTED_OUTPUT1)
        self.assertTrue(expected_dataset.equals(actual_dataset))

    def test_dataset2(self):
        """
        Test ingest_raw_uci_data.py with the default parameters.

        """

        ingest_raw_uci_data.main([self.output_path, str(TEST_DATASET2)])
        actual_dataset = pd.read_csv((Path(self.output_path) / TEST_DATASET2.name).with_suffix('.csv'))
        expected_dataset = pd.read_csv(EXPECTED_OUTPUT2)
        self.assertTrue(expected_dataset.equals(actual_dataset))

    def test_dataset3(self):
        """
        Test ingest_raw_uci_data.py with the default parameters.

        """

        ingest_raw_uci_data.main([self.output_path, str(TEST_DATASET3)])
        actual_dataset = pd.read_csv((Path(self.output_path) / TEST_DATASET3.name).with_suffix('.csv'))
        expected_dataset = pd.read_csv(EXPECTED_OUTPUT3)
        self.assertTrue(expected_dataset.equals(actual_dataset))


# Define setUp and tearDown functions outside of the class so that they are
# callable from other TestCase classes.
def setUp(self):
    self.output_path = tempfile.mkdtemp()
    self.prev_column_names = ingest_raw_uci_data.COLUMN_NAMES
    ingest_raw_uci_data.COLUMN_NAMES = TEST_COLUMNS


def tearDown(self):
    ingest_raw_uci_data.COLUMNS_NAMES = self.prev_column_names
    shutil.rmtree(self.output_path, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
