"""
Perform preprocessing activities that occur before feature selection.
Ergo, this script is designed to be run before the Feature Selection
notebook - the output of this script will be read by that notebook,
and by preprocess_stage2.py.

Inputs to this script are three individual datasets containing heart
disease data - switzerland.data, hungarian.data, and long_beach.data.

Preprocessing steps performed by this script include:
- Convert each dataset from a one-dimensional list to a dataframe.
- Create a subset of only the columns we are interested in
  (see COLUMNS_OF_INTEREST).
- Combine the three individual datasets into a single dataset with all
  rows from the individual sets.

"""

import subprocess
import sys
import re
from pathlib import Path

import pandas as pd


GIT_ROOT = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
GIT_ROOT = Path(GIT_ROOT.decode('utf-8').strip())
BUILD = GIT_ROOT / Path('build')
DATA = GIT_ROOT / Path('data')

# Path to the input datasets.
SWITZERLAND_DATASET = DATA / Path('switzerland.data')
HUNGARIAN_DATASET = DATA / Path('hungarian.data')
LONG_BEACH_DATASET = DATA / Path('long_beach.data')

# Path to the output dataset.
COMBINED_DATASET_OUTPUT_PATH = BUILD / Path('combined_data.csv')

# Path to the file containing column names for the above three datasets.
COLUMNS_FILE = DATA / Path('column_names')

# Names of columns we are interested in studying.
# Discard all other columns from the dataset.
COLUMNS_OF_INTEREST = ['age', 'sex', 'cp', 'thalrest', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'smoke', 'cigs', 'years', 'famhist', 'num']


def main():
    """
    Program's 'main' function. Main execution starts here.

    """

    switzerland_dataset = load_dataset(SWITZERLAND_DATASET)
    curated_switzerland_dataset = switzerland_dataset[COLUMNS_OF_INTEREST]

    hungarian_dataset = load_dataset(HUNGARIAN_DATASET)
    curated_hungarian_dataset = hungarian_dataset[COLUMNS_OF_INTEREST]

    long_beach_dataset = load_dataset(LONG_BEACH_DATASET)
    curated_long_beach_dataset = long_beach_dataset[COLUMNS_OF_INTEREST]

    combined_dataset = curated_switzerland_dataset.append(curated_hungarian_dataset).append(curated_long_beach_dataset)
    combined_dataset.to_csv(str(COMBINED_DATASET_OUTPUT_PATH), index=False)

    return 0


def load_dataset(path):
    """
    Load a dataset from a file containing whitespace-separated values
    and return a pandas dataframe.

    """

    with COLUMNS_FILE.open() as attributes_fp:
        attributes = attributes_fp.read().split(', ')

    with path.open() as dataset_fp:
        raw_data = dataset_fp.read()

    samples = []
    current_sample = []
    for count, data_point in enumerate(re.split(r'\s+', raw_data)):
        if count != 0 and count % len(attributes) == 0:
            samples.append(current_sample)
            current_sample = []

        if data_point == '-9':
            data_point = None

        current_sample.append(data_point)

    dataset = pd.DataFrame(data=samples, columns=attributes)
    return dataset


if __name__ == '__main__':
    sys.exit(main())
