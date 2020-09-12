"""
Perform preprocessing activities that occur after feature selection.
Ergo, this script is designed to be run after (and informed by) the
Feature Selection notebook.

"""

import subprocess
from pathlib import Path

import pandas

# Path to the root of the git repository.
GIT_ROOT = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
GIT_ROOT = Path(GIT_ROOT.decode('utf-8').strip())

# Path to the input dataset.
INPUT_DATASET_PATH = GIT_ROOT / Path('build/combined_data.csv')

# Path to the output dataset.
OUTPUT_DATASET_PATH = GIT_ROOT / Path('build/preprocessed_data.csv')

# Columns to subset from the original input dataset.
SUBSET_COLUMNS = ['age', 'sex', 'cp', 'thalrest', 'trestbps', 'restecg', 'fbs',
                  'thalach', 'exang', 'oldpeak', 'num']


def main():
    dataset = pandas.read_csv(INPUT_DATASET_PATH)

    # Discard all columns except those in SUBSET_COLUMNS.
    data_subset = dataset[SUBSET_COLUMNS]

    # Discard all rows that contain NAs.
    data_subset = data_subset.dropna()

    # Discard all rows where resting blood pressue is 0.
    data_subset = data_subset[data_subset.trestbps != 0]

    # Convert chest pain to a binary class.
    data_subset.loc[data_subset['cp'] != 4, 'cp'] = 1
    data_subset.loc[data_subset['cp'] == 4, 'cp'] = -1

    # Convert resting ECG to a binary class.
    data_subset.loc[data_subset['restecg'] != 1, 'restecg'] = -1

    # Rescale binary/ternary classes to range from -1 to 1.
    data_subset.loc[data_subset['sex'] == 0, 'sex'] = -1
    data_subset.loc[data_subset['exang'] == 0, 'exang'] = -1

    data_subset.to_csv(OUTPUT_DATASET_PATH, index=None)
    print(f'Removed {len(dataset) - len(data_subset)} rows from the original dataset.')
    print(f'Total rows in new dataset: {len(data_subset)}.')


if __name__ == '__main__':
    main()
