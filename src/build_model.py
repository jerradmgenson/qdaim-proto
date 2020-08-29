"""
Generate a machine learning model to classify subjects as positive or
negative for ischemic heart disease. Full reproducibility is provided by
placing all configuration parameters, datasets, and random number
generator seeds in the repository and associating the generated model
with the commit hash.


Configuration
=============
All configuration parameters that may affect the behavior of the model
are located in the file 'config.json' located in the root directory of
the reposistory. The parameters in this file are described individually
below.

- training_dataset: the path to the training dataset starting from the
  root of the repository.

- testing_dataset: the path to the testing dataset starting from the
  root of the reposistory.

- columns: a list of columns from the testing and training datasets to
  use as inputs to the model.

- random_seed: an integer to seed the random number generators with.

- scoring: the method to use for scoring candidate models during model
  generation. Possible values are 'accuracy', 'precision', 'sensitivity',
  'specificity', and 'informedness' (Youden's J statistic).

- algorithm: the machine learning algorithm to use for generating the
  model. Possible values are 'svm' (support vector machine),
  'knn' (k-nearest neighbors), 'rfc' (random forest classifier), and
  'sgd' (stochastic gradient descent).

- algorithm_parameters: a list of parameter dicts that is used during
  grid search to tune the hyperparameters of the model. A complete list
  of available parameters and their values can be found in the
  scikit-learn user manual located at:
  https://scikit-learn.org/stable/supervised_learning.html
  This parameter can be omitted if you wish to use the default values
  for the model's hyperparameters.


Command Line Arguments
======================
In addition to the parameters in the configuration file, some additional
arguments may be supplied via the command line. These are described below.
None of these parameters affect the behavior of the generated model.

usage: build_model.py [-h] [-o OUTPUT_PATH] [--cpu CPU]
                      [--log_level {critical,error,warning,info,debug}]

  -h, --help            show this help message and exit
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path to output the heart disease model to.
  --cpu CPU             Number of processes to use for training models.
  --log_level {critical,error,warning,info,debug}
                        Log level to configure logging with.


Output Path
===========
By default, the output path for generated models is (starting from the
reposistory root): build/heart_disease_model.dat
A CSV file containing the generated model's predictions appended to the
testing dataset is saved alongside the model. This file is given the same
name as the model, with '.dat' replaced by '_validation.csv' (by default,
'heart_disease_model_validation.csv').

"""

import random
import argparse
import logging
import pickle
import time
import subprocess
import os
import sys
import platform
import json
import datetime
from pathlib import Path
from collections import namedtuple

import numpy as np
import scipy as sp
import pandas as pd
import joblib
import threadpoolctl
import sklearn
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Path to the root of the git repository.
GIT_ROOT = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode('utf-8').strip())

# URL for the repository on Github.
GITHUB_URL = 'https://github.com/jerradmgenson/cardiac'

# Path to the configuration file
CONFIG_FILE_PATH = GIT_ROOT / 'config.json'

# Path to save generated models to by default.
DEFAULT_OUTPUT_PATH = GIT_ROOT / 'build/heart_disease_model.dat'

# Identifies a machine learning algorithm's name and sklearn class.
MLAlgorithm = namedtuple('MLAlgorithm', 'name class_')

# Machine learning algorithms that can be used to generate a model.
# Keys are three-letter algorithm abbreviations.
# Values are MLAlgorithm objects.
SUPPORTED_ALGORITHMS = {
    'svm': MLAlgorithm('support vector machine', svm.SVC),
    'knn': MLAlgorithm('k-nearest neighbors', KNeighborsClassifier),
    'rfc': MLAlgorithm('random forest', RandomForestClassifier),
    'sgd': MLAlgorithm('stochastic gradient descent', SGDClassifier),
}

# Stores values from the configuration file.
Config = namedtuple('Config',
                    'training_dataset testing_dataset columns random_seed scoring algorithm algorithm_parameters',
                    defaults=(None,))

# Collects model scores together in a single object.
Scores = namedtuple('Scores',
                    'accuracy precision sensitivity specificity informedness')

# Contains input data and target data for a single dataset.
Dataset = namedtuple('Dataset', 'inputs targets')

# Stores training and testing datasets together in a single object.
Datasets = namedtuple('Datasets', 'training testing')


def main():
    """
    Program's main function. Primary execution starts here.

    """

    start_time = time.time()
    command_line_arguments = parse_command_line()
    configure_logging(command_line_arguments.log_level)
    config = read_config_file(CONFIG_FILE_PATH)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    datasets = load_datasets(config.training_dataset, config.testing_dataset, config.columns)
    commit_hash = get_commit_hash()
    print('Training dataset: {}'.format(config.training_dataset))
    print('Testing dataset: {}'.format(config.testing_dataset))
    print('Using columns: {}'.format(config.columns))
    print('Random number generator seed: {}'.format(config.random_seed))
    print('Scoring method: {}'.format(config.scoring))
    print('Algorithm: {}'.format(config.algorithm.name))
    print('Training dataset rows: {}'.format(len(datasets.training.inputs)))
    print('Testing dataset rows: {}'.format(len(datasets.testing.inputs)))
    print('Commit hash: {}\n'.format(commit_hash))
    score_function = create_scorer(config.scoring)
    model = train_model(config.algorithm.class_,
                        score_function,
                        command_line_arguments.cpu,
                        datasets.training.inputs,
                        datasets.training.targets,
                        config.algorithm_parameters)

    model.algorithm = config.algorithm.name
    model.commit_hash = commit_hash
    model.validation = 'UNVALIDATED'
    model.repository = GITHUB_URL
    model.numpy_version = np.version.version
    model.scipy_version = sp.version.version
    model.pandas_version = pd.__version__
    model.sklearn_version = sklearn.__version__
    model.joblib_version = joblib.__version__
    model.threadpoolctl_version = threadpoolctl.__version__
    model.operating_system = platform.system() + ' ' + platform.version() + ' ' + platform.release()
    model.architecture = platform.processor()
    model.created = datetime.datetime.today().isoformat()
    username = subprocess.check_output(['git', 'config', 'user.name']).decode('utf-8').strip()
    email = subprocess.check_output(['git', 'config', 'user.email']).decode('utf-8').strip()
    model.author = '{} <{}>'.format(username, email)
    scores, predictions = validate_model(model,
                                         datasets.testing.inputs,
                                         datasets.testing.targets)

    print('\nModel scores:')
    for metric, value in scores._asdict().items():
        setattr(model, metric, value)
        print('{}:    {}'.format(metric, value))

    validation_dataset = pd.DataFrame(data=datasets.testing.inputs,
                                      index=config.columns)

    validation_dataset['target'] = datasets.testing.target
    validation_dataset['prediction'] = predictions
    save_validation(validation_dataset, command_line_arguments.output_path)
    save_model(model, command_line_arguments.output_path)
    print('\nSaved model to {}'.format(command_line_arguments.output_path))
    print('Runtime: {} seconds'.format(time.time() - start_time))

    return 0


def train_model(model_class,
                score_function,
                cpus,
                training_inputs,
                training_targets,
                parameter_grid=None):

    """


    """

    pipeline = Pipeline(steps=[('scaler', StandardScaler()),
                               ('model', model_class())])

    if parameter_grid:
        grid_estimator = GridSearchCV(pipeline,
                                      parameter_grid,
                                      scoring=score_function,
                                      n_jobs=cpus)

        grid_estimator.fit(training_inputs, training_targets)
        model = grid_estimator.best_estimator_

    else:
        model = pipeline
        model.fit(training_inputs, training_targets)

    return model


def load_datasets(training_dataset, testing_dataset, columns):
    """
    Load training and testing datasets from the filesystem and return
    them as a Datasets object.

    Args
      training_dataset: Path to the training dataset.
      testing_dataset: Path to the testing dataset.
      columns: List of columns from the datasets to use as model inputs.

    Returns
      An instance of Datasets.

    """

    training_dataset = pd.read_csv(str(training_dataset))
    testing_dataset = pd.read_csv(str(testing_dataset))
    training_subset = training_dataset[columns + ['target']]
    testing_subset = testing_dataset[columns + ['target']]

    return Datasets(training=Dataset(inputs=split_inputs(training_subset),
                                     targets=split_target(training_subset),
                    testing=Dataset(inputs=split_inputs(testing_subset),
                                    targets=split_target(testing_subset))))


def split_inputs(dataframe):
    """
    Split the input columns out of the given dataframe and return them
    as a two-dimensional numpy array. All columns except the last column
    in the dataframe are considered to be input columns.

    """

    return dataframe.to_numpy()[:, 0:-1]


def split_target(dataframe):
    """
    Split the target column out of the given dataframe and return it
    as a one-dimensional numpy array. Only the last column in the
    dataframe is considered to the target column.

    """

    return dataframe.to_numpy()[:, -1]


def save_validation(dataset, output_path):
    dataset_path = output_path.with_name(output_path.stem + '_validation.csv')
    dataset.to_csv(dataset_path, index=None)


def read_config_file(path):
    with path.open() as config_fp:
        config_json = json.load(config_fp)

    config_json['training_dataset'] = os.path.join(str(GIT_ROOT),
                                                   config_json['training_dataset'])

    config_json['testing_dataset'] = os.path.join(str(GIT_ROOT),
                                                  config_json['testing_dataset'])

    try:
        config_json['algorithm'] = SUPPORTED_ALGORITHMS[config_json['algorithm']]

    except KeyError:
        raise ValueError('Unknown machine learning algorithm `{}`'.format(config_json['algorithm']))

    return Config(**config_json)


def get_commit_hash():
    try:
        if subprocess.check_output(['git', 'diff']).strip():
            return ''

        return subprocess.check_output(['git', 'rev-parse', '--verify', 'HEAD']).decode('utf-8').strip()

    except FileNotFoundError:
        return ''


def parse_command_line():
    parser = argparse.ArgumentParser(description='Build a machine learning model to predict heart disease.')
    parser.add_argument('-o', '--output_path',
                        default=DEFAULT_OUTPUT_PATH,
                        type=Path,
                        help='Path to output the heart disease model to.')

    parser.add_argument('--cpu',
                        type=int,
                        default=os.cpu_count(),
                        help='Number of processes to use for training models.')

    parser.add_argument('--log_level',
                        choices=('critical', 'error', 'warning', 'info', 'debug'),
                        default='info',
                        help='Log level to configure logging with.')

    return parser.parse_args()


def configure_logging(log_level):
    log_level_number = getattr(logging, log_level.upper())
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level_number)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level_number)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def create_scorer(scoring):
    def calculate_score(model, inputs, targets):
        scores = validate_model(model, inputs, targets)
        return getattr(scores, scoring)


def validate_model(model, input_data, target_data):
    predictions = model.predict(input_data)
    true_positives = 0.0
    true_negatives = 0.0
    false_positives = 0.0
    false_negatives = 0.0
    for prediction, target in zip(predictions, target_data):
        if prediction == 1 and target == 1:
            true_positives += 1.0

        elif prediction == 1 and target == 0:
            false_positives += 1.0

        elif prediction == 0 and target == 1:
            false_negatives += 1.0

        else:
            true_negatives += 1.0

    accuracy = (true_positives + true_negatives) / len(input_data)
    precision = true_positives / (true_positives + false_positives)
    sensitivity = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)
    informedness = sensitivity + specificity - 1
    scores = Scores(accuracy=accuracy,
                    precision=precision,
                    sensitivity=sensitivity,
                    specificity=specificity,
                    informedness=informedness)

    return scores, predictions


def save_model(model, output_path):
    with output_path.open('wb') as output_file:
        pickle.dump(model, output_file)


if __name__ == '__main__':
    sys.exit(main())
