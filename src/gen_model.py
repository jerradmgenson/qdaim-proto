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
import re
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
GIT_ROOT = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
GIT_ROOT = Path(GIT_ROOT.decode('utf-8').strip())

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
                    ('training_dataset',
                     'testing_dataset',
                     'columns',
                     'random_seed',
                     'scoring',
                     'algorithm',
                     'algorithm_parameters'))

# Possible scoring methods that may be used for hyperparameter tuning.
SCORING_METHODS = 'accuracy precision sensitivity specificity informedness'

# Collects model scores together in a single object.
ModelScores = namedtuple('ModelScores', SCORING_METHODS)

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
    print('Training dataset: {}'.format(config.training_dataset))
    print('Testing dataset: {}'.format(config.testing_dataset))
    print('Using columns: {}'.format(config.columns))
    print('Random number generator seed: {}'.format(config.random_seed))
    print('Scoring method: {}'.format(config.scoring))
    print('Algorithm: {}'.format(config.algorithm.name))
    print('Training dataset rows: {}'.format(len(datasets.training.inputs)))
    print('Testing dataset rows: {}'.format(len(datasets.testing.inputs)))
    score_function = create_scorer(config.scoring)
    model = train_model(config.algorithm.class_,
                        datasets.training.inputs,
                        datasets.training.targets,
                        score_function,
                        command_line_arguments.cpu,
                        config.algorithm_parameters)

    model_scores, predictions = score_model(model,
                                            datasets.testing.inputs,
                                            datasets.testing.targets)

    bind_model_metadata(model, model_scores)
    validation_dataset = create_validation_dataset(datasets.testing.inputs,
                                                   datasets.testing.targets,
                                                   predictions,
                                                   config.columns)

    save_validation(validation_dataset, command_line_arguments.output_path)
    save_model(model, command_line_arguments.output_path)
    print('\nSaved model to {}'.format(command_line_arguments.output_path))
    print('Runtime: {} seconds'.format(time.time() - start_time))

    return 0


def create_validation_dataset(input_data, target_data, prediction_data, columns):
    """
    Create a Pandas DataFrame that shows how input samples and expected target
    data corresponds to the actual outputs predicted by the model.

    Args
      input_data: A 2D numpy array of inputs to the model, where each
                  row of the array represents a sample and each column
                  represents a feature.
      target_data: A 1D numpy array of model targets. Each element is an
                   expected output for the corresponding sample in the
                   input_data array.
      predicted_data: A 1D numpy array of model targets. Each element is an
                      actual model output for the corresponding sample in the
                      input_data array.
      columns: A list of column names for the model input data.

    Returns
      An instance of pandas.DataFrame.

    """

    validation_dataset = pd.DataFrame(data=input_data, columns=columns)
    validation_dataset['target'] = target_data
    validation_dataset['prediction'] = prediction_data

    return validation_dataset


def run_command(command):
    """
    Run the given command in a subprocess and return its output.

    Args
      command: The command to run as a string.

    Returns
      Standard output from the subprocess, decoded as a UTF-8 string.

    """

    return subprocess.check_output(re.split(r'\s+', command)).decode('utf-8').strip()


def bind_model_metadata(model, scores):
    """
    Generate metadata and bind it to the model. The following attributes
    will be assigned to `model`:

    - commit_hash
    - validated
    - repository
    - numpy_version
    - scipy_version
    - pandas_version
    - sklearn_version
    - joblib_version
    - threadpoolctl_version
    - operating_system
    - architecture
    - created
    - author

    Args
      model: An instance of a scikit-learn estimator.
      scores: An instance of ModelScores.

    Returns
      None

    """

    print('\nModel scores:')
    for metric, score in scores._asdict().items():
        setattr(model, metric, score)
        print('{}:    {}'.format(metric, score))

    model.commit_hash = get_commit_hash()
    model.validated = False
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
    username = run_command('git config user.name')
    email = run_command('git config user.email')
    model.author = '{} <{}>'.format(username, email)


def train_model(model_class,
                input_data,
                target_data,
                score_function,
                cpus=1,
                parameter_grid=None):

    """
    Train a machine learning model on the given data.

    Args
      model_class: A scikit-learn estimator class e.g. 'sklearn.svm.SVC'.
      input_data: A 2D numpy array of inputs to the model, where each
                  row of the array represents a sample and each column
                  represents a feature.
      target_data: A 1D numpy array of model targets. Each element is an
                   expected output for the corresponding sample in the
                   input_data array.
      score_function: A function that takes three parameters - an
                      estimator, an array of input data, and an array
                      of target data, and returns a score as a float,
                      where higher numbers are better.
      cpus: (Default=1) Number of processes to use for training the model.
      parameter_grid: (Default=None) A sequence of dicts with possible
                      hyperparameter values. Used for tuning the
                      hyperparameters. When present, grid search will be
                      used to train the model.

    Returns
      A trained scikit-learn estimator object.

    """

    pipeline = Pipeline(steps=[('scaler', StandardScaler()),
                               ('model', model_class())])

    if parameter_grid:
        grid_estimator = GridSearchCV(pipeline,
                                      parameter_grid,
                                      scoring=score_function,
                                      n_jobs=cpus)

        grid_estimator.fit(input_data, target_data)
        model = grid_estimator.best_estimator_

    else:
        model = pipeline
        model.fit(input_data, target_data)

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
                                     targets=split_target(training_subset)),
                    testing=Dataset(inputs=split_inputs(testing_subset),
                                    targets=split_target(testing_subset)))


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
    """
    Save a validation dataset alongside a model.

    Args
      dataset: Validation dataset as a pandas dataframe.
      output_path: Path that the model was saved to.

    Returns
      None

    """

    dataset_path = output_path.with_name(output_path.stem + '_validation.csv')
    dataset.to_csv(dataset_path, index=None)


def read_config_file(path):
    """
    Read a json configuration file into memory.

    Args
      path: Path to the configuration file (as a Path object).

    Returns
      A Config object.

    """

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

    if 'algorithm_parameters' not in config_json:
        config_json['algorithm_parameters'] = []

    # Prepend algorithm parameters with `model__` so they can be fed to
    # scikit-learn Pipeline without raising a ValueError.
    modified_algorithm_parameters = []
    for parameter_set in config_json['algorithm_parameters']:
        modified_parameter_set = {'model__' + key: value for key, value in parameter_set.items()}
        modified_algorithm_parameters.append(modified_parameter_set)

    config_json['algorithm_parameters'] = modified_algorithm_parameters

    return Config(**config_json)


def get_commit_hash():
    """
    Get the git commit hash of the current commit.

    Returns
      The current commit hash as a string. If there are uncommitted
      changes, or if the current working directory is not in a git
      reposistory, return the empty string instead.

    """

    try:
        if run_command('git diff'):
            return ''

        return run_command('git rev-parse --verify HEAD')

    except FileNotFoundError:
        return ''


def parse_command_line():
    """
    Parse the command line using argparse.

    Returns
      A Namespace object returned by parse_args().

    """

    description = 'Build a machine learning model to predict heart disease.'
    parser = argparse.ArgumentParser(description=description)
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
    """
    Configure the logger for the current module.

    Returns
      A Logger object for the current module.

    """

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
    """
    Create a scoring function for hyperparameter tuning.

    Args
      scoring: The scoring method that the function should use. Possible
               values are enumerated by `SCORING_METHODS`.

    Returns
      A function that can be passed to the `scoring` parameter of
      `sklearn.model_selection.GridSearchCV`.

    """

    def calculate_score(model, inputs, targets):
        model_scores, _ = score_model(model, inputs, targets)
        return getattr(model_scores, scoring)

    return calculate_score


def score_model(model, input_data, target_data):
    """
    Score the given model on a set of data. The scoring metrics used are
    accuracy, precision, sensitivity, specificity, and informedness.

    Args
      model: A trained instance of a scikit-learn estimator.
      input_data: A 2D numpy array of inputs to the model where the rows
                  are samples and the columns are features.
      target_data: A 1D numpy array of expected model outputs.

    Returns
      An instance of ModelScores.

    """

    prediction_data = model.predict(input_data)
    model_scores = ModelScores(accuracy=calculate_accuracy(target_data, prediction_data),
                               precision=calculate_precision(target_data, prediction_data),
                               sensitivity=calculate_sensitivity(target_data, prediction_data),
                               specificity=calculate_specificity(target_data, prediction_data),
                               informedness=calculate_informedness(target_data, prediction_data))

    return model_scores, prediction_data


def calculate_accuracy(target_data, prediction_data):
    """
    Calculate accuracy - the number of correct predictions divided by
    the total number of predictions

    """

    correct_predictions = np.sum(target_data == prediction_data)
    return correct_predictions / len(prediction_data)


def calculate_precision(target_data, prediction_data):
    """
    Calculate precision - the number of true positives divided by the
    total number of positive predictions.

    """

    true_positives = calculate_true_positives(target_data, prediction_data)
    false_positives = calculate_false_positives(target_data, prediction_data)
    return true_positives / (true_positives + false_positives)


def calculate_sensitivity(target_data, prediction_data):
    """
    Calculate sensitivity - the proportion of positives that are
    correctly identified.

    """

    true_positives = calculate_true_positives(target_data, prediction_data)
    false_negatives = calculate_false_negatives(target_data, prediction_data)
    return true_positives / (true_positives + false_negatives)


def calculate_specificity(target_data, prediction_data):
    """
    Calculate sensitivity - the proportion of negatives that are
    correctly identified.

    """

    true_negatives = calculate_true_negatives(target_data, prediction_data)
    false_positives = calculate_false_positives(target_data, prediction_data)
    return true_negatives / (true_negatives + false_positives)


def calculate_informedness(target_data, prediction_data):
    """
    Calculate informedness (aka Youden's J statistic).
    This is the sensitivity + specificity - 1.

    """

    return (calculate_sensitivity(target_data, prediction_data)
            + calculate_specificity(target_data, prediction_data)
            - 1)


def calculate_true_positives(target_data, prediction_data):
    """
    Calculate the number of positives (1's) in prediction_data that agree with
    the corresponding element of target_data.

    """

    return np.sum(((target_data == 1).astype(int) + (prediction_data == 1).astype(int)) == 2)


def calculate_false_positives(target_data, prediction_data):
    """
    Calculate the number of positives (1's) in prediction_data that disagree with
    the corresponding element of target_data.

    """

    return np.sum(((target_data == 0).astype(int) + (prediction_data == 1).astype(int)) == 2)


def calculate_true_negatives(target_data, prediction_data):
    """
    Calculate the number of negatives (0's) in prediction_data that agree with
    the corresponding element of target_data.

    """

    return np.sum(((target_data == 0).astype(int) + (prediction_data == 0).astype(int)) == 2)


def calculate_false_negatives(target_data, prediction_data):
    """
    Calculate the number of negatives (0's) in prediction_data that disagree with
    the corresponding element of target_data.

    """

    return np.sum(((target_data == 1).astype(int) + (prediction_data == 0).astype(int)) == 2)


def save_model(model, output_path):
    """
    Save the given model to disk.

    Args
      model: An instance of a scikit-learn estimator.
      output_path: The path to save the model to as a Path object.

    Returns
      None

    """

    with output_path.open('wb') as output_file:
        pickle.dump(model, output_file)


if __name__ == '__main__':
    sys.exit(main())
