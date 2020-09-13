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
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


# Path to the root of the git repository.
GIT_ROOT = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
GIT_ROOT = Path(GIT_ROOT.decode('utf-8').strip())

# URL for the repository on Github.
GITHUB_URL = 'https://github.com/jerradmgenson/cardiac'

# Path to the configuration file
CONFIG_FILE_PATH = GIT_ROOT / 'src/gen_model_config.json'

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
    'rrc': MLAlgorithm('ridge regression classifier', RidgeClassifier),
    'lrc': MLAlgorithm('logistic regression classifier', LogisticRegression),
    'lda': MLAlgorithm('linear discriminant analysis', LinearDiscriminantAnalysis),
    'qda': MLAlgorithm('quadratic discriminant analysis', QuadraticDiscriminantAnalysis),
    'mlp': MLAlgorithm('multilayer perceptron', MLPClassifier),
}

# Possible preprocessing methods that can be used to prepare data for
# a model.
PREPROCESSING_METHODS = {
    'none': None,
    'standard scaling': StandardScaler,
    'robust scaling': RobustScaler,
    'quantile transformer': QuantileTransformer,
    'power transformer': PowerTransformer,
    'normalize': Normalizer,
    'pca': PCA,
}


# Stores values from the configuration file.
Config = namedtuple('Config',
                    ('training_dataset',
                     'testing_dataset',
                     'random_seed',
                     'scoring',
                     'algorithm',
                     'algorithm_parameters',
                     'preprocessing_method',
                     'pca_whiten',
                     'pca_components'))

# Possible scoring methods that may be used for hyperparameter tuning.
SCORING_METHODS = 'accuracy precision hmean_precision hmean_recall sensitivity specificity informedness'

# Collects model scores together in a single object.
ModelScores = namedtuple('ModelScores', SCORING_METHODS)

# Contains input data and target data for a single dataset.
Dataset = namedtuple('Dataset', 'inputs targets')

# Stores training and testing datasets together in a single object.
Datasets = namedtuple('Datasets', 'training testing columns')


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
    datasets = load_datasets(config.training_dataset, config.testing_dataset)
    print(f'Training dataset:     {config.training_dataset}')
    print(f'Testing dataset:      {config.testing_dataset}')
    print(f'Random number seed:   {config.random_seed}')
    print(f'Scoring method:       {config.scoring}')
    print(f'Algorithm:            {config.algorithm.name}')
    print(f'Preprocessing method: {config.preprocessing_method}')
    print(f'Training samples:     {len(datasets.training.inputs)}')
    print(f'Testing samples:      {len(datasets.testing.inputs)}')
    score_function = create_scorer(config.scoring)
    model = train_model(config.algorithm.class_,
                        datasets.training.inputs,
                        datasets.training.targets,
                        score_function,
                        config.preprocessing_method,
                        n_components=config.pca_components,
                        whiten=config.pca_whiten,
                        cpus=command_line_arguments.cpu,
                        parameter_grid=config.algorithm_parameters)

    model_scores, predictions = score_model(model,
                                            datasets.testing.inputs,
                                            datasets.testing.targets)

    bind_model_metadata(model, model_scores)
    validation_dataset = create_validation_dataset(datasets.testing.inputs,
                                                   datasets.testing.targets,
                                                   predictions,
                                                   datasets.columns[:-1])

    save_validation(validation_dataset, command_line_arguments.output_path)
    save_model(model, command_line_arguments.output_path)
    print(f'\nSaved model to {command_line_arguments.output_path}')
    print(f'Runtime: {time.time() - start_time:.2} seconds')

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
        if score is None:
            continue

        setattr(model, metric, score)
        label = metric + ':'
        print(f'{label:16} {score:.4}')

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
                preprocessing_method,
                n_components=None,
                whiten=False,
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
      preprocessing_method: The method to use to preprocess the data before
                            feeding it to the model. Must be a key of
                            'PREPROCESSING_METHODS'.
      n_components: (Default=None) The number of components to keep when using
                    principal component analysis. Ignored unless
                    'preprocessing_method' is 'pca'.
      whiten: (Default=False) Whether to whiten the data when using principal
              component analysis. Ignored unless 'processing_method' is 'pca'.
      cpus: (Default=1) Number of processes to use for training the model.
      parameter_grid: (Default=None) A sequence of dicts with possible
                      hyperparameter values. Used for tuning the
                      hyperparameters. When present, grid search will be
                      used to train the model.

    Returns
      A trained scikit-learn estimator object.

    """

    pipeline_steps = []
    if preprocessing_method == 'pca':
        preprocessor = PREPROCESSING_METHODS[preprocessing_method](n_components=n_components, whiten=whiten)

        pipeline_steps.append((preprocessing_method, preprocessor))

    elif preprocessing_method == 'none':
        pass

    else:
        preprocessor = PREPROCESSING_METHODS[preprocessing_method]()
        pipeline_steps.append((preprocessing_method, preprocessor))

    pipeline_steps.append(('model', model_class()))
    pipeline = Pipeline(steps=pipeline_steps)
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


def load_datasets(training_dataset, testing_dataset):
    """
    Load training and testing datasets from the filesystem and return
    them as a Datasets object.

    Args
      training_dataset: Path to the training dataset.
      testing_dataset: Path to the testing dataset.

    Returns
      An instance of Datasets.

    """

    training_dataset = pd.read_csv(str(training_dataset))
    testing_dataset = pd.read_csv(str(testing_dataset))

    return Datasets(training=Dataset(inputs=split_inputs(training_dataset),
                                     targets=split_target(training_dataset)),
                    testing=Dataset(inputs=split_inputs(testing_dataset),
                                    targets=split_target(testing_dataset)),
                    columns=training_dataset.columns)


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

    if config_json['preprocessing_method'] not in PREPROCESSING_METHODS:
        raise ValueError('Unknown preprocessing method `{}`.'.format(config_json['preprocessing_method']))

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

    def scorer(model, inputs, targets):
        model_scores, _ = score_model(model, inputs, targets)
        score = getattr(model_scores, scoring)
        if score is None:
            raise ValueError(f'`{scoring}` can not be used with this type of classification.')

        return score

    return scorer


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
    report = sklearn.metrics.classification_report(target_data,
                                                   prediction_data,
                                                   output_dict=True)

    classes = np.unique(target_data)
    if len(classes) == 2:
        positive_class = str(np.max(classes))
        negative_class = str(np.min(classes))
        precision = report[positive_class]['precision']
        sensitivity = report[positive_class]['recall']
        specificity = report[negative_class]['recall']

    else:
        precision = None
        sensitivity = None
        specificity = None

    model_scores = ModelScores(accuracy=report['accuracy'],
                               precision=precision,
                               hmean_precision=calculate_hmean_precision(report, classes),
                               hmean_recall=calculate_hmean_recall(report, classes),
                               sensitivity=sensitivity,
                               specificity=specificity,
                               informedness=calculate_informedness(report, classes))

    return model_scores, prediction_data


def calculate_hmean_recall(classification_report, classes):
    """
    Calculate the harmonic mean of the recall values across all classes.

    Args
      classification_report: A dict returned by sklearn.metrics.classification_report().
      classes: A sequence of unique class names.

    Returns
      The harmonic mean recall as a real number.

    """

    return sp.stats.hmean([classification_report[str(x)]['recall'] for x in classes])


def calculate_hmean_precision(classification_report, classes):
    """
    Calculate the harmonic mean of the precision values across all classes.

    Args
      classification_report: A dict returned by sklearn.metrics.classification_report().
      classes: A sequence of unique class names.

    Returns
      The harmonic mean precision as a real number.

    """

    return sp.stats.hmean([classification_report[str(x)]['precision'] for x in classes])


def calculate_informedness(classification_report, classes):
    """
    Calculate informedness, otherwise known as Youden's J statistic
    generalized by the following formula:

    (sum(recalls) - number of classes / 2) * (2 / number of classes)

    Args
      classification_report: A dict returned by sklearn.metrics.classification_report().
      classes: A sequence of unique class names.

    Returns
      The calculated informedness value as a real number between -1 and 1.

    """

    recall_sum = sum(classification_report[str(c)]['recall'] for c in classes)
    return (recall_sum - len(classes) / 2) * (2 / len(classes))


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
