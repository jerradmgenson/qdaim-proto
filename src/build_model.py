import random
import argparse
import logging
import pickle
import time
import subprocess
import multiprocessing
import os
import json
from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


GIT_ROOT = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode('utf-8').strip())
CONFIG_FILE_PATH = GIT_ROOT / 'config.json'
OUTPUT_PATH = GIT_ROOT / 'build'
DATASET_COLUMNS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
GITHUB_URL = 'https://github.com/jerradmgenson/cardiac'
MODEL_BASE_NAME = 'heart_disease_model'
SVC_PARAMETER_GRID = [
    {'model__C': [0.001, 0.1, 0.5, 1, 2, 10, 100, 1000], 'model__kernel': ['linear'], 'model__cache_size': [500]},
    {'model__C': [0.001, 0.1, 0.5, 1, 2, 10, 100, 1000], 'model__kernel': ['rbf', 'sigmoid'], 'model__gamma': [0.001, 0.0001], 'model__cache_size': [500]},
    {'model__C': [0.001, 0.1, 0.5, 1, 2, 10, 100, 1000], 'model__kernel': ['poly'], 'model__gamma': [0.001, 0.0001], 'model__degree': [2, 3, 4], 'model__cache_size': [500]},
]

KNC_PARAMETER_GRID = [
    {'model__n_neighbors': [3, 5, 10, 15], 'model__weights': ['uniform', 'distance'],  'model__algorithm': ['ball_tree', 'kd_tree', 'brute'], 'model__p': [1, 2, 3]}
]

SGD_PARAMETER_GRID = [
    {'model__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'], 'model__penalty': ['l1', 'l2', 'elasticnet'], 'model__alpha': 10.0**-np.arange(1,7), 'model__max_iter': [1500, 2000]},
]

RFC_PARAMETER_GRID = [
    {'model__n_estimators': [50, 100, 200], 'model__max_features': [None, 'sqrt'], 'model__max_samples': [0.1, 0.4, 0.8, 1], 'model__criterion': ['gini', 'entropy'], 'model__min_samples_split': [2, 4, 8]},
]

Model = namedtuple('Model', 'class_ name abbreviation parameter_grid')

MODELS = (Model(svm.SVC, 'support vector machine', 'svc', SVC_PARAMETER_GRID),
          Model(KNeighborsClassifier, 'k-nearest neighbors', 'knc', KNC_PARAMETER_GRID),
          Model(RandomForestClassifier, 'random forest', 'rfc', RFC_PARAMETER_GRID),
          Model(SGDClassifier, 'stochastic gradient descent', 'sgc', SGD_PARAMETER_GRID))


Config = namedtuple('Config',
                    'input_path validation_fraction columns random_seed scoring')


Scores = namedtuple('Scores',
                    'accuracy precision sensitivity specificity informedness')


def main():
    start_time = time.time()
    command_line_arguments = parse_command_line()
    configure_logging(command_line_arguments.log_level)
    config = read_config_file(CONFIG_FILE_PATH)
    print('Loading cardiac dataset from {}'.format(config.input_path))
    cardiac_dataset = pd.read_csv(str(config.input_path))
    print('Done loading cardiac dataset.')
    reduced_dataset = cardiac_dataset[config.columns + ['target']]
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    shuffled_dataset = reduced_dataset.sample(frac=1)
    input_data = shuffled_dataset.values[:, 0:-1]
    target_data = shuffled_dataset.values[:, -1]
    validation_rows = int(len(input_data) * config.validation_fraction)
    validation_inputs = input_data[:validation_rows]
    validation_targets = target_data[:validation_rows]
    training_inputs = input_data[validation_rows:]
    training_targets = target_data[validation_rows:]
    commit_hash = get_commit_hash()
    print('Training dataset rows: {}'.format(len(training_inputs)))
    print('Validation dataset rows: {}'.format(len(validation_inputs)))
    print('Random number generator seed: {}'.format(config.random_seed))
    print('Commit hash: {}'.format(commit_hash))
    jobs = []
    for model_args in MODELS:
        if model_args.name == 'stochastic gradient descent':
            model_args.parameter_grid[0]['model__max_iter'] = [np.ceil(10**6 / len(training_inputs))]

        job = (model_args,
               training_inputs,
               training_targets,
               config.scoring)

        jobs.append(job)

    with multiprocessing.Pool(command_line_arguments.cpu) as pool:
        models = pool.map(train_model, jobs)

    for model in models:
        model.commit_hash = commit_hash
        model.validation = 'UNVALIDATED'
        model.github_url = GITHUB_URL
        scores = validate_model(model, validation_inputs, validation_targets)
        model.scores = scores
        print_model_results(model, model.name)
        output_path = (command_line_arguments.output_path
                       / (MODEL_BASE_NAME + '_{}.dat'.format(model.abbreviation)))

        save_model(model, output_path)
        print('Saved {} model to {}'.format(model.name, output_path))
        model = None

    print('Runtime: {} seconds'.format(time.time() - start_time))


def read_config_file(path):
    with path.open() as config_fp:
        config_json = json.load(config_fp)

    config_json['input_path'] = config_json['input_path'].replace('{GIT_ROOT}', str(GIT_ROOT))
    return Config(**config_json)


def get_commit_hash():
    try:
        if subprocess.check_output(['git', 'diff']).strip():
            return ''

        return subprocess.check_output(['git', 'rev-parse', '--verify', 'HEAD']).decode('utf-8').strip()

    except FileNotFoundError:
        return ''


def print_model_results(model, name):
    print('\nScores for {}:'.format(name))
    for metric, value in model.scores._asdict().items():
        print('{}:    {}'.format(metric, value))


def parse_command_line():
    parser = argparse.ArgumentParser(description='Build a machine learning model to predict heart disease.')
    parser.add_argument('-o', '--output_path',
                        default=OUTPUT_PATH,
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


def validation_fraction(n):
    x = float(n)
    if x < 0 or x > 1:
        raise ValueError('validation_fraction must not be less than 0 or greater than 1.')

    return x


def dataset_column(n):
    if n not in DATASET_COLUMNS:
        raise ValueError('column must be one of {}'.format(DATASET_COLUMNS))

    return n


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


def train_model(job):
    model_args = job[0]
    training_inputs = job[1]
    training_targets = job[2]
    scoring = job[3]
    calculate_score = create_scorer(scoring)
    base_model = model_args.class_()
    pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('model', base_model)])
    grid_estimator = GridSearchCV(pipeline, model_args.parameter_grid,
                                  scoring=calculate_score)

    grid_estimator.fit(training_inputs, training_targets)
    model = grid_estimator.best_estimator_
    model.best_score = grid_estimator.best_score_
    model.name = model_args.name
    model.abbreviation = model_args.abbreviation

    return model


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

    return Scores(accuracy=accuracy,
                  precision=precision,
                  sensitivity=sensitivity,
                  specificity=specificity,
                  informedness=informedness)


def save_model(model, output_path):
    with output_path.open('wb') as output_file:
        pickle.dump(model, output_file)


if __name__ == '__main__':
    main()
