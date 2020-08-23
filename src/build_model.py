import random
import argparse
import logging
import pickle
from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


DATASET_COLUMNS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
DEFAULT_COLUMNS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'target']
MODEL_BASE_NAME = 'heart_disease_model'
TRAINING_ITERATIONS = 1
SVC_PARAMETER_GRID = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'kernel': ['rbf', 'sigmoid'], 'gamma': [0.001, 0.0001]},
    {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'gamma': [0.001, 0.0001], 'degree': [2, 3, 4]},
]

KNC_PARAMETER_GRID = [
    {'n_neighbors': [3, 5, 10, 15], 'weights': ['uniform', 'distance'],  'algorithm': ['ball_tree', 'kd_tree', 'brute'], 'p': [1, 2], 'n_jobs': [-1]}
]

SGD_PARAMETER_GRID = [
    {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'], 'penalty': ['l1', 'l2', 'elasticnet'], 'n_jobs': [-1]},
]

RFC_PARAMETER_GRID = [
    {'n_estimators': [50, 100, 200, 1000], 'max_samples': [0.1, 0.4, 0.8, 1], 'criterion': ['gini', 'entropy'], 'n_jobs': [-1]},
]

MLP_PARAMETER_GRID = [
    {'hidden_layer_sizes': [(10,), (5,), (3,), (10, 10), (10, 5), (10, 3), (5, 5), (5, 3), (3, 3)], 'activation': ['logistic', 'tanh', 'relu'], 'solver': ['lbfgs'], 'alpha': [.0001, .001, .01]},
]

Model = namedtuple('Model', 'class_ name abbreviation parameter_grid')

MODELS = (Model(svm.SVC, 'support vector machine', 'svc', SVC_PARAMETER_GRID),
          Model(KNeighborsClassifier, 'k-nearest neighbors', 'knc', KNC_PARAMETER_GRID),
          Model(RandomForestClassifier, 'random forest', 'rfc', RFC_PARAMETER_GRID),
          Model(SGDClassifier, 'stochastic gradient descent', 'sgc', SGD_PARAMETER_GRID),
          Model(MLPClassifier, 'multilayer perceptron', 'mlp', MLP_PARAMETER_GRID))


def main():
    command_line_arguments = parse_command_line()
    configure_logging(command_line_arguments.log_level)
    print('Loading cardiac dataset from {}.'.format(command_line_arguments.input_path))
    cardiac_dataset = pd.read_csv(str(command_line_arguments.input_path))
    print('Done loading cardiac dataset.')
    if command_line_arguments.columns:
        reduced_dataset = cardiac_dataset[command_line_arguments.columns + ['target']]

    else:
        reduced_dataset = cardiac_dataset[DEFAULT_COLUMNS + ['target']]

    print('Initial dataset')
    print(reduced_dataset)
    np.random.seed(command_line_arguments.random_seed)
    shuffled_dataset = reduced_dataset.sample(frac=1)
    input_data = shuffled_dataset.values[:, 0:-1]
    target_data = shuffled_dataset.values[:, -1]
    scaler = StandardScaler().fit(input_data)
    scaled_inputs = scaler.transform(input_data)
    training_rows = int(len(scaled_inputs) * (1 - command_line_arguments.validation_fraction))
    training_inputs = scaled_inputs[:training_rows]
    training_targets = target_data[:training_rows]
    print('\nTraining data ({} rows)'.format(len(training_inputs)))
    print(training_inputs)
    validation_inputs = scaled_inputs[training_rows:]
    validation_targets = target_data[training_rows:]
    print('\nValidation data ({} rows)'.format(len(validation_inputs)))
    print(validation_inputs)
    print('\nRandom number generator seed: {}'.format(command_line_arguments.random_seed))

    for model in MODELS:
        build_model(model.class_,
                    model.name,
                    model.abbreviation,
                    model.parameter_grid,
                    scaler,
                    command_line_arguments.random_seed,
                    command_line_arguments.output_path,
                    training_inputs,
                    training_targets,
                    validation_inputs,
                    validation_targets)


def build_model(model_class,
                name,
                abbreviation,
                parameter_grid,
                scaler,
                random_seed,
                output_path,
                training_inputs,
                training_targets,
                validation_inputs,
                validation_targets):
    print('\nConstructing {} model from dataset.'.format(name))
    model = train_model(model_class,
                        parameter_grid,
                        training_inputs,
                        training_targets,
                        TRAINING_ITERATIONS)

    model.scaler = scaler
    model.random_seed = random_seed
    validate_model(model, validation_inputs, validation_targets)
    print_model_results(model, name)
    final_output_path = output_path / (MODEL_BASE_NAME + '_{}.dat'.format(abbreviation))
    save_model(model, final_output_path)
    print('Saved {} model to {}'.format(name, final_output_path))


def print_model_results(model, name):
    print('Result for {} model.'.format(name))
    print('Accuracy:    {}'.format(model.accuracy))
    print('Precision:   {}'.format(model.precision))
    print('Sensitivity: {}'.format(model.sensitivity))
    print('Specificity: {}'.format(model.specificity))


def parse_command_line():
    parser = argparse.ArgumentParser(description='Build a machine learning model to predict heart disease.')
    parser.add_argument('input_path', type=Path, help='Path to the input cardiac dataset.')
    parser.add_argument('output_path', type=Path, help='Path to output the heart disease model to.')
    parser.add_argument('--validation_fraction',
                        type=validation_fraction,
                        default=0.2,
                        help='The fraction of the dataset to use for validation as a decimal between 0 and 1.')

    parser.add_argument('--columns',
                        type=dataset_column,
                        nargs='+',
                        help='Columns in the heart disease dataset to use as inputs to the model.')

    parser.add_argument('--random_seed',
                        type=int,
                        default=random.randint(1, 10000000),
                        help='Initial integer to seed the random number generator with.')

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


def train_model(model_class, parameter_grid, input_data, target_data, iterations):
    best_model = None
    for _ in range(iterations):
        base_model = model_class()
        grid_search_model = GridSearchCV(base_model, parameter_grid)
        grid_search_model.fit(input_data, target_data)
        if best_model is None or grid_search_model.best_score_ > best_model.best_score_:
            best_model = grid_search_model

    best_model.training_inputs = input_data
    best_model.training_targets = target_data

    return best_model


def validate_model(model, input_data, target_data):
    predictions = model.predict(input_data)
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    for prediction, target in zip(predictions, target_data):
        if prediction == 1 and target == 1:
            true_positives += 1

        elif prediction == 1 and target == 0:
            false_positives += 1

        elif prediction == 0 and target == 1:
            false_negatives += 1

        else:
            true_negatives += 1

    model.accuracy = (true_positives + true_negatives) / len(input_data)
    model.precision = true_positives / (true_positives + false_positives)
    model.sensitivity = true_positives / (true_positives + false_negatives)
    model.specificity = true_negatives / (true_negatives + false_positives)
    model.validation_inputs = input_data
    model.validation_targets = target_data


def save_model(model, output_path):
    with output_path.open('wb') as output_file:
        pickle.dump(model, output_file)


if __name__ == '__main__':
    main()
