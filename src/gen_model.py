#!/usr/bin/python3
"""
Generate a machine learning model to classify subjects as positive or
negative for ischemic heart disease. Full reproducibility is provided by
placing all configuration parameters, datasets, and random number
generator seeds in the repository and associating the generated model
with the commit hash.

Copyright 2020, 2021 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.


Configuration
=============
All configuration parameters that may affect the behavior of the model
are located in the file 'config.json' located in the root directory of
the reposistory. The parameters in this file are described individually
below.

- training_dataset: the path to the training dataset starting from the
  root of the repository.

- validation_dataset: the path to the validation dataset starting from the
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
validation dataset is saved alongside the model. This file is given the same
name as the model, with '.dat' replaced by '_validation.csv' (by default,
'heart_disease_model_validation.csv').

"""

import random
import time
import sys
import datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

import util
import scoring

# URL for the repository on Github.
import outliers
GITHUB_URL = 'https://github.com/jerradmgenson/cardiac'


def main(argv):
    """
    Program's main function. Primary execution starts here.

    """

    start_time = time.time()
    command_line_arguments = util.parse_command_line(argv)
    logfile_path = command_line_arguments.target.with_name(
        command_line_arguments.target.stem + '.log')

    util.configure_logging(command_line_arguments.log_level, logfile_path)
    random.seed(command_line_arguments.random_state)
    np.random.seed(command_line_arguments.random_state)
    print('Loading datasets...')
    datasets = util.load_datasets(command_line_arguments.training,
                                  command_line_arguments.validation)

    print(f'Training dataset:      {command_line_arguments.training}')
    print(f'Validation dataset:    {command_line_arguments.validation}')
    print(f'Random state:          {command_line_arguments.random_state}')
    print(f'Scoring method:        {command_line_arguments.scoring}')
    print(f'Model:                 {command_line_arguments.model}')
    print(f'Preprocessing methods: {command_line_arguments.preprocessing}')
    print(f'Training samples:      {len(datasets.training.inputs)}')
    print(f'Validation samples:    {len(datasets.validation.inputs)}')
    score_function = scoring.scoring_methods()[command_line_arguments.scoring]
    print('Generating model...')
    preprocessing_methods = [util.PREPROCESSING_METHODS[i] for i in command_line_arguments.preprocessing]
    model = train_model(util.SUPPORTED_ALGORITHMS[command_line_arguments.model].class_,
                        datasets.training.inputs,
                        datasets.training.targets,
                        score_function,
                        preprocessing_methods=preprocessing_methods,
                        cpus=command_line_arguments.cpu,
                        parameter_grid=command_line_arguments.parameter_grid)

    model.validation = dict()
    print('Scoring model...')
    model_scores = scoring.score_model(model,
                                       datasets.validation.inputs,
                                       datasets.validation.targets)

    print('\nModel scores:')
    for metric, score in model_scores.items():
        if score:
            msg = '{metric:13} {score:.4}'.format(metric=metric + ':', score=score)
            print(msg)

    model.validation['scores'] = model_scores
    if command_line_arguments.cross_validate:
        mean_scores, std_scores = cross_validate(model,
                                                 datasets,
                                                 command_line_arguments.cross_validate)

        print(f'\n{command_line_arguments.cross_validate}-fold cross-validation scores:')
        for metric, mean_score, std_score in zip(std_scores, mean_scores.values(), std_scores.values()):
            if mean_score:
                mean_msg = '{metric:20} {score:.4}'.format(metric='mean ' + metric + ':',
                                                           score=mean_score)

                std_msg = '{metric:20} {score:.4}'.format(metric='std ' + metric + ':',
                                                          score=std_score)

                print(mean_msg)
                print(std_msg)

        model.validation['cross_validation_mean'] = mean_scores
        model.validation['cross_validation_std'] = std_scores

    if command_line_arguments.outlier_scores:
        outlier_scores = outliers.score(model, datasets,
                                        random_state=command_line_arguments.random_state)

        print('\nOutlier scores:')
        for metric, score in outlier_scores.items():
            if score:
                msg = '{metric:13} {score:.4}'.format(metric=metric + ':', score=score)
                print(msg)

        model.validation['outlier_scores'] = outlier_scores

    if command_line_arguments.print_hyperparameters:
        print(f'Model hyperparameters:\n{model.get_params()}\n')

    predictions = model.predict(datasets.validation.inputs)
    validation_dataset = create_validation_dataset(datasets.validation.inputs,
                                                   datasets.validation.targets,
                                                   predictions,
                                                   datasets.columns[:-1])

    print('\nSaving model to disk...')
    model.commit_hash = util.get_commit_hash()
    model.repository = GITHUB_URL
    model.created = datetime.datetime.today().isoformat()
    util.save_validation(validation_dataset, command_line_arguments.target)
    util.save_model(model, command_line_arguments.target)
    print(f'Saved model to {command_line_arguments.target}')
    runtime = f'Runtime: {time.time() - start_time:.2} seconds'
    print(runtime)

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

    assert len(validation_dataset) == len(input_data)
    assert len(validation_dataset.columns) == len(columns) + 2

    return validation_dataset


def train_model(model_class,
                input_data,
                target_data,
                score_function,
                preprocessing_methods=None,
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
      preprocessing_methods: Methods to use to preprocess the data before feeding
                             it to the model. Must be a member of
                             'PREPROCESSING_METHODS'. (Default=None)
      cpus: Number of processes to use for training the model. (Default=1)
      parameter_grid: A sequence of dicts with possible hyperparameter values.
                      Used for tuning the hyperparameters. When present, grid
                      search will be used to train the model. (Default=None)

    Returns
      A trained scikit-learn estimator object.

    """

    pipeline_steps = []
    preprocessing_methods = preprocessing_methods or []
    for count, method in enumerate(preprocessing_methods):
        preprocessor = method()
        pipeline_steps.append((f'preprocessing{count+1}', preprocessor))

    pipeline_steps.append(('model', model_class()))
    pipeline = Pipeline(steps=pipeline_steps)
    if parameter_grid:
        grid_estimator = sklearn.model_selection.GridSearchCV(pipeline,
                                                              parameter_grid,
                                                              scoring=score_function,
                                                              n_jobs=cpus)

        grid_estimator.fit(input_data, target_data)
        model = grid_estimator.best_estimator_

    else:
        model = pipeline
        model.fit(input_data, target_data)

    return model


def cross_validate(model, datasets, n_splits):
    """
    Cross-validate a model by splitting the dataset into training/validation
    sets numerous times and calculating summary statistics for the model scores.

    Args:
      model: A trained instance of a scikit-learn estimator.
      datasets: An instance of Datasets.
      n_splits: Number of splits (or folds) to use in cross-validation.

    Returns:
     A 2-tuple of Scores objects, where the first element is the mean of all
     the models' scores, and the second element is the standard deviation of
     all the models' scores.

    """

    inputs = np.concatenate((datasets.training.inputs,
                             datasets.validation.inputs))

    assert len(inputs) == len(datasets.training.inputs) + len(datasets.validation.inputs)
    targets = np.concatenate((datasets.training.targets,
                              datasets.validation.targets))

    assert len(targets) == len(datasets.training.targets) + len(datasets.validation.targets)
    kfold = sklearn.model_selection.KFold(n_splits=n_splits)
    scores_lists = dict()
    for training_index, testing_index in kfold.split(inputs):
        training_inputs = inputs[training_index]
        training_targets = targets[training_index]
        testing_inputs = inputs[testing_index]
        testing_targets = targets[testing_index]

        new_model = sklearn.clone(model)
        new_model.fit(training_inputs, training_targets)
        scores = scoring.score_model(new_model, testing_inputs, testing_targets)
        for metric, score in scores.items():
            if score is None or np.isnan(score):
                continue

            if metric in scores_lists:
                scores_lists[metric].append(score)

            else:
                scores_lists[metric] = [score]

    mean_scores = dict()
    std_scores = dict()
    for metric, score_list in scores_lists.items():
        assert len(score_list) <= n_splits
        mean_scores[metric] = np.mean(score_list)
        std_scores[metric] = np.std(score_list)

    return mean_scores, std_scores


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
