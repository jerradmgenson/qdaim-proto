#!/usr/bin/python3
"""
Generate a customizable classification model with a wide variety of
preprocessing and model configurations.

Copyright 2020, 2021 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

positional arguments:
  target                Output path to save the model to.
  training              Path to the training dataset.
  validation            Path to the validation dataset.

optional arguments:
  -h, --help            show this help message and exit
  --cpu CPU             Number of processes to use for training models.
  --log-level {critical,error,warning,info,debug}
                        Log level to configure logging with.
  --cross-validate CROSS_VALIDATE
                        Cross-validate the model using the specified number of folds.
  --outlier-scores      Score model on outliers in the testing data.
  --model {svm,rfc,etc,gbc,sgd,rrc,lrc,nbc,lda,qda,dtc,knn,rnc}
                        Algorithm to use to generate the model.
  --preprocessing {standard scaling,
                   robust scaling,
                   quantile transformer,
                   power transformer,
                   normalize,
                   pca,
                   ica,
                   isomap,
                   lle,
                   feature agglomeration,
                   nca,
                   factor analysis} [...]
                        Preprocessing methods to use in the generated model.
  --scoring {accuracy,
             precision,
             sensitivity,
             specificity,
             informedness,
             mcc,
             recall,
             f1_score,
             ami,
             dor,
             lr_plus,
             lr_minus,
             roc_auc}
                        Scoring method to use for model hyperparameter tuning.
  --random-state RANDOM_STATE
                        State to initialize random number generators with.
  --parameter-grid PARAMETER_GRID
                        Parameter grid to use with grid search
                        (as a json string).
  --print-hyperparameters
                        Print hyperparameter values of the final model.

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
