"""
Unit tests for gen_model.py

Copyright 2020 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import os
import unittest
import random
import subprocess
from pathlib import Path
from unittest.mock import patch, Mock

import sklearn
import numpy as np
import scipy as sp
import pandas as pd

import gen_model
import scoring


class CreateValidationDatasetTest(unittest.TestCase):
    """
    Tests for gen_model.create_validation_dataset()

    """

    def test_create_validation_dataset(self):
        """
        Test create_validation_dataset() on typical inputs.

        """

        input_data = np.array([[1, 2, 3], [4, 5, 6]])
        target_data = np.array([2, 5])
        prediction_data = np.array([3, 6])
        columns = ['col1', 'col2', 'col3']
        target_dataset = pd.DataFrame([[1, 2, 3, 2, 3], [4, 5, 6, 5, 6]],
                                      columns=columns + ['target', 'prediction'])

        validation_dataset = gen_model.create_validation_dataset(input_data,
                                                                 target_data,
                                                                 prediction_data,
                                                                 columns)

        self.assertTrue(target_dataset.eq(validation_dataset).all().all())


class GetCommitHashTest(unittest.TestCase):
    """
    Tests for gen_model.get_commit_hash()

    """

    def test_git_call_success(self):
        """
        Test get_commit_hash() when calls to git are successful.

        """

        def create_run_command_mock():
            def run_command_mock(command):
                if command == 'git diff':
                    return ''

                elif command == 'git rev-parse --verify HEAD':
                    return '26223577219e04975a8ea93b95d0ab047a0ea536'

                assert False

            return run_command_mock

        run_command_patch = patch.object(gen_model,
                                         'run_command',
                                         new_callable=create_run_command_mock)

        with run_command_patch:
            commit_hash = gen_model.get_commit_hash()

        self.assertEqual(commit_hash, '26223577219e04975a8ea93b95d0ab047a0ea536')

    def test_nonempty_git_diff(self):
        """
        Test get_commit_hash() when git diff returns a nonempty string.

        """

        run_command_patch = patch.object(gen_model,
                                         'run_command',
                                         return_value='sdfigh')

        with run_command_patch:
            commit_hash = gen_model.get_commit_hash()

        self.assertEqual(commit_hash, '')

    def test_file_not_found_error(self):
        """
        Test get_commit_hash() when git can not be found

        """

        def create_run_command_mock():
            def run_command_mock(command):
                raise FileNotFoundError()

            return run_command_mock

        run_command_patch = patch.object(gen_model,
                                         'run_command',
                                         new_callable=create_run_command_mock)

        with run_command_patch:
            commit_hash = gen_model.get_commit_hash()

        self.assertEqual(commit_hash, '')


class RunCommandTest(unittest.TestCase):
    """
    Tests for gen_model.run_command()

    """

    def test_run_command(self):
        """
        Test run_command() on a typical input.

        """

        check_output_patch = patch.object(gen_model.subprocess,
                                          'check_output',
                                          return_value=b'bbd155263aeaae63c12ad7498a0594fb2ff8d615\n')

        with check_output_patch as check_output_mock:
            command_output = gen_model.run_command('git rev-parse --verify HEAD')

        self.assertEqual(check_output_mock.call_count, 1)
        self.assertEqual(check_output_mock.call_args[0][0],
                         ['git', 'rev-parse', '--verify', 'HEAD'])

        self.assertEqual(command_output,
                         'bbd155263aeaae63c12ad7498a0594fb2ff8d615')


class TrainModelTest(unittest.TestCase):
    """
    Tests for gen_model.train_model()

    """

    def setUp(self):
        random.seed(326717227)
        sp.random.seed(326717227)

    def test_svm_classifier(self):
        """
        Test train_model() with SVM classifier.

        """

        grid = [{'model__C': [1, 2, 3], 'model__kernel': ['linear', 'rbf']}]
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1],
                           [0, 0], [0, 1], [1, 0], [1, 1],
                           [0, 0], [0, 1], [1, 0], [1, 1]])

        targets = np.array([-1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1])
        score = scoring.create_scorer('informedness')
        model = gen_model.train_model(sklearn.svm.SVC,
                                      inputs,
                                      targets,
                                      score,
                                      ['standard scaling'],
                                      parameter_grid=grid,
                                      cpus=1)

        self.assertEqual(len(model.steps), 2)
        self.assertEqual(model.steps[0][0], 'standard scaling')
        self.assertEqual(model.steps[1][0], 'model')
        self.assertTrue((model.predict(inputs) == targets).all())

    def test_qda_classifier(self):
        """
        Test train_model() with QDA classifier.

        """

        inputs = np.array([[-4], [-3], [-2], [-1], [1], [2], [3], [4]])
        targets = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
        score = scoring.create_scorer('accuracy')
        model = gen_model.train_model(sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis,
                                      inputs,
                                      targets,
                                      score,
                                      [],
                                      cpus=2)

        self.assertEqual(len(model.steps), 1)
        self.assertEqual(model.steps[0][0], 'model')
        self.assertTrue((model.predict(inputs) == targets).all())

    def test_sgd_classifier(self):
        """
        Test train_model() with SGD classifier.

        """

        inputs = np.array([[-4], [-3], [-2], [-1], [1], [2], [3], [4]])
        targets = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
        score = scoring.create_scorer('precision')
        model = gen_model.train_model(sklearn.linear_model.SGDClassifier,
                                      inputs,
                                      targets,
                                      score,
                                      ['pca', 'robust scaling'],
                                      cpus=4)

        self.assertEqual(len(model.steps), 3)
        self.assertEqual(model.steps[0][0], 'pca')
        self.assertEqual(model.steps[1][0], 'robust scaling')
        self.assertEqual(model.steps[2][0], 'model')
        self.assertTrue(model.predict(inputs).any())


class SplitInputsTests(unittest.TestCase):
    """
    Tests for gen_model.split_inputs

    """

    def test_split_inputs(self):
        """
        Test split_inputs() on a typical input.

        """

        data = pd.DataFrame([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
        inputs = gen_model.split_inputs(data)
        self.assertTrue((inputs == np.array([[0, 0], [0, 1], [1, 0], [1, 1]])).all())


class SplitTargetTests(unittest.TestCase):
    """
    Tests for gen_model.split_inputs

    """

    def test_split_target(self):
        """
        Test split_target() on a typical input.

        """

        data = pd.DataFrame([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
        inputs = gen_model.split_target(data)
        self.assertTrue((inputs == np.array([0, 1, 1, 0])).all())


class BindModelMetadataTests(unittest.TestCase):
    """
    Tests for gen_model.bind_model_metadata

    """

    def test_bind_model_metadata(self):
        """
        Test bind_model_metadata() on typical inputs.

        """

        def create_run_command_mock():
            def run_command_mock(command):
                if command == 'git config user.name':
                    return 'Joshua Norton'

                elif command == 'git config user.email':
                    return 'emperorofamerica@gmail.com'

                elif command == 'git diff':
                    return ''

                elif 'git rev-parse --verify HEAD':
                    return '26223577219e04975a8ea93b95d0ab047a0ea536'

                assert False

            return run_command_mock

        scores = dict(accuracy=1.,
                      informedness=2.,
                      mcc=3.,
                      precision=4.,
                      recall=5.,
                      f1_score=6.,
                      ami=7.,
                      sensitivity=8.,
                      specificity=9.,
                      dor=10.,
                      lr_plus=11.,
                      lr_minus=12.,
                      roc_auc=13.)

        attributes = ('commit_hash', 'validated', 'reposistory', 'numpy_version',
                      'scipy_version', 'pandas_version', 'sklearn_version',
                      'joblib_version', 'threadpoolctl_version', 'operating_system',
                      'architecture', 'created', 'author')

        attributes += tuple(scores.keys())
        model = Mock()
        run_command_patch = patch.object(gen_model,
                                         'run_command',
                                         new_callable=create_run_command_mock)

        with run_command_patch:
            gen_model.bind_model_metadata(model, scores)

        for attribute in attributes:
            self.assertTrue(hasattr(model, attribute))

    def test_run_command_raises_called_process_error(self):
        """
        Test bind_model_metadata() when run_command raises a
        subprocess.CalledProcessError on git config user.name.

        """

        def create_run_command_mock():
            def run_command_mock(command):
                if command == 'git config user.name':
                    raise subprocess.CalledProcessError(1, "['git', 'config', 'user.name']")

                elif command == 'git diff':
                    return ''

                elif 'git rev-parse --verify HEAD':
                    return '26223577219e04975a8ea93b95d0ab047a0ea536'

                assert False

            return run_command_mock

        scores = dict(accuracy=1.,
                      informedness=2.,
                      mcc=3.,
                      precision=4.,
                      recall=5.,
                      f1_score=6.,
                      ami=7.,
                      sensitivity=8.,
                      specificity=9.,
                      dor=10.,
                      lr_plus=11.,
                      lr_minus=12.,
                      roc_auc=13.)

        attributes = ('commit_hash', 'validated', 'reposistory', 'numpy_version',
                      'scipy_version', 'pandas_version', 'sklearn_version',
                      'joblib_version', 'threadpoolctl_version', 'operating_system',
                      'architecture', 'created', 'author')

        attributes += tuple(scores.keys())
        model = Mock()
        run_command_patch = patch.object(gen_model,
                                         'run_command',
                                         new_callable=create_run_command_mock)

        with run_command_patch:
            gen_model.bind_model_metadata(model, scores)

        for attribute in attributes:
            self.assertTrue(hasattr(model, attribute))

        self.assertEqual(model.author, ' <>')


class ReadConfigFileTest(unittest.TestCase):
    """
    Tests for gen_model.read_config_file()

    """

    def test_invalid_algorithm(self):
        """
        Test read_config_file() with an invalid algorithm.

        """

        def create_json_load_mock():
            def json_load_mock(args):
                return dict(training_dataset='src/tests/data/binary_training_dataset1.csv',
                            validation_dataset='src/tests/data/binary_validation_dataset1.csv',
                            algorithm='invalid_algorithm')

            return json_load_mock

        json_load_patch = patch.object(gen_model.json,
                                       'load',
                                       new_callable=create_json_load_mock)

        with json_load_patch:
            with self.assertRaises(gen_model.InvalidConfigError):
                gen_model.read_config_file(Path(os.devnull))

    def test_invalid_preprocessing_methods(self):
        """
        Test read_config_file() with invalid preprocessing methods.

        """

        def create_json_load_mock():
            def json_load_mock(args):
                return dict(training_dataset='src/tests/data/binary_training_dataset1.csv',
                            validation_dataset='src/tests/data/binary_validation_dataset1.csv',
                            algorithm='svm',
                            preprocessing_methods=['invalid_method'])

            return json_load_mock

        json_load_patch = patch.object(gen_model.json,
                                       'load',
                                       new_callable=create_json_load_mock)

        with json_load_patch:
            with self.assertRaises(gen_model.InvalidConfigError):
                gen_model.read_config_file(Path(os.devnull))

    def test_invalid_config_json(self):
        """
        Test read_config_file() when config file is not valid json.

        """

        with self.assertRaises(gen_model.InvalidConfigError):
            gen_model.read_config_file(Path(os.devnull))


class IsValidConfigTest(unittest.TestCase):
    """
    Tests for gen_model.is_valid_config()

    """

    def test_valid_config_with_algorithm_parameters(self):
        """
        Test that `is_valid_config` raises an exception when `config` is
        valid and contains algorithm parameters.

        """

        config = dict(training_dataset='',
                      validation_dataset='',
                      random_seed=str(1),
                      scoring='accuracy',
                      preprocessing_methods=['pca'],
                      pca_components=1,
                      pca_whiten=False,
                      algorithm='svm',
                      algorithm_parameters=[dict(C=[1])])

        gen_model.is_valid_config(config)

    def test_valid_config_without_algorithm_parameters(self):
        """
        Test that `is_valid_config` raises an exception when `config` is
        valid and does not contain algorithm parameters.

        """

        config = dict(training_dataset='',
                      validation_dataset='',
                      random_seed=str(1),
                      scoring='accuracy',
                      preprocessing_methods=['pca'],
                      pca_components=1,
                      pca_whiten=False,
                      algorithm='svm')

        gen_model.is_valid_config(config)

    def test_config_not_a_dict(self):
        """
        Test that `is_valid_config` raises an exception when `config` is
        not a dict.

        """

        with self.assertRaises(gen_model.InvalidConfigError):
            gen_model.is_valid_config([])

    def test_config_contains_misspelled_parameters(self):
        """
        Test that `is_valid_config` raises an exception when `config`
        contains a misspelled parameter.

        """

        config = dict(training_dataset=None,
                      validation_dataset=None,
                      random_seed=None,
                      scoring=None,
                      preprocessing_method=None,
                      pca_components=None,
                      pca_whiten=None,
                      algorithm=None)

        with self.assertRaises(gen_model.InvalidConfigError):
            gen_model.is_valid_config(config)

    def test_config_contains_missing_parameters(self):
        """
        Test that `is_valid_config` raises an exception when `config`
        contains a missing parameter.

        """

        config = dict(training_dataset=None,
                      validation_dataset=None,
                      scoring=None,
                      preprocessing_methods=None,
                      pca_components=None,
                      pca_whiten=None,
                      algorithm=None)

        with self.assertRaises(gen_model.InvalidConfigError):
            gen_model.is_valid_config(config)

    def test_config_contains_extraneous_parameters(self):
        """
        Test that `is_valid_config` raises an exception when `config`
        contains an extraneous parameter.

        """

        config = dict(training_dataset=None,
                      validation_dataset=None,
                      random_seed=None,
                      scoring=None,
                      preprocessing_methods=None,
                      pca_components=None,
                      pca_whiten=None,
                      algorithm=None,
                      extraneous_parameter=None)

        with self.assertRaises(gen_model.InvalidConfigError):
            gen_model.is_valid_config(config)

    def test_training_dataset_not_a_path(self):
        """
        Test that `is_valid_config` raises an exception when
        `training_dataset` is not a path.

        """

        config = dict(training_dataset=None,
                      validation_dataset=None,
                      random_seed=None,
                      scoring=None,
                      preprocessing_methods=None,
                      pca_components=None,
                      pca_whiten=None,
                      algorithm=None)

        with self.assertRaises(gen_model.InvalidConfigError):
            gen_model.is_valid_config(config)

    def test_validation_dataset_not_a_path(self):
        """
        Test that `is_valid_config` raises an exception when
        `validation_dataset` is not a path.

        """

        config = dict(training_dataset='',
                      validation_dataset=None,
                      random_seed=None,
                      scoring=None,
                      preprocessing_methods=None,
                      pca_components=None,
                      pca_whiten=None,
                      algorithm=None)

        with self.assertRaises(gen_model.InvalidConfigError):
            gen_model.is_valid_config(config)

    def test_random_seed_too_big(self):
        """
        Test that `is_valid_config` raises an exception when
        `random_seed` is greater than 2**32-1.

        """

        config = dict(training_dataset='',
                      validation_dataset='',
                      random_seed=str(2**32),
                      scoring=None,
                      preprocessing_methods=None,
                      pca_components=None,
                      pca_whiten=None,
                      algorithm=None)

        with self.assertRaises(gen_model.InvalidConfigError):
            gen_model.is_valid_config(config)

    def test_random_seed_too_small(self):
        """
        Test that `is_valid_config` raises an exception when
        `random_seed` is less than 0.

        """

        config = dict(training_dataset='',
                      validation_dataset='',
                      random_seed=str(-1),
                      scoring=None,
                      preprocessing_methods=None,
                      pca_components=None,
                      pca_whiten=None,
                      algorithm=None)

        with self.assertRaises(gen_model.InvalidConfigError):
            gen_model.is_valid_config(config)

    def test_random_seed_not_an_integer(self):
        """
        Test that `is_valid_config` raises an exception when
        `random_seed` is not an integer.

        """

        config = dict(training_dataset='',
                      validation_dataset='',
                      random_seed=str(1.5),
                      scoring=None,
                      preprocessing_methods=None,
                      pca_components=None,
                      pca_whiten=None,
                      algorithm=None)

        with self.assertRaises(gen_model.InvalidConfigError):
            gen_model.is_valid_config(config)

    def test_scoring_method_is_not_valid(self):
        """
        Test that `is_valid_config` raises an exception when
        `scoring_method` is not valid.

        """

        config = dict(training_dataset='',
                      validation_dataset='',
                      random_seed=str(1),
                      scoring='invalid',
                      preprocessing_methods=None,
                      pca_components=None,
                      pca_whiten=None,
                      algorithm=None)

        with self.assertRaises(gen_model.InvalidConfigError):
            gen_model.is_valid_config(config)

    def test_preprocessing_methods_not_valid(self):
        """
        Test that `is_valid_config` raises an exception when
        `preprocessing_methods` are not valid.

        """

        config = dict(training_dataset='',
                      validation_dataset='',
                      random_seed=str(1),
                      scoring='accuracy',
                      preprocessing_methods=['invalid'],
                      pca_components=None,
                      pca_whiten=None,
                      algorithm=None)

        with self.assertRaises(gen_model.InvalidConfigError):
            gen_model.is_valid_config(config)

    def test_preprocessing_methods_wrong_type(self):
        """
        Test that `is_valid_config` raises an exception when
        `preprocessing_methods` is the wrong type.

        """

        config = dict(training_dataset='',
                      validation_dataset='',
                      random_seed=str(1),
                      scoring='accuracy',
                      preprocessing_methods=1,
                      pca_components=None,
                      pca_whiten=None,
                      algorithm=None)

        with self.assertRaises(gen_model.InvalidConfigError):
            gen_model.is_valid_config(config)

    def test_pca_components_too_small(self):
        """
        Test that `is_valid_config` raises an exception when
        `pca_components` is less than or equal to 0.

        """

        config = dict(training_dataset='',
                      validation_dataset='',
                      random_seed=str(1),
                      scoring='accuracy',
                      preprocessing_methods=['pca'],
                      pca_components=0,
                      pca_whiten=None,
                      algorithm=None)

        with self.assertRaises(gen_model.InvalidConfigError):
            gen_model.is_valid_config(config)

    def test_pca_components_wrong_type(self):
        """
        Test that `is_valid_config` raises an exception when
        `pca_components` is not an integer.

        """

        config = dict(training_dataset='',
                      validation_dataset='',
                      random_seed=str(1),
                      scoring='accuracy',
                      preprocessing_methods=['pca'],
                      pca_components=[],
                      pca_whiten=None,
                      algorithm=None)

        with self.assertRaises(gen_model.InvalidConfigError):
            gen_model.is_valid_config(config)

    def test_pca_whiten_wrong_type(self):
        """
        Test that `is_valid_config` raises an exception when
        `pca_whiten` is not a boolean.

        """

        config = dict(training_dataset='',
                      validation_dataset='',
                      random_seed=str(1),
                      scoring='accuracy',
                      preprocessing_methods=['pca'],
                      pca_components=1,
                      pca_whiten=[],
                      algorithm=None)

        with self.assertRaises(gen_model.InvalidConfigError):
            gen_model.is_valid_config(config)

    def test_algorithm_is_not_valid(self):
        """
        Test that `is_valid_config` raises an exception when
        `algorithm` is not a valid value.

        """

        config = dict(training_dataset='',
                      validation_dataset='',
                      random_seed=str(1),
                      scoring='accuracy',
                      preprocessing_methods=['pca'],
                      pca_components=1,
                      pca_whiten=False,
                      algorithm='invalid')

        with self.assertRaises(gen_model.InvalidConfigError):
            gen_model.is_valid_config(config)

    def test_algorithm_parameters_wrong_type(self):
        """
        Test that `is_valid_config` raises an exception when
        `algorithm_parameters` is not the correct type.

        """

        config = dict(training_dataset='',
                      validation_dataset='',
                      random_seed=str(1),
                      scoring='accuracy',
                      preprocessing_methods=['pca'],
                      pca_components=1,
                      pca_whiten=False,
                      algorithm='qda',
                      algorithm_parameters=None)

        with self.assertRaises(gen_model.InvalidConfigError):
            gen_model.is_valid_config(config)

    def test_algorithm_parameters_extraneous_parameter(self):
        """
        Test that `is_valid_config` raises an exception when
        `algorithm_parameters` contains an extraneous parameter.

        """

        config = dict(training_dataset='',
                      validation_dataset='',
                      random_seed=str(1),
                      scoring='accuracy',
                      preprocessing_methods=['pca'],
                      pca_components=1,
                      pca_whiten=False,
                      algorithm='qda',
                      algorithm_parameters=[dict(C=[1])])

        with self.assertRaises(gen_model.InvalidConfigError):
            gen_model.is_valid_config(config)


class CrossValidateTest(unittest.TestCase):
    """
    Tests for gen_model.cross_validate()

    """

    INPUTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1]])
    TARGETS = np.array([0, 1, 1, 1, 0, 1, 1, 1])

    def test_perfect_model(self):
        """
        Test cross_validate() with a model that always predicts correctly.

        """

        def predict_mock(inputs):
            inputs = inputs.tolist()
            all_inputs = self.INPUTS.tolist()
            targets = self.TARGETS.tolist()
            predictions = []
            for input_ in inputs:
                predictions.append(targets[all_inputs.index(input_)])

            return np.array(predictions)

        model = Mock()
        model.predict = predict_mock
        datasets = gen_model.Datasets(gen_model.Dataset(self.INPUTS, self.TARGETS),
                                      gen_model.Dataset(self.INPUTS, self.TARGETS),
                                      ['x', 'y'])

        sklearn_clone_patch = patch.object(sklearn,
                                           'clone',
                                           new_callable=lambda: lambda x: x)

        with sklearn_clone_patch:
            median_scores, mad_scores = gen_model.cross_validate(model,
                                                                 datasets,
                                                                 2)

        self.assertEqual(median_scores,
                         dict(accuracy=1.0,
                              informedness=1.0,
                              mcc=1.0,
                              precision=1.0,
                              recall=1.0,
                              f1_score=1.0,
                              ami=1.0,
                              sensitivity=1.0,
                              specificity=1.0,
                              dor=np.inf,
                              lr_plus=np.inf,
                              lr_minus=0.0,
                              roc_auc=1.0))

        self.assertTrue(np.isnan(mad_scores['dor']))
        self.assertTrue(np.isnan(mad_scores['lr_plus']))
        mad_scores_sans_nan = mad_scores.copy()
        del mad_scores_sans_nan['dor']
        del mad_scores_sans_nan['lr_plus']
        self.assertEqual(mad_scores_sans_nan,
                         dict(accuracy=0.0,
                              informedness=0.0,
                              mcc=0.0,
                              precision=0.0,
                              recall=0.0,
                              f1_score=0.0,
                              ami=0.0,
                              sensitivity=0.0,
                              specificity=0.0,
                              lr_minus=0.0,
                              roc_auc=0.0))

    def test_useless_model(self):
        """
        Test cross_validate() with a model that always predicts incorrectly.

        """

        def predict_mock(inputs):
            return np.full(len(inputs), 100)

        model = Mock()
        model.predict = predict_mock
        datasets = gen_model.Datasets(gen_model.Dataset(self.INPUTS, self.TARGETS),
                                      gen_model.Dataset(self.INPUTS, self.TARGETS),
                                      ['x', 'y'])

        sklearn_clone_patch = patch.object(sklearn,
                                           'clone',
                                           new_callable=lambda: lambda x: x)

        with sklearn_clone_patch:
            median_scores, mad_scores = gen_model.cross_validate(model,
                                                                 datasets,
                                                                 2)

        self.assertEqual(median_scores,
                         dict(accuracy=0.0,
                              informedness=-1.0,
                              mcc=0.0,
                              precision=0.0,
                              recall=0.0,
                              f1_score=0.0,
                              ami=-1.1845850666627777e-15))

        self.assertEqual(mad_scores,
                         dict(accuracy=0.0,
                              informedness=0.0,
                              mcc=0.0,
                              precision=0.0,
                              recall=0.0,
                              f1_score=0.0,
                              ami=0.0))

    def test_random_model(self):
        """
        Test cross_validate() with a model that makes predictions at random.

        """

        def predict_mock(inputs):
            return np.random.randint(0, 2, len(inputs))

        np.random.seed(1)
        model = Mock()
        model.predict = predict_mock
        datasets = gen_model.Datasets(gen_model.Dataset(np.repeat(self.INPUTS, 10000, 0),
                                                        np.repeat(self.TARGETS, 10000)),
                                      gen_model.Dataset(np.repeat(self.INPUTS, 10000, 0),
                                                        np.repeat(self.TARGETS, 10000)),
                                      ['x', 'y'])

        sklearn_clone_patch = patch.object(sklearn,
                                           'clone',
                                           new_callable=lambda: lambda x: x)

        with sklearn_clone_patch:
            median_scores, mad_scores = gen_model.cross_validate(model,
                                                                 datasets,
                                                                 50)

        expected_median_scores = dict(accuracy=0.4975,
                                      informedness=-0.50171875,
                                      mcc=0.0,
                                      precision=1.0,
                                      recall=0.495,
                                      f1_score=0.6615078803758534,
                                      ami=5.184292427413571e-15,
                                      sensitivity=0.495,
                                      specificity=0.0,
                                      lr_plus=0.496875,
                                      lr_minus=np.inf,
                                      dor=0.9452974770681903,
                                      roc_auc=0.4929404761904762)

        for score_x, score_y in zip(median_scores.values(), expected_median_scores.values()):
            self.assertAlmostEqual(score_x, score_y)

        expected_mad_scores = dict(accuracy=0.007968749999999997,
                                   informedness=0.009531249999999991,
                                   mcc=0.0,
                                   precision=0.0,
                                   recall=0.00927083333333334,
                                   f1_score=0.010489296852607799,
                                   ami=1.995059231561136e-15,
                                   sensitivity=0.00927083333333334,
                                   specificity=0.0,
                                   lr_plus=0.011249999999999982,
                                   lr_minus=np.nan,
                                   dor=0.025665399623180152,
                                   roc_auc=0.003409226190476178)

        for score_x, score_y in zip(mad_scores.values(), expected_mad_scores.values()):
            if np.isnan(score_x) and np.isnan(score_y):
                continue

            self.assertAlmostEqual(score_x, score_y)

    def test_large_spread_model(self):
        """
        Test cross_validate() with a model that makes predictions at random
        with a large amount of spread.

        """

        def predict_mock(inputs):
            choice = random.randint(0, 3)
            if choice == 0:
                return np.full(len(inputs), 0)

            elif choice == 1:
                return np.full(len(inputs), 1)

            elif choice == 2:
                return np.random.randint(0, 2, len(inputs))

            elif choice == 3:
                return np.full(len(inputs), 2)

            else:
                assert False

        random.seed(3)
        np.random.seed(3)
        model = Mock()
        model.predict = predict_mock
        datasets = gen_model.Datasets(gen_model.Dataset(np.repeat(self.INPUTS, 20, 0),
                                                        np.repeat(self.TARGETS, 20)),
                                      gen_model.Dataset(np.repeat(self.INPUTS, 20, 0),
                                                        np.repeat(self.TARGETS, 20)),
                                      ['x', 'y'])

        sklearn_clone_patch = patch.object(sklearn,
                                           'clone',
                                           new_callable=lambda: lambda x: x)

        with sklearn_clone_patch:
            median_scores, mad_scores = gen_model.cross_validate(model,
                                                                 datasets,
                                                                 10)

        expected_median_scores = dict(accuracy=0.5625,
                                      informedness=0.0,
                                      mcc=0.0,
                                      precision=0.4375,
                                      recall=0.53125,
                                      f1_score=0.5227272727272727,
                                      ami=0.011661702327125191,
                                      sensitivity=0.5,
                                      specificity=0.0,
                                      lr_plus=1.0,
                                      lr_minus=np.inf,
                                      roc_auc=0.5,
                                      dor=1.9142857142857144)

        for score_a, score_b in zip(median_scores.values(), expected_median_scores.values()):
            self.assertAlmostEqual(score_a, score_b)

        expected_mad_scores = dict(accuracy=0.375,
                                   informedness=0.125,
                                   mcc=0.0,
                                   precision=0.4375,
                                   recall=0.46875,
                                   f1_score=0.4772727272727273,
                                   ami=0.02377993878345713,
                                   sensitivity=0.5,
                                   specificity=0.0,
                                   lr_plus=0.8,
                                   lr_minus=np.nan,
                                   roc_auc=0.0,
                                   dor=0.9142857142857144)

        for score_a, score_b in zip(mad_scores.values(), expected_mad_scores.values()):
            if np.isnan(score_a) and np.isnan(score_b):
                continue

            self.assertAlmostEqual(score_a, score_b)


if __name__ == '__main__':
    unittest.main()
