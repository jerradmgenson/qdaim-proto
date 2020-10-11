"""
Unit tests for util.py

Copyright 2020 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import util


class GetCommitHashTest(unittest.TestCase):
    """
    Tests for util.get_commit_hash()

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

        run_command_patch = patch.object(util,
                                         'run_command',
                                         new_callable=create_run_command_mock)

        with run_command_patch:
            commit_hash = util.get_commit_hash()

        self.assertEqual(commit_hash, '26223577219e04975a8ea93b95d0ab047a0ea536')

    def test_nonempty_git_diff(self):
        """
        Test get_commit_hash() when git diff returns a nonempty string.

        """

        run_command_patch = patch.object(util,
                                         'run_command',
                                         return_value='sdfigh')

        with run_command_patch:
            commit_hash = util.get_commit_hash()

        self.assertEqual(commit_hash, '')

    def test_file_not_found_error(self):
        """
        Test get_commit_hash() when git can not be found

        """

        def create_run_command_mock():
            def run_command_mock(command):
                raise FileNotFoundError()

            return run_command_mock

        run_command_patch = patch.object(util,
                                         'run_command',
                                         new_callable=create_run_command_mock)

        with run_command_patch:
            commit_hash = util.get_commit_hash()

        self.assertEqual(commit_hash, '')


class RunCommandTest(unittest.TestCase):
    """
    Tests for util.run_command()

    """

    def test_run_command(self):
        """
        Test run_command() on a typical input.

        """

        check_output_patch = patch.object(util.subprocess,
                                          'check_output',
                                          return_value=b'bbd155263aeaae63c12ad7498a0594fb2ff8d615\n')

        with check_output_patch as check_output_mock:
            command_output = util.run_command('git rev-parse --verify HEAD')

        self.assertEqual(check_output_mock.call_count, 1)
        self.assertEqual(check_output_mock.call_args[0][0],
                         ['git', 'rev-parse', '--verify', 'HEAD'])

        self.assertEqual(command_output,
                         'bbd155263aeaae63c12ad7498a0594fb2ff8d615')


class SplitInputsTests(unittest.TestCase):
    """
    Tests for util.split_inputs

    """

    def test_split_inputs(self):
        """
        Test split_inputs() on a typical input.

        """

        data = pd.DataFrame([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
        inputs = util.split_inputs(data)
        self.assertTrue((inputs == np.array([[0, 0], [0, 1], [1, 0], [1, 1]])).all())


class SplitTargetTests(unittest.TestCase):
    """
    Tests for util.split_inputs

    """

    def test_split_target(self):
        """
        Test split_target() on a typical input.

        """

        data = pd.DataFrame([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
        inputs = util.split_target(data)
        self.assertTrue((inputs == np.array([0, 1, 1, 0])).all())


class ReadConfigFileTest(unittest.TestCase):
    """
    Tests for util.read_config_file()

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

        json_load_patch = patch.object(util.json,
                                       'load',
                                       new_callable=create_json_load_mock)

        with json_load_patch:
            with self.assertRaises(util.InvalidConfigError):
                util.read_config_file(Path(os.devnull))

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

        json_load_patch = patch.object(util.json,
                                       'load',
                                       new_callable=create_json_load_mock)

        with json_load_patch:
            with self.assertRaises(util.InvalidConfigError):
                util.read_config_file(Path(os.devnull))

    def test_invalid_config_json(self):
        """
        Test read_config_file() when config file is not valid json.

        """

        with self.assertRaises(util.InvalidConfigError):
            util.read_config_file(Path(os.devnull))


class IsValidConfigTest(unittest.TestCase):
    """
    Tests for util.is_valid_config()

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

        util.is_valid_config(config)

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

        util.is_valid_config(config)

    def test_config_not_a_dict(self):
        """
        Test that `is_valid_config` raises an exception when `config` is
        not a dict.

        """

        with self.assertRaises(util.InvalidConfigError):
            util.is_valid_config([])

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

        with self.assertRaises(util.InvalidConfigError):
            util.is_valid_config(config)

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

        with self.assertRaises(util.InvalidConfigError):
            util.is_valid_config(config)

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

        with self.assertRaises(util.InvalidConfigError):
            util.is_valid_config(config)

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

        with self.assertRaises(util.InvalidConfigError):
            util.is_valid_config(config)

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

        with self.assertRaises(util.InvalidConfigError):
            util.is_valid_config(config)

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

        with self.assertRaises(util.InvalidConfigError):
            util.is_valid_config(config)

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

        with self.assertRaises(util.InvalidConfigError):
            util.is_valid_config(config)

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

        with self.assertRaises(util.InvalidConfigError):
            util.is_valid_config(config)

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

        with self.assertRaises(util.InvalidConfigError):
            util.is_valid_config(config)

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

        with self.assertRaises(util.InvalidConfigError):
            util.is_valid_config(config)

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

        with self.assertRaises(util.InvalidConfigError):
            util.is_valid_config(config)

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

        with self.assertRaises(util.InvalidConfigError):
            util.is_valid_config(config)

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

        with self.assertRaises(util.InvalidConfigError):
            util.is_valid_config(config)

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

        with self.assertRaises(util.InvalidConfigError):
            util.is_valid_config(config)

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

        with self.assertRaises(util.InvalidConfigError):
            util.is_valid_config(config)

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

        with self.assertRaises(util.InvalidConfigError):
            util.is_valid_config(config)

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

        with self.assertRaises(util.InvalidConfigError):
            util.is_valid_config(config)


if __name__ == '__main__':
    unittest.main()
