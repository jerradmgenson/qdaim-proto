"""
Unit tests for gen_model.py

"""

import unittest
import random
from unittest.mock import patch

import sklearn
import numpy as np
import scipy as sp
import pandas as pd

import gen_model


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

    def test_train_model_svm(self):
        """
        Test train_model() with SVM classifier.

        """

        grid = [{'model__C': [1, 2, 3], 'model__kernel': ['linear', 'rbf']}]
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1],
                           [0, 0], [0, 1], [1, 0], [1, 1],
                           [0, 0], [0, 1], [1, 0], [1, 1]])

        targets = np.array([-1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1])
        score = gen_model.create_scorer('informedness')
        model = gen_model.train_model(sklearn.svm.SVC,
                                      inputs,
                                      targets,
                                      score,
                                      'standard scaling',
                                      parameter_grid=grid)

        self.assertEqual(len(model.steps), 2)
        self.assertEqual(model.steps[0][0], 'standard scaling')
        self.assertEqual(model.steps[1][0], 'model')
        self.assertTrue((model.predict(inputs) == targets).all())

    def test_train_model_qda(self):
        """
        Test train_model() with QDA classifier.

        """

        inputs = np.array([[-4], [-3], [-2], [-1], [1], [2], [3], [4]])
        targets = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
        score = gen_model.create_scorer('accuracy')
        model = gen_model.train_model(sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis,
                                      inputs,
                                      targets,
                                      score,
                                      'none',
                                      cpus=2)

        self.assertEqual(len(model.steps), 1)
        self.assertEqual(model.steps[0][0], 'model')
        self.assertTrue((model.predict(inputs) == targets).all())

    def test_train_model_sgd(self):
        """
        Test train_model() with SGD classifier.

        """

        inputs = np.array([[-4], [-3], [-2], [-1], [1], [2], [3], [4]])
        targets = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
        score = gen_model.create_scorer('precision')
        model = gen_model.train_model(sklearn.linear_model.SGDClassifier,
                                      inputs,
                                      targets,
                                      score,
                                      'pca',
                                      cpus=4)

        self.assertEqual(len(model.steps), 2)
        self.assertEqual(model.steps[0][0], 'pca')
        self.assertEqual(model.steps[1][0], 'model')
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


class SplitTargetsTests(unittest.TestCase):
    """
    Tests for gen_model.split_inputs

    """

    def test_split_targets(self):
        """
        Test split_inputs() on a typical input.

        """

        data = pd.DataFrame([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
        inputs = gen_model.split_target(data)
        self.assertTrue((inputs == np.array([0, 1, 1, 0])).all())


if __name__ == '__main__':
    unittest.main()
