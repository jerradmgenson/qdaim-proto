"""
Unit tests for gen_model.py

Copyright 2020, 2021 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import unittest
import random
import subprocess
from unittest.mock import patch, Mock

import sklearn
import numpy as np
import scipy as sp
import pandas as pd

import util
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
        score = scoring.scoring_methods()['informedness']
        model = gen_model.train_model(sklearn.svm.SVC,
                                      inputs,
                                      targets,
                                      score,
                                      preprocessing_methods=[util.PREPROCESSING_METHODS['standard scaling']],
                                      parameter_grid=grid,
                                      cpus=1)

        self.assertEqual(len(model.steps), 2)
        self.assertEqual(model.steps[0][0], 'preprocessing1')
        self.assertEqual(model.steps[1][0], 'model')
        self.assertTrue((model.predict(inputs) == targets).all())

    def test_qda_classifier(self):
        """
        Test train_model() with QDA classifier.

        """

        inputs = np.array([[-4], [-3], [-2], [-1], [1], [2], [3], [4]])
        targets = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
        score = scoring.scoring_methods()['accuracy']
        model = gen_model.train_model(sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis,
                                      inputs,
                                      targets,
                                      score,
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
        score = scoring.scoring_methods()['precision']
        model = gen_model.train_model(sklearn.linear_model.SGDClassifier,
                                      inputs,
                                      targets,
                                      score,
                                      preprocessing_methods=[util.PREPROCESSING_METHODS['robust scaling']],
                                      cpus=4)

        self.assertEqual(len(model.steps), 2)
        self.assertEqual(model.steps[0][0], 'preprocessing1')
        self.assertEqual(model.steps[1][0], 'model')
        self.assertTrue(model.predict(inputs).any())


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
        datasets = util.Datasets(util.Dataset(self.INPUTS, self.TARGETS),
                                 util.Dataset(self.INPUTS, self.TARGETS),
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
        datasets = util.Datasets(util.Dataset(self.INPUTS, self.TARGETS),
                                 util.Dataset(self.INPUTS, self.TARGETS),
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
        datasets = util.Datasets(util.Dataset(np.repeat(self.INPUTS, 10000, 0),
                                              np.repeat(self.TARGETS, 10000)),
                                 util.Dataset(np.repeat(self.INPUTS, 10000, 0),
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
        datasets = util.Datasets(util.Dataset(np.repeat(self.INPUTS, 20, 0),
                                              np.repeat(self.TARGETS, 20)),
                                 util.Dataset(np.repeat(self.INPUTS, 20, 0),
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
