"""
Unit tests for outliers.py

Copyright 2020 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import unittest
from unittest.mock import Mock

import numpy as np
from scipy import stats

import outliers
import util


class AdjustedBoxplotTestCase(unittest.TestCase):
    """
    Tests for outliers.adjusted_boxplot

    """

    def test_3d_array_raises_value_error(self):
        """
        Test that a 3D x1 array raises a ValueError.

        """

        x1 = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        x2 = x1
        with self.assertRaises(ValueError):
            outliers.adjusted_boxplot(x1, x2)

    def test_unequal_array_ndim_raises_value_error(self):
        """
        Test that ValueError raised when x1.ndim != x2.ndim.

        """

        x1 = [1, 2, 3, 4, 5, 6]
        x2 = [[1, 2], [3, 4], [5, 6]]
        with self.assertRaises(ValueError):
            outliers.adjusted_boxplot(x1, x2)

    def test_empty_array_raises_index_error(self):
        """
        Test that passing empty arrays raises an IndexError.

        """

        with self.assertRaises(IndexError):
            outliers.adjusted_boxplot([], [])

    def test_unequal_array_columns_raise_value_error(self):
        """
        Test that ValueError is raised when x1.shape[1] != x2.shape[1]

        """

        x1 = [[1, 2, 3], [4, 5, 6]]
        x2 = [[1, 2], [3, 4], [5, 6]]
        with self.assertRaises(ValueError):
            outliers.adjusted_boxplot(x1, x2)

    def test_1d_gaussian_with_two_outliers(self):
        """
        Test that outliers are found in 1D gaussian data when two
        are expected.

        """

        dist1 = stats.norm(loc=100, scale=15)
        x1 = dist1.rvs(size=1000, random_state=1)
        dist2 = stats.norm(loc=100, scale=3)
        x2 = dist2.rvs(size=100, random_state=1)
        x2 = np.concatenate([x2, [-50, 150]])
        x2_outliers = outliers.adjusted_boxplot(x1, x2)
        self.assertEqual(x2_outliers.shape, x2.shape)
        self.assertEqual(np.sum(x2_outliers), 2)
        self.assertTrue(x2_outliers[-1])
        self.assertTrue(x2_outliers[-2])

    def test_2d_gaussian_with_two_outliers(self):
        """
        Test that outliers are found in 2D gaussian data when two
        are expected.

        """

        dist1 = stats.norm(loc=100, scale=15)
        x1 = np.reshape(dist1.rvs(size=10000, random_state=1), (1000, 10))
        dist2 = stats.norm(loc=100, scale=3)
        x2 = np.reshape(dist2.rvs(size=1000, random_state=1), (100, 10))
        x2[0][0] = -50
        x2[-1][-1] = 150
        x2_outliers = outliers.adjusted_boxplot(x1, x2)
        self.assertEqual(x2_outliers.shape, x2.shape)
        self.assertEqual(np.sum(x2_outliers.flatten()), 2)
        self.assertTrue(x2_outliers[0][0])
        self.assertTrue(x2_outliers[-1][-1])

    def test_1d_beta_with_three_outliers(self):
        """
        Test that outliers are found in 1D beta distributed data when
        three are expected.

        """

        dist1 = stats.beta(a=2, b=5)
        x1 = dist1.rvs(size=1000, random_state=1)
        dist2 = stats.beta(a=2, b=5)
        x2 = dist2.rvs(size=50, random_state=1)
        x2 = np.concatenate([x2, [0, -0.1, 1, 1.1]])
        x2_outliers = outliers.adjusted_boxplot(x1, x2)
        self.assertEqual(x2_outliers.shape, x2.shape)
        self.assertEqual(np.sum(x2_outliers), 3)
        self.assertTrue(x2_outliers[-1])
        self.assertTrue(x2_outliers[-2])
        self.assertTrue(x2_outliers[-3])

    def test_2d_beta_with_two_outliers(self):
        """
        Test that outliers are found in 2D beta distributed data when
        two are expected.

        """

        dist1 = stats.beta(a=2, b=5)
        x1 = np.reshape(dist1.rvs(size=10000, random_state=1), (1000, 10))
        dist2 = stats.beta(a=2, b=5)
        x2 = np.reshape(dist2.rvs(size=500, random_state=1), (50, 10))
        x2[0][0] = 1
        x2[-1][-1] = -0.1
        x2_outliers = outliers.adjusted_boxplot(x1, x2)
        self.assertEqual(x2_outliers.shape, x2.shape)
        self.assertEqual(np.sum(x2_outliers.flatten()), 2)
        self.assertTrue(x2_outliers[0][0])
        self.assertTrue(x2_outliers[-1][-1])

    def test_1d_chi2_with_three_outliers(self):
        """
        Test that outliers are found in 1D Chi-square distributed data
        when three are expected.

        """

        dist1 = stats.chi2(df=2)
        x1 = dist1.rvs(size=1000, random_state=1)
        dist2 = stats.chi2(df=2)
        x2 = dist2.rvs(size=50, random_state=1)
        x2 = np.concatenate([x2, [-1, 14, 15]])
        x2_outliers = outliers.adjusted_boxplot(x1, x2)
        self.assertEqual(x2_outliers.shape, x2.shape)
        self.assertEqual(np.sum(x2_outliers), 3)
        self.assertTrue(x2_outliers[-1])
        self.assertTrue(x2_outliers[-2])
        self.assertTrue(x2_outliers[-3])

    def test_2d_chi2_with_two_outliers(self):
        """
        Test that outliers are found in 2D Chi-square distributed data
        when two are expected.

        """

        dist1 = stats.chi2(df=2)
        x1 = np.reshape(dist1.rvs(size=10000, random_state=1), (1000, 10))
        dist2 = stats.chi2(df=2)
        x2 = np.reshape(dist2.rvs(size=500, random_state=1), (50, 10))
        x2[0][0] = 20
        x2[-1][-1] = -1
        x2_outliers = outliers.adjusted_boxplot(x1, x2)
        self.assertEqual(x2_outliers.shape, x2.shape)
        self.assertEqual(np.sum(x2_outliers.flatten()), 2)
        self.assertTrue(x2_outliers[0][0])
        self.assertTrue(x2_outliers[-1][-1])


class RandomCutTestCase(unittest.TestCase):
    """
    Tests for outliers.random_cut

    """

    def test_1d_array_raises_value_error(self):
        """
        Test that a 1D x1 array raises a ValueError.

        """

        x1 = [1, 2, 3, 4, 5]
        x2 = x1
        with self.assertRaises(ValueError):
            outliers.random_cut(x1, x2)

    def test_3d_array_raises_value_error(self):
        """
        Test that a 3D x1 array raises a ValueError.

        """

        x1 = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        x2 = x1
        with self.assertRaises(ValueError):
            outliers.random_cut(x1, x2)

    def test_unequal_array_columns_raise_value_error(self):
        """
        Test that ValueError is raised when x1.shape[1] != x2.shape[1]

        """

        x1 = [[1, 2, 3], [4, 5, 6]]
        x2 = [[1, 2], [3, 4], [5, 6]]
        with self.assertRaises(ValueError):
            outliers.random_cut(x1, x2)

    def test_n_trees_less_than_one_raises_value_error(self):
        """
        Test that ValueError is raised when n_trees < 1.

        """

        x1 = [[1, 2], [3, 4]]
        x2 = x1
        with self.assertRaises(ValueError):
            outliers.random_cut(x1, x2, n_trees=0)

    def test_n_trees_float_raises_value_error(self):
        """
        Test that ValueError is raised when n_trees is a float.

        """

        x1 = [[1, 2], [3, 4]]
        x2 = x1
        with self.assertRaises(ValueError):
            outliers.random_cut(x1, x2, n_trees=100.1)

    def test_tree_size_less_than_one_raises_value_error(self):
        """
        Test that ValueError is raised when tree_size < 1.

        """

        x1 = [[1, 2], [3, 4]]
        x2 = x1
        with self.assertRaises(ValueError):
            outliers.random_cut(x1, x2, tree_size=0)

    def test_tree_size_float_raises_value_error(self):
        """
        Test that ValueError is raised when tree_size is a float.

        """

        x1 = [[1, 2], [3, 4]]
        x2 = x1
        with self.assertRaises(ValueError):
            outliers.random_cut(x1, x2, tree_size=256.1)

    def test_tree_size_greater_than_x1_rows_raises_value_error(self):
        """
        Test that ValueError is raised when tree_size > len(x1).

        """

        x1 = [[1, 2], [3, 4]]
        x2 = x1
        with self.assertRaises(ValueError):
            outliers.random_cut(x1, x2, tree_size=256)

    def test_k_less_than_or_equal_0_raises_value_error(self):
        """
        Test that ValueError is raised when k is less than or equal to 0.

        """

        x1 = [[1, 2], [3, 4]]
        x2 = x1
        with self.assertRaises(ValueError):
            outliers.random_cut(x1, x2, k=0)

    def test_multivariate_gaussian_with_six_outliers(self):
        """
        Test that outliers are found in 2D normally distributed
        multivariate data when six are expected.

        """

        np.random.seed(1)
        dist = stats.multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
        x1 = np.reshape(dist.rvs(size=4500, random_state=1), (3000, 3))
        x2 = np.reshape(dist.rvs(size=150, random_state=1), (100, 3))
        x2_outliers = outliers.random_cut(x1, x2)
        self.assertEqual(x2_outliers.shape[0], x2.shape[0])
        self.assertTrue(np.array_equal(np.where(x2_outliers)[0],
                                       np.array([22, 56, 57, 64, 84, 85])))

    def test_dirichlet_with_nine_outliers(self):
        """
        Test that outliers are found in 2D Dirichlet distributed
        data when nine are expected.

        """

        np.random.seed(1)
        dist = stats.dirichlet(alpha=(1, 2, 3), seed=1)
        x1 = np.reshape(dist.rvs(size=3000, random_state=1), (3000, 3))
        x2 = np.reshape(dist.rvs(size=100, random_state=1), (100, 3))
        x2_outliers = outliers.random_cut(x1, x2)
        self.assertEqual(x2_outliers.shape[0], x2.shape[0])
        self.assertTrue(np.array_equal(np.where(x2_outliers)[0],
                                       np.array([1, 16, 19, 41, 48, 59, 65, 85, 87])))

    def test_wishart_with_nine_outliers(self):
        """
        Test that outliers are found in 2D Wishart distributed
        data when nine are expected.

        """

        np.random.seed(1)
        dist = stats.wishart(df=2, scale=2, seed=1)
        x1 = np.reshape(dist.rvs(size=9000, random_state=1), (3000, 3))
        x2 = np.reshape(dist.rvs(size=300, random_state=1), (100, 3))
        x2_outliers = outliers.random_cut(x1, x2)
        self.assertEqual(x2_outliers.shape[0], x2.shape[0])
        self.assertTrue(np.array_equal(np.where(x2_outliers)[0],
                                       np.array([13, 36, 50, 57, 66, 70, 83, 84, 91, 99])))

    def test_beta_with_three_outliers(self):
        """
        Test that outliers are found in 2D beta distributed data when
        three are expected.

        """

        np.random.seed(1)
        dist1 = stats.beta(a=2, b=5)
        x1 = np.reshape(dist1.rvs(size=10000, random_state=1), (1000, 10))
        dist2 = stats.beta(a=2, b=5)
        x2 = np.reshape(dist2.rvs(size=500, random_state=1), (50, 10))
        x2[0][0] = 1
        x2[-1][-1] = -0.1
        x2_outliers = outliers.random_cut(x1, x2)
        self.assertEqual(x2_outliers.ndim, 1)
        self.assertEqual(x2_outliers.shape[0], x2.shape[0])
        self.assertEqual(np.sum(x2_outliers), 3)
        self.assertTrue(x2_outliers[0])
        self.assertTrue(x2_outliers[-1])

    def test_chi2_with_four_outliers(self):
        """
        Test that outliers are found in 2D Chi-square distributed data
        when four are expected.

        """

        np.random.seed(1)
        dist1 = stats.chi2(df=2)
        x1 = np.reshape(dist1.rvs(size=10000, random_state=1), (1000, 10))
        dist2 = stats.chi2(df=2)
        x2 = np.reshape(dist2.rvs(size=500, random_state=1), (50, 10))
        x2[0][0] = 20
        x2[-1][-1] = -1
        x2_outliers = outliers.random_cut(x1, x2)
        self.assertEqual(x2_outliers.ndim, 1)
        self.assertEqual(x2_outliers.shape[0], x2.shape[0])
        self.assertEqual(np.sum(x2_outliers), 4)
        self.assertTrue(x2_outliers[0])
        self.assertTrue(x2_outliers[-1])


class IsNumericTestCase(unittest.TestCase):
    """
    Tests for outliers.is_numeric

    """

    def test_empty_array_raises_value_error(self):
        """
        Test that is_numeric raises ValueError when called with
        an empty array.

        """

        with self.assertRaises(ValueError):
            outliers.is_numeric([[]])

    def test_1d_array_raises_value_error(self):
        """
        Test that is_numeric raises ValueError when called with
        a 1D array.

        """

        with self.assertRaises(ValueError):
            outliers.is_numeric([1, 2, 3, 4])

    def test_3d_array_raises_value_error(self):
        """
        Test that is_numeric raises ValueError when called with
        a 3D array.

        """

        with self.assertRaises(ValueError):
            outliers.is_numeric([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    def test_frac_less_than_0_raises_value_error(self):
        """
        Test that is_numeric raises ValueError when called with
        frac < 0.

        """

        with self.assertRaises(ValueError):
            outliers.is_numeric([[1, 2]], frac=-1)

    def test_frac_greater_than_1_raises_value_error(self):
        """
        Test that is_numeric raises ValueError when called with
        frac > 1.

        """

        with self.assertRaises(ValueError):
            outliers.is_numeric([[1, 2]], frac=1.1)

    def test_default_parameters_correct_classification(self):
        """
        Test that is_numeric correctly classifies columns in a 2D array
        with default parameters.

        """

        x = np.array([[0, .23, 1, 1],
                      [1, .04, 1, 2],
                      [1, .89, 1, 3],
                      [0, .15, 0, 4],
                      [1, .23, 0, 5],
                      [1, .76, 0, 6],
                      [1, .12, 1, 7],
                      [0, .09, 1, 8],
                      [1, .13, 1, 9],
                      [1, .44, 0, 0]])

        x_numeric = outliers.is_numeric(x)
        self.assertEqual(x_numeric.ndim, 1)
        self.assertEqual(x_numeric.shape[0], x.shape[1])
        self.assertTrue(np.array_equal(x_numeric,
                                       np.array([True, True, True, True])))

    def test_frac_20_correct_classification(self):
        """
        Test that is_numeric correctly classifies columns in a 2D array
        with frac=.2

        """

        x = np.array([[0, .23, 1, 1],
                      [1, .04, 1, 2],
                      [1, .89, 1, 3],
                      [0, .15, 0, 4],
                      [1, .23, 0, 5],
                      [1, .76, 0, 6],
                      [1, .12, 1, 7],
                      [0, .09, 1, 8],
                      [1, .13, 1, 9],
                      [1, .44, 0, 0]])

        x_numeric = outliers.is_numeric(x, frac=.2)
        self.assertEqual(x_numeric.ndim, 1)
        self.assertEqual(x_numeric.shape[0], x.shape[1])
        self.assertTrue(np.array_equal(x_numeric,
                                       np.array([False, True, False, True])))


class LocateTestCase(unittest.TestCase):
    """
    Tests for outliers.locate

    """

    def test_gaussian_with_two_outliers(self):
        """
        Test that outliers are found in 2D gaussian data when two
        are expected.

        """

        np.random.seed(1)
        dist1 = stats.norm(loc=100, scale=15)
        x1 = np.reshape(dist1.rvs(size=10000, random_state=1), (1000, 10))
        dist2 = stats.norm(loc=100, scale=3)
        x2 = np.reshape(dist2.rvs(size=1000, random_state=1), (100, 10))
        x2[0][0] = -50
        x2[-1][-1] = 150
        x2_outliers = outliers.locate(x1, x2)
        self.assertEqual(x2_outliers.ndim, 1)
        self.assertEqual(x2_outliers.shape[0], x2.shape[0])
        self.assertEqual(np.sum(x2_outliers), 2)
        self.assertTrue(x2_outliers[0])
        self.assertTrue(x2_outliers[-1])

    def test_beta_with_two_outliers(self):
        """
        Test that outliers are found in 2D beta distributed data when
        two are expected.

        """

        np.random.seed(1)
        dist1 = stats.beta(a=2, b=5)
        x1 = np.reshape(dist1.rvs(size=10000, random_state=1), (1000, 10))
        dist2 = stats.beta(a=2, b=5)
        x2 = np.reshape(dist2.rvs(size=500, random_state=1), (50, 10))
        x2[0][0] = 1
        x2[-1][-1] = -0.1
        x2_outliers = outliers.locate(x1, x2)
        self.assertEqual(x2_outliers.ndim, 1)
        self.assertEqual(x2_outliers.shape[0], x2.shape[0])
        self.assertEqual(np.sum(x2_outliers), 3)
        self.assertTrue(x2_outliers[0])
        self.assertTrue(x2_outliers[-1])

    def test_chi2_with_two_outliers(self):
        """
        Test that outliers are found in 2D Chi-square distributed data
        when two are expected.

        """

        np.random.seed(1)
        dist1 = stats.chi2(df=2)
        x1 = np.reshape(dist1.rvs(size=10000, random_state=1), (1000, 10))
        dist2 = stats.chi2(df=2)
        x2 = np.reshape(dist2.rvs(size=500, random_state=1), (50, 10))
        x2[0][0] = 20
        x2[-1][-1] = -1
        x2_outliers = outliers.locate(x1, x2)
        self.assertEqual(x2_outliers.ndim, 1)
        self.assertEqual(x2_outliers.shape[0], x2.shape[0])
        self.assertEqual(np.sum(x2_outliers), 4)
        self.assertTrue(x2_outliers[0])
        self.assertTrue(x2_outliers[-1])


class ScoreTestCase(unittest.TestCase):
    """
    Tests for outliers.score

    """

    def test_perfect_classifier(self):
        """
        Test with a model that always classifies correctly.

        """

        np.random.seed(1)
        dist1 = stats.norm(loc=100, scale=15)
        x1_inputs = np.reshape(dist1.rvs(size=10000, random_state=1), (1000, 10))
        x1_targets = np.random.randint(-1, 1, size=1000)
        dist2 = stats.norm(loc=100, scale=3)
        x2_inputs = np.reshape(dist2.rvs(size=1000, random_state=1), (100, 10))
        x2_targets = np.random.randint(-1, 1, size=100)
        x2_inputs[0][0] = -50
        x2_inputs[-1][-1] = 150
        datasets = util.Datasets(training=util.Dataset(inputs=x1_inputs,
                                                       targets=x1_targets),
                                 validation=util.Dataset(inputs=x2_inputs,
                                                         targets=x2_targets),
                                 columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])

        model = Mock()
        model.predict = Mock(return_value=np.array([x2_targets[0], x2_targets[-1]]))
        scores = outliers.score(model, datasets)
        self.assertTrue(np.isnan(scores.pop('informedness')))
        self.assertDictEqual(scores,
                             {'accuracy': 1.0,
                              'mcc': 0.0,
                              'precision': 1.0,
                              'recall': 1.0,
                              'f1_score': 1.0,
                              'ami': 1.0,
                              'outliers': 2.0})

    def test_useless_classifier(self):
        """
        Test with a model that always classifies incorrectly.

        """

        np.random.seed(1)
        dist1 = stats.norm(loc=100, scale=15)
        x1_inputs = np.reshape(dist1.rvs(size=10000, random_state=1), (1000, 10))
        x1_targets = np.random.randint(-1, 1, size=1000)
        dist2 = stats.norm(loc=100, scale=3)
        x2_inputs = np.reshape(dist2.rvs(size=1000, random_state=1), (100, 10))
        x2_targets = np.random.randint(-1, 1, size=100)
        x2_inputs[0][0] = -50
        x2_inputs[-1][-1] = 150
        datasets = util.Datasets(training=util.Dataset(inputs=x1_inputs,
                                                       targets=x1_targets),
                                 validation=util.Dataset(inputs=x2_inputs,
                                                         targets=x2_targets),
                                 columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])

        model = Mock()
        model.predict = Mock(return_value=np.array([2, 3]))
        scores = outliers.score(model, datasets)
        self.assertDictEqual(scores,
                             {'accuracy': 0.0,
                              'informedness': -np.inf,
                              'mcc': 0.0,
                              'precision': 0.0,
                              'recall': 0.0,
                              'f1_score': 0.0,
                              'ami': 0.0,
                              'outliers': 2.0})

    def test_half_correct_classifier(self):
        """
        Test with a model that classifies correctly 50% of the time.

        """

        np.random.seed(1)
        dist1 = stats.norm(loc=100, scale=15)
        x1_inputs = np.reshape(dist1.rvs(size=10000, random_state=1), (1000, 10))
        x1_targets = np.random.randint(-1, 1, size=1000)
        dist2 = stats.norm(loc=100, scale=3)
        x2_inputs = np.reshape(dist2.rvs(size=1000, random_state=1), (100, 10))
        x2_targets = np.random.randint(-1, 1, size=100)
        x2_inputs[0][0] = -50
        x2_inputs[-1][-1] = 150
        datasets = util.Datasets(training=util.Dataset(inputs=x1_inputs,
                                                       targets=x1_targets),
                                 validation=util.Dataset(inputs=x2_inputs,
                                                         targets=x2_targets),
                                 columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])

        model = Mock()
        model.predict = Mock(return_value=np.array([x2_targets[0], 1]))
        scores = outliers.score(model, datasets)
        self.assertTrue(np.isnan(scores.pop('dor')))
        self.assertDictEqual(scores,
                             {'accuracy': 0.5,
                              'informedness': -0.5,
                              'mcc': 0.0,
                              'precision': 0.0,
                              'recall': 0.0,
                              'f1_score': 0.0,
                              'ami': 0.0,
                              'sensitivity': 0.0,
                              'specificity': 0.5,
                              'lr_plus': 0.0,
                              'lr_minus': 2.0,
                              'outliers': 2.0})

    def test_datasets_with_zero_outliers(self):
        """
        Test with datasets that have zero outliers.

        """

        np.random.seed(1)
        dist = stats.uniform(loc=100, scale=15)
        x1_inputs = np.reshape(dist.rvs(size=10000, random_state=1), (1000, 10))
        x1_targets = np.repeat([0, 1], 500)
        np.random.shuffle(x1_targets)
        x2_inputs = np.reshape(dist.rvs(size=1000, random_state=1), (100, 10))
        x2_targets = np.repeat([0, 1], 50)
        np.random.shuffle(x2_targets)
        datasets = util.Datasets(training=util.Dataset(inputs=x1_inputs,
                                                       targets=x1_targets),
                                 validation=util.Dataset(inputs=x2_inputs,
                                                         targets=x2_targets),
                                 columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])

        model = Mock()
        scores = outliers.score(model, datasets)
        self.assertDictEqual(scores, dict())
