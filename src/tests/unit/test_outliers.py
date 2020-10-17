"""
Unit tests for outliers.py

Copyright 2020 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import unittest

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
        Test that outliers are found in 2D gaussian data when two
        are expected.

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

    def test_unequal_array_ndim_raises_value_error(self):
        """
        Test that ValueError raised when x1.ndim != x2.ndim.

        """

        x1 = [1, 2, 3, 4, 5, 6]
        x2 = [[1, 2], [3, 4], [5, 6]]
        with self.assertRaises(ValueError):
            outliers.random_cut(x1, x2)

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
                                       np.array([ 1, 16, 19, 41, 48, 59, 65, 85, 87])))

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


if __name__ == '__main__':
    unittest.main()
