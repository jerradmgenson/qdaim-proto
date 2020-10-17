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
        x2 = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
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

    def test_1d_gaussian_with_outliers(self):
        """
        Test that outliers are found in 1D gaussian data when some
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

    def test_2d_gaussian_with_outliers(self):
        """
        Test that outliers are found in 2D gaussian data when some
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

    def test_1d_gaussian_with_no_outliers(self):
        """
        Test that no outliers are found in 1D gaussian data when none
        are expected.

        """

        dist1 = stats.norm(loc=100, scale=15)
        x1 = dist1.rvs(size=1000, random_state=1)
        dist2 = stats.norm(loc=100, scale=3)
        x2 = dist2.rvs(size=100, random_state=1)
        x2_outliers = outliers.adjusted_boxplot(x1, x2)
        self.assertEqual(x2_outliers.shape, x2.shape)
        self.assertFalse(np.any(x2_outliers))

    def test_2d_gaussian_with_no_outliers(self):
        """
        Test that no outliers are found in 2D gaussian data when none
        are expected.

        """

        dist1 = stats.norm(loc=100, scale=15)
        x1 = np.reshape(dist1.rvs(size=10000, random_state=1), (1000, 10))
        dist2 = stats.norm(loc=100, scale=3)
        x2 = np.reshape(dist2.rvs(size=1000, random_state=1), (100, 10))
        x2_outliers = outliers.adjusted_boxplot(x1, x2)
        self.assertEqual(x2_outliers.shape, x2.shape)
        self.assertFalse(np.any(x2_outliers.flatten()))

    def test_1d_beta_with_outliers(self):
        """
        Test that outliers are found in 1D beta distributed data when
        some are expected.

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

    def test_2d_beta_with_outliers(self):
        """
        Test that outliers are found in 2D gaussian data when some
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

    def test_1d_beta_without_outliers(self):
        """
        Test that no outliers are found in 1D beta distributed data
        when none are expected.

        """

        dist1 = stats.beta(a=2, b=5)
        x1 = dist1.rvs(size=1000, random_state=1)
        dist2 = stats.beta(a=2, b=5)
        x2 = dist2.rvs(size=50, random_state=1)
        x2_outliers = outliers.adjusted_boxplot(x1, x2)
        self.assertEqual(x2_outliers.shape, x2.shape)
        self.assertFalse(np.any(x2_outliers))

    def test_2d_beta_without_outliers(self):
        """
        Test that no outliers are found in 2D beta distributed data
        when none are expected.

        """

        dist1 = stats.beta(a=2, b=5)
        x1 = np.reshape(dist1.rvs(size=10000, random_state=1), (1000, 10))
        dist2 = stats.beta(a=2, b=5)
        x2 = np.reshape(dist2.rvs(size=500, random_state=1), (50, 10))
        x2_outliers = outliers.adjusted_boxplot(x1, x2)
        self.assertEqual(x2_outliers.shape, x2.shape)
        self.assertFalse(np.any(x2_outliers.flatten()))

    def test_1d_chi2_with_outliers(self):
        """
        Test that outliers are found in 1D Chi-square distributed data
        when some are expected.

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

    def test_2d_chi2_with_outliers(self):
        """
        Test that outliers are found in 2D Chi-square distributed data
        when some are expected.

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

    def test_1d_chi2_without_outliers(self):
        """
        Test that no outliers are found in 1D Chi-square distributed
        data when none are expected.

        """

        dist1 = stats.chi2(df=2)
        x1 = dist1.rvs(size=1000, random_state=1)
        dist2 = stats.chi2(df=2)
        x2 = dist2.rvs(size=50, random_state=1)
        x2_outliers = outliers.adjusted_boxplot(x1, x2)
        self.assertEqual(x2_outliers.shape, x2.shape)
        self.assertFalse(np.any(x2_outliers))

    def test_2d_chi2_without_outliers(self):
        """
        Test that no outliers are found in 2D Chi-square distributed
        data when none are expected.

        """

        dist1 = stats.chi2(df=2)
        x1 = np.reshape(dist1.rvs(size=10000, random_state=1), (1000, 10))
        dist2 = stats.chi2(df=2)
        x2 = np.reshape(dist2.rvs(size=500, random_state=1), (50, 10))
        x2_outliers = outliers.adjusted_boxplot(x1, x2)
        self.assertEqual(x2_outliers.shape, x2.shape)
        self.assertFalse(np.any(x2_outliers.flatten()))


if __name__ == '__main__':
    unittest.main()
