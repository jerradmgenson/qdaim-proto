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


class MahalanobisDistanceTest(unittest.TestCase):
    """
    Testcase for outliers.mahalanobis_distance

    """

    def test_normal_distribution(self):
        """
        Test mahalanobis_distance on normally-distributed data.

        """

        np.random.seed(1)
        actual_outliers = np.array([[100, -100, 100, 100, -100],
                                    [13, 0, -13, 0, 3],
                                    [-5, -10, 7, -10, 5]])

        possible_outliers = np.concatenate([np.random.normal(size=(100, 5)),
                                            actual_outliers])

        gaussian = np.random.normal(size=(10000, 5))
        mahalanobis = outliers.mahalanobis_distance(gaussian, possible_outliers)
        predicted_outliers = possible_outliers[mahalanobis]
        self.assertTrue((predicted_outliers == actual_outliers).all())

    def test_hyperbolic_distribution(self):
        """
        Test mahalanobis_distance on hyperbolically-distributed data.

        """

        np.random.seed(1)
        actual_outliers = np.array([[100, -100, 100, 100, -100],
                                    [13, 0, -13, 0, 3],
                                    [-5, -10, 7, -10, 5]])

        hypsecant = stats.hypsecant()
        possible_outliers = np.concatenate([hypsecant.pdf(np.random.rand(100, 5)),
                                            actual_outliers])

        hyperbolic = hypsecant.pdf(np.random.normal(size=(10000, 5)))
        mahalanobis = outliers.mahalanobis_distance(hyperbolic, possible_outliers)
        predicted_outliers = possible_outliers[mahalanobis]
        self.assertTrue((predicted_outliers == actual_outliers).all())

    def test_logistic_distribution(self):
        """
        Test mahalanobis_distance on logistically-distributed data.

        """

        np.random.seed(1)
        actual_outliers = np.array([[100, -100, 100, 100, -100],
                                    [13, 0, -13, 0, 3],
                                    [-5, -10, 7, -10, 5]])

        logistic = stats.logistic()
        possible_outliers = np.concatenate([logistic.pdf(np.random.rand(100, 5)),
                                            actual_outliers])

        logdata = logistic.pdf(np.random.normal(size=(10000, 5)))
        mahalanobis = outliers.mahalanobis_distance(logdata, possible_outliers)
        predicted_outliers = possible_outliers[mahalanobis]
        self.assertTrue((predicted_outliers == actual_outliers).all())



if __name__ == '__main__':
    unittest.main()
