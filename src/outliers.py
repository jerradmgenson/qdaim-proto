"""
A library for locating outliers in multivariate systems with correlated
features and scoring models based on their performance on outliers.

Copyright 2020 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import logging

import numpy as np
from scipy.stats import chi2
from scipy.spatial import distance

import scoring


def score(model, datasets, p=.003):  # pylint: disable=C0103
    """
    Score model on only the outliers in a dataset.

    Args:
      model: A trained instance of a scikit-learn estimator.
      datasets: An instance of Datasets.
      p: Significance level to use for determining if a sample is an outlier.
         (Default=.003)

    Returns:
      A scores dict returned by `score_model`.

    """

    outliers = mahalanobis_distance(datasets.training.inputs,
                                    datasets.validation.inputs,
                                    p=p)

    outliers_count = np.sum(outliers)
    if outliers_count == 0:
        logger = logging.getLogger(__name__)
        logger.warning('No outliers could be found.')
        return {}

    scores = scoring.score_model(model,
                                 datasets.validation.inputs[outliers],
                                 datasets.validation.targets[outliers])

    scores['outliers'] = float(outliers_count)
    scores['p'] = p

    return scores


def mahalanobis_distance(x1, x2, p=.003):  # pylint: disable=C0103
    """
    Find outliers in a multivariate system using the Mahalanobis distance.

    Calculates the Mahalanobis distance from each sample in x2 to x1,
    then performs a Chi-Squared test on each distance to determine the p-value.
    It considers any samples with a p-value less than the given significance
    level to be outliers.

    This is a fast and statistically rigorous method of finding outliers in a
    multivariate system where the features may be correlated. However, it
    requires that the data conform to an elliptical distribution and that the
    inverse covariance matrix is defined for x1. Otherwise, it will fail to find
    the outliers.

    Args:
      x1: n x m array to use as the distribution for calculating the
          Mahalanobis distance.
      x2: k x m array of samples to test for outliers. (Default=a1)
      p: Significance level to use for determining if a sample is an outlier.
         (Default=.003)

    Returns:
      k x 1 boolean array where True elements correspond to outliers in x2.

    """

    try:
        distances = distance.cdist(x2, x1, metric='mahalanobis').diagonal()

    except np.linalg.LinAlgError as linalg_error:
        logger = logging.getLogger(__name__)
        logger.warning(str(linalg_error))
        return np.array([])

    p_values = chi2.cdf(distances, len(x2[0]) - 1)

    return p_values < p
