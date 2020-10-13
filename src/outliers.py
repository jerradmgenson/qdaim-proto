"""
A library for locating outliers in multivariate systems with correlated
features and scoring models based on their performance on outliers.

Copyright 2020 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import math
import logging

import numpy as np
from scipy.stats import chi2
from scipy.spatial import distance
import sklearn
from sklearn import cluster
import kneed

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
    silhouette = None
    if outliers_count == 0:
        outliers, silhouette = spatial_clustering(datasets.training.inputs,
                                                  datasets.validation.inputs)

        import pdb; pdb.set_trace()
        outliers_count = np.sum(outliers)
        if outliers_count == 0:
            logger = logging.getLogger(__name__)
            logger.warning('No outliers found.')
            return dict()

    scores = scoring.score_model(model,
                                 datasets.validation.inputs[outliers],
                                 datasets.validation.targets[outliers])

    scores['outliers'] = float(outliers_count)
    scores['p'] = p
    if silhouette:
        scores['silhouette'] = silhouette

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

    p_values = 1 - chi2.cdf(distances, len(x2[0]) - 1)

    return p_values < p


def spatial_clustering(x1, x2=None):  # pylint: disable=C0103
    """
    Find outliers in a multivariate system using spatial clustering.

    Finds outliers using a spatial clustering approach. First, it uses
    density-based spatial clustering of applications with noise (DBSCAN) to find
    core samples in x1. Then it calculates the Euclidean distance from each
    sample in x2 to the nearest core sample in x1 and checks if the distance is
    greater than the eps parameter chosen for the DBSCAN model. It considers any
    such samples to be outliers.

    This is a robust method of locating outliers - it does not require that the
    data conform to any particular distribution, it does not require that the
    inverse covariance matrix be defined, and it is insensitive to noise.
    However, as a heuristical method, it is less statistically rigorous than
    other methods and is also comparatively expensive in terms of space and time
    complexity.

    Args:
      x1: n x m array to use as the basis for the core samples.
      x2: k x m array of samples to test for outliers. (Default=x1)

    Returns:
      A 2-tuple of
      (k x 1 boolean array where True elements correspond to outliers in x2,
       mean silhouette coefficient of the DBSCAN model)

    """

    if x2 is None:
        x2 = np.copy(x1)

    robust_scaler = sklearn.preprocessing.RobustScaler().fit(x1)
    x1 = robust_scaler.transform(x1)
    x2 = robust_scaler.transform(x2)
    min_samples_range = range(3, 2 * x1.shape[1] + 1)
    parameter_grid = []
    for min_samples in min_samples_range:
        parameter_grid.append(dict(min_samples=[min_samples],
                                   eps=[estimate_eps(x1, min_samples)]))

    grid_estimator = sklearn.model_selection.GridSearchCV(cluster.DBSCAN(),
                                                          parameter_grid,
                                                          scoring=scoring.silhouette_coefficient)

    grid_estimator.fit(x1)
    model = grid_estimator.best_estimator_
    core_samples = model.components_
    distances = distance.cdist(x2, core_samples)
    min_distances = distances.min(axis=1)

    return min_distances > model.eps, grid_estimator.best_score_


def estimate_eps(x, min_samples):  # pylint: disable=C0103
    """
    Estimate the optimal eps value for DBSCAN using the Kneedle algorithm.

    Source: https://ieeexplore.ieee.org/document/5961514

    Args:
      x: 2-dimensional array used to fit the DBSCAN model.
      min_samples: value for the min_samples parameter of the DBSCAN model.

    Returns:
      Estimated optimal value for eps as a float.

    """

    nearest_neighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=min_samples+1)
    distances, _ = nearest_neighbors.fit(x).kneighbors(x)
    distances = np.sort(distances[:,min_samples], axis=0)
    kneedle = kneed.KneeLocator(np.arange(distances.shape[0]),
                                distances,
                                S=1.0,
                                curve='convex',
                                direction='increasing')

    return kneedle.knee_y
