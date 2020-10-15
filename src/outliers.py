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
from scipy import stats
from scipy.spatial import distance
import pandas as pd
import sklearn
from sklearn import cluster
import kneed
import rrcf
from statsmodels.stats.stattools import medcouple

import scoring


def score(model, datasets, alpha=.003, method='random_forest'):  # pylint: disable=C0103
    """
    Score model on only the outliers in a dataset.

    Args:
      model: A trained instance of a scikit-learn estimator.
      datasets: An instance of Datasets.
      alpha: Significance level to use for identifying an outlier.
             (Default=.003)

    Returns:
      A scores dict returned by `score_model`.

    """

    outliers = locate(datasets.training.inputs, datasets.validation.inputs,
                      method=method,
                      alpha=alpha)

    outlier_count = np.sum(outliers)
    if outlier_count == 0:
        logger = logging.getLogger(__name__)
        logger.warning('No outliers found.')
        return dict()

    scores = scoring.score_model(model,
                                 datasets.validation.inputs[outliers],
                                 datasets.validation.targets[outliers])

    scores['outliers'] = float(outlier_count)
    scores['alpha'] = alpha

    return scores


def locate(x1, x2, method='random_forest', alpha=.003):
    combined_datasets = np.concatenate([x1, x2])
    numeric_columns = is_numeric(combined_datasets)
    univariate_outliers = adjusted_boxplot(x1, x2)
    univariate_outliers = np.logical_and(univariate_outliers,
                                         np.tile(numeric_columns, (univariate_outliers.shape[0], 1)))

    outliers = np.any(univariate_outliers, axis=1)
    if method == 'mahalanobis':
        outliers += mahalanobis_distance(x1, x2, alpha=alpha)

    elif method == 'random_forest':
        outliers += random_cut_forest(x1, x2)

    elif method == 'clustering':
        outliers += spatial_clustering(x1, x2)

    else:
        raise ValueError(f'`{method}` not a recognized method.')

    return outliers


def is_numeric(x, frac=.05):
    numeric_columns = np.any(x.T != x.T.astype(np.int), axis=1)
    max_categories = x.shape[0] * frac
    numeric_columns += np.array([len(np.unique(x)) > max_categories for x in x.T])

    return numeric_columns


def adjusted_boxplot(x1, x2):
    """
    Reference: https://d-scholarship.pitt.edu/7948/1/Seo.pdf

    """

    q1 = np.quantile(x1, .25, axis=0)
    q3 = np.quantile(x1, .75, axis=0)
    iqr = q3 - q1
    mc = medcouple(x1, axis=0)
    lower_fence = np.zeros(mc.shape)
    upper_fence = np.zeros(mc.shape)
    np.copyto(lower_fence,
              q1 - 1.5 * np.exp(-3.5 * mc) * iqr,
              where=mc >= 0)

    np.copyto(lower_fence,
              q1 - 1.5 * np.exp(-4 * mc) * iqr,
              where=mc < 0)

    np.copyto(upper_fence,
              q3 + 1.5 * np.exp(4 * mc) * iqr,
              where=mc >= 0)

    np.copyto(upper_fence,
              q3 + 1.5 * np.exp(3.5 * mc) * iqr,
              where=mc < 0)

    return (x2 < lower_fence) + (x2 > upper_fence)


def mahalanobis_distance(x1, x2, alpha=.003):  # pylint: disable=C0103
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
      alpha: Significance level to use for identifying an outlier.
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

    p_values = 1 - stats.chi2.cdf(distances, len(x2[0]) - 1)

    return p_values < alpha


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

    Reference: https://ieeexplore.ieee.org/document/5961514

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


def random_cut_forest(x1, x2, n_trees=100, tree_size=256):  # pylint: disable=C0103
    """
    Find outliers in a multivariate system using Robust Random Cut Forest.

    First construct a forest of random cut trees from x1 and calculate the
    mean codisp for each sample in x1 for each tree that it is in. Then insert
    each sample from x2 into each tree one by one and calculate the mean codisp
    for each sample. Samples with codisp greater than the 75th percentile of
    mean x1 codisps + IQR * 1.5 are considered to be outliers.

    Reference: http://proceedings.mlr.press/v48/guha16.pdf

    Args:
      x1: n x m array to use as the basis for the forest..
      x2: k x m array of samples to test for outliers.

    Returns:
      k x 1 boolean array where True elements correspond to outliers in x2

    """

    forest = []
    while len(forest) < n_trees:
        ixs = np.random.choice(x1.shape[0], size=(x1.shape[0] // tree_size, tree_size),
                               replace=False)

        trees = [rrcf.RCTree(x1[ix], index_labels=ix) for ix in ixs]
        forest.extend(trees)

    x1_mean_codisp = pd.Series(0.0, index=np.arange(x1.shape[0]))
    index = np.zeros(x1.shape[0])
    for tree in forest:
        codisp = pd.Series({leaf: tree.codisp(leaf) for leaf in tree.leaves})
        x1_mean_codisp[codisp.index] += codisp
        np.add.at(index, codisp.index.values, 1)

    x1_mean_codisp /= index
    x2_mean_codisp = np.zeros(x2.shape[0])
    for sample_index, sample in enumerate(x2):
        sample_mean_codisp = 0
        for tree in forest:
            tree.insert_point(sample, index='sample')
            sample_mean_codisp += tree.codisp('sample')
            tree.forget_point('sample')

        sample_mean_codisp /= len(forest)
        x2_mean_codisp[sample_index] = sample_mean_codisp

    iqr = stats.iqr(x1_mean_codisp)
    outliers = x2_mean_codisp > np.quantile(x1_mean_codisp, 0.75) + 1.5 * iqr

    return outliers
