"""
A library for locating and scoring outliers in univariate and multivariate
systems with unknown distributions and highly correlated features.

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
import rrcf
from statsmodels.stats.stattools import medcouple

import scoring


def score(model, datasets):  # pylint: disable=C0103
    """
    Score model on only the outliers in a dataset.

    Args:
      model: A trained instance of a scikit-learn estimator.
      datasets: An instance of Datasets.

    Returns:
      A scores dict returned by `score_model`.

    """

    outliers = locate(datasets.training.inputs, datasets.validation.inputs)
    outlier_count = np.sum(outliers)
    if outlier_count == 0:
        logger = logging.getLogger(__name__)
        logger.warning('No outliers found.')
        return dict()

    scores = scoring.score_model(model,
                                 datasets.validation.inputs[outliers],
                                 datasets.validation.targets[outliers])

    # All values in `scores` need to be a float so that the string formatter in
    # gen_model.bind_model_metadata() can handle them correctly.
    scores['outliers'] = float(outlier_count)

    return scores


def locate(x1, x2):
    """
    Locate outlier rows in array x2 with respect to array x1 using univariate
    and multivariate methods for outlier detection.

    Args:
      x1: n x m array to use as the basis for identifying outlier rows.
      x2: k x m array to test for outlier rows.

    Returns:
      k x 1 boolean array where True elements indicate outlier rows in x2.

    """

    combined_datasets = np.concatenate([x1, x2])
    numeric_columns = is_numeric(combined_datasets)
    univariate_outliers = adjusted_boxplot(x1, x2)

    # Only consider univariate outliers in numeric columns.
    univariate_outliers = np.logical_and(univariate_outliers,
                                         np.tile(numeric_columns, (univariate_outliers.shape[0], 1)))

    # If any column in a row is an outlier, consider the entire row an outlier.
    outliers = np.any(univariate_outliers, axis=1)
    outliers += random_cut_forest(x1, x2)

    return outliers


def is_numeric(x, frac=.05):
    """
    Test if the columns in array x are numeric using the following heuristic:
    - A column is numeric if it contains real numbers.
    - A column is numeric if it has more unique values than len(col) * frac.
    - Otherwise, the column is non-numeric.

    Args:
      x: n x m array to check for numeric columns.
      frac: Value to use for frac in the numeric test. (Default=.05)

    Returns:
      k x 1 boolean array where True elements indicate numeric columns.

    """

    # For each column, test if the column contains real numbers.
    numeric_columns = np.any(x.T != x.T.astype(np.int), axis=1)

    # For each column, test if the column has more unique values than
    # len(col) * frac.
    max_categories = x.shape[0] * frac
    numeric_columns += np.array([len(np.unique(x)) > max_categories for x in x.T])

    return numeric_columns


def adjusted_boxplot(x1, x2):  # pylint: disable=C0103
    """
    Locate outliers in a univariate system using the adjusted boxplot method.

    Tests each column of each row for outliers.

    This method is non-parametric and robust with respect to outliers
    (i.e. extreme outliers do not mask less extreme outliers) and the
    distribution of the data (i.e. it does not fail on skewed and non-Gaussian
    distributions).

    Args:
      x1: n x m array to use as the basis for identifying outliers.
      x2: k x m array to test for outliers.

    Returns:
      k x m boolean array where True elements indicate outliers.

    References:
      https://www.researchgate.net/profile/Mia_Hubert/publication/4749681_An_Adjusted_Boxplot_for_Skewed_Distributions/links/59e35504458515393d5b8743/An-Adjusted-Boxplot-for-Skewed-Distributions.pdf
      https://d-scholarship.pitt.edu/7948/1/Seo.pdf

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


def random_cut_forest(x1, x2, n_trees=100, tree_size=256):  # pylint: disable=C0103
    """
    Find outliers in a multivariate system using Robust Random Cut Forest.

    This method, like adjusted_boxplot, is robust with respect to outliers and
    the data distribution. Although it is parametric, it is not very sensitive
    to the values of the parameters, and the default values should work in most
    cases. However, it is very inefficient, and will likely be too slow to be
    practical for large datasets.

    Unlike adjusted_boxplot, it tests rows as a whole to check if they are
    outliers, not individual columns.

    Args:
      x1: n x m array to use as the basis for the forest.
      x2: k x m array of samples to test for outliers.

    Returns:
      k x 1 boolean array where True elements correspond to outliers in x2.

    References:
      http://proceedings.mlr.press/v48/guha16.pdf
      https://joss.theoj.org/papers/10.21105/joss.01336

    """

    # Construct a forest of random cut trees from x1 and calculate the mean
    # codisp for each row in x1 for each tree that it is in.
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

    # Insert each row from x2 into each tree one by one and calculate
    # the mean codisp for each row.
    x2_mean_codisp = np.zeros(x2.shape[0])
    for sample_index, sample in enumerate(x2):
        sample_mean_codisp = 0
        for tree in forest:
            tree.insert_point(sample, index='sample')
            sample_mean_codisp += tree.codisp('sample')
            tree.forget_point('sample')

        sample_mean_codisp /= len(forest)
        x2_mean_codisp[sample_index] = sample_mean_codisp

    # Rows with codisp greater than the 75th percentile of
    # mean x1 codisps + IQR * 1.5 are considered to be outliers.
    iqr = stats.iqr(x1_mean_codisp)
    outliers = x2_mean_codisp > np.quantile(x1_mean_codisp, 0.75) + 1.5 * iqr

    return outliers
