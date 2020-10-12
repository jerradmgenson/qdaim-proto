import math

import numpy as np
from scipy.spatial import distance
from scipy.stats import chi2
import sklearn
from sklearn.pipeline import Pipeline
from sklearn import cluster
import kneed

import scoring


def score(model, datasets):
    """
    Score model on only the outliers in a dataset.

    Args:
      model: A trained instance of a scikit-learn estimator.
      datasets: An instance of Datasets.

    Returns:
      A scores dict returned by `score_model`.

    """

    outliers = mahalanobis_distance(datasets.training.inputs,
                                    datasets.validation.inputs)

    outlier_count = np.sum(outliers)
    if outlier_count > 0:
        scores = scoring.score_model(model,
                                     datasets.validation.inputs[outliers],
                                     datasets.validation.targets[outliers])

        return scores, outlier_count, None

    if hasattr(model, 'steps') and len(model.steps) >= 2:
        # Assume that `model` is a pipeline and the first step is preprocessing.
        outliers, outlier_model = spatial_clustering(datasets.training.inputs,
                                                     datasets.validation.inputs,
                                                     preprocessor=model.steps[0][1])

    else:
        outliers, outlier_model = spatial_clustering(datasets.training.inputs,
                                                     datasets.validation.inputs)

    outlier_count = np.sum(outliers)
    scores = scoring.score_model(model,
                                 datasets.validation.inputs[outliers],
                                 datasets.validation.targets[outliers])

    return scores, outlier_count, outlier_model


def spatial_clustering(a1, a2=None, preprocessor=None):
    """
    Find outliers in a multivariate system using spatial clustering.

    Finds outliers using a spatial clustering approach. First, it uses
    density-based spatial clustering of applications with noise (DBSCAN) to find
    core samples in a1. Then it calculates the Euclidean distance from each
    sample in a2 to the nearest core sample in a1 and checks if the distance is
    greater than the eps parameter chosen for the DBSCAN model. It considers any
    such samples to be outliers.

    This is a robust method of locating outliers - it does not require that the
    data conform to any particular distribution, it does not require that the
    inverse covariance matrix be defined, and it is insensitive to noise.
    However, as a heuristical method, it is less statistically rigorous than
    other methods and is also comparatively expensive in terms of space and time
    complexity.

    Args:
      a1: n x m array to use as the basis for the core samples.
      a2: k x m array of samples to test for outliers. (Default=a1)
      preprocessor: A scikit-learn Transformer object to use in a pipeline when
                    constructing the DBSCAN model. (Default=None)

    Returns:
      A 2-tuple of
      (k x 1 boolean array where True elements correspond to outliers in a2,
       DBSCAN model that was used to produce core samples)

    """

    if a2 is None:
        a2 = a1

    nearest_neighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=2)
    a1_distances, _ = nearest_neighbors.fit(a1).kneighbors(a1)
    a1_distances = np.sort(a1_distances, axis=0)
    a1_distances = a1_distances[:,1]
    kneedle = kneed.KneeLocator(range(len(a1_distances)),
                                a1_distances,
                                S=1.0,
                                curve='convex',
                                direction='increasing')

    eps = kneedle.knee_y
    pipeline_steps = []
    if preprocessor:
        pipeline_steps.append(('preprocessor', preprocessor))

    pipeline_steps.append(('model', cluster.DBSCAN(eps=eps)))
    pipeline = Pipeline(steps=pipeline_steps)
    parameter_grid = dict(
        model__min_samples=range(a1.shape[1] + 1, 2 * a1.shape[1] + 1)
    )

    grid_estimator = sklearn.model_selection.GridSearchCV(pipeline,
                                                          parameter_grid,
                                                          scoring=scoring.mean_silhouette_coefficient)

    grid_estimator.fit(a1)
    model = grid_estimator.best_estimator_.steps[-1][1]
    core_samples = model.components_
    if preprocessor:
        a2 = grid_estimator.best_estimator_.steps[0][1].transform(a2)

    distances = distance.cdist(a2, core_samples)
    min_distances = distances.min(axis=1)

    return min_distances > eps, grid_estimator.best_estimator_


def mahalanobis_distance(a1, a2=None, p=.003):
    """
    Find outliers in a multivariate system using the Mahalanobis distance.

    Calculates the Mahalanobis distance from each sample in a2 to a1, then
    performs a Chi-Squared test on each distance to determine the p-value.
    It considers any samples with a p-value less than the given significance
    level to be outliers.

    This is a fast and statistically rigorous method of finding outliers in a
    multivariate system where the features may be correlated. However, it
    requires that the data conform to an elliptical distribution and that the
    inverse covariance matrix is defined for a1. Otherwise, it will fail to find
    the outliers.

    Args:
      a1: n x m array to use as the distribution for calculating the
          Mahalanobis distance.
      a2: k x m array of samples to test for outliers. (Default=a1)
      p: Significance level to use for determining if a sample is an outlier.
         (Default=.003)

    Returns:
      k x 1 boolean array where True elements correspond to outliers in a2.

    """

    if a2 is None:
        a2 = a1

    try:
        distances = distance.cdist(a2, a1, metric='mahalanobis').diagonal()

    except np.linalg.LinAlgError as linalg_error:
        logger = logging.getLogger(__name__)
        logger.warning(str(linalg_error))
        return np.array([])

    p_values = chi2.cdf(distances, len(a2[0]) - 1)

    return p_values < p
