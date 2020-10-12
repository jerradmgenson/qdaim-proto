import math

import numpy as np
from scipy.spatial import distance
from scipy.stats import chi2
import sklearn
from sklearn.pipeline import Pipeline
from sklearn import cluster

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

    if hasattr(model, 'steps') and len(model.steps) >= 2:
        # Assume that `model` is a pipeline and the first step is preprocessing.
        outliers, outlier_model = spacial_clustering(datasets.training.inputs,
                                                     datasets.validation.inputs,
                                                     preprocessor=model.steps[0][1])

    else:
        outliers, outlier_model = spacial_clustering(datasets.training.inputs,
                                                     datasets.validation.inputs)

    outlier_count = np.sum(outliers)
    scores = scoring.score_model(model,
                                 datasets.validation.inputs[outliers],
                                 datasets.validation.targets[outliers])

    return scores, outlier_count, outlier_model


def spacial_clustering(a1, a2=None, preprocessor=None):
    """
    Find outliers in a multivariate system using spacial clustering.

    Finds outliers using a spacial clustering approach. First, it uses
    density-based spacial clustering of applications with noise (DBSCAN) to find
    core samples in a1. Then it calculates the Euclidean distance from each
    sample in a2 to the nearest core sample in a1 and checks if the distance is
    greater than the eps parameter chosen for the DBSCAN model. It considers any
    such samples to be outliers.

    Args:
      a1: n x m array to use as the basis for the core samples.
      a2: k x m array of samples to test for outliers. (Default=a1)
      preprocessor: A scikit-learn Transformer object to use in a pipeline when
                    constructing the DBSCAN model. (Default=None)

    Returns:
      A k x 1 boolean array where True elements correspond to outliers in a2.

    """

    if a2 is None:
        a2 = a1

    pipeline_steps = []
    if preprocessor:
        pipeline_steps.append(('preprocessor', preprocessor))

    pipeline_steps.append(('model', cluster.DBSCAN()))
    pipeline = Pipeline(steps=pipeline_steps)
    parameter_grid = dict(
        model__eps=[0.05, 0.1, 0.5, 1, 5, 10, 15],
        model__min_samples=list(range(2, int(math.sqrt(len(a1))), 2))
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

    return min_distances > model.eps, grid_estimator.best_estimator_
