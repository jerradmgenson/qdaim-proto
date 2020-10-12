import math

import numpy as np
from scipy.spatial import distance
from scipy.stats import chi2
import sklearn
from sklearn.pipeline import Pipeline
from sklearn import cluster

import scoring


def score_outliers(model, datasets):
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
        outliers, outlier_model = find_outliers(datasets.training.inputs,
                                                datasets.validation.inputs,
                                                preprocessor=model.steps[0][1])

    else:
        outliers, outlier_model = find_outliers(datasets.training.inputs,
                                                datasets.validation.inputs)

    outlier_count = np.sum(outliers)
    scores = scoring.score_model(model,
                                 datasets.validation.inputs[outliers],
                                 datasets.validation.targets[outliers])

    return scores, outlier_count, outlier_model


def find_outliers(x, y, preprocessor=None, p=.001):
    """
    Find outliers in array `y` based on core samples in array `x`.

    Finds outliers using an anomaly detection system approach. First, it uses
    density-based spacial clustering of applications with noise (DBSCAN) to find
    core samples in the training dataset. Then it calculates the Euclidean
    distance from each sample in the testing dataset to the nearest core sample
    in the training dataset. Finally, it performs a Chi-Squared test to determine
    if the distances between the core samples and any of the testing samples is
    statistically significant. It considers any samples it finds to be outliers.

    Args:
      x: An n x m array of training input samples.
      y: A k x m array of testing input samples.
      preprocessor: (Default=None) A scikit-learn Transformer object to use in a
                    pipeline with the DBSCAN model.
      p: (Default=.001) The significance level to use in the Chi-Squared test
         for outliers.

    Returns:
      A k x 1 boolean array where True values correspond to outliers in `y`.

    """

    pipeline_steps = []
    if preprocessor:
        pipeline_steps.append(('preprocessor', preprocessor))

    pipeline_steps.append(('model', cluster.DBSCAN()))
    pipeline = Pipeline(steps=pipeline_steps)
    parameter_grid = dict(
        model__eps=[0.05, 0.1, 0.5, 1, 5],
        model__min_samples=list(range(2, int(math.sqrt(len(x)))))
    )

    grid_estimator = sklearn.model_selection.GridSearchCV(pipeline,
                                                          parameter_grid,
                                                          scoring=scoring.mean_silhouette_coefficient)

    grid_estimator.fit(x)
    model = grid_estimator.best_estimator_.steps[-1][1]
    core_samples = model.components_
    if preprocessor:
        y = preprocessor.transform(y)

    distances = distance.cdist(y, core_samples)
    min_distances = distances.min(axis=1)
    p_values = 1 - chi2.cdf(min_distances, len(y[0]) - 1)

    return p_values < p, model
